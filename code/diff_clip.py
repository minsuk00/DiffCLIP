import warnings

warnings.filterwarnings(
    "ignore", message=".*`Transformer2DModelOutput` is deprecated and will be removed in version 1.0.0.*"
)
# sync_dist turned off performance
warnings.filterwarnings(
    "ignore",
    ".*sync_dist=True.* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
import torchvision.datasets as datasets

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPModel
from attention_processor import (
    IPAttnProcessor2_0 as IPAttnProcessor,
    AttnProcessor2_0 as AttnProcessor,
)

import os
import json
from PIL import Image
from typing import Any, Literal, Mapping
import wandb
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from time import time
from pprint import pprint
import itertools
from safetensors import safe_open

from utils import collect_features


# Path Configuration
ROOT_DIR = "/home/choi/DiffCLIP"
IP_ADAPTER_PATH = "/scratch/choi/model/IP-Adapter/models/ip-adapter_sd15.bin"
SD_PATH = "/scratch/choi/model/stable-diffusion-v1-5"
CLIP_PATH = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"  # or hf repo name
OUTPUT_PATH = "/scratch/choi/output/DiffCLIP"
DATASET_PATH = "/scratch/choi/dataset/ImageNet100"

config = {
    "minibatch_size": 6,
    "gradient_accum_step": 14,
    # "minibatch_size": 12,
    # "gradient_accum_step": 21,
    "timestep_range": (400, 600),  # 0 for full range
    "clip-resolution": 224,
    "diffusion-resolution": 512,
    # "diffusion-resolution": 224,
    "weight_decay": 1e-2,
    "learning_rate": 5e-5,
    "diffusion_loss_scale": 0.001,
    # "diffusion_loss_scale": 0,
    "ckpt_only_clip_weights": True,
    "check_1nn": False,
    "check_zero_shot": True,
}
config["filename"] = f"diff-loss-{config['diffusion_loss_scale']}"


class MyDataset(Dataset):
    def __init__(self, is_train: bool):
        super().__init__()
        self.mode = "train" if is_train else "val"

        # self.tokenizer = None
        self.image_root_path = f"{DATASET_PATH}/{self.mode}"
        self.data = json.load(
            open(f"{DATASET_PATH}/_img_text_pair_{self.mode}.json")
        )  # list of dict: [{'class_id': 'n03785016','image_file': 'n03785016_5681.JPEG','text': 'a photo of moped'}]

        self.text_embedding_dict_L = torch.load(os.path.join(ROOT_DIR, "IN100_text_embedding_dict_L.pt"))
        self.text_embedding_dict_with_projection_H = torch.load(
            os.path.join(ROOT_DIR, "IN100_text_embedding_dict_with_projection_H.pt")
        )

        self.transform = T.Compose(
            [
                T.Resize(config["diffusion-resolution"], antialias=True),
                T.CenterCrop(config["diffusion-resolution"]),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # T.Normalize([0.5], [0.5]),
            ]
        )
        self.clip_image_processor = CLIPImageProcessor({"shortest_edge": config["clip-resolution"]})

    def __getitem__(self, idx):
        class_id, image_file, caption = self.data[idx].values()

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, class_id, image_file))

        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        text_sd = self.text_embedding_dict_L[class_id]
        text_clip = self.text_embedding_dict_with_projection_H[class_id]

        # tokenize caption (precomputed)

        return {
            "image": image,
            # "text_input_ids": text_input_ids,
            "clip_image": clip_image.squeeze(),
            "text_embeddings_sd": text_sd,
            "text_embeddings_clip": text_clip,
        }

    def __len__(self):
        return len(self.data)


class ImageProjModel(nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class IPAdapter(nn.Module):
    def __init__(self, unet, image_proj_model, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model

        # init adapter modules
        attn_procs = {}
        unet_sd = unet.state_dict()
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # state_dict = torch.load(ckpt_path, map_location="cpu")
        if os.path.splitext(ckpt_path)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        # print(f"Successfully loaded weights from checkpoint {ckpt_path}")


class IN100_DataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.image_root_path = DATASET_PATH
        self.nn_transform = T.Compose(
            [
                T.Resize(256, interpolation=3),
                T.CenterCrop(224),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.zero_shot_transform = T.Compose(
            [
                T.Resize(256, interpolation=3),
                T.CenterCrop(224),
                T.ToImage(),
            ]
        )

    def setup(self, stage=None):
        self.train_dataset = MyDataset(is_train=True)
        self.val_dataset = MyDataset(is_train=False)

        ### 1nn evaluation dataset
        if config["check_1nn"]:
            self.nn_train_dataset = datasets.ImageFolder(self.image_root_path + "/train", transform=self.nn_transform)
            # indices = torch.load("../imagenet_indices.pth")
            # nn_train_dataset = torch.utils.data.Subset(nn_train_dataset, indices)
            # TODO: random indexing? Subset? 1nn take too long
            self.nn_val_dataset = datasets.ImageFolder(self.image_root_path + "/val", transform=self.nn_transform)
            self.train_sampler = torch.utils.data.DistributedSampler(self.nn_train_dataset)

        if config["check_zero_shot"]:
            self.zero_shot_dataset = datasets.ImageFolder(
                self.image_root_path + "/val", transform=self.zero_shot_transform
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=config["minibatch_size"] * config["gradient_accum_step"],
            drop_last=True,
            num_workers=4,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            # shuffle=True,
            batch_size=config["minibatch_size"] * config["gradient_accum_step"],
            drop_last=True,
            num_workers=4,
        )
        return val_loader

    def nn_train_dataloader(self):
        return DataLoader(
            self.nn_train_dataset,
            batch_size=512,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            sampler=self.train_sampler,
        )

    def nn_val_dataloader(self):
        return DataLoader(
            self.nn_val_dataset,
            batch_size=512,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

    def zero_shot_dataloader(self):
        return DataLoader(
            self.zero_shot_dataset,
            batch_size=512,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )


class DiffCLIP(L.LightningModule):
    def __init__(self, datamodule):
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule"])
        self.datamodule = datamodule
        self.automatic_optimization = False

        self.noise_scheduler = DDPMScheduler.from_pretrained(SD_PATH, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(SD_PATH, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(SD_PATH, subfolder="unet")
        self.clip_model = CLIPModel.from_pretrained(CLIP_PATH)

        self.vae.requires_grad_(False)
        unet.requires_grad_(False)
        # self.clip_model.requires_grad_(True)
        del self.clip_model.text_model
        del self.clip_model.text_projection
        self.clip_model.train()
        # self.clip_model.vision_model.requires_grad_(True)
        # self.clip_model.visual_projection.requires_grad_(True)
        # self.clip_model.logit_scale.requires_grad_(True)

        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=self.clip_model.config.projection_dim,
            clip_extra_context_tokens=4,
        )
        self.ip_adapter = IPAdapter(unet, image_proj_model, IP_ADAPTER_PATH)

    def setup(self, stage: str) -> None:
        ### zero-shot dataset
        if config["check_zero_shot"]:
            self.processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    # CLIP image encoder with projection
    def forward(self, clip_images) -> Any:
        res = self.clip_model.visual_projection(self.clip_model.vision_model(clip_images.to(self.device)).pooler_output)
        return res

    def shared_step(self, batch, is_train=True):
        opt = self.optimizers()
        opt.zero_grad()

        # gather gt from other gpu's for larger batch size effect
        text_embeds = self.all_gather(batch["text_embeddings_clip"]).reshape(-1, batch["text_embeddings_clip"].size(1))
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        # text_embeds = batch["text_embeddings_clip"] / batch["text_embeddings_clip"].norm(p=2, dim=-1, keepdim=True)
        for i in range(config["gradient_accum_step"]):
            idx_start = config["minibatch_size"] * i
            idx_end = config["minibatch_size"] * (i + 1)
            minibatch_images = batch["image"][idx_start:idx_end]
            minibatch_clip_images = batch["clip_image"][idx_start:idx_end]
            minibatch_text_embeddings_sd = batch["text_embeddings_sd"][idx_start:idx_end]

            with torch.no_grad():
                latents = self.vae.encode(minibatch_images.to(self.device)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # self.print(latents.shape)  # torch.Size([12, 4, 64, 64])
            # Sample a random timestep for each image
            if config["timestep_range"] == 0:
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
            else:
                timesteps = torch.randint(*config["timestep_range"], (bsz,), device=latents.device)
            timesteps = timesteps.long()
            mean_timestep = torch.mean(timesteps, dtype=float).item()

            # Add noise to the latents according to the noise magnitude at each timestep (forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            image_embeds = self.forward(minibatch_clip_images)

            noise_pred = self.ip_adapter(noisy_latents, timesteps, minibatch_text_embeddings_sd, image_embeds)
            mse_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            # accelerator.print(clip_model.module.logit_scale.exp())
            logit_scale = self.clip_model.logit_scale.exp()
            logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
            # logits_per_image = torch.matmul(image_embeds, text_embeds.t())
            clip_loss = nn.functional.cross_entropy(
                logits_per_image,
                idx_start
                + self.trainer.local_rank * config["gradient_accum_step"] * config["minibatch_size"]
                + torch.arange(len(logits_per_image), device=self.device),
            )
            loss = clip_loss + config["diffusion_loss_scale"] * mse_loss

            if is_train:
                self.manual_backward(loss)

        out = {
            "mse_loss": mse_loss,
            "clip_loss": clip_loss,
            "loss": loss,
            "logit_scale": logit_scale,
            "mean_timestep": mean_timestep,
        }
        if is_train:
            opt.step()
        return out

    def training_step(self, batch, b_idx):
        out = self.shared_step(batch)
        log = {}
        for key, value in out.items():
            log["train_" + key] = value

        self.log_dict(
            log,
            prog_bar=True,
            batch_size=config["gradient_accum_step"] * config["minibatch_size"],
        )
        # return out["loss"] # automatic optimization off

    def validation_step(self, batch):
        out = self.shared_step(batch, is_train=False)
        log = {}
        for key, value in out.items():
            log["val_" + key] = value
        self.log_dict(
            log,
            prog_bar=True,
            batch_size=config["gradient_accum_step"] * config["minibatch_size"],
        )

    def on_validation_epoch_end(self) -> None:
        if config["check_1nn"]:
            self.evaluate_nn()
        if config["check_zero_shot"]:
            self.evaluate_zero_shot()

    def configure_optimizers(self):
        params_to_opt = itertools.chain(
            self.ip_adapter.image_proj_model.parameters(),
            self.ip_adapter.adapter_modules.parameters(),
            self.clip_model.vision_model.parameters(),
            self.clip_model.visual_projection.parameters(),
            [self.clip_model.logit_scale],
            # [clip_model.logit_scale, clip_model.mse_loss_scale],
        )
        optimizer = torch.optim.AdamW(params_to_opt, lr=config["learning_rate"], weight_decay=config["weight_decay"])
        return optimizer

    # perform 1nn
    def evaluate_nn(self):
        start_time = time()
        X_train, Y_train = collect_features(
            self.clip_model.vision_model, self.datamodule.nn_train_dataloader(), self.device
        )
        X_train = self.all_gather(X_train).reshape(-1, X_train.size(1))
        Y_train = self.all_gather(Y_train).reshape(-1)
        X_test, Y_test = collect_features(
            self.clip_model.vision_model, self.datamodule.nn_val_dataloader(), self.device
        )

        BATCH_SIZE = 256
        corrects = []
        d_train = X_train.T.pow(2).sum(dim=0, keepdim=True)
        for i in range(0, X_test.shape[0], BATCH_SIZE):
            X_batch = X_test[i : i + BATCH_SIZE]
            Y_batch = Y_test[i : i + BATCH_SIZE]
            d_batch = X_batch.pow(2).sum(dim=1)[:, None]
            distance = d_batch - torch.mm(X_batch, X_train.T) * 2 + d_train
            corrects.append((Y_batch == Y_train[distance.argmin(dim=1)]).detach())
        corrects = torch.cat(corrects)
        acc = corrects.float().mean()

        if self.trainer.sanity_checking:
            self.initial_nn_acc = acc
        self.print(
            f"epoch: {self.current_epoch}, 1nn acc: {acc}, elapsed-time: {timedelta(seconds=(time()-start_time))}"
        )
        self.log_dict(
            {"1nn-accuracy": acc, "initial-1nn-acc": self.initial_nn_acc},
            logger=True,
        )

    # evaluate zero-shot classification accuracy
    def evaluate_zero_shot(self):
        start_time = time()
        correct_predictions = 0
        total_images = 0
        text_features = torch.load("../IN100_classnames_text_features.pt")  # precomputed
        text_features = text_features.to(self.device)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        for image_tensors, labels in self.datamodule.zero_shot_dataloader():
            image_tensors = image_tensors.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                image_tensors = self.processor.preprocess(images=image_tensors, return_tensors="pt").pixel_values
                image_features = self.forward(image_tensors)
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity between image and text features
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Get the class with the highest similarity score for each image in the batch
            predicted_classes = similarity.argmax(dim=-1)

            # Accuracy calculation with gt
            correct_predictions += (predicted_classes == labels).sum().item()

            total_images += len(image_tensors)

        acc = correct_predictions / total_images
        if self.trainer.sanity_checking:
            self.initial_zero_shot_acc = acc
        self.print(
            f"epoch: {self.current_epoch}, zero-shot acc: {acc}, elapsed-time: {timedelta(seconds=(time()-start_time))}"
        )
        self.log_dict(
            {"zero-shot-acc": acc, "initial-zero-shot-acc": self.initial_zero_shot_acc},
            logger=True,
        )

    def on_save_checkpoint(self, checkpoint):
        if config["ckpt_only_clip_weights"]:
            # Filter state_dict
            filtered_state_dict = {k: v for k, v in self.state_dict().items() if "clip_model.vision_model" in k}
            checkpoint["state_dict"] = filtered_state_dict


def create_trainer() -> L.Trainer:
    save_dir = "{save_root}/clip-finetuning/{dirname}".format(
        save_root=OUTPUT_PATH, dirname=config["now"] + "_" + config["filename"]
    )
    wandb_logger = WandbLogger(
        project="DiffCLIP",
        name="clip-finetuning/" + config["now"] + "_" + config["filename"],
        log_model=True,
        config=config,
        save_code=True,
        save_dir=save_dir,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=1,
        # monitor="val_loss",
        # mode="min",
        # save_last=True,
        # save_last=False,
        # filename="{epoch}_{val_loss:.4f}",
        filename="{epoch}",
        save_weights_only=True if config["ckpt_only_clip_weights"] else False,
    )
    trainer = L.Trainer(
        max_epochs=100,
        logger=(
            CSVLogger(save_dir="{save_root}/trash".format(save_root=OUTPUT_PATH))
            if config["test_mode"]
            else wandb_logger
        ),
        log_every_n_steps=10,
        callbacks=[lr_monitor_callback, ckpt_callback],
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=3,
        strategy="ddp",
        # strategy="ddp_find_unused_parameters_true",
        precision="16-mixed",
    )
    if trainer.is_global_zero and not config["test_mode"]:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    return trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Finetuning CLIP with Diffusion Representations")
    parser.add_argument(
        "-T",
        "--test_mode",
        action="store_true",
    )
    parser.add_argument(
        "-C",
        "--ckpt",
        type=str,
        default="None",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config.update(vars(args))
    config.update({"now": datetime.now().strftime("%Y-%m-%d_%H:%M:%S")})

    trainer = create_trainer()
    if trainer.is_global_zero:
        print("===========================")
        pprint(config)
        print(
            "\n##### Effective Batch Size: {bs}".format(
                bs=config["gradient_accum_step"] * config["minibatch_size"] * trainer.num_devices
            )
        )
        print("===========================")

    in100_datamodule = IN100_DataModule()
    model = DiffCLIP(in100_datamodule)

    trainer.fit(model, datamodule=in100_datamodule, ckpt_path=None if config["ckpt"] == "None" else config["ckpt"])

    wandb.finish()


if __name__ == "__main__":
    main()
