import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPModel
from ip_adapter.attention_processor import (
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
from datetime import datetime
from pprint import pprint
import itertools
from safetensors import safe_open


config = {
    "minibatch_size": 8,  # TODO: see how much i can go
    "gradient_accum_step": 1,
    "timestep_range": (400, 600),  # 0 for full range
    "resolution": 224,  # TODO: change?
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
}


# TODO: seperate to data module
class MyDataset(Dataset):
    def __init__(self, is_train: bool):
        super().__init__()
        IMAGE_ROOT_PATH = "/scratch/choi/dataset/ImageNet100"
        self.mode = "train" if is_train else "val"

        # self.tokenizer = None
        self.image_root_path = f"{IMAGE_ROOT_PATH}/{self.mode}"
        self.data = json.load(
            open(f"{IMAGE_ROOT_PATH}/_img_text_pair_{self.mode}.json")
        )  # list of dict: [{'class_id': 'n03785016','image_file': 'n03785016_5681.JPEG','text': 'a photo of moped'}]

        self.text_embedding_dict_L = torch.load("IN100_text_embedding_dict_L.pt")
        self.text_embedding_dict_with_projection_H = torch.load("IN100_text_embedding_dict_with_projection_H.pt")

        self.transform = T.Compose(
            [
                # T.Resize(config["resolution"], interpolation=T.InterpolationMode.BILINEAR),
                T.Resize(config["resolution"], antialias=True),
                T.CenterCrop(config["resolution"]),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.5], [0.5]),
            ]
        )
        self.clip_image_processor = CLIPImageProcessor()

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
            "clip_image": clip_image,
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

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


# TODO: use diffclip or clipvisualmodel when doing linear probing?
class DiffCLIP(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # CLIP_PATH = "/scratch/choi/model/IP-Adapter/models/image_encoder"
        IP_ADAPTER_PATH = "/scratch/choi/model/IP-Adapter/models/ip-adapter_sd15.bin"
        SD_PATH = "/scratch/choi/model/stable-diffusion-v1-5"

        # TODO: set requires grad, del text model?
        self.noise_scheduler = DDPMScheduler.from_pretrained(SD_PATH, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(SD_PATH, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(SD_PATH, subfolder="unet")
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

        image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.clip_model.config.projection_dim,
            clip_extra_context_tokens=4,
        )
        self.ip_adapter = IPAdapter(self.unet, image_proj_model, IP_ADAPTER_PATH)

    def setup(self, stage: str) -> None:
        self.train_dataset = MyDataset(is_train=True)
        self.val_dataset = MyDataset(is_train=False)

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=config["minibatch_size"] * config["gradient_accum_step"],  # TODO: check batch size?
            num_workers=4,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=config["minibatch_size"] * config["gradient_accum_step"],
            num_workers=4,
        )
        return val_loader

    def forward(self) -> Any:
        pass

    def shared_step(self, batch):
        opt = self.optimizers()
        text_embeds = batch["text_embeddings_clip"] / batch["text_embeddings_clip"].norm(p=2, dim=-1, keepdim=True)
        for i in range(config["gradient_accum_step"]):
            idx_start = config["minibatch_size"] * i
            idx_end = config["minibatch_size"] * (i + 1)
            minibatch_images = batch["images"][idx_start:idx_end]
            minibatch_clip_images = batch["clip_images"][idx_start:idx_end]
            minibatch_drop_image_embeds = batch["drop_image_embeds"][idx_start:idx_end]
            minibatch_text_embeddings_sd = batch["text_embeddings_sd"][idx_start:idx_end]

            with torch.no_grad():
                latents = self.vae.encode(minibatch_images.to(self.device)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            if config["timestep_range"] == 0:
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
            else:
                timesteps = torch.randint(*config["timestep_range"], (bsz,), device=latents.device)
            timesteps = timesteps.long()
            mean_timestep = torch.mean(timesteps, dtype=float).item()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            image_embeds = self.clip_model.module.visual_projection(
                self.clip_model.module.vision_model(minibatch_clip_images.to(self.device)).pooler_output
            )

            noise_pred = self.ip_adapter(noisy_latents, timesteps, minibatch_text_embeddings_sd, image_embeds)

            mse_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            # accelerator.print(clip_model.module.logit_scale.exp())
            logit_scale = clip_model.module.logit_scale.exp()
            logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
            # logits_per_image = torch.matmul(image_embeds, text_embeds.t())
            clip_loss = nn.functional.cross_entropy(
                logits_per_image,
                idx_start + torch.arange(len(logits_per_image), device=accelerator.device),
            )
            loss = clip_loss + mse_loss * 1e-3

            loss = self.compute_loss(batch)
            self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        out = {}
        return out

    def training_step(self, batch):
        out = self.shared_step(batch)

        # return out["loss"]

    # def validation_step(self, batch):
    #     out = self.shared_step(batch)

    def configure_optimizers(self):
        # TODO: check param to update
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

    # TODO: customize save model behavior? save only CLIP? add config.json?


def create_trainer() -> L.Trainer:
    save_dir = "./wandb/clip-finetuning/{dirname}".format(dirname=config["now"])
    wandb_logger = WandbLogger(
        project="clip-finetuning",
        name=config["now"],
        log_model=True,
        config=config,
        save_code=True,
        save_dir=save_dir,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = ModelCheckpoint(
        # save_top_k=2,
        monitor="val_loss",  # TODO: configure
        mode="min",
        save_last=True,
        filename="{epoch}_{val_loss:.4f}",
    )
    trainer = L.Trainer(
        max_epochs=100,
        logger=None if config["test_mode"] else wandb_logger,
        log_every_n_steps=10,
        callbacks=[lr_monitor_callback, ckpt_callback],
        check_val_every_n_epoch=1,
        accelerator="gpu",
        # devices=3,
        devices=[1],
        # strategy="ddp",
        # precision="16-mixed",
        # accumulate_grad_batches=8 #TODO: gradient accumulation
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
        print("===========================")

    model = DiffCLIP()
    trainer.fit(model, ckpt_path=None if config["ckpt"] == "None" else config["ckpt"])

    wandb.finish()


if __name__ == "__main__":
    main()
