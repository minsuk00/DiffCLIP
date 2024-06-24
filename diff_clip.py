import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from transformers import CLIPImageProcessor

import os
import json
from PIL import Image
from typing import Any, Literal, Mapping
import wandb
import argparse
from pathlib import Path
from datetime import datetime
from pprint import pprint


config = {
    "batch_size": 8,  # TODO: see how much i can go
    "gradient_accum_step": 1,
    "timestep_range": (400, 600),
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


class DiffCLIP(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        self.train_dataset = MyDataset(is_train=True)
        self.val_dataset = MyDataset(is_train=False)

        # TODO: vae, tokenizer, etc initialize?

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=config["batch_size"] * config["gradient_accum_step"],
            num_workers=4,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=config["batch_size"] * config["gradient_accum_step"],
            num_workers=4,
        )
        return val_loader

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)

    def shared_step(self, batch):
        pass

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor | Mapping[str, Any] | None:
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> torch.Tensor | Mapping[str, Any] | None:
        return super().validation_step(*args, **kwargs)

    def configure_optimizers(self):
        return super().configure_optimizers()


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
