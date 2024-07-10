import warnings

warnings.filterwarnings(action="ignore", message=".*will be defined as the corresponding NumPy scalar.*")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import v2 as T

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from flash.core.optimizers import LARS

from transformers import CLIPVisionModel

from typing import Any
import os
from datetime import datetime, timedelta
import argparse
import wandb
from pathlib import Path
import numpy as np
import time

from utils import accuracy


class FeatureDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


@torch.no_grad()
def collect_features(model, loader, device):
    model.eval()
    model.to(device)
    print(f"collecting features on device: {device}...")
    X, Y = [], []
    total_len = len(loader)
    start_time = time.time()
    avg_time_per_batch = 0
    for cnt, batch in enumerate(loader):
        temp_time = time.time()
        x, y = batch
        # x, y = convert_tensor(batch, device=device)
        x = x.to(device)
        y = y.to(device)
        x = model(x)
        X.append(x.pooler_output.detach().cpu())
        Y.append(y.detach().cpu())
        avg_time_per_batch = (avg_time_per_batch * cnt + (time.time() - temp_time)) / (cnt + 1)
        eta_seconds = (total_len - (cnt + 1)) * avg_time_per_batch
        print(
            f"collect done: {cnt+1} / {total_len}, eta: {timedelta(seconds=eta_seconds)}, time_passed: {timedelta(seconds=(time.time()-start_time))}",
            end="\r",
        )

    X = torch.cat(X).detach().numpy()
    Y = torch.cat(Y).detach().numpy()

    print(f"\ntotal time: {timedelta(seconds=time.time()-start_time)}")
    return X, Y


class CLIPVisionLinEval(L.LightningModule):
    def __init__(self, ckpt=None):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained("/scratch/choi/model/CLIP-ViT-H-14-laion2B-s32B-b79K")
        # load finetuned clip model
        if ckpt:
            checkpoint = torch.load(ckpt, map_location="cpu")
            clipmodel_keys = {
                k.replace("clip_model.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("clip_model.vision_model")
            }
            self.model.load_state_dict(clipmodel_keys)

            # self.model = diff_clip_model.clip_model.vision_model
        self.model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1280, affine=False, eps=1e-6),
            torch.nn.Linear(1280, 100),  # TODO: nb_classes 100 or 1000
        )
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        for _, p in self.model.head.named_parameters():
            p.requires_grad = True
        self.model.eval()
        self.model.head.train()
        # trunc_normal_(model.head[1].weight, std=0.01)

        # def hook(module, _, output):
        #     return module.head(output.pooler_output)

        # self.model.register_forward_hook(hook)

    def setup(self, stage: str) -> None:
        DATA_PATH = "/scratch/choi/dataset/ImageNet100"
        self.loss_fn = nn.CrossEntropyLoss()

        transform_train = T.Compose(
            [
                # Excluded random augmentation as collecting feature beforehand
                # T.RandomResizedCrop(224, interpolation=3),
                # T.RandomHorizontalFlip(),
                T.Resize(256, interpolation=3),
                T.CenterCrop(224),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        transform_val = T.Compose(
            [
                T.Resize(256, interpolation=3),
                T.CenterCrop(224),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        train_dataset = ImageFolder(os.path.join(DATA_PATH, "train"), transform=transform_train)
        val_dataset = ImageFolder(os.path.join(DATA_PATH, "val"), transform=transform_val)
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=512, num_workers=4, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        val_sampler = torch.utils.data.DistributedSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=512, num_workers=4, pin_memory=True, drop_last=False, sampler=val_sampler
        )
        X_train, Y_train = collect_features(self.model.vision_model, train_loader, self.device)
        X_val, Y_val = collect_features(self.model.vision_model, val_loader, self.device)
        self.train_dataset = FeatureDataset(X_train, Y_train)
        self.val_dataset = FeatureDataset(X_val, Y_val)

        # X_train, Y_train = self.all_gather(collect_features(self.model.vision_model, train_loader, self.device))
        # X_train = X_train.flatten(0, 1)
        # Y_train = Y_train.flatten(0, 1)
        # self.train_dataset = FeatureDataset(X_train, Y_train)
        # X_val, Y_val = self.all_gather(collect_features(self.model.vision_model, val_loader, self.device))
        # X_val = X_val.flatten(0, 1)
        # Y_val = Y_val.flatten(0, 1)
        # self.val_dataset = FeatureDataset(X_val, Y_val)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=128,
            num_workers=4,
            drop_last=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=128,
            num_workers=4,
            drop_last=False,
            pin_memory=True,
        )
        return val_loader

    def forward(self, embeddings):
        return self.model.head(embeddings)

    def training_step(self, batch):
        clip_embeddings, gt_labels = batch
        outputs = self(clip_embeddings)
        loss = self.loss_fn(outputs, gt_labels)

        return loss

    def validation_step(self, batch):
        clip_embeddings, gt_labels = batch
        outputs = self(clip_embeddings)
        loss = self.loss_fn(outputs, gt_labels)

        acc1, acc5 = accuracy(outputs, gt_labels, topk=(1, 5))

        # sync_dist = self.trainer.current_epoch == (self.trainer.max_epochs - 1)
        self.log_dict({"Acc@1": acc1, "Acc@5": acc5, "loss": loss}, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = LARS(self.model.head.parameters(), lr=0.05, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            anneal_strategy="cos",
            total_steps=self.trainer.max_epochs * len(self.train_dataloader()),
            max_lr=0.1,
        )
        return [optimizer], [scheduler]


def create_trainer(output_file_name, args) -> L.Trainer:
    save_root_dir = "/scratch/choi/output/DiffCLIP"
    save_dir = "{save_root}/clip-lineval/{now}_{dirname}".format(
        save_root=save_root_dir, now=args.now, dirname=output_file_name
    )
    wandb_logger = WandbLogger(
        project="DiffCLIP",
        name="clip-lineval/" + args.now + "_" + output_file_name,
        log_model=True,
        config=vars(args),
        save_code=True,
        save_dir=save_dir,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    trainer = L.Trainer(
        max_epochs=50,
        logger=(
            CSVLogger(save_dir="{save_root}/trash".format(save_root=save_root_dir)) if args.test_mode else wandb_logger
        ),
        # log_every_n_steps=10,
        callbacks=[lr_monitor_callback],
        accelerator="gpu",
        devices=3,
        # strategy="ddp",
        # precision="16-mixed",
        enable_checkpointing=False,
        use_distributed_sampler=False,
    )
    if trainer.is_global_zero and not args.test_mode:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    return trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Linear evaluation of CLIP")
    parser.add_argument(
        "-T",
        "--test_mode",
        action="store_true",
    )
    parser.add_argument(
        "-C",
        "--ckpt",
        type=str,
        help="CLIP ckpt path to evaluate.",
        default=None,
    )
    args = parser.parse_args()
    if args.ckpt is None:
        print("Warning. CKPT set to default CLIP. Provide ckpt arg if you intend to evaluate a fine-tuned CLIP")
    return args


def main():
    start_time = time.time()
    # OUTPUT_FILE_NAME = "CLIP-mse-loss-0.001-epoch-1"
    OUTPUT_FILE_NAME = "default-clip"

    L.seed_everything(1)
    args = parse_args()
    args.now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    trainer = create_trainer(OUTPUT_FILE_NAME, args)

    model = CLIPVisionLinEval(args.ckpt)
    trainer.fit(model)
    print(f"elapsed-time: {timedelta(seconds=(time.time()-start_time))}")
    wandb.finish()


if __name__ == "__main__":
    main()
