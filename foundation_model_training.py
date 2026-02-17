import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from monai import transforms
from monai.data import (
    Dataset,
    DataLoader,
    pad_list_data_collate
    )
from monai.networks.nets import resnet10, ViTAutoEnc

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer


def dataset():
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=['dce']),
            transforms.EnsureChannelFirstd(keys=['dce']),
            transforms.Orientationd(keys=['dce'], axcodes="RAS"),
            transforms.Spacingd(keys=['dce'], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
            transforms.NormalizeIntensityd(keys=['dce'], nonzero=True, channel_wise=True),
            transforms.ResizeWithPadOrCropd(keys=['dce'], spatial_size=(128, 128, 128)),
            transforms.RandSpatialCropd(keys=['dce'], roi_size=(128, 128, 128), random_center=True),
            transforms.RandGaussianNoised(keys=['dce'], prob=0.2),
            transforms.ToTensord(keys=['dce'])
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=['dce']),
            transforms.EnsureChannelFirstd(keys=['dce']),
            transforms.Orientationd(keys=['dce'], axcodes="RAS"),
            transforms.Spacingd(keys=['dce'], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
            transforms.NormalizeIntensityd(keys=['dce'], nonzero=True, channel_wise=True),
            transforms.ResizeWithPadOrCropd(keys=['dce'], spatial_size=(128, 128, 128)),
            transforms.RandSpatialCropd(keys=['dce'], roi_size=(128, 128, 128), random_center=True),
            transforms.ToTensord(keys=['dce'])
        ]
    )

    def generate_splits(data_path):
        subjects = []
        for i in os.listdir(data_path):
            subject = {
                'dce': os.path.join(data_path+i), #, i+'-t1c.nii.gz'),
                }
            subjects.append(subject)
        return subjects
    
    train_subjects = generate_splits('/path/to/train/')
    val_subjects = generate_splits('/path/to/val/')

    dataset = Dataset(data=train_subjects, transform=train_transforms) #, cache_num=24, cache_rate=1, num_workers=2)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=pad_list_data_collate, drop_last=True)

    dataset = Dataset(data=val_subjects, transform=val_transforms) #, cache_num=24, cache_rate=1, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=pad_list_data_collate, drop_last=True)

    return train_loader, val_loader

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.proj(x)

class SimCLRModule(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = resnet10(spatial_dims=3, n_input_channels=1, feed_forward=False)
        self.projection = ProjectionHead(in_dim=512, out_dim=128)
        self.lr = lr
        self.temperature = 0.07

    def forward(self, x):
        return self.projection(self.encoder(x))

    def info_nce_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        representations = torch.cat([z1, z2], dim=0)
        similarity = torch.mm(representations, representations.T)
        sim_exp = torch.exp(similarity / self.temperature)
        mask = ~torch.eye(sim_exp.shape[0], dtype=torch.bool, device=sim_exp.device)
        sim_exp = sim_exp.masked_select(mask).view(sim_exp.shape[0], -1)
        positives = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        loss = -torch.log(positives / sim_exp.sum(dim=-1)[:z1.size(0)])
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x1 = batch["dce"]#torch.cat([batch["t1c"], batch["t1n"], batch["t2f"], batch["t2w"]], dim=1)
        x2 = x1 + 0.01 * torch.randn_like(x1)
        z1 = self(x1)
        z2 = self(x2)
        loss = self.info_nce_loss(z1, z2)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x1 = batch["dce"]#torch.cat([batch["t1c"], batch["t1n"], batch["t2f"], batch["t2w"]], dim=1)
        x2 = x1 + 0.01 * torch.randn_like(x1)  # Second view

        z1 = self(x1)
        z2 = self(x2)
        val_loss = self.info_nce_loss(z1, z2)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    train_loader, val_loader = dataset()

    model = SimCLRModule()

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min")],
        precision=16,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    # CUDA_VISIBLE_DEVICES=0 python3 foundation_model_training.py
