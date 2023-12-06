from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from unflatten import unflatten

from inv3d_util.misc import median_blur

from .densenetccnl import dnetccnl
from .grad_loss import GradLoss
from .recon_loss import UnwarpLoss
from .unetnc import UnetGenerator


class LitDewarpNetWC(pl.LightningModule):
    dataset_options = {"resolution": 128, "extra_features": ["wc"]}

    train_options = {"max_epochs": 300, "batch_size": 40, "early_stopping_patience": 25}

    def __init__(self):
        super().__init__()
        self.model = UnetGenerator(input_nc=3, output_nc=3, num_downs=7)
        self.activation_fn = nn.Hardtanh(0, 1.0)
        self.grad_loss_fn = GradLoss(window_size=5, padding=2)

    def forward(self, image, **kwargs):
        return self.activation_fn(self.model(image))

    def training_step(self, batch, batch_idx):
        images = batch["input"]["image"]
        labels = batch["train"]["wc"]

        outputs = self.activation_fn(self.model(images))

        loss_l1 = F.l1_loss(outputs, labels)
        loss_grad = self.grad_loss_fn(outputs, labels)

        lambda_var = min(0.2 * (self.current_epoch // 50 + 1), 1.0)
        loss = loss_l1 + lambda_var * loss_grad

        self.log("train/dewarpnet_wc_loss", loss)

        return {"loss": loss, "output": outputs}

    def validation_step(self, batch, batch_idx):
        images = batch["input"]["image"]
        labels = batch["train"]["wc"]

        outputs = self.activation_fn(self.model(images))

        self.log("val/mse_loss", F.mse_loss(outputs, labels), sync_dist=True)
        self.log("val/l1_loss", F.l1_loss(outputs, labels), sync_dist=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=True)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/mse_loss",
        }


class LitDewarpNetBM(pl.LightningModule):
    dataset_options = {"resolution": 128, "extra_features": ["wc", "recon"]}

    train_options = {"max_epochs": 300, "batch_size": 40, "early_stopping_patience": 25}

    def __init__(self):
        super().__init__()
        self.model = dnetccnl(
            img_size=self.dataset_options["resolution"],
            in_channels=3,
            out_channels=2,
            filters=32,
        )
        self.unwarp_loss_fn = UnwarpLoss()

    def forward(self, image, **kwargs):
        bm = self.model(image)

        # convert BM from DewarpNet format to standard format
        bm = (bm + 1) / 2
        bm = rearrange(bm, "n c h w -> n c w h")
        bm = torch.roll(bm, shifts=1, dims=1)

        bm = median_blur(bm)
        bm = torch.clamp(bm, min=0, max=1)
        return bm

    def training_step(self, batch, batch_idx):
        wc_data = batch["train"]["wc"]
        bm_data = batch["train"]["bm"]
        recon_images = batch["train"]["recon"]

        # convert BM to DewarpNet format
        bm_data = (bm_data * 2) - 1
        bm_data = rearrange(bm_data, "n c h w -> n c w h")
        bm_data = torch.roll(bm_data, shifts=1, dims=1)

        outputs = self.model(wc_data)

        loss_B = F.l1_loss(outputs, bm_data)
        loss_D = self.unwarp_loss_fn(inp_img=recon_images, pred=outputs, label=bm_data)

        loss = (10.0 * loss_B) + (0.5 * loss_D)

        self.log("train/dewarpnet_bm_loss_B", loss_B)
        self.log("train/dewarpnet_bm_loss_D", loss_D)
        self.log("train/dewarpnet_bm_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        wc_data = batch["train"]["wc"]
        bm_data = batch["train"]["bm"]

        # convert BM to DewarpNet format
        bm_data = (bm_data * 2) - 1
        bm_data = rearrange(bm_data, "n c h w -> n c w h")
        bm_data = torch.roll(bm_data, shifts=1, dims=1)

        outputs = self.model(wc_data)

        self.log("val/mse_loss", F.mse_loss(outputs, bm_data))
        self.log("val/l1_loss", F.l1_loss(outputs, bm_data))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=True)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/mse_loss",
        }


class LitDewarpNetJoint(pl.LightningModule):
    dataset_options = {"resolution": 128, "extra_features": ["wc", "recon"]}

    train_options = {"max_epochs": 300, "batch_size": 40, "early_stopping_patience": 25}

    def __init__(self, wc_dir: Optional[Path] = None, bm_dir: Optional[Path] = None):
        super().__init__()

        if wc_dir is None:
            self.wc_model = LitDewarpNetWC()
        else:
            [wc_checkpoint] = Path(wc_dir).rglob("checkpoint*ckpt")
            self.wc_model = LitDewarpNetWC.load_from_checkpoint(wc_checkpoint)
            print(f"INFO: Loading WC weights {wc_checkpoint}")

        if bm_dir is None:
            self.bm_model = LitDewarpNetBM()
        else:
            [bm_checkpoint] = Path(bm_dir).rglob("checkpoint*ckpt")
            self.bm_model = LitDewarpNetBM.load_from_checkpoint(bm_checkpoint)
            print(f"INFO: Loading BM weights {bm_checkpoint}")

        self.unwarp_loss_fn = UnwarpLoss()
        self.mse_loss_fn = nn.MSELoss()
        self.l1_loss_fn = nn.L1Loss()

    def forward(self, image, **kwargs):
        return self.bm_model(self.wc_model(image))

    def training_step(self, batch, batch_idx):
        # first stage
        wc_result = self.wc_model.training_step(batch=batch, batch_idx=batch_idx)
        wc_loss = wc_result["loss"]
        wc_output = wc_result["output"]

        # second stage
        bm_input = unflatten(
            {
                "train.wc": wc_output,
                "train.bm": batch["train"]["bm"],
                "train.recon": batch["train"]["recon"],
            }
        )
        bm_loss = self.bm_model.training_step(batch=bm_input, batch_idx=batch_idx)

        loss = 0.5 * wc_loss + 0.5 * bm_loss

        self.log("train/dewarpnet_joint_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["input"]["image"]
        bm_data = batch["train"]["bm"]

        outputs = self.bm_model(self.wc_model(images))

        self.log("val/mse_loss", F.mse_loss(outputs, bm_data))
        self.log("val/l1_loss", F.l1_loss(outputs, bm_data))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=True)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/mse_loss",
        }
