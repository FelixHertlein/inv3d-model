import torch
from einops import repeat
from pytorch_lightning import LightningModule

from inv3d_util.mapping import create_identity_map


class LitIdentity(LightningModule):
    dataset_options = {"resolution": 128}

    train_options = {
        "batch_size": 1,
        "max_epochs": 1,
        "early_stopping_patience": None,
    }

    def __init__(self):
        super().__init__()

    def forward(self, image, **kwargs):
        bm = create_identity_map(128)
        bm = torch.from_numpy(bm)
        bm = repeat(bm, "h w c -> n c h w", n=image.shape[0])
        return bm

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        self.log("val/mse_loss", 0, batch_size=1, sync_dist=True)
        self.log("val/l1_loss", 0, batch_size=1, sync_dist=True)

    def configure_optimizers(self):
        return None
