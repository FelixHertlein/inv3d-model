import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from inv3d_util.mapping import apply_map_torch


class UnwarpLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp_img, pred, label):
        # convert BM from DewarpNet to standard format
        pred = (pred + 1) / 2
        pred = rearrange(pred, "n c h w -> n c w h")
        pred = torch.roll(pred, shifts=1, dims=1)

        # convert BM from DewarpNet to standard format
        label = (label + 1) / 2
        label = rearrange(label, "n c h w -> n c w h")
        label = torch.roll(label, shifts=1, dims=1)

        unwarped_prediction = apply_map_torch(inp_img, pred)
        unwarped_label = apply_map_torch(inp_img, label)
        loss_fn = nn.MSELoss()
        unwarp_loss = loss_fn(unwarped_prediction, unwarped_label)

        return unwarp_loss.float()
