import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(threshold=sys.maxsize)


def sobel(window_size):
    assert window_size % 2 != 0
    ind = window_size // 2

    matx = []
    maty = []
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gx_ij = 0
            else:
                gx_ij = i / float(i * i + j * j)
            row.append(gx_ij)
        matx.append(row)
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gy_ij = 0
            else:
                gy_ij = j / float(i * i + j * j)
            row.append(gy_ij)
        maty.append(row)

    if window_size == 3:
        mult = 2
    elif window_size == 5:
        mult = 20
    elif window_size == 7:
        mult = 780

    matx = np.array(matx) * mult
    maty = np.array(maty) * mult

    return torch.Tensor(matx), torch.Tensor(maty)


def create_window(window_size, channel):
    windowx, windowy = sobel(window_size)
    windowx, windowy = windowx.unsqueeze(0).unsqueeze(0), windowy.unsqueeze(
        0
    ).unsqueeze(0)
    windowx = torch.Tensor(windowx.expand(channel, 1, window_size, window_size))
    windowy = torch.Tensor(windowy.expand(channel, 1, window_size, window_size))

    return windowx, windowy


def gradient(img, windowx, windowy, window_size, padding, channel):
    if channel > 1:  # do convolutions on each channel separately and then concatenate
        gradx = torch.ones(img.shape)
        grady = torch.ones(img.shape)
        if img.is_cuda:
            gradx = gradx.cuda(img.get_device())
            grady = grady.cuda(img.get_device())

        for i in range(channel):
            gradx[:, i, :, :] = F.conv2d(
                img[:, i, :, :].unsqueeze(1), windowx, padding=padding, groups=1
            ).squeeze(
                1
            )  # fix the padding according to the kernel size
            grady[:, i, :, :] = F.conv2d(
                img[:, i, :, :].unsqueeze(1), windowy, padding=padding, groups=1
            ).squeeze(1)

    else:
        gradx = F.conv2d(img, windowx, padding=padding, groups=1)
        grady = F.conv2d(img, windowy, padding=padding, groups=1)

    return gradx, grady


class GradLoss(torch.nn.Module):
    def __init__(self, window_size=3, padding=1):
        super(GradLoss, self).__init__()
        self.window_size = window_size
        self.padding = padding
        self.channel = 1  # out channel
        self.windowx, self.windowy = create_window(window_size, self.channel)

    def forward(self, pred, label):
        (batch_size, channel, _, _) = pred.size()
        if pred.is_cuda:
            self.windowx = self.windowx.cuda(pred.get_device())
            self.windowx = self.windowx.type_as(pred)
            self.windowy = self.windowy.cuda(pred.get_device())
            self.windowy = self.windowy.type_as(pred)

        pred_gradx, pred_grad_y = gradient(
            pred, self.windowx, self.windowy, self.window_size, self.padding, channel
        )
        label_gradx, label_grad_y = gradient(
            label, self.windowx, self.windowy, self.window_size, self.padding, channel
        )

        l1_loss = nn.L1Loss()
        grad_loss = l1_loss(pred_gradx, label_gradx) + l1_loss(
            pred_grad_y, label_grad_y
        )

        return grad_loss
