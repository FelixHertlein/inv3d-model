from typing import *

import lpips
import pandas as pd
import torch
import tqdm
from cachetools import cached
from einops import rearrange
from Levenshtein import distance
from PIL import Image
from pytesseract import pytesseract
from pytorch_msssim import ms_ssim

from inv3d_util.image import scale_image
from inv3d_util.mapping import scale_map_torch
from inv3d_util.misc import to_numpy_image
from inv3d_util.parallel import process_tasks


@cached({})
def get_lpips_metric():
    return lpips.LPIPS(net="alex").cuda()


def score_all(states: List[Dict]) -> pd.DataFrame:
    _scale_images(states)

    _score_epe(states)
    _score_msssim(states)
    _score_lpips(states)
    _score_textual(states, num_workers=16)

    dfs = [
        pd.DataFrame.from_dict({k: [v] for k, v in state["results"].items()})
        for state in states
    ]

    return pd.concat(dfs, ignore_index=True)


def _scale_images(states: List[Dict]):
    for state in tqdm.tqdm(states, "Scale images"):
        true_image = to_numpy_image(state["true_image"])
        norm_image = to_numpy_image(state["norm_image"])

        true_image = scale_image(true_image, area=598400)
        norm_image = scale_image(norm_image, area=598400)

        state["true_image_scaled"] = true_image
        state["norm_image_scaled"] = norm_image


@torch.no_grad()
def _score_msssim(states: List[Dict]):
    for state in tqdm.tqdm(states, "Calcualte msssim"):
        true_image = state["true_image_scaled"]
        norm_image = state["norm_image_scaled"]

        true_image = (
            rearrange(torch.from_numpy(true_image), "h w c -> 1 c h w").cuda().float()
            / 255
        )
        norm_image = (
            rearrange(torch.from_numpy(norm_image), "h w c -> 1 c h w").cuda().float()
            / 255
        )

        score = ms_ssim(true_image, norm_image, data_range=1, size_average=False)
        state["results"]["ms_ssim"] = float(score.cpu().numpy())


@torch.no_grad()
def _score_lpips(states: List[Dict]):
    for state in tqdm.tqdm(states, "Calcualte lpips"):
        true_image = state["true_image_scaled"]
        norm_image = state["norm_image_scaled"]

        true_image = (
            rearrange(torch.from_numpy(true_image), "h w c -> 1 c h w").cuda().float()
            / 255
        )
        norm_image = (
            rearrange(torch.from_numpy(norm_image), "h w c -> 1 c h w").cuda().float()
            / 255
        )

        score = get_lpips_metric()(true_image, norm_image)
        state["results"]["lpips"] = float(score.cpu().numpy())


def _score_textual(states: List[Dict], num_workers: int):
    print("Calculating textual scores")

    tasks = [
        {
            "true_image": to_numpy_image(state["true_image"]),
            "norm_image": to_numpy_image(state["norm_image"]),
        }
        for state in states
        if state["text_evaluation"]
    ]

    all_results = process_tasks(
        _score_textual_task, tasks, num_workers=num_workers, use_indexes=True
    )
    for idx, results in all_results.items():
        states[idx]["results"].update(**results)


def _score_textual_task(task: Dict):
    true_text = pytesseract.image_to_string(
        Image.fromarray(task["true_image"]), lang="eng"
    )
    norm_text = pytesseract.image_to_string(
        Image.fromarray(task["norm_image"]), lang="eng"
    )

    text_distance = distance(true_text, norm_text)

    return {"ed": text_distance, "cer": text_distance / len(true_text)}


def _score_epe(states: List[Dict]):
    states = [state for state in states if state["true_bm"] is not None]

    for state in tqdm.tqdm(states, "Calcualte EPE"):
        out_bm = state["out_bm"].cuda()
        true_bm = state["true_bm"].cuda()
        out_bm = scale_map_torch(out_bm, resolution=true_bm.shape[-2:])
        score = torch.linalg.norm(true_bm - out_bm, ord=2, dim=1).mean(dim=(1, 2))
        state["results"]["epe"] = float(score.cpu().numpy())
