from copy import deepcopy
from random import seed, shuffle
from typing import *

from torch.utils.data import Dataset
from unflatten import unflatten

from inv3d_util.load import load_json
from inv3d_util.path import check_dir, list_dirs

from .loaders import *


class Inv3DDataset(Dataset):
    def __init__(
        self,
        source_dir: Path,
        resolution: int,
        template: str = "full",
        extra_features: Optional[List[str]] = None,
        limit_samples: Optional[int] = None,
        repeat_samples: Optional[int] = None,
        **kwargs
    ):
        self.source_dir = check_dir(source_dir)
        self.resolution = resolution
        self.template = template
        self.extra_features = [] if extra_features is None else extra_features

        min_max_values = load_json(self.source_dir.parent / "wc_min_max.json")
        self.wc_min_values = tuple(min_max_values["min"])
        self.wc_max_values = tuple(min_max_values["max"])

        self.samples = list(sorted([s.name for s in list_dirs(self.source_dir)]))

        if limit_samples:
            self.samples = self.samples[:limit_samples]

        if repeat_samples:
            self.samples = self.samples * repeat_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_dir = self.source_dir / sample

        data = {}
        data["sample"] = str(sample)
        data["index"] = idx
        data["input.template"] = prepare_template(
            sample_dir / "flat_template.png",
            sample_dir / "flat_information_delta.png",
            sample_dir / "flat_text_mask.png",
            self.template,
            resolution=self.resolution,
        )
        data["input.image"] = prepare_masked_image_jitter(
            sample_dir / "warped_document.png",
            sample_dir / "warped_UV.npz",
            resolution=self.resolution,
        )

        data["train.bm"] = prepare_bm(
            self.source_dir / sample / "warped_BM.npz", self.resolution
        )

        if "wc" in self.extra_features:
            data["train.wc"] = prepare_wc(
                self.source_dir / sample / "warped_WC.npz",
                self.wc_min_values,
                self.wc_max_values,
                self.resolution,
            )

        if "recon" in self.extra_features:
            data["train.recon"] = prepare_masked_image(
                sample_dir / "warped_recon.png",
                sample_dir / "warped_UV.npz",
                resolution=self.resolution,
            )

        if "mask" in self.extra_features:
            data["train.mask"] = prepare_mask(
                sample_dir / "warped_UV.npz", resolution=self.resolution
            )

        return unflatten(data)


class Inv3DTestDataset(Dataset):
    def __init__(
        self,
        source_dir: Path,
        resolution: int,
        template: str = "full",
        num_text_evals: int = 64,
        limit_samples: Optional[int] = None,
        **kwargs
    ):
        assert template in ["full", "white", "struct", "text", "random"]

        self.source_dir = check_dir(source_dir)
        self.resolution = resolution
        self.use_random_template = template == "random"
        self.template = "full" if self.use_random_template else template
        self.num_text_evals = num_text_evals
        self.samples = list(sorted([s.name for s in list_dirs(self.source_dir)]))

        if limit_samples:
            self.samples = self.samples[:limit_samples]

        if self.use_random_template:
            self.samples_shuffled = deepcopy(self.samples)
            seed(42)
            shuffle(self.samples_shuffled)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_dir = self.source_dir / sample

        template_dir = (
            self.source_dir / self.samples_shuffled[idx]
            if self.use_random_template
            else sample_dir
        )

        data = {}
        data["sample"] = str(sample)
        data["index"] = idx
        data["input.template"] = prepare_template(
            template_dir / "flat_template.png",
            template_dir / "flat_information_delta.png",
            template_dir / "flat_text_mask.png",
            variation=self.template,
            resolution=self.resolution,
        )
        data["input.image"] = prepare_masked_image(
            sample_dir / "warped_document.png",
            sample_dir / "warped_UV.npz",
            resolution=self.resolution,
        )

        data["eval.text_evaluation"] = idx < self.num_text_evals
        data["eval.true_bm"] = prepare_bm(self.source_dir / sample / "warped_BM.npz")
        data["eval.true_image"] = prepare_image(sample_dir / "flat_document.png")
        data["eval.orig_image"] = prepare_masked_image(
            sample_dir / "warped_document.png", sample_dir / "warped_UV.npz"
        )
        return unflatten(data)


class Inv3DRealDataset(Dataset):
    def __init__(
        self,
        source_dir: Path,
        resolution: int,
        template: str = "full",
        limit_samples: Optional[int] = None,
        **kwargs
    ):
        assert template in ["full", "white", "struct", "text", "random"]

        self.source_dir = check_dir(source_dir)
        self.resolution = resolution
        self.use_random_template = template == "random"
        self.template = "full" if self.use_random_template else template
        self.samples = list(sorted(self.source_dir.rglob("warped_document_*.jpg")))

        if limit_samples:
            self.samples = self.samples[:limit_samples]

        if self.use_random_template:
            self.samples_shuffled = deepcopy(self.samples)
            seed(42)
            shuffle(self.samples_shuffled)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_dir = sample.parent

        template_dir = (
            self.samples_shuffled[idx].parent
            if self.use_random_template
            else sample_dir
        )

        data = {}
        data["sample"] = str(sample)
        data["index"] = idx
        data["input.template"] = prepare_template(
            template_dir / "flat_template.png",
            template_dir / "flat_information_delta.png",
            template_dir / "flat_text_mask.png",
            variation=self.template,
            resolution=self.resolution,
        )
        data["input.image"] = prepare_image(sample, resolution=self.resolution)

        data["eval.text_evaluation"] = True
        data["eval.true_image"] = prepare_image(sample_dir / "flat_document.png")
        data["eval.orig_image"] = prepare_image(sample)

        return unflatten(data)
