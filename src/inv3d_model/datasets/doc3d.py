from typing import *

from torch.utils.data import Dataset
from unflatten import unflatten

from inv3d_util.mapping import apply_map
from inv3d_util.path import check_dir, check_file, list_files

from .loaders import *


class Doc3DDataset(Dataset):
    def __init__(
        self,
        source_file: Path,
        resolution: int,
        extra_features: Optional[List[str]] = None,
        limit_samples: Optional[int] = None,
        repeat_samples: Optional[int] = None,
        **kwargs,
    ):
        source_file = check_file(source_file, suffix=".txt", exist=True)

        self.source_dir = source_file.parent
        self.resolution = resolution
        self.extra_features = [] if extra_features is None else extra_features

        self.wc_min_values = (-0.67492497, -1.2289206, -1.2442188)
        self.wc_max_values = (0.6436657, 1.2396319, 1.2539363)

        self.samples = [s for s in source_file.read_text().split("\n") if s != ""]

        if limit_samples:
            self.samples = self.samples[:limit_samples]

        if repeat_samples:
            self.samples = self.samples * repeat_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        data = {}
        data["sample"] = str(sample)
        data["index"] = idx
        data["input.image"] = prepare_masked_image_jitter(
            self.source_dir / f"img/{sample}.png",
            self.source_dir / f"uv/{sample}.exr",
            resolution=self.resolution,
        )

        data["train.bm"] = prepare_bm(
            self.source_dir / f"bm/{sample}.mat", self.resolution
        )

        if "wc" in self.extra_features:
            data["train.wc"] = prepare_wc(
                self.source_dir / f"wc/{sample}.exr",
                self.wc_min_values,
                self.wc_max_values,
                self.resolution,
            )

        if "recon" in self.extra_features:
            data["train.recon"] = prepare_masked_image(
                self.source_dir / f"recon/{sample[:-4]}chess480001.png",
                self.source_dir / f"uv/{sample}.exr",
                resolution=self.resolution,
            )

        if "mask" in self.extra_features:
            data["train.mask"] = prepare_mask(
                self.source_dir / f"uv/{sample}.exr", resolution=self.resolution
            )

        return unflatten(data)


class Doc3DTestDataset(Dataset):
    def __init__(
        self,
        source_file: Path,
        resolution: int,
        limit_samples: Optional[int] = None,
        **kwargs,
    ):
        source_file = check_file(source_file, suffix=".txt", exist=True)

        self.source_dir = source_file.parent
        self.resolution = resolution
        self.samples = [s for s in source_file.read_text().split("\n") if s != ""]

        if limit_samples:
            self.samples = self.samples[:limit_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        data = {}
        data["sample"] = str(sample)
        data["index"] = idx
        data["input.image"] = prepare_masked_image(
            self.source_dir / f"img/{sample}.png",
            self.source_dir / f"uv/{sample}.exr",
            resolution=self.resolution,
        )

        data["eval.text_evaluation"] = False
        data["eval.true_bm"] = prepare_bm(self.source_dir / f"bm/{sample}.mat")
        data["eval.orig_image"] = prepare_masked_image(
            self.source_dir / f"img/{sample}.png", self.source_dir / f"uv/{sample}.exr"
        )

        # true image is not available for doc3D
        albedo_image = prepare_masked_image(
            self.source_dir / f"alb/{sample}.png", self.source_dir / f"uv/{sample}.exr"
        )
        true_image = apply_map(
            rearrange(albedo_image, "c h w -> h w c"),
            rearrange(data["eval.true_bm"], "c h w -> h w c"),
            resolution=(920, 650),
        )  # best approximation of evaluation size of 598400

        true_image = rearrange(true_image, "h w c -> c h w")
        data["eval.true_image"] = true_image

        return unflatten(data)


class Doc3DRealDataset(Dataset):
    def __init__(
        self,
        source_dir: Path,
        resolution: int,
        limit_samples: Optional[int] = None,
        **kwargs,
    ):
        self.source_dir = check_dir(source_dir, exist=True)
        self.resolution = resolution
        self.samples = list_files(self.source_dir / "crop", suffixes=[".png"])
        self.ocr_ids = [
            1,
            9,
            10,
            12,
            19,
            20,
            21,
            22,
            23,
            24,
            30,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            44,
            45,
            46,
            47,
            49,
        ]

        if limit_samples:
            self.samples = self.samples[:limit_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        scan_id = sample.stem.split("_")[0]

        data = {}
        data["sample"] = str(sample)
        data["index"] = idx
        data["input.image"] = prepare_image(sample, resolution=self.resolution)

        data["eval.text_evaluation"] = int(scan_id) in self.ocr_ids
        data["eval.orig_image"] = prepare_image(sample)
        data["eval.true_image"] = prepare_image(
            sample.parent.parent / f"scan/{scan_id}.png"
        )

        return unflatten(data)
