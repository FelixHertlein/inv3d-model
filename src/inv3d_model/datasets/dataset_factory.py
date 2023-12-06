from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

import dpath.util
from torch.utils.data import Dataset

from inv3d_util.load import load_yaml


class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATE = "val"
    TEST = "test"


@dataclass
class FactoryElement:
    create_fn: Callable
    source_base: str


class DatasetFactory:
    _all_datasets: Dict[str, FactoryElement] = {}

    def __init__(self, sources_file: Path) -> None:
        self.sources = load_yaml(sources_file)

    @classmethod
    def get_all_datasets(cls) -> List[str]:
        return sorted(list(cls._all_datasets.keys()))

    @classmethod
    def register_dataset(
        cls,
        name: str,
        dataset_class: Type[Dataset],
        source: Optional[str] = None,
        **init_kwargs,
    ):
        if name in cls._all_datasets:
            raise ValueError(f"Dataset '{name}' is already registered!")

        if source is None:
            source = name

        cls._all_datasets[name] = FactoryElement(
            create_fn=partial(dataset_class, **init_kwargs), source_base=source
        )

    def create(self, name: str, split: DatasetSplit, **dataset_kwargs) -> Dataset:
        if name not in self._all_datasets:
            raise ValueError(
                f"Dataset '{name}' is unknown! Cannot create a new instance!"
            )

        create_fn = self._all_datasets[name].create_fn
        source_base = self._all_datasets[name].source_base

        source_key = f"{source_base}/{split.value}_dir"
        source = dpath.util.get(self.sources, source_key, default=None)

        if source is None:
            raise ValueError(
                f"Source '{source_key}' is not provided by the source file! Cannot create dataset '{name}!'"
            )

        return create_fn(source, **dataset_kwargs)
