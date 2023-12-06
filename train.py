import cv2

cv2.setNumThreads(0)

import argparse
import json
import sys
from pathlib import Path

project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir / "src"))

from inv3d_model.model_zoo import ModelZoo


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument(
        "--model",
        type=str,
        choices=list(zoo.list_models(verbose=False)),
        required=True,
        help="Select the model for training.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(zoo.list_datasets(verbose=False)),
        required=True,
        help="Select the dataset to train on.",
    )

    parser.add_argument(
        "--version",
        type=str,
        required=False,
        default=None,
        help="Specify a version id for given training. Optional.",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="The index of the GPU to use for training.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="The number of workers as an integer.",
    )

    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Enable fast development run (default is False).",
    )

    parser.add_argument(
        "--model_kwargs",
        type=json.loads,  # Assumes model_kwargs is a JSON string
        default=None,
        help="Optional model keyword arguments as a JSON string.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from a previous run (default is False).",
    )

    return parser


# Usage:
if __name__ == "__main__":
    zoo = ModelZoo(
        root_dir=project_dir / "models", sources_file=project_dir / "sources.yaml"
    )

    parser = create_arg_parser()
    args = parser.parse_args()

    train_name = f"{args.model}@{args.dataset}"

    if args.version:
        train_name += f"@{args.version}"

    zoo.train_model(
        name=train_name,
        gpus=args.gpu,
        num_workers=args.num_workers,
        fast_dev_run=args.fast_dev_run,
        model_kwargs=args.model_kwargs,
        resume=args.resume,
    )
