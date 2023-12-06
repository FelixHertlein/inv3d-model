import cv2

cv2.setNumThreads(0)

import argparse
import sys
from pathlib import Path

project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir / "src"))

from inv3d_model.model_zoo import ModelZoo


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluation script")

    parser.add_argument(
        "--trained_model",
        type=str,
        choices=list(zoo.list_trained_models(verbose=False)),
        required=True,
        help="Select the model for evaluation.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(zoo.list_datasets(verbose=False)),
        required=True,
        help="Select the dataset to evaluate on.",
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

    return parser


# Usage:
if __name__ == "__main__":
    zoo = ModelZoo(
        root_dir=project_dir / "models", sources_file=project_dir / "sources.yaml"
    )

    parser = create_arg_parser()
    args = parser.parse_args()

    zoo.eval_model(
        trained_model=args.trained_model,
        dataset=args.dataset,
        gpu=args.gpu,
        num_workers=args.num_workers,
    )
