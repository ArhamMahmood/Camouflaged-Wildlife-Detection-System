
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Automatically locate the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Allow importing dataset.py from the project root
sys.path.insert(0, str(PROJECT_DIR))

from dataset import COD10KDataset


def visualize_sample(dataset_dir, split="Train", index=0):

    # Resolve the dataset directory supplied by the user
    dataset_dir = Path(dataset_dir).expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    dataset = COD10KDataset(
        root_dir=dataset_dir,
        split=split,
        image_size=(352, 352)
    )

    if index < 0 or index >= len(dataset):
        raise IndexError(
            f"Index must be between 0 and {len(dataset) - 1}"
        )

    image, mask = dataset[index]

    image = image.permute(1, 2, 0).numpy()
    mask = mask.squeeze(0).numpy()

    # Create side-by-side visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground-Truth Mask")
    plt.axis("off")

    plt.tight_layout()

    # Automatically save inside the project's results folder
    output_dir = PROJECT_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{split.lower()}_sample_{index}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Visualization completed successfully!")
    print(f"Dataset: {dataset_dir}")
    print(f"Split: {split}")
    print(f"Sample index: {index}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Visualize a COD10K image and its ground-truth mask."
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the extracted COD10K-v3 dataset directory."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="Train",
        choices=["Train", "Test"]
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0
    )

    args = parser.parse_args()

    visualize_sample(
        dataset_dir=args.dataset_dir,
        split=args.split,
        index=args.index
    )
