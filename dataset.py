
# ==========================================
# COD10K CUSTOM PYTORCH DATASET
# ==========================================

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class COD10KDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="Train",
        image_size=(352, 352)
    ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size

        # ImageNet normalization parameters
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(3, 1, 1)

        # Define image and mask directories
        self.image_dir = self.root_dir / split / "Image"
        self.mask_dir = self.root_dir / split / "GT_Object"

        # Validate directories
        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)

        if not self.mask_dir.exists():
            raise FileNotFoundError(self.mask_dir)

        # Build image-mask pairs using filenames
        self.samples = []

        for image_path in sorted(self.image_dir.glob("*.jpg")):

            mask_path = self.mask_dir / f"{image_path.stem}.png"

            if mask_path.exists():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError("No valid image-mask pairs found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        # Read image in BGR format
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read ground-truth mask in grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        # Resize image using bilinear interpolation
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LINEAR
        )

        # Resize mask using nearest-neighbor interpolation
        mask = cv2.resize(
            mask,
            self.image_size,
            interpolation=cv2.INTER_NEAREST
        )

        # Convert pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Convert image from HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Convert arrays into PyTorch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Apply ImageNet normalization to the image
        image = (image - self.mean) / self.std

        return image, mask
