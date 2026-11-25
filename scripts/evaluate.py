import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34

from py_sod_metrics import MAE, Smeasure, Emeasure, WeightedFmeasure


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(
            x, size=skip.shape[2:],
            mode="bilinear", align_corners=False
        )
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResNet34UNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        # Checkpoint supplies the trained weights.
        backbone = resnet34(weights=None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.center = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)

        self.final_upsample = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.center(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)

        output = F.interpolate(
            d1, size=input_size,
            mode="bilinear", align_corners=False
        )

        return self.final_upsample(output)


# ============================================================
# COD10K TEST DATASET
# ============================================================

class COD10KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=352):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

        self.pairs = []

        for image_path in sorted(self.image_dir.glob("*.jpg")):
            mask_path = self.mask_dir / (image_path.stem + ".png")
            if mask_path.exists():
                self.pairs.append((image_path, mask_path))

        if not self.pairs:
            raise RuntimeError(
                f"No matching image-mask pairs found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask


# ============================================================
# EVALUATION
# ============================================================

def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    project_dir = Path(args.project_dir)
    checkpoint_path = project_dir / "models" / "best_model.pth"

    test_image_dir = Path(args.data_dir) / "Test" / "Image"
    test_mask_dir = Path(args.data_dir) / "Test" / "GT_Object"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Loading test dataset...")

    dataset = COD10KDataset(
        test_image_dir,
        test_mask_dir,
        image_size=352
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    print("Test images:", len(dataset))

    print("\nLoading trained checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model = ResNet34UNet(pretrained=False)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(device)
    model.eval()

    print("Architecture:", checkpoint["architecture"])
    print("Best epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["best_val_loss"])
    print("Device:", device)

    mae_metric = MAE()
    sm_metric = Smeasure()
    em_metric = Emeasure()
    wf_metric = WeightedFmeasure()

    print("\nRunning test evaluation...")

    with torch.inference_mode():
        for batch_index, (images, masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            predictions = probabilities.squeeze(1).cpu().numpy()
            ground_truths = masks.squeeze(1).cpu().numpy()

            for pred, gt in zip(predictions, ground_truths):
                pred = pred.astype(np.float32, copy=False)
                gt = gt.astype(bool, copy=False)

                # Preserve original float probability maps.
                # Do not apply uint8 conversion or min-max normalization.
                mae_metric.step(pred, gt, normalize=False)
                sm_metric.step(pred, gt, normalize=False)
                em_metric.step(pred, gt, normalize=False)
                wf_metric.step(pred, gt, normalize=False)

            processed = min(
                (batch_index + 1) * args.batch_size,
                len(dataset)
            )

            if (batch_index + 1) % 25 == 0 or processed == len(dataset):
                print(f"Processed {processed}/{len(dataset)}")

    mae_results = mae_metric.get_results()
    sm_results = sm_metric.get_results()
    em_results = em_metric.get_results()
    wf_results = wf_metric.get_results()


    metrics = {
        "Model": "ResNet34-UNet",
        "Test Images": len(dataset),
        "S-measure": float(np.mean(sm_results["sm"])),
        "Mean E-measure": float(np.mean(em_results["em"]["curve"])),
        "Adaptive E-measure": float(em_results["em"]["adp"]),
        "Weighted F-measure": float(np.mean(wf_results["wfm"])),
        "MAE": float(np.mean(mae_results["mae"]))
    }


    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([metrics])

    csv_path = results_dir / "final_test_metrics.csv"
    md_path = results_dir / "final_test_metrics.md"

    results_df.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Final COD10K Test Performance\n\n"
        + results_df.to_markdown(index=False, floatfmt=".6f")
        + "\n\nEvaluation protocol: float probability predictions "
        "with `normalize=False`; all test images included.\n",
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("FINAL COD10K TEST PERFORMANCE")
    print("=" * 65)
    print(results_df.to_string(index=False))

    print("\nSaved:", csv_path)
    print("Saved:", md_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet34 U-Net on COD10K"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to COD10K-v3 dataset directory"
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)

    evaluate(parser.parse_args())
