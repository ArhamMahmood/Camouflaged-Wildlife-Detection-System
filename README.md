# Camouflaged Wildlife Detection System

A deep-learning project for **camouflaged object segmentation** using the COD10K dataset and a ResNet34 U-Net model. The project covers dataset preparation, model training, quantitative evaluation, and an interactive Gradio demonstration deployed on Hugging Face Spaces.

## Live Demo

**Try the application:** [Camouflaged Wildlife Detection System](https://arhammah-camouflaged-wildlife-detection-system.hf.space)

Upload an image to view:
- **Prediction probability map** — visualizes the model's predicted foreground probabilities.
- **Predicted binary mask** — converts the probability map into a foreground/background mask using the selected threshold.
- **Detection overlay** — overlays the predicted foreground region on the original image.

The public Space uses Hugging Face ZeroGPU. The application may take time to wake after inactivity, and free GPU availability is subject to the platform's usage limits.

## 1. Project Overview

Camouflaged object detection (COD) is a computer-vision task that identifies objects whose colors, patterns, or textures closely resemble their surroundings. This project trains a segmentation model to predict the pixel-level foreground mask of camouflaged objects in an image.

### Objectives

- Prepare and inspect the COD10K dataset.
- Implement a PyTorch dataset loader for image/ground-truth mask pairs.
- Visualize images and their annotations.
- Train a ResNet34 U-Net segmentation model.
- Evaluate the model using standard camouflaged-object-detection metrics.
- Provide an interactive Gradio application for demonstration and inference.

## 2. Dataset

This project uses **COD10K (Camouflaged Object Detection)**.

- Official dataset page: https://dengpingfan.github.io/pages/COD.html#download
- Original authors' repository: https://github.com/DengPingFan/SINet
- Kaggle mirror used during this project: https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset/data

The official research page is the primary reference. The Kaggle page is a mirror, not the original authors' repository.

### Dataset split

| Split | Images |
|---|---:|
| Training | 6,000 |
| Testing | 4,000 |
| **Total** | **10,000** |

The dataset contains RGB images and corresponding ground-truth annotations. The project uses the `Image/` and `GT_Object/` directories.

### Expected dataset structure

```text
COD10K-v3/
├── Train/
│   ├── Image/
│   └── GT_Object/
└── Test/
    ├── Image/
    └── GT_Object/
```

Image and mask filenames must have matching file stems. The dataset is **not included** in this GitHub repository; users must obtain it separately and configure their own dataset path.

### Dataset usage

Review the dataset authors' terms before downloading or using COD10K. The dataset documentation indicates non-commercial research use; commercial use requires permission from the authors. Do not assume that public availability grants commercial rights. Refer to the official dataset page for current terms and contact the authors for permission where required.

## 3. Model Architecture

| Setting | Value |
|---|---|
| Architecture | ResNet34 U-Net |
| Encoder | ImageNet-pretrained ResNet34 |
| Input resolution | 352 × 352 |
| Output | Single-channel segmentation logits |
| Loss | 0.5 Binary Cross-Entropy + 0.5 Dice |
| Optimizer | AdamW |
| Training epochs | 20 |
| Selected checkpoint | Epoch 8 |

The best checkpoint was selected using validation loss. Training used a 5,400-image training subset and a 600-image validation subset.

### Training configuration

| Setting | Value |
|---|---:|
| Training images | 5,400 |
| Validation images | 600 |
| Batch size | 8 |
| Input size | 352 × 352 |
| Encoder learning rate | 0.0001 |
| Decoder learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss | 0.5 BCE + 0.5 Dice |

Automatic mixed precision and recovery checkpoints were used during training.

## 4. Test Evaluation

The selected epoch-8 checkpoint was evaluated on the 4,000 images in the COD10K v3 test folder.

| Metric | Result | Direction |
|---|---:|---|
| S-measure (Sα) | 0.837567 | Higher is better |
| Mean E-measure (Eφ) | 0.849610 | Higher is better |
| Adaptive E-measure | 0.817374 | Higher is better |
| Weighted F-measure (Fω) | 0.235738 | Higher is better |
| MAE | 0.035727 | Lower is better |

These are the results produced by this project's evaluation workflow. Results from other papers or implementations should not be treated as directly comparable unless the dataset split, evaluation protocol, and metric implementation are the same.

### Evaluation artifacts

The evaluation workflow produced the following project artifacts:

- `results/final_test_metrics.csv`
- `results/final_test_metrics.md`
- `results/training_history.csv`
- `scripts/evaluate.py` — standalone test-set evaluation script

Training and validation loss/accuracy visualizations were also generated during experimentation.

## 5. Try the Gradio Application

Open the public demo:

**[Launch Camouflaged Wildlife Detection System](https://arhammah-camouflaged-wildlife-detection-system.hf.space)**

### Usage instructions

1. Open the live demo link.
2. Upload an image containing a scene in which a camouflaged object may be present.
3. Adjust the prediction threshold if needed. A higher threshold generally selects fewer pixels as foreground; a lower threshold generally selects more.
4. Click **Generate Prediction**.
5. Inspect the probability map, predicted binary mask, and detection overlay.

The model resizes the uploaded image to 352 × 352 for inference, applies ImageNet normalization, and resizes the predicted map back to the original image dimensions for display.

### Deployment

- Platform: Hugging Face Spaces
- SDK: Gradio
- Hardware: ZeroGPU (free-tier Space)
- Public application URL: https://arhammah-camouflaged-wildlife-detection-system.hf.space

The Space may enter an idle/sleeping state and need time to start when opened. ZeroGPU access is subject to Hugging Face's current free-tier quota and availability. Therefore, uninterrupted 24/7 inference is not guaranteed.

## 6. Repository Structure

The main project and deployment files include:

```text
Camouflaged-Wildlife-Detection-System/
├── app.py
├── dataset.py
├── requirements.txt
├── README.md
├── models/
│   ├── unet.py
│   └── best_model.pth
├── notebooks/
├── scripts/
│   └── evaluate.py
├── results/
└── .gitignore
```

The COD10K dataset is kept outside the repository. Large model weights may be stored using Git LFS; check that the checkpoint is available to anyone expected to reproduce local inference.

## 7. Environment and Installation

### Requirements

- Python 3.10 or newer (use a Python version supported by the selected PyTorch build).
- PyTorch and a compatible Torchvision version.
- NumPy, Pillow, OpenCV, and the packages listed in `requirements.txt`.
- A CUDA-capable GPU is recommended for training. CPU inference is possible but may be slower.

### Local setup

Clone the repository:

```bash
git clone https://github.com/ArhamMahmood/Camouflaged-Wildlife-Detection-System.git
cd Camouflaged-Wildlife-Detection-System
```

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA-specific PyTorch installation, select the correct command for your operating system and CUDA environment from the official PyTorch guide: https://pytorch.org/get-started/locally/

### Dataset preparation and visualization

1. Download COD10K from the official dataset page and extract it.
2. Confirm that the extracted folder contains `Train/Image`, `Train/GT_Object`, `Test/Image`, and `Test/GT_Object`.
3. Run the visualization script from the repository root:

```bash
python scripts/visualize_dataset.py --dataset_dir "PATH_TO_COD10K-v3" --split Train --index 0
```

Replace `PATH_TO_COD10K-v3` with the path to your extracted dataset folder. The script saves a visualization under `results/`.

### Google Colab

The project can also be used in Google Colab. Clone the repository, install the requirements, mount Google Drive if the dataset is stored there, and configure the dataset path in the relevant notebook or script. Do not use another person's local or Drive path.

## 8. Application Limitations

- **Research prototype:** This application is an experimental demonstration, not a production wildlife-monitoring system.
- **Prediction uncertainty:** The model may miss camouflaged objects, include background pixels, or produce incomplete/incorrect masks.
- **Dataset coverage:** Performance depends on the visual patterns and object/background conditions represented in COD10K. Generalization to other environments, species, cameras, lighting, and image quality is not guaranteed.
- **Single-image segmentation:** The application predicts a foreground mask; it does not identify species, verify animal identity, or provide a bounding-box-based object detector.
- **Input resizing:** Images are resized to 352 × 352 for model inference. Fine details may be lost during resizing.
- **Threshold sensitivity:** The selected threshold changes the binary mask and can affect the apparent size of the predicted region.
- **Compute availability:** The public demo uses free ZeroGPU hardware. Cold starts, queueing, temporary unavailability, and usage quotas may affect response time.
- **Evaluation scope:** Reported metrics describe the project's evaluation run and should not be interpreted as a guarantee of performance on arbitrary user images.

## 9. Disclaimer

**This tool is provided for research and educational purposes only.** It is not intended to serve as definitive wildlife identification, legal evidence, or a substitute for expert review. Predictions must be independently checked by qualified people before being used to support environmental, legal, conservation, or operational decisions. The developers make no guarantee that every camouflaged object will be detected accurately.

Users are responsible for ensuring they have the right to upload and process any image they submit. Do not upload confidential, sensitive, or personally identifying material.

## Project Use and Permissions Disclaimer

**This project is provided strictly for educational, learning, and non-commercial research purposes. It is not intended or authorized for commercial use.**

You may review the project to learn about camouflaged object detection, deep learning, PyTorch, model training, and Gradio deployment. However, do not commercially use, redistribute, publish, modify for commercial purposes, or reuse substantial parts of this project's code, implementation, documentation, or other original project materials without obtaining prior written permission from the **original author and creator of this project**.

If you wish to use, adapt, reproduce, redistribute, or otherwise incorporate any part of this project into another project, please contact the original author and creator to request permission first.

This notice applies to the original materials created for this project. Third-party materials, libraries, pretrained weights, and datasets remain subject to their respective owners' licenses and terms. In particular, use of the COD10K dataset is governed by the dataset authors' terms; contact the COD10K authors for permissions relating to that dataset. Permission from this project's creator does not replace permission required from third-party rights holders.

## 10. Troubleshooting

| Issue | Suggested check |
|---|---|
| Dataset directory not found | Confirm that the supplied path points to the extracted `COD10K-v3` directory. |
| No matching image-mask pairs | Check that the expected image and mask directories exist and filenames match. |
| `ModuleNotFoundError` | Activate the intended environment and install the dependencies from `requirements.txt`. |
| Model checkpoint not found | Confirm that `models/best_model.pth` exists and is accessible. |
| Hugging Face Space is starting or sleeping | Wait for the Space to initialize; free-tier hardware may have cold starts. |
| Prediction fails | Review the Space runtime logs and confirm that the uploaded image is a supported image file. |

## 11. Project Status

### Task 1 — Baseline Environment and Data Preparation
- Development environment configured.
- COD10K dataset downloaded, extracted, and inspected.
- Image-mask pairs checked.
- Custom PyTorch dataset and DataLoader implemented and tested.
- Image and ground-truth mask visualization implemented.

### Task 2 — Model Training and Metric Evaluation
- ResNet34 U-Net implemented and trained.
- Best checkpoint selected using validation loss (epoch 8).
- Test evaluation completed on 4,000 COD10K v3 images.
- S-measure, E-measure, weighted F-measure, and MAE calculated.
- Training history and evaluation artifacts generated.

### Task 3 — Gradio Deployment and Project Handover
- Gradio application implemented to accept user-uploaded images.
- Application returns a probability map, binary segmentation mask, and detection overlay.
- Application deployed to a public Hugging Face Space using free ZeroGPU hardware.
- README updated with demo link, usage instructions, limitations, and research disclaimer.

## 12. Acknowledgements

We acknowledge the COD10K dataset authors for providing the dataset for camouflaged object detection research.

- COD10K research page: https://dengpingfan.github.io/pages/COD.html#download
- Original SINet repository: https://github.com/DengPingFan/SINet

## Project Links

- **Live Gradio demo:** https://arhammah-camouflaged-wildlife-detection-system.hf.space
- **GitHub repository:** https://github.com/ArhamMahmood/Camouflaged-Wildlife-Detection-System
- **Hugging Face Space repository:** https://huggingface.co/spaces/ArhamMah/camouflaged-wildlife-detection-system
