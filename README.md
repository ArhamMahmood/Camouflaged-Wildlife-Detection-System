# Camouflaged Wildlife Detection System

A PyTorch-based project focused on detecting camouflaged objects using the COD10K dataset.

The goal is to develop a deep learning system that can identify animals and other objects that blend into their surrounding environments.

**Project Status:** In Progress

---

## 1. Run the Visualization Script

The project includes a Python script that loads a COD10K image and its corresponding ground-truth mask and saves them side by side.

### Step 1: Clone the Repository

```bash
git clone https://github.com/ArhamMahmood/Camouflaged-Wildlife-Detection-System.git

cd Camouflaged-Wildlife-Detection-System
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download the COD10K Dataset

Download the dataset from the [official COD10K page](https://dengpingfan.github.io/pages/COD.html#download).

Extract the dataset and locate the `COD10K-v3` directory.

### Step 4: Run the Script

From the project root, execute:

**Windows:**
```bash
python scripts/visualize_dataset.py --dataset_dir "D:/Datasets/COD10K-v3" --split Train --index 0
```

**Linux:**
```bash
python scripts/visualize_dataset.py --dataset_dir "/home/user/datasets/COD10K-v3" --split Train --index 0
```

**Google Colab:**

Clone the repository and install dependencies:

```python
!git clone https://github.com/ArhamMahmood/Camouflaged-Wildlife-Detection-System.git

%cd Camouflaged-Wildlife-Detection-System

!pip install -r requirements.txt
```

If the dataset is stored in Google Drive, mount it:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Run the visualization:

```python
!python scripts/visualize_dataset.py \
    --dataset_dir "/content/drive/MyDrive/COD10K-v3" \
    --split Train \
    --index 0
```

Replace the dataset path with your own extraction location.

### Step 5: View the Result

The generated visualization is automatically saved inside:

```text
results/train_sample_0.png
```

In Google Colab, display it using:

```python
from IPython.display import display, Image

display(Image(filename="results/train_sample_0.png"))
```

**Available arguments:**

| Argument | Description | Default |
|---|---|---|
| `--dataset_dir` | Path to COD10K-v3 | Required |
| `--split` | Train or Test | Train |
| `--index` | Sample index | 0 |

The script does not contain machine-specific dataset paths. You only need to provide the dataset location when running it.

---

## 2. Dataset Information

This project uses the **COD10K (Camouflaged Object Detection) dataset**.

- Training images: 6,000
- Testing images: 4,000
- Total images: 10,000
- Annotations: Ground-truth object masks

**Dataset resources:**

- [Official COD10K Dataset Page](https://dengpingfan.github.io/pages/COD.html#download)
- [Original SINet GitHub Repository](https://github.com/DengPingFan/SINet)
- [Kaggle Dataset Mirror](https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset/data)

The dataset is not included in this GitHub repository. Users must download it separately.

**Usage note:** Review the dataset's usage terms before using it. The dataset documentation specifies non-commercial research use; commercial use requires permission from the authors.

---

## 3. Dataset Structure

After extraction, the expected dataset structure is:

```text
COD10K-v3/
├── Info/
├── Readme.txt
├── Train/
│   ├── Image/
│   ├── GT_Object/
│   ├── GT_Instance/
│   └── GT_Edge/
└── Test/
    ├── Image/
    ├── GT_Object/
    ├── GT_Instance/
    └── GT_Edge/
```

The current implementation uses:

- `Image/` for input images.
- `GT_Object/` for ground-truth masks.

Image and mask filenames must match.

---

## 4. Project Structure

```text
Camouflaged-Wildlife-Detection-System/
├── data/
├── notebooks/
│   └── 01_COD10K_Dataset_Preparation.ipynb
├── scripts/
│   └── visualize_dataset.py
├── models/
├── results/
├── dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

| File/Folder | Purpose |
|---|---|
| `dataset.py` | Custom PyTorch Dataset class |
| `notebooks/` | Dataset preparation notebook |
| `scripts/` | Visualization script |
| `data/` | Reserved for dataset-related files |
| `models/` | Reserved for model files |
| `results/` | Generated visualizations and future results |
| `requirements.txt` | Required Python packages |

---

## 5. Dataset Preparation

The custom `COD10KDataset` class in `dataset.py` handles:

- Loading image-mask pairs.
- Converting images from BGR to RGB.
- Resizing images and masks to 352 × 352.
- Normalizing pixel values to [0, 1].
- Converting images and masks into PyTorch tensors.

Images use bilinear interpolation, while masks use nearest-neighbor interpolation to preserve their labels.

**Expected tensor shapes:**

- Image: `[3, 352, 352]`
- Mask: `[1, 352, 352]`

The dataset preparation notebook also includes dataset inspection, image-mask verification, DataLoader testing, and visualization.

---

## 6. Requirements

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Pillow

Install dependencies using:

```bash
pip install -r requirements.txt
```

A GPU is recommended for future model training but is not required for the visualization script.

---

## 7. Current Development Progress

### Task 1 — Baseline Environment and Data Preparation

- [x] Environment setup
- [x] COD10K dataset preparation
- [x] Image-mask pair verification
- [x] Custom PyTorch Dataset class
- [x] DataLoader testing
- [x] Image and mask visualization
- [x] Initial project structure

### Upcoming Work

**Task 2 — Model Training and Evaluation**
- Model implementation and training
- Evaluation using standard COD metrics
- Results comparison

**Task 3 — Gradio Deployment**
- Develop the Gradio application
- Deploy to Hugging Face Spaces
- Final project handover

---

**Repository:** https://github.com/ArhamMahmood/Camouflaged-Wildlife-Detection-System
