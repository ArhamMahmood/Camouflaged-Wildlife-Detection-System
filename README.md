Camouflaged Wildlife Detection System
A PyTorch-based Camouflaged Object Detection (COD) project using the COD10K dataset. This project aims to prepare and develop a deep learning system that identifies objects, particularly wildlife, that blend into their surrounding environments.
GitHub Repository:
https://github.com/ArhamMahmood/Camouflaged-Wildlife-Detection-System
1. Project Overview
Camouflaged Object Detection (COD) is a computer vision task that identifies objects that have similar colors, patterns, or textures to their surroundings.
Unlike ordinary object detection, camouflaged object detection is challenging because objects can be difficult to distinguish from the background.
This project uses the COD10K dataset to develop and evaluate a deep learning-based camouflaged object detection system.
Project Objectives
Prepare and organize the COD10K dataset.
Implement a custom PyTorch Dataset class for loading image-mask pairs.
Visualize input images alongside their ground-truth masks.
Develop and train a camouflaged object detection model.
Evaluate the model using standard COD evaluation metrics.
Develop a Gradio application for model demonstration.
2. Dataset Information
COD10K Dataset
The project uses the COD10K (Camouflaged Object Detection) dataset.
Official dataset page:
https://dengpingfan.github.io/pages/COD.html#download
Original authors' GitHub repository:
https://github.com/DengPingFan/SINet
Kaggle mirror used for the current dataset copy:
https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset/data
The official research page is the primary reference for dataset information and author-provided downloads. The Kaggle link is a mirror, not the original authors' repository.
Dataset Statistics
Split
Number of Images
Training
6,000
Testing
4,000
Total
10,000

The dataset includes RGB images and corresponding ground-truth annotations.
Dataset Usage Terms
Please review the COD10K dataset's usage terms before downloading or using it.
The dataset documentation states that it is available for non-commercial research use. Commercial use requires permission from the dataset authors.
For commercial-use inquiries, contact the authors at:
dengpfan@gmail.com
Do not assume that using a publicly accessible dataset automatically grants commercial permission.
Important
The dataset is not included in this GitHub repository because of its size and distribution requirements.
Each user must obtain the dataset separately and provide its local directory path when running the project.
3. Repository Structure
After cloning the repository, the project structure is:
Camouflaged-Wildlife-Detection-System/
│
├── data/
│   └── .gitkeep
│
├── notebooks/
│   └── 01_COD10K_Dataset_Preparation.ipynb
│
├── scripts/
│   └── visualize_dataset.py
│
├── models/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── dataset.py
├── requirements.txt
├── .gitignore
└── README.md
Directory Description
File/Directory
Purpose
data/
Reserved for dataset-related files. The COD10K archive is stored separately.
notebooks/
Jupyter notebooks for dataset preparation and experiments.
scripts/
Standalone Python scripts.
models/
Reserved for model definitions and checkpoints.
results/
Stores generated visualizations and future evaluation results.
dataset.py
Custom PyTorch Dataset implementation.
requirements.txt
Python dependency list.
.gitignore
Excludes datasets, generated files, and temporary files from Git.
4. Requirements
Before running the project, install:
Python 3.10 or newer
PyTorch
Torchvision
OpenCV
NumPy
Matplotlib
Pillow
The project can be executed on:
Windows
Linux
Google Colab
Other compatible Python environments
A CUDA-compatible GPU is recommended for future model training, but it is not required just to run the dataset visualization script.
5. Setup Instructions — Local Computer
Follow these steps to run the project on your own computer.
Step 1: Clone the GitHub Repository
Open a terminal or command prompt and execute:
git clone https://github.com/ArhamMahmood/Camouflaged-Wildlife-Detection-System.git
Move into the project directory:
cd Camouflaged-Wildlife-Detection-System
Step 2: Create a Virtual Environment
Create a Python virtual environment:
python -m venv venv
Activate it.
Windows:
venv\Scripts\activate
Linux:
source venv/bin/activate
Step 3: Install Dependencies
Install the required packages:
pip install -r requirements.txt
If your environment requires a specific PyTorch build for CUDA, install the appropriate PyTorch version using the official installation selector:
https://pytorch.org/get-started/locally/
Step 4: Download the COD10K Dataset
Obtain the COD10K dataset from the official research page:
https://dengpingfan.github.io/pages/COD.html#download
Alternatively, consult the Kaggle mirror listed in the Dataset Information section.
Download and extract the dataset.
Step 5: Verify the Dataset Structure
After extraction, the dataset should have this structure:
COD10K-v3/
│
├── Info/
├── Readme.txt
│
├── Train/
│   ├── Image/
│   ├── GT_Edge/
│   ├── GT_Instance/
│   └── GT_Object/
│
└── Test/
├── Image/
├── GT_Edge/
├── GT_Instance/
└── GT_Object/
Important: The directory supplied to the script must be the COD10K-v3 folder itself, not its Train or Test subfolder.
The current implementation uses:
Train/Image/ and Test/Image/ for RGB images.
Train/GT_Object/ and Test/GT_Object/ for ground-truth object masks.
Image and mask filenames must have matching file stems.
Step 6: Run the Visualization Script
From the project root directory, execute:
python scripts/visualize_dataset.py --dataset_dir "PATH_TO_COD10K-v3" --split Train --index 0
Replace PATH_TO_COD10K-v3 with the actual dataset directory.
Example — Windows:
python scripts/visualize_dataset.py --dataset_dir "D:/Datasets/COD10K-v3" --split Train --index 0
Example — Linux:
python scripts/visualize_dataset.py --dataset_dir "/home/user/datasets/COD10K-v3" --split Train --index 0
Step 7: View the Generated Visualization
After successful execution, the visualization will be saved automatically inside:
results/
└── train_sample_0.png
The generated image contains:
Original RGB image.
Corresponding ground-truth object mask.
You can open the PNG file using any image viewer.
6. Setup Instructions — Google Colab
The project can also be executed using Google Colab without requiring a local Python installation.
Step 1: Open Google Colab
Visit:
https://colab.research.google.com/
Create a new notebook.
Step 2: Clone the GitHub Repository
Run:
!git clone https://github.com/ArhamMahmood/Camouflaged-Wildlife-Detection-System.git
Move into the project directory:
%cd Camouflaged-Wildlife-Detection-System
Step 3: Install Dependencies
Run:
!pip install -r requirements.txt
Colab may already have some dependencies installed. Installing the requirements ensures the project dependencies are available.
Step 4: Mount Google Drive
If the dataset is stored in Google Drive, mount it:
from google.colab import drive

drive.mount('/content/drive')
Follow the authorization instructions.
Step 5: Download and Extract COD10K
Download COD10K from the official dataset page or an available mirror.
Extract it into Google Drive.
For example, if the extracted directory is:
/content/drive/MyDrive/COD10K-v3/
Make sure it contains the Train, Test, and annotation directories described above.
Step 6: Run the Visualization Script
From the project root directory, run:
!python scripts/visualize_dataset.py
--dataset_dir "/content/drive/MyDrive/COD10K-v3"
--split Train
--index 0
Change the dataset path if you extracted the dataset somewhere else.
Step 7: Display the Generated Image in Colab
Run:
from IPython.display import display, Image

display(Image(filename="results/train_sample_0.png"))
You should see the original image and its corresponding ground-truth mask.
7. Dataset Loading Implementation
The project contains a custom PyTorch Dataset class in:
dataset.py
The class is named:
COD10KDataset
Main Responsibilities
The custom Dataset class:
Locates the image and mask directories.
Finds matching image-mask pairs using their filenames.
Loads RGB images using OpenCV.
Loads ground-truth object masks.
Resizes images and masks to 352 × 352.
Converts images and masks into PyTorch tensors.
Normalizes pixel values to the range [0, 1].
Image and Mask Processing
Different COD10K images have different dimensions.
To make them compatible with batching, the implementation resizes them to:
352 × 352
Interpolation methods:
Images: Bilinear interpolation.
Masks: Nearest-neighbor interpolation.
Nearest-neighbor interpolation is used for masks to preserve their binary label values.
Tensor Shapes
Data
Tensor Shape
Image
[3, 352, 352]
Mask
[1, 352, 352]

Both tensors use float32.
8. Visualization Script
The standalone visualization script is located at:
scripts/visualize_dataset.py
It uses the custom COD10KDataset class to load a sample image and its corresponding ground-truth mask.
Available Arguments
Argument
Description
Default
--dataset_dir
Path to the extracted COD10K-v3 directory
Required
--split
Dataset split: Train or Test
Train
--index
Index of the sample to visualize
0
Examples
Visualize the first training sample:
python scripts/visualize_dataset.py --dataset_dir "/path/to/COD10K-v3" --split Train --index 0
Visualize the tenth training sample:
python scripts/visualize_dataset.py --dataset_dir "/path/to/COD10K-v3" --split Train --index 9
Visualize the first testing sample:
python scripts/visualize_dataset.py --dataset_dir "/path/to/COD10K-v3" --split Test --index 0
Output
Generated visualizations are saved in the repository's results/ directory.
The output filename is generated using the selected split and sample index.
For example:
results/
├── train_sample_0.png
├── train_sample_9.png
└── test_sample_0.png
9. Dataset Preparation Notebook
The repository also includes:
notebooks/01_COD10K_Dataset_Preparation.ipynb
This notebook documents the initial dataset preparation workflow, including:
Environment setup and dataset access.
Dataset directory inspection.
Image-mask pairing verification.
Dataset statistics.
Image and mask dimension inspection.
Custom PyTorch Dataset implementation and testing.
DataLoader testing.
Side-by-side image and ground-truth mask visualization.
To run the notebook locally, install Jupyter:
pip install jupyter
Then launch it:
jupyter notebook
Open the notebook from the notebooks/ directory.
In the notebook, configure the dataset directory variable to match the location where COD10K was extracted. Notebook cells may require adapting their dataset paths to the current environment.
10. Model Training and Evaluation
Model Architecture
The project uses a ResNet34-based U-Net architecture for pixel-level camouflaged object segmentation.
- Architecture: ResNet34 U-Net
- Encoder: ImageNet-pretrained ResNet34
- Input resolution: 352 × 352
- Output: Single-channel segmentation logits
- Loss function: Combined Binary Cross-Entropy and Dice loss
- Optimizer: AdamW
- Training epochs: 20
- Best checkpoint: Epoch 8
Training Configuration
Setting	Value
Training images	5,400
Validation images	600
Batch size	8
Input size	352 × 352
Encoder learning rate	0.0001
Decoder learning rate	0.001
Weight decay	0.0001
Loss	0.5 BCE + 0.5 Dice


The training process used automatic mixed precision and checkpoint recovery. The best checkpoint was selected using validation loss.
Final Test Performance
The best checkpoint (epoch 8) was evaluated on the 4,000 images in the COD10K v3 test folder.
Metric	ResNet34 U-Net
S-measure (Sα) ↑	0.837567
Mean E-measure (Eφ) ↑	0.849610
Adaptive E-measure ↑	0.817374
Weighted F-measure (Fω) ↑	0.235738
MAE ↓	0.035727


Reference Results from Published COD10K Research
The following fully supervised methods and scores are reported in the COD10K comparison table of the IJCAI 2026 paper.
Model	Backbone	S-measure ↑	E-measure ↑	Weighted F-measure ↑	MAE ↓
SINet (2020)	ResNet50	0.776	0.864	0.631	0.043
UGTR (2021)	ResNet50	0.817	0.852	0.666	0.036
ZoomNet (2022)	ResNet50	0.838	0.911	0.729	0.029
FEDER (2023)	Res2Net50	0.844	0.911	0.736	0.029
Our ResNet34 U-Net	ResNet34	0.837567	0.849610	0.235738	0.035727


Comparison note: The published paper reports the standard COD10K benchmark results, while this project evaluated the full 4,000-image COD10K v3 test folder. The benchmark table identifies 2,026 COD10K test images. Therefore, these values are reference results and should not be interpreted as a controlled, like-for-like comparison. Evaluation implementations and test protocols may also differ.

Published reference: From Language to Segmentation: Collaborative Category-Guided Unsupervised Camouflaged Object Detection with SAM3 — IJCAI 2026.
Training Curves
The following training visualizations are generated and saved in the results/ directory:
- training_validation_loss.png — training and validation loss across 20 epochs.
- training_validation_accuracy.png — training and validation pixel accuracy across 20 epochs.
The lowest validation loss was 0.251776 at epoch 8. Validation loss fluctuated and generally increased after that point while training loss continued to decrease, indicating overfitting. Accordingly, the epoch-8 checkpoint was retained as the best model.
Evaluation Artifacts
The evaluation workflow generates:
- results/final_test_metrics.csv
- results/final_test_metrics.md
- results/training_history.csv
- scripts/evaluate.py — standalone test-set evaluation script.
The model checkpoint is saved as models/best_model.pth when included in the repository. The COD10K dataset itself is not included.
11. Troubleshooting
Error: Dataset directory not found
Cause: The supplied dataset path is incorrect.
Solution: Check the dataset extraction location and ensure the path points directly to COD10K-v3.
Error: No valid image-mask pairs found
Cause: The image and mask directories may be missing, empty, or contain filenames that do not match.
Solution: Verify that:
Train/Image/
Train/GT_Object/
Test/Image/
Test/GT_Object/
exist and that matching image and mask files have the same filename stem.
Error: ModuleNotFoundError
Cause: A required Python dependency is missing.
Solution:
pip install -r requirements.txt
Error: Visualization file not found
Cause: The script may not have completed successfully, or the output may have been saved under a different split or sample index.
Solution: Check the terminal output for the generated file location and inspect the project's results/ directory.
Error: Index out of range
Cause: The requested sample index is outside the selected dataset split.
Solution: Use a valid index:
Training indices: 0–5999
Testing indices: 0–3999
12. Current Project Status
Task 1 — Baseline Environment and Data Preparation
Development environment setup.
COD10K dataset downloaded and extracted.
Dataset directory structure inspected.
Image-mask pairs verified.
Custom PyTorch Dataset class implemented.
DataLoader tested.
Image and ground-truth mask visualization implemented.
Project directory structure prepared.
README and setup instructions documented.
Note: GitHub publication and final repository verification should be completed separately.
Task 2 — Model Training and Metric Evaluation
- ResNet34 U-Net model implemented and trained.
- Best checkpoint selected using validation loss (epoch 8).
- Test evaluation completed on 4,000 COD10K v3 images.
- S-measure, E-measure, weighted F-measure, and MAE calculated.
- Standalone evaluation script and metrics reports generated.
- Training and validation curves generated.
Upcoming Development Phase
Task 3 — Gradio Deployment and Handover
Build an interactive Gradio application.
Deploy the application to Hugging Face Spaces.
Prepare project documentation and handover.
13. Important Notes
The COD10K dataset is not stored in this GitHub repository.
Users must download and extract the dataset separately.
Dataset paths must be supplied according to the user's own environment.
The visualization script does not require editing when changing computers or Colab environments.
The notebook may require updating its dataset directory variable.
Review the dataset license and obtain required permissions before commercial use.
14. Acknowledgements
We acknowledge the COD10K dataset authors for providing the dataset for camouflaged object detection research.
Dataset research page:
https://dengpingfan.github.io/pages/COD.html#download
Original SINet implementation:
https://github.com/DengPingFan/SINet

Repository
GitHub: https://github.com/ArhamMahmood/Camouflaged-Wildlife-Detection-System
