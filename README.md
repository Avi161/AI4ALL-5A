# AI4ALL-5A (FloodNet_Colab.ipynb)
# 🌊 FloodNet: Disaster Prediction using Satellite Imagery

**FloodNet** is a deep learning project capable of detecting flood events by fusing multi-modal satellite imagery. It combines **Sentinel-1 (SAR)** and **Sentinel-2 (Optical)** data to predict high-risk zones, even in cloudy conditions where traditional optical sensors fail.

This repository contains the code for preprocessing, training, and evaluating the **SimpleCNN** binary classification model.

## 🚀 Key Features
* **Multi-Modal Sensor Fusion:** Stacks Synthetic Aperture Radar (SAR) and Optical spectral bands for robust detection.
* **Custom Preprocessing:** Handles distinct data distributions (Log-scaling for SAR, Min-Max for Optical).
* **Binary Classification:** Predicts "Flood" (1) or "No Flood" (0) for specific geographic tiles.
* **Performance Metrics:** Includes automated generation of Confusion Matrices, ROC Curves, and F1 Scores.

---

## 🛠️ Prerequisites

This project is designed to run on **Google Colab** to leverage free GPU resources.

### Libraries Used
* `torch` & `torchvision` (Deep Learning)
* `rasterio` (Satellite Image Processing)
* `numpy` & `pandas` (Data Manipulation)
* `matplotlib` & `seaborn` (Visualization)
* `scikit-learn` (Metrics)

---

## 📂 Dataset & Directory Structure

To replicate this project, you must structure your data exactly as the code expects in your **Google Drive**.

1.  **Download the Data:** This project uses the **SEN12-FLOOD** dataset.
2.  **Google Drive Setup:**
    Create a folder in your Drive named `AI4ALL_Project`. Inside, ensure your file structure looks like this:



/content/drive/MyDrive/FloodNet/    # Or your configured root directory
└── SEN12FLOOD/                     # Main Dataset Directory
    ├── S2list.json                 # Critical metadata file listing all scenes
    ├── [Scene_Folder_ID_1]/        # e.g. '2018_12_18_...'
    │   ├── [Prefix]_B2.tif         # Individual band files
    │   ├── [Prefix]_B3.tif
    │   ├── [Prefix]_B4.tif
    │   └── ...
    ├── [Scene_Folder_ID_2]/
    └── ...

Note: If you store your data elsewhere, you must update the root_dir variable in the FloodNet_Colab.ipynb notebook.

**⚙️ Installation & Usage**
1. Open in Google Colab
Upload the FloodNet_Colab.ipynb file to your Google Drive or open it directly from this repository using Colab.

2. Install Dependencies
The first cell of the notebook installs the required geospatial libraries. Run it once at the start of your session:

**Python**

!pip install rasterio
3. Mount Google Drive
The code requires access to your Drive to load the dataset. Run the mounting cell and authorize access:

**Python**

from google.colab import drive
drive.mount('/content/drive')
4. Train the Model
Run the training cells. The script will:

Load and preprocess the S1 and S2 images.

Train the SimpleCNN for 15 Epochs.

Save the best model weights based on Validation Accuracy.

5. Evaluate
The final cells will generate:

Confusion Matrix: To visualize True Positives vs. False Negatives.

ROC Curve: To check the area under the curve (AUC).

Sample Predictions: Visual comparison of Ground Truth vs. Model Prediction.

**🧠 Model Architecture**
The model is a custom Convolutional Neural Network (CNN) optimized for binary classification.

Input: Stacked Tensor (Sentinel-1 Bands + Sentinel-2 Bands)

Hidden Layers:

3x Convolutional Layers (Feature Extraction)

ReLU Activations

Max Pooling (Downsampling)

Dropout (0.5) to prevent overfitting

Output: Sigmoid Activation (Probability Score 0.0 - 1.0)

Hyperparameters:

Optimizer: Adam

Learning Rate: 1e-4

Batch Size: 16

Loss Function: Binary Cross Entropy (BCEWithLogitsLoss)

**📊 Results**
Accuracy: 91.5%

**Recall:** 92.7%

**AUC:** 0.979

**📜 Citations**
If you use this code or methodology, please credit the original data sources:

SEN12-FLOOD: A SAR and Multispectral Benchmark for Flood Detection

NASA Earth Applied Sciences (Flood Detection)
