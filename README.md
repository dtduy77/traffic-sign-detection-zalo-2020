# Zalo Traffic Sign Detection 2020 🚦

This project implements a complete pipeline for the **Zalo AI Challenge 2020: Traffic Sign Detection** task, utilizing the **YOLOv26m** architecture (via Ultralytics). We chose providing **YOLOv26m** specifically because it offers the best balance between **high detection performance** and **inference speed**. It covers dataset preparation, robust preprocessing, training, and an advanced evaluation system with Vietnamese support.

## 📂 Project Structure

```
├── download_dataset.py          # Script to download dataset from Kaggle
├── preprocess_clean_augment.py  # Data cleaning and augmentation pipeline
├── train.ipynb                  # Training notebook (YOLO model training)
├── evaluation.py                # Inference and side-by-side visualization
├── requirements.txt             # Project dependencies
├── zalo_traffic_signs_v2_saved/ # Training outputs (weights, plots, logs)
└── result_side_by_side/         # Final evaluation results (visualizations)
```

## 🚀 Setup & Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/dtduy77/traffic-sign-detection-zalo-2020.git
   cd traffic-sign-detection-zalo-2020
   ```
2. **Create a Virtual Environment** (Recommended):

   * This prevents conflicts with your system's Python environment.
   * **For Windows:**

     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   * **For macOS / Linux:**

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *Key libraries: `ultralytics`, `opencv-python`, `albumentations`, `matplotlib`, `kagglehub`.*
4. **Download Weights**:

   * Download trained weights from [Google Drive](https://drive.google.com/drive/folders/1evX39Wb18gJlqRDeGu20eSk3arssSNC5?usp=sharing).
   * Place them in: `zalo_traffic_signs_v2_saved/weights/`.
   * *Recommendation: Use `best.pt` for evaluation and prediction.*
5. **Prepare Dataset**:

   * Run `python download_dataset.py` to fetch the data.
   * Run `python preprocess_clean_augment.py` to generate the high-quality training set.

## 📊 Exploratory Data Analysis (EDA)

Before training, we conducted a thorough analysis using `EDA.ipynb` to understand the dataset characteristics:

1. **Class Distribution**: Identified significant class imbalance, necessitating the use of augmentation strategies for rare classes.
2. **Box Quality**: Discovered "small boxes" (noise) and duplicate annotations that could hinder training stability.
3. **Visualization**: Implemented custom visualization with **Vietnamese font support** to inspect bounding boxes and labels accurately.

## 🛠️ Data Preprocessing Pipeline

We implemented a rigorous preprocessing pipeline in `preprocess_clean_augment.py` to ensure dataset quality:

1. **Noise Removal**: Filtered out "small boxes" (area < 40 px²) which are often noise or unrecognizable.
2. **Duplicate Suppression**: Removed overlapping bounding boxes of the same class with IoU > 0.7.
3. **Class Balancing**: Analyzed class distribution and applied **Albumentations** (Crop, Blur, Brightness) to augment rare classes, aiming for 80% sample count of the majority class.

## 🧠 Training Guide

The model training is handled via `train.ipynb`.

### 1. Dataset Setup Options

* **Option 1 (Fastest - Recommended)**: Use the **pre-cleaned & augmented** dataset hosted on my private Kaggle.
  * **Action**: Contact me at `duythduong.2003@gmail.com` to request the API Key.
  * **Usage**: Once you have the key, input it when running the download cell in `train.ipynb` (or `download_dataset.py`).
* **Option 2 (Manual Processing)**:
  1. Run `python download_dataset.py` to fetch the **raw** dataset.
  2. Run `python preprocess_clean_augment.py` to clean and augment the data.
  3. Upload the generated `zalo_clean_augmented_dataset` folder to your working environment (Colab/Kaggle).
  4. **Important**: Update the `downloaded_path` variable in **Step 2** of `train.ipynb` with the path to your uploaded dataset.

### 2. Training Process

* **Start from Scratch**:
  * Simply run all cells in `train.ipynb`.
  * This will initialize the YOLO model and start training for 100 epochs.
* **Resume Training**:
  1. Locate your last checkpoint (e.g., `zalo_traffic_signs_v2_saved/weights/last.pt`).
  2. Update the model loading line in `train.ipynb`:
     ```python
     model = YOLO('path/to/last.pt')
     ```
  3. Set `resume=True` in the `model.train(...)` function.
* **Completion**:
  * Once trained, results and weights are saved in `zalo_traffic_signs_v2_saved`.
  * Download this folder to keep your best model (`best.pt`) for inference.

### 3. Training Results (Best Model at Epoch 90)

* **Precision (B):** 0.956
* **Recall (B):** 0.897
* **mAP50 (B):** 0.950
* **mAP50-95 (B):** 0.697

The training demonstrated stable convergence and high accuracy across the diverse traffic sign categories.

## 📊 Evaluation & Visualization

The `evaluation.py` script provides a robust inference pipeline:

* **Batch Processing**: Automatically processes all images in the test folder.
* **Side-by-Side Comparison**: Stitches the **Original Image** and **Model Prediction** for easy manual verification.
* **Vietnamese Support**: Uses custom PIL-based text rendering to correctly display Vietnamese class names on bounding boxes.
* **Smart Label Placement**: Algorithms to prevent label overlap for crowded scenes.

**Run Evaluation:**

```bash
python evaluation.py
```

## ✨ Demo Results

Here are real-time detection results on Vietnamese streets:

<div align="center">
  <table border="0">
    <thead>
      <tr>
        <th width="45%" align="center">Original Video</th>
        <th width="45%" align="center">Prediction (YOLOv26m)</th>
      </tr>
    </thead>
    <tbody>
      <!-- Video 1 -->
      <tr>
        <td align="center"><video src="videos/7491066896592.mp4" width="100%" controls autoplay loop muted></video></td>
        <td align="center"><img src="videos/result/7491066896592_result.gif" width="100%" /></td>
      </tr>
      <!-- Video 2 -->
      <tr>
        <td align="center"><video src="videos/7491078013748.mp4" width="100%" controls autoplay loop muted></video></td>
        <td align="center"><img src="videos/result/7491078013748_result.gif" width="100%" /></td>
      </tr>
      <!-- Video 3 -->
      <tr>
        <td align="center"><video src="videos/7491078439517.mp4" width="100%" controls autoplay loop muted></video></td>
        <td align="center"><img src="videos/result/7491078439517_result.gif" width="100%" /></td>
      </tr>
      <!-- Video 4 -->
      <tr>
        <td align="center"><video src="videos/7491106513667.mp4" width="100%" controls autoplay loop muted></video></td>
        <td align="center"><img src="videos/result/7491106513667_result.gif" width="100%" /></td>
      </tr>
      <!-- Video 5 -->
      <tr>
        <td align="center"><video src="videos/7491106769782.mp4" width="100%" controls autoplay loop muted></video></td>
        <td align="center"><img src="videos/result/7491106769782_result.gif" width="100%" /></td>
      </tr>
    </tbody>
  </table>
</div>

> *The model successfully detects and translates traffic signs in real-time with high confidence.*

* **Small Object Limitations**: There are a few instances where extremely small bounding boxes are missed. This is an expected trade-off and is considered acceptable for this application.
* **Overall**: The model is highly robust and performs well on the public test set.
