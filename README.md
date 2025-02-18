# Mask Detection Project

## Introduction
This project focuses on detecting whether individuals in an image or video are wearing masks. It utilizes **YOLOv5**, a state-of-the-art object detection model, trained on a dataset containing annotated images of people with and without masks.

## Repository Structure
- **best.pt**: The trained YOLOv5 model weights.
- **streamlit_app.py**: A web application built with Streamlit for real-time mask detection.
- **Yolov5_Mask_Detection.ipynb**: A Jupyter Notebook containing the training pipeline and evaluation.

### YOLOv5 Data Structure
Each image has a corresponding label file containing:
- Object class.
- Normalized bounding box coordinates (center_x, center_y, width, height).

**face-mask.yaml** defines:
- **train**: Path to training images.
- **val**: Path to validation images.
- **test**: Path to test images.
- **nc**: Number of classes (2).
- **names**: Class names (`mask`, `no-mask`).

## Model Training (YOLOv5)
### YOLOv5 Architecture:
- **Backbone (CSPDarknet53)**: Extracts features from images.
- **Neck (PANet)**: Merges multi-scale information.
- **Head**: Predicts bounding boxes and confidence scores.

![Capture d’écran 2025-02-18 140342 (1)](https://github.com/user-attachments/assets/613f6c91-2464-4514-b701-1ae1d98ff7cc)

### Training Process:
- **GPU Acceleration**: Initially on Google Colab, later on VS Code for efficiency.
- **Hyperparameters**:
  - Image Size: `640x640`
  - Epochs: `100`
  - Batch Size: `16`
- **Evaluation Metrics**: Precision, Recall, mAP@IoU.

## Model Inference
- Tested on **real-time video** and **static images**.
- Deployed via **Streamlit** for interactive detection.

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run streamlit_app.py
```

### 3. Run Inference on Image
```bash
python detect.py --weights best.pt --img 640 --conf 0.5 --source image.jpg
```

### 4. Run Inference on Video
```bash
python detect.py --weights best.pt --img 640 --conf 0.5 --source VideoMask.mp4
```

## Future Improvements
- Train on a **larger dataset** to improve generalization.
- Optimize inference speed for real-time detection.
- Deploy as a **web service** using FastAPI or Flask.


Would you like any modifications or additions? 🚀
