# Dental Radiography Analysis: Hybrid YOLO + U-Net Pipeline

## Overview
This project implements a hybrid deep learning pipeline for the detection and segmentation of dental abnormalities in radiography images. By combining **YOLOv8** (You Only Look Once) for object detection and **U-Net** for semantic segmentation, the system achieves precise localization and pixel-level delineation of dental conditions.

The pipeline is designed to identify the following conditions:
*   **Cavities**
*   **Fillings**
*   **Impacted Teeth**
*   **Implants**

## Key Features
*   **Hybrid Architecture**: Uses YOLOv8 to detect Regions of Interest (ROI) and U-Net to perform fine-grained segmentation within those ROIs. This "zoom-in" approach reduces background noise and improves segmentation accuracy.
*   **Automated Data Preparation**: Automatically converts bounding box annotations (CSV) into binary masks for U-Net and standard labels for YOLO.
*   **Robust Preprocessing**: detailed image augmentation pipeline using `Albumentations` (RandomBrightness, GaussNoise, MotionBlur, CLAHE, etc.).
*   **Dual Training**: Trains both specific object detection (YOLO) and segmentation (U-Net) models.
*   **Visualization Tools**: Includes comprehensive visualization for training metrics, YOLO detections, U-Net masks, and the combined hybrid inference results.

## Dataset
The project utilizes the **Dental Radiography** dataset hosted on Kaggle.
*   **Source**: [imtkaggleteam/dental-radiography](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography)
*   The dataset is automatically downloaded using `kagglehub`.

## Requirements
*   Python 3.8+
*   TensorFlow / Keras
*   Ultralytics (YOLOv8)
*   OpenCV
*   Albumentations
*   Pandas, NumPy, Matplotlib
*   KaggleHub

Install dependencies via pip:
```bash
pip install albumentations opencv-python tensorflow ultralytics kagglehub pandas matplotlib tqdm
```

## Project Workflow
1.  **Data Download**: Fetches dataset from KaggleHub.
2.  **Mask Generation**: Converts bounding box coordinates from `_annotations.csv` into binary masks for semantic segmentation training.
3.  **Data Augmentation**: Applies heavy augmentation to training images to improve model generalization.
4.  **U-Net Training**: Trains a custom U-Net model on the full or cropped images to learn pixel-wise features.
5.  **YOLO Training**: Formats data for YOLOv8 and fine-tunes a `yolov8n` model to detect dental classes.
6.  **Hybrid Inference**:
    *   **Step 1**: YOLO detects the tooth/abnormality bounding box.
    *   **Step 2**: The image is cropped to this box.
    *   **Step 3**: U-Net segments the specific abnormality processing only the relevant area.
    *   **Step 4**: The mask is projected back onto the original image.

## Usage

### Running the Pipeline
Run the main script to download data, train models, and visualize results:

```bash
python main.py
```
*(Note: Ensure your code is saved in a file, e.g., `main.py`)*

### Inference Example
To run inference on a new image using the trained models:

```python
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

# Load models
yolo_model = YOLO("yolo_model.pt")
unet_model = load_model("unet_model.h5")

# Run Hybrid Inference
# (Ensure the hybrid_inference function is available in your scope)
result_image = hybrid_inference(yolo_model, unet_model, "path/to/xray.jpg")

# Display
plt.imshow(result_image)
plt.show()
```

## Model Architecture details

### U-Net
*   **Encoder**: Standard convolutional blocks with MaxPooling.
*   **Decoder**: Transpose convolutions with skip connections (concatenation) from the encoder.
*   **Loss**: Binary Crossentropy.
*   **Optimizer**: Adam (lr=1e-4).

### YOLOv8
*   **Variant**: YOLOv8 Nano (`yolov8n.pt`) for speed and efficiency.
*   **Classes**: 4 (Cavity, Fillings, Impacted Tooth, Implant).
*   **Training**: 50 Epochs.

## Results
The script automatically plots:
*   Training/Validation Loss and Accuracy curves for U-Net.
*   Sample predictions showing YOLO bounding boxes vs. U-Net masks vs. Hybrid output.
*   IoU (Intersection over Union) metrics for validation.

## License
Dataset provided by `imtkaggleteam` on Kaggle. Please refer to the source for specific licensing terms. Code provided as an educational reference for hybrid computer vision pipelines.
