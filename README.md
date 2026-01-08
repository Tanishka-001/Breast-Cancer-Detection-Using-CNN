# Breast Cancer Detection using CNN (IDC Histopathology Images)

## Executive Summary
This project applies a Convolutional Neural Network (CNN) to classify breast histopathology image patches as **Invasive Ductal Carcinoma (IDC) positive or negative**. The goal is to assist early cancer detection by automating image-based diagnosis and reducing dependence on manual visual inspection.

---

## Problem Statement
Invasive Ductal Carcinoma (IDC) accounts for nearly **80% of breast cancer cases**. Diagnosis from biopsy images is time-consuming and heavily dependent on expert interpretation. Manual analysis may lead to variability in results and delayed treatment decisions, especially in resource-constrained settings.

Traditional machine learning approaches struggle to capture complex spatial patterns in medical images, making deep learning-based solutions more suitable.

---

## Solution Overview
A CNN-based binary image classification pipeline was developed to:
- Learn hierarchical features from histopathology images
- Accurately distinguish cancerous and non-cancerous tissue
- Evaluate performance using recall-focused medical metrics

---

## Dataset
- **Source:** Kaggle – Breast Histopathology Images  
- **Link:** https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images  
- **Image Size Used:** 50 × 50 RGB  
- **Classes:**
  - IDC (−): Non-cancerous tissue
  - IDC (+): Cancerous tissue

⚠️ The dataset is **not included** in this repository due to Kaggle licensing restrictions.

---

## Methodology

### Data Preprocessing
- Image loading using OpenCV
- Resizing all images to 50×50 pixels
- Binary labeling (IDC positive / negative)
- Train–test split (70% training, 30% testing)
- One-hot encoding of labels

### Model Architecture
- Multiple convolutional layers with ReLU activation
- Batch normalization for training stability
- Max pooling for spatial downsampling
- Dropout for regularization
- Fully connected layers for classification
- Softmax output layer (2 classes)

### Training
- Optimizer: Adam (learning rate = 0.0001)
- Loss function: Binary Cross-Entropy
- Epochs: 50
- Batch size: 35

---

## Results

| Metric | Value |
|------|------|
Accuracy | **0.87**
Recall (IDC +) | **~0.95**
Precision | ~0.89
F1-Score | ~0.88

High recall ensures minimal false negatives, which is critical in medical diagnosis scenarios.

---

## Key Insights
- CNNs effectively learn discriminative tissue patterns from histopathology images
- Recall and ROC-based evaluation are more meaningful than accuracy alone for cancer detection
- Regularization techniques significantly improved model generalization

---

## Technologies Used
**Python | TensorFlow | Keras | OpenCV | NumPy | Pandas | Matplotlib | Seaborn | Scikit-learn**

---

## How to Run
1. Download the dataset from Kaggle  
2. Place the images inside a `data/` directory  
3. Run the Python script or Jupyter notebook  

---

## Limitations and Future Work

### Limitations
- Patch-level classification rather than whole-slide images
- Class imbalance affects threshold sensitivity

### Future Improvements
- Grad-CAM for model interpretability
- Transfer learning using pre-trained CNNs
- Whole-slide image classification
- Clinical validation on additional datasets

---

## Demo
```python
input_img = X_test[4000:4001]
prediction = model.predict(input_img).argmax()
print("Predicted class:", prediction)
