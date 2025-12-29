# Pneumonia Detection using Deep Transfer Learning 🫁

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
Pneumonia is a life-threatening infectious disease affecting the lungs. Early and accurate diagnosis is crucial for effective treatment. This project implements a **Deep Learning** solution to automatically detect Pneumonia from Chest X-Ray images.

Using **Transfer Learning** with a pre-trained **ResNet18** architecture, the model is optimized for **high sensitivity (Recall)** to ensure no Pneumonia cases are missed during screening.

## 📂 Dataset
The project utilizes the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
* **Data Source:** Guangzhou Women and Children’s Medical Center.
* **Classes:** `Normal` vs. `Pneumonia` (Binary Classification).
* **Structure:** Divided into Train, Test, and Validation sets.

## 🧠 Methodology
We employed a Transfer Learning approach to leverage features learned from the ImageNet dataset:
1.  **Backbone:** ResNet18 (Pre-trained).
2.  **Modifications:** The final fully connected layer was replaced to output 2 classes.
3.  **Preprocessing:** Images resized to 224x224, normalized using ImageNet statistics, and augmented to prevent overfitting.
4.  **Loss Function:** CrossEntropyLoss.

## 📊 Results & Analysis
The model was evaluated on the test set with a focus on medical diagnostic safety (minimizing False Negatives).

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 1.00 | 0.39 | 0.56 | 234 |
| **Pneumonia** | **0.73** | **1.00** | **0.85** | 390 |
| | | | | |
| **Accuracy** | | | **0.77** | 624 |
| **Macro Avg** | 0.87 | 0.69 | 0.70 | 624 |
| **Weighted Avg** | 0.83 | 0.77 | 0.74 | 624 |

### 🩺 Clinical Interpretation
* **Perfect Recall for Pneumonia (1.00):** The model correctly identified **100% of the Pneumonia cases**. In a medical context, this is the most critical metric because missing a positive case (False Negative) can be fatal.
* **Trade-off:** To achieve perfect sensitivity, the model behaves conservatively, leading to a lower recall for the 'Normal' class. This means it acts as a highly safe **screening tool**, flagging potential risks for further review by radiologists.

## 🚀 Installation & Usage
This project is designed to run on **Google Colab** or a local machine with GPU support.

### Prerequisites
* Python 3.x
* PyTorch, Torchvision
* Kaggle API

### Running on Google Colab
1.  Clone this repository.
2.  Upload `model.py`, `train.py`, and `utils.py` to your Colab workspace.
3.  Open `main.ipynb` and run the cells sequentially.
4.  Ensure you have your `kaggle.json` token ready for dataset download.
