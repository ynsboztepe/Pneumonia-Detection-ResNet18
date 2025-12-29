# Pneumonia Detection using Deep Transfer Learning 🫁

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
Pneumonia is a life-threatening infectious disease affecting the lungs. Early and accurate diagnosis is crucial for effective treatment. This project implements a **Deep Learning** solution to automatically detect Pneumonia from Chest X-Ray images.

Using **Transfer Learning** with a pre-trained **ResNet18** architecture, the model achieves high accuracy with computationally efficient training, making it suitable for assisting radiologists in diagnostic processes.

## 📂 Dataset
The project utilizes the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
* **Data Source:** Guangzhou Women and Children’s Medical Center.
* **Classes:** `Normal` vs. `Pneumonia` (Binary Classification).
* **Structure:** Divided into Train, Test, and Validation sets.

## 🧠 Methodology
We employed a Transfer Learning approach to leverage features learned from the ImageNet dataset:
1.  **Backbone:** ResNet18 (Pre-trained).
2.  **Modifications:** The final fully connected layer was replaced to output 2 classes.
3.  **Preprocessing:** Images resized to 224x224, normalized using ImageNet statistics, and augmented (RandomFlip, Rotation) to prevent overfitting.
4.  **Optimizer:** Adam (`lr=0.0001`).
5.  **Loss Function:** CrossEntropyLoss.

## 🚀 Installation & Usage
This project is designed to run on **Google Colab** or a local machine with GPU support.

### Prerequisites
* Python 3.x
* PyTorch, Torchvision
* Matplotlib, Seaborn
* Kaggle API (for downloading data)

### Running on Google Colab
1.  Clone this repository.
2.  Upload `model.py`, `train.py`, and `utils.py` to your Colab workspace.
3.  Open `main.ipynb` and run the cells sequentially.
4.  Ensure you have your `kaggle.json` token ready for dataset download.

## 📊 Results
After training for **10 epochs**, the model achieved the following performance on the test set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **~90%** (Update this with your value) |
| **Precision** | **0.XX** (Update this) |
| **Recall** | **0.XX** (Update this) |
| **F1-Score** | **0.XX** (Update this) |

*Confusion Matrix visualization is included in the notebook.*

## 📁 File Structure
