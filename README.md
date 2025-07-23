# Real-Time Voice Emotion Detection 🎙️🧠

This project detects human emotions from voice using machine learning (Random Forest on RAVDESS dataset).

## 📁 Dataset
- RAVDESS Speech-only Dataset
- Not included in this repo (due to size). [Download RAVDESS Speech](https://zenodo.org/record/1188976) and extract manually.

## 📦 Files
- `dataset_preparation.py` – Feature extraction, model training
- `emotion_model.pkl` – Trained model
- `ravdess_dataset.pkl` – Processed dataset
- `inference_demo.py` – Predict emotion from a `.wav` file
- `requirements.txt` – Dependencies list

## 🔧 How to Use

1. Train the model using `dataset_preparation.py` in Colab
2. Predict emotion with `inference_demo.py`
3. Upload your own `.wav` file and test!

## 📌 Requirements
Install dependencies:

```bash
pip install -r requirements.txt
