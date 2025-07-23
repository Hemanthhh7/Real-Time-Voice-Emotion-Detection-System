# dataset_preparation.py

import os
import zipfile
import glob
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define paths
zip_path = "Audio_Speech_Actors_01-24.zip"  # Place this zip file in the root directory
extract_path = "ravdess"

# Step 1: Unzip dataset
def unzip_dataset(zip_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Dataset unzipped!")
    except FileNotFoundError:
        print(f"❌ Error: {zip_path} not found.")
    except zipfile.BadZipFile:
        print(f"❌ Error: {zip_path} is not a valid zip file.")

# Step 2: Emotion label map
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Step 3: Extract MFCC features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# Step 4: Load dataset and extract features
def process_dataset(extract_path):
    X, y = [], []

    for file_path in glob.glob(f"{extract_path}/Actor_*/**/*.wav", recursive=True):
        file_name = os.path.basename(file_path)
        try:
            emotion_code = file_name.split('-')[2]
            emotion_label = emotion_map.get(emotion_code)

            if emotion_label:
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(emotion_label)
        except IndexError:
            print(f"⚠️ Skipping invalid file: {file_name}")

    return np.array(X), np.array(y)

# Step 5: Train & save model and dataset
def train_model(X, y):
    if len(X) == 0:
        print("❌ No features found. Aborting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    joblib.dump((X_train, X_test, y_train, y_test), "ravdess_dataset.pkl")
    print("✅ Dataset saved as 'ravdess_dataset.pkl'")

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "emotion_model.pkl")
    print("✅ Model saved as 'emotion_model.pkl'")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc:.2%}")

# Main flow
if __name__ == "__main__":
    unzip_dataset(zip_path, extract_path)
    X, y = process_dataset(extract_path)
    print(f"✅ Features extracted: {len(X)}")
    train_model(X, y)
