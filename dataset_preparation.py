# dataset_preparation.py
import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data"  # folder with audio files

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def load_data():
    features = []
    labels = []
    for folder in os.listdir(DATA_PATH):
        for file in os.listdir(os.path.join(DATA_PATH, folder)):
            if file.endswith(".wav"):
                path = os.path.join(DATA_PATH, folder, file)
                feature = extract_features(path)
                features.append(feature)
                labels.append(folder)  # folder name as emotion
    return np.array(features), np.array(labels)

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "emotion_model.pkl")
print("✅ Model trained and saved as emotion_model.pkl")
