# dataset_preparation.py
!pip install resampy
!pip install librosa --upgrade

import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import zipfile
import glob # Import glob
!pip install resampy
!pip install librosa --upgrade

# ✅ Step 1: Install dependencies (if needed)
!pip install librosa scikit-learn joblib resampy --quiet

# ✅ Step 2: Upload the ZIP Dataset (manual upload required)
from google.colab import files
# uploaded = files.upload()  # Upload `Audio_Speech_Actors_01-24.zip` - This line is commented out as it requires user interaction

# ✅ Step 3: Unzip the dataset
zip_path = "Audio_Speech_Actors_01-24.zip"  # uploaded file - Assuming the file has been uploaded
extract_path = "/content/ravdess"

# Ensure the extract_path exists
os.makedirs(extract_path, exist_ok=True)


try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset unzipped!")
except FileNotFoundError:
    print(f"❌ Error: {zip_path} not found. Please upload the zip file.")
    # Exit or handle the error appropriately if the file is not found
except zipfile.BadZipFile:
    print(f"❌ Error: {zip_path} is not a valid zip file. Please check the uploaded file.")
    # Exit or handle the error appropriately if the file is a bad zip file


# ✅ Step 4: Prepare emotion label mapping
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

# ✅ Step 5: Extract MFCC features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

X, y = [], []

# Use glob to find all .wav files in the extracted directory
for file_path in glob.glob(f"{extract_path}/Actor_*/**/*.wav", recursive=True):
    file_name = os.path.basename(file_path)
    # Assuming the file name format is correct: 03-01-04-01-02-02-01.wav
    try:
        emotion_code = file_name.split('-')[2]
        emotion_label = emotion_map.get(emotion_code)

        if emotion_label: # Only process if a valid emotion label is found
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(emotion_label)
        else:
            print(f"⚠️ Could not determine emotion for file: {file_name}")

    except IndexError:
        print(f"⚠️ Skipping file with unexpected name format: {file_name}")


print(f"✅ Extracted features from {len(X)} files.")


# ✅ Step 6: Train-test split
# Check if any features were extracted before splitting
if len(X) > 0:
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ✅ Step 7: Save the dataset
    joblib.dump((X_train, X_test, y_train, y_test), "ravdess_dataset.pkl")
    print("✅ Dataset saved as 'ravdess_dataset.pkl'")

    # ✅ Step 8: Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "emotion_model.pkl")
    print("✅ Model trained and saved as 'emotion_model.pkl'")

    # ✅ Step 9: Evaluate model accuracy
    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.2%}")

else:
    print("❌ No features were extracted. Cannot proceed with training and evaluation.")
