# app.py

import streamlit as st
import librosa
import numpy as np
import joblib
import soundfile as sf
import tempfile

st.set_page_config(page_title="Real-Time Voice Emotion Detection", layout="centered")
st.title("🎙️ Real-Time Voice Emotion Detection System")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("emotion_model.pkl")
    return model

model = load_model()

# Extract features
def extract_features(audio_file):
    audio, sr = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

# UI
uploaded_file = st.file_uploader("🎤 Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        features = extract_features(tmp.name)
        prediction = model.predict(features)[0]

    st.success(f"🧠 Predicted Emotion: **{prediction.upper()}**")
else:
    st.info("Please upload a .wav file to get started.")
