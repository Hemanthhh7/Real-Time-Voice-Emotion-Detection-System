
import streamlit as st
import numpy as np
import sounddevice as sd
import librosa
import joblib
import tempfile
import wavio

# Load trained model
model = joblib.load("emotion_model.pkl")

st.title("🎙️ Real-Time Voice Emotion Detection")

DURATION = 4  # seconds
SAMPLE_RATE = 22050

def record_audio():
    st.info("Recording for 4 seconds...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio.flatten()

def extract_mfcc(audio, sample_rate=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)
if st.button("🎤 Record Voice"):
    audio_data = record_audio()
    st.audio(audio_data, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
        wavio.write(f.name, audio_data.reshape(-1, 1), SAMPLE_RATE, sampwidth=2)
        f.seek(0)
        st.audio(f.name)

    features = extract_mfcc(audio_data)
    prediction = model.predict([features])[0]
    st.success(f"🗣️ Detected Emotion: **{prediction}**")
