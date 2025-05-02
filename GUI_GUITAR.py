import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
import tempfile
import os
import sounddevice as sd
from scipy.io.wavfile import write

# Load model
model = load_model("guitar_chord_detection_model.h5")

# Label mapping
chords = ['Am', 'Bb', 'Bdim', 'c', 'Dm', 'Em', 'F', 'G']
label_to_index = {label: idx for idx, label in enumerate(chords)}

# Constants
sample_rate = 22050
duration = 5  # in seconds
n_mels = 128
target_frames = 128

# Helper functions
def load_wav_file(file_path):
    y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    expected_len = sr * duration
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    return y

def wav_to_mel_spec(file_path):
    y = load_wav_file(file_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < target_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :target_frames]
    return np.expand_dims(mel_db, axis=-1)

def predict_chord(file_path):
    spec = wav_to_mel_spec(file_path)
    spec = np.expand_dims(spec, axis=0)
    prediction = model.predict(spec)
    predicted_index = np.argmax(prediction)
    return list(label_to_index.keys())[list(label_to_index.values()).index(predicted_index)]

# Streamlit UI
st.title("🎸 Guitar Chord Detector")
st.write("Upload a `.wav` file or record audio to predict the chord.")

option = st.radio("Choose input method:", ["Upload File", "Record Audio"])

# Upload File Mode
if option == "Upload File":
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        chord = predict_chord(temp_path)
        st.success(f"🎵 Predicted Chord: **{chord}**")
        os.remove(temp_path)

# Record Audio Mode
if option == "Record Audio":
    st.info("Click below to record 5 seconds of audio...")

    if st.button("🎙️ Record Now"):
        with st.spinner("Recording..."):
            recording = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
            sd.wait()

            temp_path = "recorded_audio.wav"
            write(temp_path, sample_rate, recording)

            chord = predict_chord(temp_path)
            st.success(f"🎵 Predicted Chord: **{chord}**")
            os.remove(temp_path)

