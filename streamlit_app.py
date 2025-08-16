import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import tempfile

# === Constants ===
MODEL_PATH = "trained_model.h5"
MAX_TIMESTEPS = 258  # Based on your training
MAX_FREQ_BINS = 128  # Based on your training

# === Streamlit UI ===
st.set_page_config(page_title="Real vs Fake Audio Detection", layout="centered")
st.title("🎙️ Real vs Fake Audio Detector")
st.write("Upload an audio file (.flac or .wav) under **10 seconds**, and the model will predict whether it's **Real** or **Fake**.")

uploaded_file = st.file_uploader("Choose an audio file (<10 seconds)", type=["flac", "wav"])

# === Helper: Clean audio ===
def clean_audio(audio_bytes, sr=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
        temp_in.write(audio_bytes)
        temp_in_path = temp_in.name

    # Load and denoise
    y, sr = librosa.load(temp_in_path, sr=sr)
    reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)

    # Save to temporary WAV
    temp_clean_path = temp_in_path.replace(".wav", "_cleaned.wav")
    sf.write(temp_clean_path, reduced_noise, sr)

    # Trim silence with pydub
    audio = AudioSegment.from_wav(temp_clean_path)

    def detect_leading_silence(sound, silence_thresh=-40.0, chunk_size=1):
        trim_ms = 0
        while trim_ms < len(sound) and sound[trim_ms:trim_ms+chunk_size].dBFS < silence_thresh:
            trim_ms += chunk_size
        return trim_ms

    def trim_silence(audio_segment):
        start_trim = detect_leading_silence(audio_segment)
        end_trim = detect_leading_silence(audio_segment.reverse())
        duration = len(audio_segment)
        trimmed = audio_segment[start_trim:duration - end_trim]
        return trimmed

    trimmed = trim_silence(audio)
    final_path = temp_clean_path.replace("_cleaned.wav", "_trimmed.flac")
    trimmed.export(final_path, format="flac")

    return final_path

# === Feature Extraction ===
def extract_features(audio_path, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    log_power_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec.T, log_power_spec.T

# === Normalization ===
def normalize_feature(feature):
    scaler = MinMaxScaler()
    return scaler.fit_transform(feature.T).T

# === Combine, pad, reshape ===
def prepare_input(mel_spec, log_power_spec):
    min_len = min(mel_spec.shape[0], log_power_spec.shape[0])
    mel_spec = normalize_feature(mel_spec[:min_len])
    log_power_spec = normalize_feature(log_power_spec[:min_len])
    combined = np.stack([mel_spec, log_power_spec], axis=-1)  # Shape: (timesteps, freq_bins, 2)

    pad_t = max(0, MAX_TIMESTEPS - combined.shape[0])
    pad_f = max(0, MAX_FREQ_BINS - combined.shape[1])
    padded = np.pad(combined, ((0, pad_t), (0, pad_f), (0, 0)), mode='constant')

    transposed = np.transpose(padded, (1, 0, 2))  # From (T, F, C) → (F, T, C)
    return np.expand_dims(transposed, axis=0)  # Add batch dimension

# === Load model once
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH, compile=False)

# === Run prediction
def run_prediction(cleaned_audio_path):
    mel, log = extract_features(cleaned_audio_path)
    model_input = prepare_input(mel, log)
    model = load_model_cached()
    prediction = model.predict(model_input)
    prob = prediction[0][0]
    label = "Fake" if prob > 0.5 else "Real"
    return label, prob

# === Main inference logic with duration check ===
if uploaded_file is not None:
    # Temporarily save uploaded file to check duration
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    # Load and check duration
    y, sr = librosa.load(temp_audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    if duration > 10:
        st.error("❌ Audio too long. Please upload an audio file shorter than 10 seconds.")
        os.remove(temp_audio_path)
    else:
        st.audio(temp_audio_path, format="audio/wav")
        with st.spinner("Cleaning and analyzing..."):
            cleaned_audio_path = clean_audio(open(temp_audio_path, 'rb').read())
            label, prob = run_prediction(cleaned_audio_path)

        st.success(f"✅ Prediction: **{label}** (Confidence: {prob:.2f})")

        os.remove(temp_audio_path)
        os.remove(cleaned_audio_path)
