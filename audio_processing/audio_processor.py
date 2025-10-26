import os
import ffmpeg
import librosa
import numpy as np
import torch
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# =========================================================
# CONFIG
# =========================================================
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"

# =========================================================
# UTILITY: Safe audio load using soundfile + resample
# =========================================================
def safe_load_audio(audio_path, sr=16000):
    # Load audio with soundfile
    y, orig_sr = sf.read(audio_path, dtype="float32")
    
    # Convert to mono if stereo
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    
    # Resample if needed
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y, sr

# =========================================================
# STEP 1: Extract Audio from Video
# =========================================================
def extract_audio_from_video(video_path, output_dir="audio_temp"):
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, os.path.basename(video_path).replace(".mp4", ".wav"))

    print(" Extracting audio track...")
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=44100)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    return audio_path

# =========================================================
# STEP 2: Extract Required Audio Features
# =========================================================
def extract_audio_features(audio_path):
    print(" Computing selected audio features...")

    # Load audio safely
    y, sr = safe_load_audio(audio_path, sr=16000)
    duration = len(y) / sr

    # --- Mean Pitch → Gender Estimation ---
    pitch = librosa.yin(y, fmin=80, fmax=800)
    mean_pitch = np.mean(pitch)
    gender_est = "male" if mean_pitch < 180 else "female"

    # --- Spectral Bandwidth (Fullness of spectrum) ---
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    return {
        "duration_sec": round(duration, 2),
        "mean_pitch": float(mean_pitch),
        "gender_est": gender_est,
        "spectral_bandwidth": float(spectral_bandwidth)
    }

# =========================================================
# STEP 3: Audio Emotion Recognition
# =========================================================
def analyze_audio_sentiment(audio_path):
    model_name = "superb/hubert-base-superb-er"  # pretrained on IEMOCAP emotions

    print(" Loading emotion recognition model (first run may take 1–2 minutes)...")
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)

    # Use safe_load_audio instead of librosa.load to avoid aifc
    y, sr = safe_load_audio(audio_path, sr=16000)

    inputs = extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()
        predicted_id = np.argmax(probs)
        label = model.config.id2label[predicted_id]

    print(f" Predicted Emotion: {label}")
    return {"predicted_emotion": label}

# =========================================================
# STEP 4: Combine Results
# =========================================================
def analyze_video_audio(video_path):
    print(f"\n Processing video: {os.path.basename(video_path)}")
    audio_path = extract_audio_from_video(video_path)

    features = extract_audio_features(audio_path)
    sentiment = analyze_audio_sentiment(audio_path)

    results = {**features, **sentiment}

    print("\n Final Audio Analysis Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results

# =========================================================
# STEP 5: Run Script
# =========================================================
if __name__ == "__main__":
    video_path = r".\data\videos\v0002.mp4"  # Change path if needed
    if not os.path.exists(video_path):
        print(" Please provide a valid .mp4 video path.")
    else:
        analyze_video_audio(video_path)