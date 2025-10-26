import os
import ffmpeg
import librosa
import numpy as np
import torch
import soundfile as sf
import json
import sys
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# =========================================================
# CONFIG
# =========================================================
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"

# Emotion mapping from model abbreviations to full names
EMOTION_MAPPING = {
    "hap": "happy",
    "sad": "sad", 
    "ang": "angry",
    "neu": "neutral",
    "exc": "excited",
    "fru": "frustrated",
    "fea": "fearful",
    "dis": "disgusted"
}

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
        
        # Map abbreviation to full emotion name
        full_emotion = EMOTION_MAPPING.get(label, label)

    return {"predicted_emotion": full_emotion}

# =========================================================
# STEP 4: Combine Results
# =========================================================
def analyze_video_audio(video_path):
    """Analyze video audio and return JSON results"""
    try:
        audio_path = extract_audio_from_video(video_path)
        features = extract_audio_features(audio_path)
        sentiment = analyze_audio_sentiment(audio_path)
        
        # Combine results into clean JSON format
        results = {
            "duration_sec": features["duration_sec"],
            "gender_estimation": features["gender_est"],
            "mean_pitch": features["mean_pitch"],
            "spectral_bandwidth": features["spectral_bandwidth"],
            "emotion": sentiment["predicted_emotion"]
        }
        
        return results
        
    except Exception as e:
        return {
            "error": str(e),
            "duration_sec": 0,
            "gender_estimation": "unknown",
            "mean_pitch": 0,
            "spectral_bandwidth": 0,
            "emotion": "unknown"
        }

# =========================================================
# STEP 5: Command Line Interface
# =========================================================
def main():
    """Main function for command line usage"""
    if len(sys.argv) != 2:
        print("Usage: python3 audio_processor.py <video_path>")
        print("Example: python3 audio_processor.py video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        error_result = {
            "error": f"Video file not found: {video_path}",
            "duration_sec": 0,
            "gender_estimation": "unknown",
            "mean_pitch": 0,
            "spectral_bandwidth": 0,
            "emotion": "unknown"
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)
    
    # Analyze video and output JSON
    result = analyze_video_audio(video_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()