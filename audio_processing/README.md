# Audio Processor

Agent-based pipeline for extracting, analyzing, and interpreting audio from video content. This module isolates the soundtrack, computes acoustic features, and detects emotional tone — enabling multimodal systems to understand not just what's shown, but how it sounds.

## What It Does

Extracts audio tracks from videos using ffmpeg and computes key acoustic features including:

- **duration_sec** – total audio length
- **mean_pitch** – average voice pitch (Hz)
- **spectral_bandwidth** – frequency spread and fullness
- **gender_estimation** – inferred from pitch range
- **emotion** – classified using the pretrained HuBERT model (superb/hubert-base-superb-er)

Outputs structured JSON for easy downstream processing by orchestrators or feature extraction agents.

## Quick Start

```python
from audio_agent.audio_processor import analyze_video_audio

# Analyze a video file
results = analyze_video_audio("./data/videos/ad_sample.mp4")
print(results)
```

### Example Output

```json
{
  "duration_sec": 14.32,
  "gender_estimation": "female",
  "mean_pitch": 227.8,
  "spectral_bandwidth": 1812.4,
  "emotion": "happy"
}
```

## How It Works

### 1. Extract audio track

```python
ffmpeg.input(video_path).output("audio.wav", ac=1, ar=44100).run()
```

### 2. Compute signal-level metrics

```python
y, sr = librosa.load("audio.wav", sr=16000)
mean_pitch = np.mean(librosa.yin(y, fmin=80, fmax=800))
spectral_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
```

### 3. Classify emotional tone with HuBERT

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-er")
model = AutoModelForAudioClassification.from_pretrained("superb/hubert-base-superb-er")

emotion = model.config.id2label[
    int(model(**extractor(y, sampling_rate=sr, return_tensors="pt")).logits.argmax())
]
```

## Result

Produces a concise, machine-readable JSON object describing the audio's mood and structure, suitable for integration into:

- Ad analysis pipelines
- Multimodal retrieval systems
- Emotion-aware video classifiers

## Agent Summary

| Property | Description |
|----------|-------------|
| **Agent Name** | audio_processor |
| **Purpose** | Extracts, analyzes, and classifies audio from videos |
| **Input** | Path to video file (.mp4, .mov, etc.) |
| **Output** | JSON: duration, gender, pitch, bandwidth, emotion |
| **Dependencies** | ffmpeg-python, librosa, soundfile, torch, transformers, numpy |

## Installation

```bash
pip install ffmpeg-python librosa soundfile torch transformers numpy
```

Note: You'll also need ffmpeg installed on your system. Install via:
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
