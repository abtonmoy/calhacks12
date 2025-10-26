# CalHacks 12 - Human Detection & Audio Analysis System

This repository contains a complete system for human detection in images and audio analysis in videos, developed for CalHacks 12.

## Features

- **Advanced Human Detection**: Multi-method face detection using OpenCV Haar cascades and MediaPipe
- **Audio Emotion Recognition**: Emotion detection from video audio using HuBERT
- **Flask API Server**: RESTful API for React/TypeScript frontend integration
- **Fast Processing**: Optimized for speed with minimal dependencies
- **Accurate Detection**: Combines multiple detection methods for better accuracy
- **Clean Architecture**: Simplified, maintainable codebase

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Install FFmpeg (macOS)
brew install ffmpeg
```

## API Server Setup

### Start the API Server

```bash
# Use the helper script
./server.sh start

# OR manually
python3 api_server.py
```

Server runs on `http://localhost:5001`

### API Endpoints

- **POST** `/api/detect-human` - Detect humans in uploaded image
- **POST** `/api/analyze-audio` - Analyze audio from uploaded video
- **GET** `/api/health` - Health check endpoint

See `API_SETUP.md` for detailed documentation.

## Usage

### Human Detection (CLI)

```bash
python3 findHuman_agent/human_detector.py image.jpg
```

Output:

```json
{
  "human_present": 1,
  "num_people": 6
}
```

### Audio Processing (CLI)

```bash
python3 audio_processing/audio_processor.py video.mp4
```

Output:

```json
{
  "duration_sec": 15.03,
  "gender_estimation": "female",
  "mean_pitch": 226.96,
  "spectral_bandwidth": 2021.3,
  "emotion": "happy"
}
```

### React Frontend Integration

See `frontend_example/APIExample.tsx` for a complete React/TypeScript component example.

## Project Structure

```
├── api_server.py                    # Flask API server
├── server.sh                        # Server management script
├── findHuman_agent/
│   └── human_detector.py            # Human detection module
├── audio_processing/
│   └── audio_processor.py           # Audio analysis module
├── requirements.txt                 # Dependencies
├── API_SETUP.md                     # API documentation
├── SERVER_CONTROLS.md              # Server management guide
└── ORCHESTRATION_ANALYSIS.md       # Architecture analysis
```

## Dependencies

- opencv-python==4.8.1.78
- numpy==1.24.3
- mediapipe>=0.10.5
- ffmpeg-python
- librosa
- torch
- soundfile
- transformers
- flask
- flask-cors

## Documentation

- `API_SETUP.md` - Complete API setup and usage guide
- `SERVER_CONTROLS.md` - How to start/stop/restart the server
- `ORCHESTRATION_ANALYSIS.md` - Architecture and design decisions
- `TEST_RESULTS.md` - Test results and validation

## Performance

- **Human Detection**: ~0.1-0.2 seconds per image
- **Audio Analysis**: 30-60 seconds per video (depending on length)
- **Accuracy**: Multi-method detection with duplicate removal
- **Memory**: Minimal footprint with optimized dependencies

## Contributing

This project was developed for CalHacks 12. Feel free to fork and contribute improvements!

## License

MIT License
