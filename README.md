# CalHacks 12 - Marketing Signal Detection System

This repository contains a comprehensive AI-powered system for extracting marketing signals, promotional content, and brand information from videos and images.

## 🎯 Overview

This system processes video files and images to extract rich marketing intelligence including:

- **Visual Analysis**: Brand names, products, promotional offers, CTAs
- **Audio Analysis**: Voice characteristics, emotion detection, audio sentiment
- **Human Detection**: Presence of people in frames
- **Text Extraction**: OCR for images, visual text detection for videos
- **Promotional Intelligence**: Promo codes, deadlines, discount types

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (macOS)
brew install ffmpeg

# Create .env file with your Anthropic API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Run the Flask API Server

```bash
# Start the server
./server.sh start

# Server runs on http://localhost:5001
```

### API Endpoints

- `POST /api/detect-human` - Simple human detection in images
- `POST /api/analyze-image` - Full image analysis (OCR + Claude API)
- `POST /api/analyze-audio` - Audio emotion analysis
- `POST /api/analyze-video` - Complete video analysis (Visual + Audio)
- `GET /api/health` - Health check

## 📦 Components

### Video Processing Pipeline

- **Frame Deduplication**: ChromaDB with CLIP embeddings
- **Vision Analysis**: Claude Vision API
- **Audio Analysis**: HuBERT emotion recognition
- **Human Detection**: OpenCV + MediaPipe

### Image Processing Pipeline

- **Human Detection**: OpenCV Haar Cascades
- **Text Extraction**: EasyOCR
- **Signal Extraction**: Claude API

## 📊 Output Format

### Image Analysis Response

```json
{
  "human_present": 1,
  "num_people": 3,
  "extracted_text": "50% OFF SALE",
  "brand_name_text": "BrandName",
  "promo_present": true,
  "promo_text": "50% OFF SALE",
  "promo_code": "SAVE50",
  "price_value": "50%",
  "cta_present": true,
  "cta_type": "shop_now"
}
```

### Video Analysis Response

See `API_ENDPOINTS.md` for complete video analysis format.

## 📖 Documentation

- `API_ENDPOINTS.md` - Complete API documentation
- `SERVER_CONTROLS.md` - Server management guide
- `video_orc_agent_main.py` - Video processing agent
- `img_orc_agent.py` - Image processing agent

## 🎯 Use Cases

- Marketing campaign analysis
- Competitive intelligence
- Promotional code extraction
- Brand monitoring
- Content audit
- Sentiment analysis

## 📝 License

MIT License
