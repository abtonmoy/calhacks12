# Module 3: Human/Safety/Emotion Detection

This module analyzes images for human presence, facial expressions, and NSFW content to ensure ad safety and compliance.

## Features

- **Human Detection**: Detects faces and people in images using OpenCV Haar cascades or RetinaFace
- **Emotion Analysis**: Classifies facial expressions (happy, neutral, sad, angry, surprised)
- **NSFW Detection**: Identifies inappropriate content for ad safety
- **Batch Processing**: Efficiently processes multiple faces to avoid redundant computations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from module3_human_safety import HumanSafetyEmotionDetector

detector = HumanSafetyEmotionDetector()
result = detector.analyze_image("path/to/image.jpg")

print(result)
# Output:
# {
#   "human_present": 1,
#   "num_people": 1,
#   "model_expression": "happy",
#   "smile_present": 1,
#   "nsfw_flag": 0,
#   "nsfw_score": 0.04
# }
```

## Components

- `human_detector.py`: Face and person detection
- `emotion_classifier.py`: Facial expression analysis
- `nsfw_detector.py`: Content safety classification
- `module3_human_safety.py`: Main orchestrator class
