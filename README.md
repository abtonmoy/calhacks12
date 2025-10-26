# CalHacks 12 - Human Detection System

This repository contains a simplified human detection system developed for CalHacks 12, focusing on efficient and accurate human/face detection in images.

## Features

- **Advanced Human Detection**: Multi-method face detection using OpenCV Haar cascades and MediaPipe
- **Fast Processing**: Optimized for speed with minimal dependencies
- **Accurate Detection**: Combines multiple detection methods for better accuracy
- **Clean Architecture**: Simplified, maintainable codebase

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from module3_human_safety import HumanDetector

detector = HumanDetector()
result = detector.analyze_image("path/to/image.jpg")

print(result)
# Output:
# {
#   "human_present": 1,
#   "num_people": 3,
#   "face_boxes": [[x, y, w, h], ...],
#   "body_boxes": [],
#   "total_faces": 3,
#   "total_bodies": 0
# }
```

## Components

- `advanced_detector.py`: Multi-method face detection (OpenCV + MediaPipe)
- `module3_human_safety.py`: Main orchestrator class for human detection
- `test_module3.py`: Test suite for validation
- `requirements.txt`: Minimal dependencies

## Dependencies

- opencv-python==4.8.1.78
- numpy==1.24.3
- mediapipe==0.10.3

## Testing

Run the test suite to validate the system:

```bash
python3 test_module3.py
```

## Performance

- **Detection Speed**: ~0.1-0.2 seconds per image
- **Accuracy**: Multi-method detection with duplicate removal
- **Memory**: Minimal footprint with optimized dependencies

## Project Structure

```
├── advanced_detector.py          # Multi-method face detection
├── module3_human_safety.py       # Main orchestrator
├── test_module3.py               # Test suite
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
└── test_images/                  # Sample images for testing
```

## Contributing

This project was developed for CalHacks 12. Feel free to fork and contribute improvements!