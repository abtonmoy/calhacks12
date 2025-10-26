# API Server Setup Guide

This guide shows you how to set up the Flask API server to serve your React/TypeScript frontend.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python3 api_server.py
```

The server will start on `http://localhost:5001`

### 3. Test the API

#### Health Check

```bash
curl http://localhost:5001/api/health
```

#### Detect Humans (Image Upload)

```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5001/api/detect-human
```

#### Analyze Audio (Video Upload)

```bash
curl -X POST -F "file=@path/to/video.mp4" http://localhost:5001/api/analyze-audio
```

## 📡 API Endpoints

### POST `/api/detect-human`

Detect humans in uploaded image.

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)
- Allowed formats: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`

**Response:**

```json
{
  "human_present": 1,
  "num_people": 6
}
```

**Error Response:**

```json
{
  "error": "No file provided",
  "human_present": 0,
  "num_people": 0
}
```

### POST `/api/analyze-audio`

Analyze audio from uploaded video.

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (video file)
- Allowed formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

**Response:**

```json
{
  "duration_sec": 15.03,
  "gender_estimation": "female",
  "mean_pitch": 226.96,
  "spectral_bandwidth": 2021.3,
  "emotion": "happy"
}
```

**Error Response:**

```json
{
  "error": "No file provided",
  "duration_sec": 0,
  "gender_estimation": "unknown",
  "mean_pitch": 0,
  "spectral_bandwidth": 0,
  "emotion": "unknown"
}
```

### GET `/api/health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "message": "API server is running"
}
```

## 🔗 Frontend Integration

### React/TypeScript Example

See `frontend_example/APIExample.tsx` for a complete React component example.

**Key Points:**

1. Set your API base URL: `const API_BASE_URL = 'http://localhost:5000/api'`
2. Use `FormData` to upload files
3. Handle the JSON responses

**Example Fetch Call:**

```typescript
const handleDetectHuman = async () => {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch("http://localhost:5000/api/detect-human", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();
  console.log(result);
};
```

## ⚙️ Configuration

### CORS Settings

The API server has CORS enabled for `localhost:3000` (React dev server). To allow other origins, modify `api_server.py`:

```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173"]
    }
})
```

### File Upload Limits

Default max file size is 100MB. To change, modify `MAX_FILE_SIZE` in `api_server.py`.

### Upload Folders

Files are temporarily stored in:

- Images: `./uploads/`
- Audio temp files: `./audio_temp/`

Both folders are auto-created on first run.

## 🧪 Testing

Test with your existing files:

```bash
# Test human detection
curl -X POST -F "file=@findHuman_agent/swim.jpg" http://localhost:5000/api/detect-human

# Test audio analysis
curl -X POST -F "file=@v0020.mp4" http://localhost:5000/api/analyze-audio
```

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'flask'"

```bash
pip install flask flask-cors
```

### Error: "Failed to load image"

- Check that the file is a valid image format
- Ensure file is not corrupted

### Error: "Could not open video"

- Check that FFmpeg is installed: `which ffmpeg`
- Install FFmpeg: `brew install ffmpeg` (macOS)

### CORS errors in browser

- Ensure React app is running on `localhost:3000`
- Check browser console for specific error messages
- Verify API server is running on `localhost:5000`

## 📝 Notes

- The API server runs on port 5000 by default
- File uploads are automatically cleaned up after processing
- Audio analysis may take 30-60 seconds depending on video length and emotion model download
- First run of emotion recognition will download the model (only happens once)
