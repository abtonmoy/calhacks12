# Flask API Server - Start/Stop Guide

## 🚀 Quick Start

### Option 1: Use the Helper Script (Recommended)

```bash
# Start server
./server.sh start

# Stop server
./server.sh stop

# Check status
./server.sh status

# Restart server
./server.sh restart
```

### Option 2: Manual Control

#### Start Server

```bash
python3 api_server.py
```

- Server runs on `http://localhost:5001`
- Press `Ctrl+C` to stop

#### Background

```bash
python3 api_server.py &
```

#### Stop Server (Manual)

```bash
# Find process ID
lsof -ti:5001

# Kill the process
kill -9 $(lsof -ti:5001)
```

## 📡 Server Endpoints

### Health Check

```bash
curl http://localhost:5001/api/health
```

### Detect Humans in Image

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5001/api/detect-human
```

### Analyze Video Audio

```bash
curl -X POST -F "file=@video.mp4" http://localhost:5001/api/analyze-audio
```

## 🛠️ Troubleshooting

### Port 5001 Already in Use

```bash
# Find what's using the port
lsof -i :5001

# Kill it
kill -9 $(lsof -ti:5001)
```

### Can't Connect to Server

```bash
# Check if server is running
./server.sh status

# Check server logs
python3 api_server.py  # Run in foreground to see logs
```

### Clear Temporary Files

```bash
# Clean up uploads folder
rm -rf uploads/*

# Clean up audio temp files
rm -rf audio_temp/*
```

## 📝 Notes

- Server runs on port **5001** (avoiding macOS AirPlay port 5000)
- React frontend should connect to `http://localhost:5001/api`
- Temporary files are auto-cleaned after processing
- Debug mode is ON for development (not production-ready)
