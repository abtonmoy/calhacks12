# Deduplication Agent

Agent-based pipeline for deduplicating video frames and images using CLIP embeddings. Extracts unique frames, stores them on disk, and groups them by source video for downstream feature extraction.

---

## What It Does

1. **Extracts frames** from videos (using ffmpeg) at configurable intervals
2. **Detects duplicates** using CLIP embeddings and cosine similarity
3. **Stores unique frames** as JPEG files on disk
4. **Groups frames by source** for easy batch processing in your model

**Result**: Only unique, deduplicated frames grouped by video source, ready to feed into feature extraction models.

---

## Quick Start

```python
from pipeline import VisualDeduplicationPipeline

# Initialize
pipeline = VisualDeduplicationPipeline(
    db_path="./chroma_visual_db",
    frame_interval=1.0,                      # 1 frame/sec
    similarity_threshold=0.9,                # 90% similar = duplicate
    frames_storage_path="./frames_storage"
)

# Process files
pipeline.process_batch(["video1.mp4", "video2.mp4", "image1.jpg"])

# Get frames grouped by source (ready for your model)
frames_by_source = pipeline.get_all_frames_for_model()

for video_path, (frame_arrays, metadata) in frames_by_source.items():
    # frame_arrays: List[np.ndarray] - RGB arrays (H, W, 3)
    # metadata: List[Dict] - frame info (index, id, source, etc.)

    for frame in frame_arrays:
        features = your_model.extract(frame)
```

---

## Core API

### Processing Files

```python
# Process single file
result = pipeline.process_file("data/videos/sample.mp4")

# Process multiple files
results = pipeline.process_batch(["video1.mp4", "video2.mp4", "image1.jpg"])
```

### Getting Frames for Your Model

#### Method 1: Get All Frames (Recommended)

```python
# Returns: Dict[source_path -> (frame_arrays, metadata)]
frames_by_source = pipeline.get_all_frames_for_model()

for video_path, (frames, metadata) in frames_by_source.items():
    print(f"{video_path}: {len(frames)} unique frames")

    # frames is List[np.ndarray] - sorted chronologically
    # metadata is List[Dict] - corresponding frame info

    for frame, meta in zip(frames, metadata):
        # frame.shape = (H, W, 3) in RGB format
        features = your_model(frame)
```

#### Method 2: Get Frames from Specific Video

```python
# Get frames from one specific video
video_path = "data/videos/sample.mp4"
frame_arrays, metadata = pipeline.get_frame_arrays_by_source(video_path)

# frame_arrays: List[np.ndarray] sorted by frame_index
# metadata: List[Dict] with frame info

for frame, meta in zip(frame_arrays, metadata):
    print(f"Frame {meta['frame_index']}: shape {frame.shape}")
    features = your_model(frame)
```

#### Method 3: Get Metadata First, Load Later

```python
# Memory-efficient: get metadata, load frames on demand
grouped = pipeline.get_frames_grouped_by_source()

for video_path, frames_info in grouped.items():
    print(f"{video_path}: {len(frames_info)} frames")

    for frame_info in frames_info:
        # Load frame when needed
        frame = pipeline.get_frame_by_id(frame_info['id'])
```

### Statistics

```python
# Database stats
stats = pipeline.get_database_stats()
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Frames on disk: {stats['saved_frames_on_disk']}")

# All source files with frame counts
sources = pipeline.get_all_source_files()
for source in sources:
    print(f"{source['source_path']}: {source['frame_count']} frames")
```

---

## Complete Example

```python
from pipeline import VisualDeduplicationPipeline
from pathlib import Path

# 1. Initialize
pipeline = VisualDeduplicationPipeline(
    db_path="./chroma_visual_db",
    frame_interval=1.0,
    similarity_threshold=0.9
)

# 2. Collect files
files = list(Path("data/videos").glob("*.mp4"))
files.extend(Path("data/images").glob("*.jpg"))

# 3. Process (deduplication happens here)
results = pipeline.process_batch([str(f) for f in files])

# 4. Get deduplicated frames grouped by source
frames_by_source = pipeline.get_all_frames_for_model()

# 5. Feed into your feature extraction model
for video_path, (frames, metadata) in frames_by_source.items():
    print(f"\nProcessing: {video_path}")
    print(f"  Frames: {len(frames)}")
    print(f"  Indices: {[m['frame_index'] for m in metadata]}")

    # Process frames from this video
    video_features = []
    for frame in frames:
        # frame is np.ndarray (H, W, 3) RGB
        features = your_model.extract_features(frame)
        video_features.append(features)

    # Save/aggregate features
    save_features(video_path, video_features, metadata)
```

---

## Configuration

Edit `config.py`:

```python
# Database
DB_PATH = "./chroma_visual_db"
FRAMES_STORAGE_PATH = "./frames_storage"

# Frame extraction
FRAME_INTERVAL = 1.0          # 1 frame per second
USE_FFMPEG = True             # Use ffmpeg (faster than OpenCV)

# Deduplication
SIMILARITY_THRESHOLD = 0.90   # 90% similar = duplicate
EMBEDDING_MODEL = "clip-ViT-B-32"
```

**Key parameters:**

- `FRAME_INTERVAL`: Higher = fewer frames extracted (faster, but might miss content)
- `SIMILARITY_THRESHOLD`: Lower = stricter duplicate detection (0.95 = very similar needed)
- `EMBEDDING_MODEL`: `clip-ViT-B-32` (fast) or `clip-ViT-L-14` (more accurate)

---

## File Structure

```
deduplication_agent/
├── pipeline.py              # Main pipeline
├── config.py               # Configuration
├── example.py              # Usage examples
├── requirements.txt        # Dependencies
│
├── data/
│   ├── images/            # Input images
│   └── videos/            # Input videos
│
├── chroma_visual_db/      # ChromaDB (embeddings + metadata)
├── frames_storage/        # Saved frame JPEGs
│   ├── abc123.jpg
│   ├── def456.jpg
│   └── ...
│
└── logs/
    ├── processing_log.json
    └── frames_info.json
```

---

## How Deduplication Works

```
For each frame f:
1. Extract frame from video (ffmpeg)
2. Compute perceptual hash (quick check)
3. Compute CLIP embedding
4. Query ChromaDB: similarity = cosine_similarity(embedding, database)
5. If similarity >= 0.9: SKIP (duplicate)
6. Else: STORE embedding + save frame JPEG
```

**Result**: Only unique frames stored, duplicates automatically skipped.

---

## API Reference

```python
# Processing
pipeline.process_file(file_path: str) -> Dict
pipeline.process_batch(file_paths: List[str]) -> List[Dict]

# Get frames for model (grouped by source)
pipeline.get_all_frames_for_model() -> Dict[str, Tuple[List[np.ndarray], List[Dict]]]
pipeline.get_frame_arrays_by_source(source_path: str) -> Tuple[List[np.ndarray], List[Dict]]
pipeline.get_frames_grouped_by_source() -> Dict[str, List[Dict]]
pipeline.get_frames_from_video(video_path: str) -> List[Dict]

# Individual frame retrieval
pipeline.get_frame_by_id(frame_id: str) -> Optional[np.ndarray]

# Statistics
pipeline.get_database_stats() -> Dict
pipeline.get_all_source_files() -> List[Dict]

# Search
pipeline.search_similar_frames(query_frame: np.ndarray, top_k: int) -> List[Dict]

# Export
pipeline.export_log(output_path: str)
pipeline.export_frames_info(output_path: str)
```

---

## Key Features

✅ **Frames grouped by source** - Easy to process videos separately  
✅ **Chronologically sorted** - Frames ordered by frame_index  
✅ **No duplicates** - Automatic deduplication via CLIP embeddings  
✅ **Ready-to-use arrays** - Frames returned as RGB numpy arrays  
✅ **Metadata preserved** - Know which frame came from which video  
✅ **Persistent storage** - Process once, retrieve anytime

---

## Common Use Cases

### Video Dataset Preprocessing

```python
# Remove redundant frames from video dataset
pipeline.process_batch(all_videos)
frames = pipeline.get_all_frames_for_model()

for video, (frames, meta) in frames.items():
    features = model.extract(frames)
```

### Multi-Video Feature Extraction

```python
# Extract features per video, maintaining source grouping
frames_by_video = pipeline.get_all_frames_for_model()

for video_path, (frames, metadata) in frames_by_video.items():
    video_features = [model(frame) for frame in frames]
    aggregate_video_features(video_path, video_features)
```

### Frame-Level Analysis

```python
# Analyze specific frames from specific videos
frames = pipeline.get_frames_from_video("data/videos/video1.mp4")

for frame_info in frames:
    frame = pipeline.get_frame_by_id(frame_info['id'])
    result = analyze_frame(frame)
```
