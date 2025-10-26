# 🚀 QUICKSTART GUIDE

**Get started with ADLOVIN in 5 minutes!**

This guide covers the essentials to get you running video and image analysis quickly.

---

## ⚡ Prerequisites

Before starting, ensure you have:

- **Python 3.8+** installed ([Download](https://www.python.org/downloads/))
- **Anthropic API Key** ([Get one here](https://console.anthropic.com/))
- **FFmpeg** installed (see Quick Install below)

---

## 📦 Quick Install

### 1. Install FFmpeg

**Windows:**

```bash
# Download from: https://ffmpeg.org/download.html
# Extract and add to PATH, or:
winget install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
sudo apt update && sudo apt install ffmpeg
```

**Verify:**

```bash
ffmpeg -version
```

### 2. Install Python Dependencies

```bash
# Core dependencies
pip install anthropic python-dotenv opencv-python pillow numpy chromadb easyocr

# Video processing
pip install imageio imageio-ffmpeg

# Audio processing
pip install ffmpeg-python librosa soundfile torch transformers
```

### 3. Set Up API Key

Create a `.env` file in the project root:

```bash
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

### 4. Create Directories

```bash
mkdir -p data/videos data/images output
```

---

## 🎬 Running Your First Analysis

### Process a Single Image

```bash
# Place your image in data/images/ or specify path directly
python img_orc_agent.py path/to/your/image.jpg
```

**Output:** `./output/results.json`

### Process a Single Video

```bash
# Basic video analysis (visual only)
python video_orc_agent_main.py path/to/your/video.mp4

# Output: ./output/{filename}_enhanced_results.json
```

### Process Multiple Images (Batch)

```bash
# Place all images in data/images/ directory
python img_orc_agent.py

# Output: ./output/results.json (contains all images)
```

### Process Multiple Videos (Batch)

**Option 1: Create Quick Batch Script**

Create `batch_videos.py`:

```python
#!/usr/bin/env python3
from pathlib import Path
from video_orc_agent_main import EnhancedVisionOrchestrator
from audio_processor import analyze_video_audio
import json

# Find all videos
video_dir = Path("data/videos")
videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))

print(f"Found {len(videos)} videos to process\n")

orchestrator = EnhancedVisionOrchestrator()

for i, video_path in enumerate(videos, 1):
    print(f"\n[{i}/{len(videos)}] Processing: {video_path.name}")

    # Visual analysis
    visual = orchestrator.process(str(video_path))

    # Audio analysis
    audio = analyze_video_audio(str(video_path))

    # Save combined results
    output_file = Path("output") / f"{video_path.stem}_complete.json"
    combined = {
        "video": video_path.name,
        "visual": visual["consolidated_video_analysis"],
        "audio": audio
    }

    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"✓ Saved: {output_file}")

print(f"\n✓ All {len(videos)} videos processed!")
```

**Run:**

```bash
python batch_videos.py
```

**Option 2: Simple Bash Loop (Linux/Mac)**

```bash
#!/bin/bash
for video in data/videos/*.mp4; do
    echo "Processing: $video"
    python video_orc_agent_main.py "$video"
done
```

---

## 🎯 Common Use Cases

### 1. Analyze Marketing Ad Video (Visual + Audio)

```bash
python video_orc_agent_main.py data/videos/nike_ad.mp4 -o results/nike_analysis.json
```

**Extracts:**

- Brand names
- Promo codes
- Deadlines
- CTAs
- Human presence
- Visual elements

**For audio too:**

```python
from audio_processor import analyze_video_audio
audio_results = analyze_video_audio("data/videos/nike_ad.mp4")
print(f"Emotion: {audio_results['emotion']}")
print(f"Gender: {audio_results['gender_estimation']}")
```

### 2. Batch Process Marketing Campaign (10+ Videos)

```bash
# Place all campaign videos in data/videos/
python batch_videos.py

# Results saved to output/ directory
# Each video gets: {name}_complete.json
```

**Check all promo codes found:**

```bash
grep -h "promo_code" output/*_complete.json | sort -u
```

### 2. Extract Promo Codes from Images

```bash
# Add promotional images to data/images/
python img_orc_agent.py

# Check results
cat output/results.json | grep -A 5 "promo_code"
```

### 3. Complete Batch Analysis (All Videos + Audio)

Create `analyze_all.py`:

```python
#!/usr/bin/env python3
"""Batch process all videos with visual + audio analysis"""
from pathlib import Path
from video_orc_agent_main import EnhancedVisionOrchestrator
from audio_processor import analyze_video_audio
import json

def analyze_all_videos():
    video_dir = Path("data/videos")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Find all videos
    videos = list(video_dir.glob("*.mp4")) + \
             list(video_dir.glob("*.avi")) + \
             list(video_dir.glob("*.mov"))

    if not videos:
        print(f"No videos found in {video_dir}")
        return

    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(videos)} videos")
    print(f"{'='*70}\n")

    orchestrator = EnhancedVisionOrchestrator()
    summary = []

    for i, video_path in enumerate(videos, 1):
        video_name = video_path.stem
        print(f"[{i}/{len(videos)}] {video_path.name}")

        try:
            # Visual analysis
            visual = orchestrator.process(str(video_path))

            # Audio analysis
            audio = analyze_video_audio(str(video_path))

            # Combine results
            result = {
                "video": video_path.name,
                "brand": visual["consolidated_video_analysis"].get("brand_name_text"),
                "promo_code": visual["consolidated_video_analysis"].get("promo_code"),
                "emotion": audio.get("emotion"),
                "gender": audio.get("gender_estimation"),
                "duration": audio.get("duration_sec"),
                "human_present": visual["consolidated_video_analysis"].get("human_present")
            }

            # Save individual result
            output_file = output_dir / f"{video_name}_complete.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "visual": visual["consolidated_video_analysis"],
                    "audio": audio
                }, f, indent=2)

            summary.append(result)
            print(f"  ✓ Brand: {result['brand']}, Emotion: {result['emotion']}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            summary.append({
                "video": video_path.name,
                "error": str(e)
            })

    # Save summary
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(videos)} videos processed")
    print(f"Summary saved: {summary_file}")
    print(f"{'='*70}\n")

    # Print quick insights
    brands = set(r["brand"] for r in summary if r.get("brand"))
    emotions = [r["emotion"] for r in summary if r.get("emotion")]

    print("Quick Insights:")
    print(f"  Unique brands: {len(brands)}")
    print(f"  Most common emotion: {max(set(emotions), key=emotions.count) if emotions else 'N/A'}")
    print(f"  Videos with promos: {sum(1 for r in summary if r.get('promo_code'))}")
    print(f"  Videos with humans: {sum(1 for r in summary if r.get('human_present'))}")

if __name__ == "__main__":
    analyze_all_videos()
```

**Run:**

```bash
python analyze_all.py
```

**Output:**

- Individual files: `output/{video_name}_complete.json`
- Summary: `output/batch_summary.json`

---

## 📊 Understanding the Output

### Image Output Format

```json
{
  "image_name": "promo_banner.jpg",
  "human_present": 1,
  "num_people": 2,
  "brand_name_text": "Nike",
  "promo_present": true,
  "promo_code": "SAVE20",
  "promo_deadline": "December 31",
  "cta_text": "Shop Now"
}
```

### Video Output Format

```json
{
  "consolidated_video_analysis": {
    "brand_name_text": "Nike",
    "product_name": "Air Max",
    "promo_present": true,
    "promo_code": "NIKE25",
    "promo_deadline": "Limited time",
    "discount_type": "percentage",
    "price_value": "25%",
    "cta_present": true,
    "cta_type": "shop_now",
    "human_present": true
  },
  "analysis": {
    "frames_analyzed": 25,
    "total_tokens_used": 50000,
    "estimated_cost_usd": 0.15
  }
}
```

---

## 🎛️ Quick Configuration

### Adjust Processing Speed vs Detail

**Faster (fewer frames, lower cost):**

```bash
python video_orc_agent_main.py video.mp4 --frame-interval 1.0 --similarity-threshold 0.95
```

**More Detail (more frames, higher cost):**

```bash
python video_orc_agent_main.py video.mp4 --frame-interval 0.2 --similarity-threshold 0.85
```

### Batch Process with Custom Settings

```bash
# Edit batch_videos.py and change initialization:
orchestrator = EnhancedVisionOrchestrator(
    frame_interval=0.5,           # Extract every 0.5 seconds
    similarity_threshold=0.92     # Moderate deduplication
)
```

### Custom Output Location

```bash
python video_orc_agent_main.py video.mp4 -o custom/path/results.json
```

### Process Only Audio (Skip Visual)

```python
# quick_audio_only.py
from pathlib import Path
from audio_processor import analyze_video_audio
import json

for video in Path("data/videos").glob("*.mp4"):
    print(f"Processing audio: {video.name}")
    audio = analyze_video_audio(str(video))

    output = Path("output") / f"{video.stem}_audio.json"
    with open(output, 'w') as f:
        json.dump(audio, f, indent=2)
```

---

## 🐛 Quick Troubleshooting

### "ANTHROPIC_API_KEY not found"

```bash
# Create .env file with your key
echo "ANTHROPIC_API_KEY=sk-ant-xxxxx" > .env
```

### "FFmpeg not found"

```bash
# Install FFmpeg (see step 1 above)
# Or update path in audio_processor.py line 13
```

### "CUDA not available" warning

```bash
# This is OK! Will use CPU (slower but works)
# To enable GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt
```

---

## 📝 Quick Tips

1. **Start small**: Test with a short video (5-10 seconds) first
2. **Check costs**: Use `--frame-interval 1.0` for testing to reduce API costs
3. **Batch at night**: Run batch processing overnight for large datasets
4. **Check outputs**: Results are in `./output/` directory
5. **View logs**: Console output shows detailed progress

---

## 📁 Quick Directory Setup

```bash
# Create all needed directories at once
mkdir -p data/{videos,images} output frames_storage audio_temp chroma_visual_db

# Verify structure
tree -L 2 -d
```

Expected structure:

```
.
├── data/
│   ├── videos/     # Put videos here
│   └── images/     # Put images here
├── output/         # Results go here
├── frames_storage/ # Auto-created
├── audio_temp/     # Auto-created
└── chroma_visual_db/ # Auto-created
```

---

## 🎯 Most Common Commands

```bash
# ===== SINGLE FILE PROCESSING =====

# Single image
python img_orc_agent.py image.jpg

# Single video (visual only)
python video_orc_agent_main.py video.mp4

# Single video (visual + audio manually)
python video_orc_agent_main.py video.mp4
python -c "from audio_processor import analyze_video_audio; print(analyze_video_audio('video.mp4'))"

# ===== BATCH PROCESSING =====

# Batch images from data/images/
python img_orc_agent.py

# Batch videos (complete analysis)
python analyze_all.py

# Batch videos (simple visual-only loop)
for video in data/videos/*.mp4; do
  python video_orc_agent_main.py "$video"
done

# ===== CUSTOM SETTINGS =====

# Single video with custom settings
python video_orc_agent_main.py video.mp4 \
  --frame-interval 0.5 \
  --similarity-threshold 0.9 \
  -o results/output.json

# ===== VIEW RESULTS =====

# Check results (pretty print)
cat output/results.json | python -m json.tool | less

# Search for promo codes
grep -r "promo_code" output/*.json

# Search for specific brand
grep -r "Nike" output/*.json

# View batch summary
cat output/batch_summary.json | python -m json.tool
```

---

## ⏱️ Expected Processing Times

| Task                             | Time           | Cost (approx) |
| -------------------------------- | -------------- | ------------- |
| Single image                     | 3-6 seconds    | $0.001        |
| 10-second video (visual only)    | 30-60 seconds  | $0.10-0.20    |
| 30-second video (visual only)    | 45-90 seconds  | $0.30-0.60    |
| 30-second video (visual + audio) | 50-100 seconds | $0.30-0.60    |
| Audio only (30 seconds)          | 5-10 seconds   | Free (local)  |
| Batch 10 images                  | 30-60 seconds  | $0.01         |
| Batch 10 videos (30s each)       | 10-15 minutes  | $3-6          |

_Costs assume Claude Sonnet 4 pricing (~$3 per million tokens)_
_Audio processing is FREE (runs locally, no API calls)_

### Batch Processing Tips

**For 10+ videos:**

- Run overnight or during off-hours
- Monitor first 2-3 videos to verify settings
- Use `--frame-interval 0.5` or higher to reduce cost
- Audio analysis adds minimal time (~5-10s per video)

---

## 🔗 Next Steps

Once you have the basics working:

1. **Read the full README.md** for advanced features
2. **Explore batch processing** for multiple files
3. **Try audio analysis** for video emotion detection
4. **Customize parameters** for your specific use case
5. **Set up automation** for regular processing jobs

---

## 💡 Quick Examples

### Extract All Promo Codes (Batch)

```bash
# Process all videos
python analyze_all.py

# Extract all promo codes found
python -c "
import json
from pathlib import Path

promos = []
for file in Path('output').glob('*_complete.json'):
    with open(file) as f:
        data = json.load(f)

    visual = data.get('visual', {})
    if visual.get('promo_present') and visual.get('promo_code'):
        promos.append({
            'video': file.stem.replace('_complete', ''),
            'brand': visual.get('brand_name_text'),
            'code': visual.get('promo_code'),
            'deadline': visual.get('promo_deadline')
        })

print('\nPROMO CODES FOUND:')
print('='*60)
for p in promos:
    print(f\"{p['brand']}: {p['code']} (Expires: {p['deadline']})\")
"
```

### Find Videos by Emotion (Batch Audio Analysis)

```python
# emotion_filter.py
import json
from pathlib import Path

def find_by_emotion(target_emotion="happy"):
    matches = []

    for file in Path("output").glob("*_complete.json"):
        with open(file) as f:
            data = json.load(f)

        audio = data.get("audio", {})
        if audio.get("emotion") == target_emotion:
            matches.append({
                "video": file.stem.replace("_complete", ""),
                "brand": data.get("visual", {}).get("brand_name_text"),
                "emotion": audio.get("emotion"),
                "gender": audio.get("gender_estimation")
            })

    print(f"\nVideos with '{target_emotion}' emotion:")
    for m in matches:
        print(f"  {m['video']}: {m['brand']} ({m['gender']} voice)")

find_by_emotion("happy")
```

### Find Videos with Humans

```bash
# After processing, search for human presence
python -c "
import json
from pathlib import Path

for file in Path('output').glob('*_enhanced_results.json'):
    with open(file) as f:
        data = json.load(f)
    if data['consolidated_video_analysis'].get('human_present'):
        print(f'{file.name}: Humans detected')
"
```

### Cost Estimation Before Processing

```python
import cv2

video = "data/videos/ad.mp4"
cap = cv2.VideoCapture(video)
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
cap.release()

# Rough estimate: 1 frame per 0.5s, 70% unique, 2500 tokens/frame
unique_frames = int((duration / 0.5) * 0.3)
tokens = unique_frames * 2500
cost = (tokens / 1_000_000) * 3

print(f"Estimated frames: {unique_frames}")
print(f"Estimated cost: ${cost:.2f}")
```

---

## 🆘 Need Help?

1. **Check console output** - shows detailed progress and errors
2. **Read error messages** - usually indicate what's wrong
3. **Verify API key** - ensure it's valid and has credits
4. **Check file paths** - make sure files exist and are readable
5. **Review full README.md** - contains detailed troubleshooting

---

## ✅ Quick Verification Checklist

Before reporting issues, verify:

- [ ] Python 3.8+ installed (`python --version`)
- [ ] FFmpeg installed (`ffmpeg -version`)
- [ ] `.env` file exists with valid API key
- [ ] Dependencies installed (`pip list | grep anthropic`)
- [ ] Input files exist and are readable
- [ ] Output directory is writable
- [ ] Sufficient disk space available

---
