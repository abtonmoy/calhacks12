# API Endpoints Documentation

## 🎯 Unified API Design

The API supports both **video** and **image** uploads with automatic routing to the appropriate analysis agent.

### 🔍 Endpoint Comparison

| Endpoint             | Input | Agent Used                   | Output                   |
| -------------------- | ----- | ---------------------------- | ------------------------ |
| `/api/detect-human`  | Image | `SimpleHumanDetector`        | Human detection          |
| `/api/analyze-audio` | Video | `audio_processor`            | Audio features + emotion |
| `/api/analyze-video` | Video | `EnhancedVisionOrchestrator` | Full video analysis      |

## 📡 Endpoints

### POST `/api/detect-human`

**Purpose:** Detect humans in uploaded image

**Request:**

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5001/api/detect-human
```

**Response:**

```json
{
  "human_present": 1,
  "num_people": 6
}
```

---

### POST `/api/analyze-audio`

**Purpose:** Analyze audio features and emotion from video

**Request:**

```bash
curl -X POST -F "file=@video.mp4" http://localhost:5001/api/analyze-audio
```

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

---

### POST `/api/analyze-video` (NEW)

**Purpose:** Full video analysis with marketing signals, human detection, and branding

**Request:**

```bash
curl -X POST -F "file=@video.mp4" http://localhost:5001/api/analyze-video
```

**Response:**

```json
{
  "success": true,
  "file_info": {
    "file_path": "uploads/video.mp4",
    "file_name": "video.mp4",
    "file_type": "video",
    "processed_at": "2025-01-15T10:30:00",
    "processing_time_seconds": 45.2
  },
  "deduplication": {
    "total_frames_extracted": 150,
    "unique_frames_stored": 45,
    "duplicate_frames_skipped": 105,
    "skip_ratio": "70.0%"
  },
  "analysis": {
    "frames_analyzed": 45,
    "frames_with_brands": 42,
    "frames_with_promos": 15,
    "frames_with_promo_codes": 8,
    "frames_with_humans": 30,
    "total_tokens_used": 125000,
    "estimated_cost_usd": 0.375
  },
  "consolidated_video_analysis": {
    "brand_name_text": "BrandName",
    "product_name": "Product",
    "industry": "Fashion",
    "promo_present": true,
    "promo_text": "Get 50% off",
    "promo_code": "SAVE50",
    "promo_deadline": "12/31/2024",
    "discount_type": "percentage",
    "price_value": "50%",
    "cta_present": true,
    "cta_type": "shop_now",
    "cta_text": "Shop Now",
    "text_density": "high",
    "brand_text_contrast": "high",
    "visual_elements": ["logo", "product", "price_tag"],
    "color_scheme": "vibrant colors",
    "aesthetic_style": "modern minimalist",
    "logos_detected": ["BrandLogo"],
    "narrative_arc": "intro → product → promo → cta",
    "human_present": true,
    "_consolidation_meta": {
      "total_frames_analyzed": 45,
      "frames_with_brand": 42,
      "frames_with_product": 40,
      "frames_with_promo": 15,
      "frames_with_cta": 38,
      "frames_with_humans": 30,
      "unique_brands_detected": 1,
      "unique_products_detected": 1
    }
  },
  "frames": [...],
  "marketing_insights": {...}
}
```

---

### GET `/api/health`

**Purpose:** Health check endpoint

**Response:**

```json
{
  "status": "ok",
  "message": "API server is running"
}
```

## 🎯 When to Use Which Endpoint?

### Use `/api/detect-human` when:

- ✅ You only need to know if humans are in an image
- ✅ Quick, lightweight check
- ✅ Budget-conscious (no API costs)

### Use `/api/analyze-audio` when:

- ✅ You need audio emotion analysis
- ✅ Gender estimation from voice
- ✅ Audio feature extraction

### Use `/api/analyze-video` when:

- ✅ You need comprehensive video analysis
- ✅ Marketing signal detection
- ✅ Brand/product identification
- ✅ Promo code extraction
- ✅ CTA detection
- ✅ Narrative arc analysis
- ✅ Full visual understanding

## 💰 Cost Considerations

| Endpoint             | Cost              | Speed          | Depth         |
| -------------------- | ----------------- | -------------- | ------------- |
| `/api/detect-human`  | Free              | Fast (<1s)     | Basic         |
| `/api/analyze-audio` | Free              | Medium (30s)   | Audio only    |
| `/api/analyze-video` | ~$0.30-1.00/video | Slow (60-120s) | Comprehensive |

**Note:** Video analysis uses Claude Vision API (approximately $0.30-1.00 per video depending on length).

## 🔄 Future Enhancement

Consider adding a **smart routing endpoint**:

```
POST /api/analyze
```

Automatically detects if upload is image or video and routes to appropriate agent.
