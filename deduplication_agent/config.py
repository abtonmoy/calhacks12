"""
Configuration settings for Visual Deduplication Pipeline
"""

# Database settings
DB_PATH = "./chroma_visual_db"
COLLECTION_NAME = "visual_embeddings"

# Frame extraction settings
FRAME_INTERVAL = 1.0  # Extract 1 frame per second from videos
USE_FFMPEG = True     # Use ffmpeg for video processing (recommended)

# Similarity detection settings
SIMILARITY_THRESHOLD = 0.90  # 0.90 = 90% similar = duplicate
EMBEDDING_MODEL = "clip-ViT-B-32"  # Options: "clip-ViT-B-32", "clip-ViT-L-14"

# Input/Output paths
DATA_DIR = "./data"
IMAGE_DIR = "./data/images"
VIDEO_DIR = "./data/videos"
LOG_DIR = "./logs"

# Processing settings
BATCH_SIZE = 10  # Process N files at a time
ENABLE_LOGGING = True