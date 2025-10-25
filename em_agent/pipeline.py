"""
Intelligent Visual Deduplication & Feature Extraction System
Core Pipeline: Input Handling, Frame Extraction, Similarity Detection
"""

import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json

import cv2
import numpy as np
from PIL import Image
import imagehash
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import subprocess
import tempfile
import shutil


@dataclass
class VisualMetadata:
    """Metadata for processed visual content"""
    source_path: str
    frame_index: Optional[int]
    timestamp: str
    perceptual_hash: str
    similarity_score: float
    is_duplicate: bool


class InputHandlingAgent:
    """Agent 1: Handles input validation and type detection"""
    
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    
    def __init__(self):
        self.processed_files = []
    
    def validate_and_classify(self, file_path: str) -> Tuple[bool, str, str]:
        """
        Validates input file and classifies as image or video
        
        Returns:
            (is_valid, file_type, error_message)
        """
        path = Path(file_path)
        
        if not path.exists():
            return False, None, f"File not found: {file_path}"
        
        if not path.is_file():
            return False, None, f"Not a file: {file_path}"
        
        ext = path.suffix.lower()
        
        if ext in self.SUPPORTED_IMAGE_FORMATS:
            return True, "image", ""
        elif ext in self.SUPPORTED_VIDEO_FORMATS:
            return True, "video", ""
        else:
            return False, None, f"Unsupported format: {ext}"
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Extract basic file metadata"""
        path = Path(file_path)
        stat = path.stat()
        
        return {
            "path": str(path.absolute()),
            "name": path.name,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }


class FrameExtractionAgent:
    """Agent 2: Extracts frames from videos or processes images"""
    
    def __init__(self, frame_interval: float = 1.0):
        """
        Args:
            frame_interval: Seconds between frame extractions for videos
        """
        self.frame_interval = frame_interval
    
    def extract_frames(self, file_path: str, file_type: str) -> List[Tuple[np.ndarray, int, str]]:
        """
        Extract frames from video or load image
        
        Returns:
            List of (frame_array, frame_index, perceptual_hash)
        """
        if file_type == "image":
            return self._process_image(file_path)
        elif file_type == "video":
            return self._process_video(file_path)
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def _process_image(self, file_path: str) -> List[Tuple[np.ndarray, int, str]]:
        """Process single image"""
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        # Convert to RGB for consistency
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        phash = self._compute_perceptual_hash(img_rgb)
        
        return [(img_rgb, 0, phash)]
    
    def _process_video(self, file_path: str) -> List[Tuple[np.ndarray, int, str]]:
        """Extract frames from video at specified interval"""
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {file_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps * self.frame_interval)
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                phash = self._compute_perceptual_hash(frame_rgb)
                frames.append((frame_rgb, extracted_count, phash))
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def _compute_perceptual_hash(self, frame: np.ndarray) -> str:
        """Compute perceptual hash for quick duplicate detection"""
        pil_img = Image.fromarray(frame)
        phash = imagehash.phash(pil_img)
        return str(phash)


class SimilarityDetectionAgent:
    """Agent 3: Detects duplicates using ChromaDB and embeddings"""
    
    def __init__(
        self, 
        db_path: str = "./chroma_db",
        collection_name: str = "visual_embeddings",
        similarity_threshold: float = 0.9,
        embedding_model: str = "clip-ViT-B-32"
    ):
        """
        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the collection
            similarity_threshold: Cosine similarity threshold for duplicates (0-1)
            embedding_model: Model for computing visual embeddings
        """
        self.similarity_threshold = similarity_threshold
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        print("Model loaded successfully")
    
    def compute_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Compute CLIP embedding for a frame"""
        # Convert to PIL Image
        pil_img = Image.fromarray(frame)
        
        # Compute embedding
        embedding = self.model.encode(pil_img, convert_to_numpy=True)
        return embedding
    
    def check_similarity(self, embedding: np.ndarray, metadata: Dict) -> Tuple[bool, float]:
        """
        Check if embedding is similar to existing ones in database
        
        Returns:
            (is_duplicate, max_similarity_score)
        """
        # Query collection
        if self.collection.count() == 0:
            return False, 0.0
        
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=1
        )
        
        if not results['distances'] or len(results['distances'][0]) == 0:
            return False, 0.0
        
        # ChromaDB returns distances, convert to similarity
        # Cosine distance = 1 - cosine similarity
        max_distance = results['distances'][0][0]
        max_similarity = 1.0 - max_distance
        
        is_duplicate = max_similarity >= self.similarity_threshold
        
        return is_duplicate, max_similarity
    
    def store_embedding(self, embedding: np.ndarray, metadata: VisualMetadata) -> str:
        """Store embedding in ChromaDB"""
        # Generate unique ID
        unique_id = hashlib.md5(
            f"{metadata.source_path}_{metadata.frame_index}_{metadata.timestamp}".encode()
        ).hexdigest()
        
        # Store in collection
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[{
                "source_path": metadata.source_path,
                "frame_index": str(metadata.frame_index) if metadata.frame_index is not None else "0",
                "timestamp": metadata.timestamp,
                "perceptual_hash": metadata.perceptual_hash,
                "similarity_score": metadata.similarity_score,
                "is_duplicate": metadata.is_duplicate
            }],
            ids=[unique_id]
        )
        
        return unique_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "total_embeddings": self.collection.count(),
            "similarity_threshold": self.similarity_threshold
        }


class VisualDeduplicationPipeline:
    """Main pipeline orchestrating all agents"""
    
    def __init__(
        self,
        db_path: str = "./chroma_db",
        frame_interval: float = 0.3,
        similarity_threshold: float = 0.9
    ):
        self.input_agent = InputHandlingAgent()
        self.frame_agent = FrameExtractionAgent(frame_interval=frame_interval)
        self.similarity_agent = SimilarityDetectionAgent(
            db_path=db_path,
            similarity_threshold=similarity_threshold
        )
        
        self.processing_log = []
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file through the complete pipeline
        
        Returns:
            Processing summary with statistics
        """
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print(f"{'='*60}")
        
        # Agent 1: Input Handling
        print("\n[Agent 1] Input Handling & Validation...")
        is_valid, file_type, error = self.input_agent.validate_and_classify(file_path)
        
        if not is_valid:
            return {"success": False, "error": error}
        
        file_info = self.input_agent.get_file_info(file_path)
        print(f"✓ File type: {file_type}")
        print(f"✓ Size: {file_info['size_mb']:.2f} MB")
        
        # Agent 2: Frame Extraction
        print(f"\n[Agent 2] Frame Extraction...")
        frames = self.frame_agent.extract_frames(file_path, file_type)
        print(f"✓ Extracted {len(frames)} frame(s)")
        
        # Agent 3: Similarity Detection & Storage
        print(f"\n[Agent 3] Similarity Detection & Deduplication...")
        
        results = {
            "file_path": file_path,
            "file_type": file_type,
            "total_frames": len(frames),
            "unique_frames": 0,
            "duplicate_frames": 0,
            "frames_processed": []
        }
        
        for frame, idx, phash in frames:
            # Compute embedding
            embedding = self.similarity_agent.compute_embedding(frame)
            
            # Check similarity
            metadata = VisualMetadata(
                source_path=file_path,
                frame_index=idx,
                timestamp=datetime.now().isoformat(),
                perceptual_hash=phash,
                similarity_score=0.0,
                is_duplicate=False
            )
            
            is_dup, sim_score = self.similarity_agent.check_similarity(embedding, metadata.__dict__)
            metadata.similarity_score = sim_score
            metadata.is_duplicate = is_dup
            
            if is_dup:
                results["duplicate_frames"] += 1
                status = f"⊗ SKIP (duplicate, similarity: {sim_score:.3f})"
            else:
                # Store unique embedding
                unique_id = self.similarity_agent.store_embedding(embedding, metadata)
                results["unique_frames"] += 1
                status = f"✓ STORED (id: {unique_id[:8]}...)"
            
            print(f"  Frame {idx}: {status}")
            
            results["frames_processed"].append({
                "frame_index": idx,
                "perceptual_hash": phash,
                "is_duplicate": is_dup,
                "similarity_score": sim_score
            })
        
        # Summary
        skip_ratio = results["duplicate_frames"] / results["total_frames"] if results["total_frames"] > 0 else 0
        
        print(f"\n{'─'*60}")
        print(f"Summary:")
        print(f"  Total frames: {results['total_frames']}")
        print(f"  Unique: {results['unique_frames']}")
        print(f"  Duplicates skipped: {results['duplicate_frames']}")
        print(f"  Skip ratio: {skip_ratio:.1%}")
        print(f"{'─'*60}")
        
        results["skip_ratio"] = skip_ratio
        results["success"] = True
        
        self.processing_log.append(results)
        return results
    
    def process_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files"""
        results = []
        for fp in file_paths:
            result = self.process_file(fp)
            results.append(result)
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics"""
        return self.similarity_agent.get_stats()
    
    def export_log(self, output_path: str = "processing_log.json"):
        """Export processing log to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.processing_log, f, indent=2)
        print(f"\nProcessing log exported to: {output_path}")


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    import config
    
    # Initialize pipeline
    pipeline = VisualDeduplicationPipeline(
        db_path="./chroma_visual_db",
        frame_interval=0.3,  # 3 frame per second for videos
        similarity_threshold=0.9  # 90% similarity threshold
    )
    
    # Collect all files from data/images and data/videos
    files_to_process = []
    
    # Get all images
    image_dir = Path("data/images")
    if image_dir.exists():
        image_files = list(image_dir.glob("*.*"))
        image_files = [str(f) for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}]
        files_to_process.extend(image_files)
        print(f"Found {len(image_files)} images")
    
    # Get all videos
    video_dir = Path("data/videos")
    if video_dir.exists():
        video_files = list(video_dir.glob("*.*"))
        video_files = [str(f) for f in video_files if f.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}]
        files_to_process.extend(video_files)
        print(f"Found {len(video_files)} videos")
    
    if not files_to_process:
        print("\n⚠ No files found in data/images or data/videos")
        print("Please add some images or videos to process")
    else:
        print(f"\n🚀 Processing {len(files_to_process)} total files...\n")
        
        # Process all files
        results = pipeline.process_batch(files_to_process)
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        
        total_frames = sum(r.get('total_frames', 0) for r in results if r.get('success'))
        unique_frames = sum(r.get('unique_frames', 0) for r in results if r.get('success'))
        duplicate_frames = sum(r.get('duplicate_frames', 0) for r in results if r.get('success'))
        
        print(f"\nTotal frames processed: {total_frames}")
        print(f"Unique frames stored: {unique_frames}")
        print(f"Duplicates skipped: {duplicate_frames}")
        if total_frames > 0:
            print(f"Skip ratio: {(duplicate_frames/total_frames)*100:.1f}%")
        
        # Get database stats
        stats = pipeline.get_database_stats()
        print(f"\nDatabase embeddings: {stats['total_embeddings']}")
        
        # Export processing log
        pipeline.export_log("logs/processing_log.json")
        print("\n✅ Done! Check logs/processing_log.json for details")