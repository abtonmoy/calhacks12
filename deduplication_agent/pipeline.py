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
import base64
from io import BytesIO

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
    frame_path: Optional[str] = None  # Path to saved frame image


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
    """Agent 2: Extracts frames from videos using ffmpeg or processes images"""
    
    def __init__(self, frame_interval: float = 1.0, use_ffmpeg: bool = True):
        """
        Args:
            frame_interval: Seconds between frame extractions for videos
            use_ffmpeg: Use ffmpeg for video processing (recommended)
        """
        self.frame_interval = frame_interval
        self.use_ffmpeg = use_ffmpeg
        
        # Check ffmpeg availability
        if use_ffmpeg:
            try:
                subprocess.run(['ffmpeg', '-version'], 
                             capture_output=True, check=True)
                print("✓ ffmpeg detected")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠ ffmpeg not found, falling back to OpenCV")
                self.use_ffmpeg = False
    
    def extract_frames(self, file_path: str, file_type: str) -> List[Tuple[np.ndarray, int, str]]:
        """
        Extract frames from video or load image
        
        Returns:
            List of (frame_array, frame_index, perceptual_hash)
        """
        if file_type == "image":
            return self._process_image(file_path)
        elif file_type == "video":
            if self.use_ffmpeg:
                return self._process_video_ffmpeg(file_path)
            else:
                return self._process_video_opencv(file_path)
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def _process_image(self, file_path: str) -> List[Tuple[np.ndarray, int, str]]:
        """Process single image using PIL for better format support"""
        try:
            # Use PIL for better format support
            pil_img = Image.open(file_path).convert('RGB')
            img_array = np.array(pil_img)
            phash = self._compute_perceptual_hash(img_array)
            return [(img_array, 0, phash)]
        except Exception as e:
            raise ValueError(f"Failed to load image {file_path}: {e}")
    
    def _process_video_ffmpeg(self, file_path: str) -> List[Tuple[np.ndarray, int, str]]:
        """
        Extract frames from video using ffmpeg (RECOMMENDED)
        
        Much faster and more reliable than OpenCV
        """
        frames = []
        temp_dir = None
        
        try:
            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp(prefix='ffmpeg_frames_')
            output_pattern = os.path.join(temp_dir, 'frame_%04d.jpg')
            
            # Get video info first
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate,duration',
                '-of', 'json',
                file_path
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(probe_result.stdout)
            
            # Parse frame rate
            if probe_data.get('streams'):
                fps_str = probe_data['streams'][0].get('r_frame_rate', '30/1')
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = 30.0  # fallback
            
            # Extract frames at specified interval using ffmpeg
            # -vf fps=1/N extracts 1 frame every N seconds
            extract_fps = 1.0 / self.frame_interval
            
            cmd = [
                'ffmpeg',
                '-i', file_path,
                '-vf', f'fps={extract_fps}',
                '-q:v', '2',  # High quality JPEG
                '-f', 'image2',
                output_pattern,
                '-hide_banner',
                '-loglevel', 'error'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Load extracted frames
            frame_files = sorted(Path(temp_dir).glob('frame_*.jpg'))
            
            for idx, frame_file in enumerate(frame_files):
                # Load frame using PIL
                pil_img = Image.open(frame_file).convert('RGB')
                frame_array = np.array(pil_img)
                phash = self._compute_perceptual_hash(frame_array)
                frames.append((frame_array, idx, phash))
            
            return frames
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise ValueError(f"ffmpeg failed: {error_msg}")
        except Exception as e:
            raise ValueError(f"Frame extraction failed: {e}")
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _process_video_opencv(self, file_path: str) -> List[Tuple[np.ndarray, int, str]]:
        """
        Extract frames from video using OpenCV (FALLBACK)
        
        Use only if ffmpeg is not available
        """
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {file_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # fallback
        
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
    
    def get_video_info(self, file_path: str) -> Dict[str, Any]:
        """Get video metadata using ffprobe"""
        if not self.use_ffmpeg:
            return {"error": "ffmpeg not available"}
        
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration,size,bit_rate:stream=width,height,r_frame_rate,codec_name',
            '-of', 'json',
            file_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception as e:
            return {"error": str(e)}


class SimilarityDetectionAgent:
    """Agent 3: Detects duplicates using ChromaDB and embeddings"""
    
    def __init__(
        self, 
        db_path: str = "./chroma_db",
        collection_name: str = "visual_embeddings",
        similarity_threshold: float = 0.9,
        embedding_model: str = "clip-ViT-B-32",
        frames_storage_path: str = "./frames_storage"
    ):
        """
        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the collection
            similarity_threshold: Cosine similarity threshold for duplicates (0-1)
            embedding_model: Model for computing visual embeddings
            frames_storage_path: Directory to store extracted frames
        """
        self.similarity_threshold = similarity_threshold
        self.frames_storage_path = Path(frames_storage_path)
        self.frames_storage_path.mkdir(parents=True, exist_ok=True)
        
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
    
    def save_frame(self, frame: np.ndarray, unique_id: str) -> str:
        """
        Save frame image to disk
        
        Returns:
            Path to saved frame
        """
        frame_filename = f"{unique_id}.jpg"
        frame_path = self.frames_storage_path / frame_filename
        
        # Convert numpy array to PIL Image and save
        pil_img = Image.fromarray(frame)
        pil_img.save(frame_path, "JPEG", quality=95)
        
        return str(frame_path)
    
    def load_frame(self, frame_path: str) -> np.ndarray:
        """
        Load frame image from disk
        
        Returns:
            Frame as numpy array
        """
        pil_img = Image.open(frame_path).convert('RGB')
        return np.array(pil_img)
    
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
    
    def store_embedding(self, embedding: np.ndarray, frame: np.ndarray, metadata: VisualMetadata) -> str:
        """
        Store embedding and frame in database
        
        Returns:
            Unique ID of stored embedding
        """
        # Generate unique ID
        unique_id = hashlib.md5(
            f"{metadata.source_path}_{metadata.frame_index}_{metadata.timestamp}".encode()
        ).hexdigest()
        
        # Save the actual frame image
        frame_path = self.save_frame(frame, unique_id)
        metadata.frame_path = frame_path
        
        # Store in collection
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[{
                "source_path": metadata.source_path,
                "frame_index": str(metadata.frame_index) if metadata.frame_index is not None else "0",
                "timestamp": metadata.timestamp,
                "perceptual_hash": metadata.perceptual_hash,
                "similarity_score": metadata.similarity_score,
                "is_duplicate": metadata.is_duplicate,
                "frame_path": frame_path  # Store frame path in metadata
            }],
            ids=[unique_id]
        )
        
        return unique_id
    
    def get_frame_by_id(self, frame_id: str) -> Optional[np.ndarray]:
        """
        Retrieve frame by its unique ID
        
        Returns:
            Frame as numpy array or None if not found
        """
        try:
            result = self.collection.get(ids=[frame_id])
            if result and result['metadatas']:
                frame_path = result['metadatas'][0].get('frame_path')
                if frame_path and os.path.exists(frame_path):
                    return self.load_frame(frame_path)
        except Exception as e:
            print(f"Error retrieving frame {frame_id}: {e}")
        return None
    
    def get_all_frames(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all stored frames with their metadata
        
        Returns:
            List of dictionaries with frame data and metadata
        """
        try:
            # Get all items from collection
            result = self.collection.get(limit=limit)
            
            frames_data = []
            for i, frame_id in enumerate(result['ids']):
                metadata = result['metadatas'][i]
                frame_path = metadata.get('frame_path')
                
                frame_info = {
                    'id': frame_id,
                    'source_path': metadata.get('source_path'),
                    'frame_index': metadata.get('frame_index'),
                    'timestamp': metadata.get('timestamp'),
                    'perceptual_hash': metadata.get('perceptual_hash'),
                    'frame_path': frame_path,
                    'frame_exists': os.path.exists(frame_path) if frame_path else False
                }
                
                frames_data.append(frame_info)
            
            return frames_data
        except Exception as e:
            print(f"Error getting all frames: {e}")
            return []
    
    def get_frames_by_source(self, source_path: str) -> List[Dict[str, Any]]:
        """
        Get all frames from a specific source file (video or image)
        
        Args:
            source_path: Path to the source file
            
        Returns:
            List of frames from that source, sorted by frame index
        """
        try:
            # Get all frames
            all_frames = self.collection.get()
            
            # Filter by source path
            matching_frames = []
            for i, frame_id in enumerate(all_frames['ids']):
                metadata = all_frames['metadatas'][i]
                if metadata.get('source_path') == source_path:
                    frame_path = metadata.get('frame_path')
                    frame_info = {
                        'id': frame_id,
                        'source_path': metadata.get('source_path'),
                        'frame_index': int(metadata.get('frame_index', 0)),
                        'timestamp': metadata.get('timestamp'),
                        'perceptual_hash': metadata.get('perceptual_hash'),
                        'frame_path': frame_path,
                        'frame_exists': os.path.exists(frame_path) if frame_path else False
                    }
                    matching_frames.append(frame_info)
            
            # Sort by frame index
            matching_frames.sort(key=lambda x: x['frame_index'])
            
            return matching_frames
        except Exception as e:
            print(f"Error getting frames by source: {e}")
            return []
    
    def get_all_sources(self) -> List[Dict[str, Any]]:
        """
        Get list of all unique source files with frame counts
        
        Returns:
            List of source files with metadata
        """
        try:
            all_frames = self.collection.get()
            
            # Group by source
            sources = {}
            for i, frame_id in enumerate(all_frames['ids']):
                metadata = all_frames['metadatas'][i]
                source = metadata.get('source_path')
                
                if source not in sources:
                    sources[source] = {
                        'source_path': source,
                        'frame_count': 0,
                        'frame_ids': []
                    }
                
                sources[source]['frame_count'] += 1
                sources[source]['frame_ids'].append(frame_id)
            
            return list(sources.values())
        except Exception as e:
            print(f"Error getting sources: {e}")
            return []
    
    def search_similar_frames(self, query_frame: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar frames to a query frame
        
        Args:
            query_frame: Frame to search for
            top_k: Number of similar frames to return
            
        Returns:
            List of similar frames with similarity scores
        """
        # Compute embedding for query frame
        query_embedding = self.compute_embedding(query_frame)
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        similar_frames = []
        for i in range(len(results['ids'][0])):
            frame_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            similarity = 1.0 - distance
            metadata = results['metadatas'][0][i]
            
            similar_frames.append({
                'id': frame_id,
                'similarity': similarity,
                'source_path': metadata.get('source_path'),
                'frame_index': metadata.get('frame_index'),
                'frame_path': metadata.get('frame_path')
            })
        
        return similar_frames
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        total_frames = self.collection.count()
        
        # Count actual saved frames on disk
        saved_frames = len(list(self.frames_storage_path.glob("*.jpg")))
        
        return {
            "total_embeddings": total_frames,
            "saved_frames_on_disk": saved_frames,
            "frames_storage_path": str(self.frames_storage_path),
            "similarity_threshold": self.similarity_threshold
        }


class VisualDeduplicationPipeline:
    """Main pipeline orchestrating all agents"""
    
    def __init__(
        self,
        db_path: str = "./chroma_db",
        frame_interval: float = 0.3,
        similarity_threshold: float = 0.9,
        use_ffmpeg: bool = True,
        frames_storage_path: str = "./frames_storage"
    ):
        self.input_agent = InputHandlingAgent()
        self.frame_agent = FrameExtractionAgent(
            frame_interval=frame_interval,
            use_ffmpeg=use_ffmpeg
        )
        self.similarity_agent = SimilarityDetectionAgent(
            db_path=db_path,
            similarity_threshold=similarity_threshold,
            frames_storage_path=frames_storage_path
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
                # Store unique embedding AND frame
                unique_id = self.similarity_agent.store_embedding(embedding, frame, metadata)
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
    
    def get_all_stored_frames(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all stored frames with metadata
        
        Returns:
            List of frame data with paths and metadata
        """
        return self.similarity_agent.get_all_frames(limit=limit)
    
    def get_frames_from_video(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Get all frames extracted from a specific video
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of frames from that video, sorted by frame index
        """
        return self.similarity_agent.get_frames_by_source(video_path)
    
    def get_all_source_files(self) -> List[Dict[str, Any]]:
        """
        Get list of all source files (videos/images) with frame counts
        
        Returns:
            List of source files with metadata
        """
        return self.similarity_agent.get_all_sources()
    
    def get_frames_grouped_by_source(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all frames grouped by their source video/image
        Perfect for batch processing in downstream models
        
        Returns:
            Dictionary mapping source_path -> list of frames (sorted by frame_index)
            {
                "data/videos/video1.mp4": [frame0, frame1, frame2, ...],
                "data/videos/video2.mp4": [frame0, frame1, ...],
                "data/images/img1.jpg": [frame0],
            }
        """
        grouped = {}
        sources = self.get_all_source_files()
        
        for source_info in sources:
            source_path = source_info['source_path']
            frames = self.get_frames_from_video(source_path)
            grouped[source_path] = frames
        
        return grouped
    
    def get_frame_arrays_by_source(self, source_path: str) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Get actual frame images (as numpy arrays) from a specific source
        Ready to feed into your next model
        
        Args:
            source_path: Path to the source video/image
            
        Returns:
            (list_of_frame_arrays, list_of_metadata)
            - frame_arrays: List of numpy arrays (H, W, 3) in chronological order
            - metadata: Corresponding metadata for each frame
        """
        frames_info = self.get_frames_from_video(source_path)
        
        frame_arrays = []
        metadata = []
        
        for frame_info in frames_info:
            frame_array = self.get_frame_by_id(frame_info['id'])
            if frame_array is not None:
                frame_arrays.append(frame_array)
                metadata.append(frame_info)
        
        return frame_arrays, metadata
    
    def get_all_frames_for_model(self) -> Dict[str, Tuple[List[np.ndarray], List[Dict[str, Any]]]]:
        """
        Get ALL frames grouped by source, as numpy arrays
        Perfect for feeding into your feature extraction model
        
        Returns:
            Dictionary mapping source_path -> (frames, metadata)
            {
                "data/videos/video1.mp4": ([array1, array2, ...], [meta1, meta2, ...]),
                "data/videos/video2.mp4": ([array1, array2, ...], [meta1, meta2, ...]),
            }
        """
        result = {}
        sources = self.get_all_source_files()
        
        for source_info in sources:
            source_path = source_info['source_path']
            frames, metadata = self.get_frame_arrays_by_source(source_path)
            result[source_path] = (frames, metadata)
        
        return result
    
    def get_frame_by_id(self, frame_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a specific frame by its ID
        
        Returns:
            Frame as numpy array or None
        """
        return self.similarity_agent.get_frame_by_id(frame_id)
    
    def search_similar_frames(self, query_frame: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find frames similar to a query frame
        
        Args:
            query_frame: Frame to search for (numpy array)
            top_k: Number of similar frames to return
            
        Returns:
            List of similar frames with similarity scores
        """
        return self.similarity_agent.search_similar_frames(query_frame, top_k)
    
    def export_frames_info(self, output_path: str = "frames_info.json"):
        """Export all frames information to JSON"""
        frames_data = self.get_all_stored_frames()
        with open(output_path, 'w') as f:
            json.dump(frames_data, f, indent=2)
        print(f"\nFrames info exported to: {output_path}")
    
    def export_log(self, output_path: str = "processing_log.json"):
        """Export processing log to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.processing_log, f, indent=2)
        print(f"\nProcessing log exported to: {output_path}")


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Initialize pipeline
    pipeline = VisualDeduplicationPipeline(
        db_path="./chroma_visual_db",
        frame_interval=0.3,  # 1 frame per 0.3 second for videos
        similarity_threshold=0.9,  # 90% similarity threshold
        frames_storage_path="./frames_storage"  # Where to save frames
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
        print(f"Saved frames on disk: {stats['saved_frames_on_disk']}")
        print(f"Frames storage path: {stats['frames_storage_path']}")
        
        # Export logs
        pipeline.export_log("logs/processing_log.json")
        pipeline.export_frames_info("logs/frames_info.json")
        
        print("\n✅ Done! Check logs/ for details")
        
        # Example: Get all stored frames
        print("\n" + "="*60)
        print("RETRIEVING FRAMES GROUPED BY SOURCE")
        print("="*60)
        
        # Method 1: Get frames grouped by source (metadata only)
        grouped_frames = pipeline.get_frames_grouped_by_source()
        print(f"\nFound {len(grouped_frames)} source files:")
        for source_path, frames in grouped_frames.items():
            print(f"  {source_path}: {len(frames)} frames")
        
        # Method 2: Get actual frame arrays ready for your model
        print("\n--- Loading frame arrays for model input ---")
        frames_for_model = pipeline.get_all_frames_for_model()
        
        for source_path, (frame_arrays, metadata) in frames_for_model.items():
            print(f"\n{source_path}:")
            print(f"  Loaded {len(frame_arrays)} frames")
            if frame_arrays:
                print(f"  Frame shape: {frame_arrays[0].shape}")
                print(f"  Frame indices: {[m['frame_index'] for m in metadata]}")
        
        # Example: Process frames from a specific video in your model
        if frames_for_model:
            first_video = list(frames_for_model.keys())[0]
            video_frames, video_metadata = frames_for_model[first_video]
            
            print(f"\n--- Example: Processing {first_video} ---")
            print(f"Ready to feed {len(video_frames)} frames into your feature extraction model")
            
            # Your model processing would look like:
            # for frame_array in video_frames:
            #     features = your_model.extract_features(frame_array)
            #     # Process features...
            
            # Save first frame as example
            if video_frames:
                retrieved_img = Image.fromarray(video_frames[0])
                retrieved_img.save("logs/retrieved_example.jpg")
                print("✓ Saved first frame to logs/retrieved_example.jpg")