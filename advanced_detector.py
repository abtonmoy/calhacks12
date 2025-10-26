"""
Advanced Multi-Method Face Detection
Tries multiple detection methods and parameters to find all faces
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import time


class AdvancedFaceDetector:
    """Advanced face detection using multiple methods and parameters"""
    
    def __init__(self):
        """Initialize advanced detector"""
        self.opencv_detector = None
        self.mediapipe_detector = None
        self._load_detectors()
    
    def _load_detectors(self):
        """Load all available detectors"""
        # Load OpenCV detector directly
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not self.face_cascade.empty():
                self.opencv_detector = "loaded"
                print("✅ OpenCV detector loaded")
            else:
                print("❌ OpenCV detector failed to load cascade")
        except Exception as e:
            print(f"❌ OpenCV detector failed: {e}")
        
        # Load MediaPipe detector
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mediapipe_detector = self.mp_face_detection.FaceDetection(
                model_selection=0,  # Try close-range model
                min_detection_confidence=0.1  # Lower threshold
            )
            print("✅ MediaPipe detector loaded")
        except Exception as e:
            print(f"❌ MediaPipe detector failed: {e}")
    
    def detect_faces_opencv_multiple_params(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV with multiple parameter sets"""
        if self.opencv_detector is None:
            return []
        
        all_faces = []
        
        # Try different parameter combinations
        param_sets = [
            {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": (20, 20)},
            {"scaleFactor": 1.05, "minNeighbors": 5, "minSize": (30, 30)},
            {"scaleFactor": 1.2, "minNeighbors": 2, "minSize": (15, 15)},
            {"scaleFactor": 1.1, "minNeighbors": 4, "minSize": (25, 25)},
        ]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for params in param_sets:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=params["scaleFactor"],
                minNeighbors=params["minNeighbors"],
                minSize=params["minSize"],
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                all_faces.append((int(x), int(y), int(w), int(h)))
        
        # Remove duplicates (faces that overlap significantly)
        return self._remove_duplicate_faces(all_faces)
    
    def detect_faces_mediapipe_multiple_params(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe with multiple parameter sets"""
        if self.mediapipe_detector is None:
            return []
        
        all_faces = []
        
        # Try different confidence thresholds
        confidence_thresholds = [0.1, 0.2, 0.3, 0.4]
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        for threshold in confidence_thresholds:
            # Create new detector with different threshold
            try:
                import mediapipe as mp
                temp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=threshold
                )
                
                results = temp_detector.process(rgb_image)
                
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        
                        x = int(bbox.xmin * width)
                        y = int(bbox.ymin * height)
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)
                        
                        # Ensure coordinates are within image bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, width - x)
                        h = min(h, height - y)
                        
                        if w > 0 and h > 0:
                            all_faces.append((x, y, w, h))
                
                temp_detector.close()
                
            except Exception as e:
                print(f"MediaPipe detection with threshold {threshold} failed: {e}")
        
        # Remove duplicates
        return self._remove_duplicate_faces(all_faces)
    
    def _remove_duplicate_faces(self, faces: List[Tuple[int, int, int, int]], overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Remove duplicate face detections"""
        if len(faces) <= 1:
            return faces
        
        # Calculate overlap between faces
        unique_faces = []
        
        for face in faces:
            x1, y1, w1, h1 = face
            is_duplicate = False
            
            for unique_face in unique_faces:
                x2, y2, w2, h2 = unique_face
                
                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    union_area = area1 + area2 - intersection_area
                    
                    overlap = intersection_area / union_area if union_area > 0 else 0
                    
                    if overlap > overlap_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def detect_faces_combined(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Combine results from multiple detection methods"""
        all_faces = []
        
        # Get faces from OpenCV with multiple parameters
        opencv_faces = self.detect_faces_opencv_multiple_params(image)
        all_faces.extend(opencv_faces)
        
        # Get faces from MediaPipe with multiple parameters
        mediapipe_faces = self.detect_faces_mediapipe_multiple_params(image)
        all_faces.extend(mediapipe_faces)
        
        # Remove duplicates
        unique_faces = self._remove_duplicate_faces(all_faces)
        
        return unique_faces
    
    def analyze_image_comprehensive(self, image_path: str) -> Dict:
        """Comprehensive analysis using all available methods"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Comprehensive analysis of {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Test each method individually
        results = {}
        
        # OpenCV with default parameters
        if self.opencv_detector:
            start_time = time.time()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            opencv_default_faces = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
            opencv_time = time.time() - start_time
            results['opencv_default'] = {
                'faces': opencv_default_faces,
                'count': len(opencv_default_faces),
                'time': opencv_time
            }
        
        # OpenCV with multiple parameters
        start_time = time.time()
        opencv_multi = self.detect_faces_opencv_multiple_params(image)
        opencv_multi_time = time.time() - start_time
        results['opencv_multi'] = {
            'faces': opencv_multi,
            'count': len(opencv_multi),
            'time': opencv_multi_time
        }
        
        # MediaPipe with multiple parameters
        start_time = time.time()
        mediapipe_multi = self.detect_faces_mediapipe_multiple_params(image)
        mediapipe_multi_time = time.time() - start_time
        results['mediapipe_multi'] = {
            'faces': mediapipe_multi,
            'count': len(mediapipe_multi),
            'time': mediapipe_multi_time
        }
        
        # Combined results
        start_time = time.time()
        combined_faces = self.detect_faces_combined(image)
        combined_time = time.time() - start_time
        results['combined'] = {
            'faces': combined_faces,
            'count': len(combined_faces),
            'time': combined_time
        }
        
        # Print results
        print(f"\nDetection Results:")
        for method, result in results.items():
            print(f"  {method}: {result['count']} faces in {result['time']:.3f}s")
        
        # Use combined results as final result
        human_present = 1 if len(combined_faces) > 0 else 0
        num_people = len(combined_faces)
        
        final_result = {
            "human_present": human_present,
            "num_people": num_people,
            "face_boxes": combined_faces,
            "total_faces": len(combined_faces),
            "method": "combined",
            "detailed_results": results
        }
        
        return final_result


def test_advanced_detection():
    """Test advanced face detection"""
    print("Advanced Multi-Method Face Detection Test")
    print("=" * 60)
    
    detector = AdvancedFaceDetector()
    
    image_path = "i0001.png"
    
    try:
        result = detector.analyze_image_comprehensive(image_path)
        
        print(f"\nFinal Results:")
        print(f"  Human Present: {result['human_present']}")
        print(f"  Number of People: {result['num_people']}")
        print(f"  Total Faces: {result['total_faces']}")
        
        print(f"\nFace Locations:")
        for i, (x, y, w, h) in enumerate(result['face_boxes']):
            print(f"  Face {i+1}: ({x}, {y}, {w}, {h})")
        
        # Expected vs Actual
        expected_people = 4
        detected_people = result['num_people']
        accuracy = detected_people / expected_people * 100
        
        print(f"\nAccuracy Analysis:")
        print(f"  Expected people: {expected_people}")
        print(f"  Detected people: {detected_people}")
        print(f"  Detection accuracy: {accuracy:.1f}%")
        
        if detected_people >= expected_people:
            print(f"  ✅ Successfully detected all expected people!")
        elif detected_people > 2:
            print(f"  ⚠️ Detected more than basic OpenCV but still missing some")
        else:
            print(f"  ❌ Still missing faces - may need manual inspection")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    test_advanced_detection()
