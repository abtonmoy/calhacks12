#!/usr/bin/env python3
"""
Simple Human Detection Script
Single file that takes an image and outputs JSON results
"""

import cv2
import numpy as np
import json
import sys
import os
from typing import List, Tuple, Dict


class SimpleHumanDetector:
    """Simple human detection using OpenCV with multiple parameter sets"""
    
    def __init__(self):
        """Initialize the detector"""
        # Load OpenCV face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier")
    
    def detect_faces(self, image_path: str) -> Dict:
        """
        Detect faces in an image using multiple parameter sets
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with detection results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_faces = []
        
        # Try different parameter combinations for better detection
        param_sets = [
            {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": (20, 20)},
            {"scaleFactor": 1.05, "minNeighbors": 5, "minSize": (30, 30)},
            {"scaleFactor": 1.2, "minNeighbors": 2, "minSize": (15, 15)},
            {"scaleFactor": 1.1, "minNeighbors": 4, "minSize": (25, 25)},
        ]
        
        for params in param_sets:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=params["scaleFactor"],
                minNeighbors=params["minNeighbors"],
                minSize=params["minSize"],
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                all_faces.append([int(x), int(y), int(w), int(h)])
        
        # Remove duplicates
        unique_faces = self._remove_duplicates(all_faces)
        
        # Prepare results
        result = {
            "human_present": 1 if len(unique_faces) > 0 else 0,
            "num_people": len(unique_faces)
        }
        
        return result
    
    def _remove_duplicates(self, faces: List[List[int]], overlap_threshold: float = 0.3) -> List[List[int]]:
        """Remove duplicate face detections - improved algorithm"""
        if len(faces) <= 1:
            return faces
        
        # Sort faces by area (largest first) to keep the best detection
        faces_with_area = [(face, face[2] * face[3]) for face in faces]
        faces_with_area.sort(key=lambda x: x[1], reverse=True)
        
        unique_faces = []
        
        for face, area in faces_with_area:
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
                    
                    # Calculate overlap ratio using union area for better accuracy
                    union_area = area1 + area2 - intersection_area
                    overlap_ratio = intersection_area / union_area if union_area > 0 else 0
                    
                    if overlap_ratio > overlap_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces


def main():
    """Main function for command line usage"""
    if len(sys.argv) != 2:
        print("Usage: python3 human_detector.py <image_path>")
        print("Example: python3 human_detector.py swim.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        detector = SimpleHumanDetector()
        result = detector.detect_faces(image_path)
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "human_present": 0,
            "num_people": 0
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
