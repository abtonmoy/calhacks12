"""
Module 3: Human Detection
Simplified version focusing only on human/face detection
"""

import cv2
import numpy as np
from typing import Dict, Optional
import os
from advanced_detector import AdvancedFaceDetector


class HumanDetector:
    """
    Simplified Module 3: Human Detection Only
    
    This class focuses on:
    - Human/face detection
    - Face counting and bounding boxes
    """
    
    def __init__(self):
        """
        Initialize the Human Detector
        """
        # Initialize human detector
        self.human_detector = AdvancedFaceDetector()
        
        print("Initialized HumanDetector:")
        print("  - Human Detection: Advanced Multi-Method")
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze an image for human presence and faces
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing:
            - human_present: 0 or 1
            - num_people: Number of people detected
            - face_boxes: List of face bounding boxes
            - body_boxes: List of body bounding boxes
            - total_faces: Total number of faces
            - total_bodies: Total number of bodies
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"Analyzing image: {image_path}")
        
        # Step A: Detect humans and faces
        print("Step A: Detecting humans and faces...")
        human_results = self.human_detector.analyze_image_comprehensive(image_path)
        
        # Extract results
        face_boxes = human_results.get("face_boxes", [])
        body_boxes = human_results.get("body_boxes", [])
        total_faces = human_results.get("total_faces", 0)
        total_bodies = human_results.get("total_bodies", 0)
        
        # Determine if humans are present
        human_present = 1 if (total_faces > 0 or total_bodies > 0) else 0
        num_people = max(total_faces, total_bodies)  # Use the higher count
        
        print("Analysis complete!")
        
        return {
            "human_present": human_present,
            "num_people": num_people,
            "face_boxes": face_boxes,
            "body_boxes": body_boxes,
            "total_faces": total_faces,
            "total_bodies": total_bodies
        }


# For backward compatibility, keep the old class name
HumanSafetyEmotionDetector = HumanDetector


if __name__ == "__main__":
    # Test the detector
    detector = HumanDetector()
    
    # Test with available images
    test_images = ["i0001.png", "i0004.png", "i0018.png", "weed.jpg"]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*50}")
            print(f"Testing: {image_path}")
            print('='*50)
            
            try:
                result = detector.analyze_image(image_path)
                print(f"✅ Human Present: {result['human_present']}")
                print(f"✅ Number of People: {result['num_people']}")
                print(f"✅ Face Boxes: {len(result['face_boxes'])}")
                print(f"✅ Body Boxes: {len(result['body_boxes'])}")
                
            except Exception as e:
                print(f"❌ Error analyzing {image_path}: {e}")
        else:
            print(f"⚠️  Image not found: {image_path}")