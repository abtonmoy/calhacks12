"""
Test Suite for Module 3: Human/Safety/Emotion Detection
"""

import unittest
import cv2
import numpy as np
import os
import tempfile
from module3_human_safety import HumanSafetyEmotionDetector
from human_detector import HumanDetector
from emotion_classifier import EmotionClassifier
from nsfw_detector import NSFWDetector


class TestHumanDetector(unittest.TestCase):
    """Test cases for HumanDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = HumanDetector(method="opencv")
        
    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.method, "opencv")
    
    def test_detect_faces_empty_image(self):
        """Test face detection on empty image"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(empty_image)
        self.assertEqual(len(faces), 0)
    
    def test_detect_faces_text_image(self):
        """Test face detection on image with text only"""
        text_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(text_image, "Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        faces = self.detector.detect_faces(text_image)
        # Should not detect faces in text-only image
        self.assertIsInstance(faces, list)
    
    def test_analyze_image_nonexistent(self):
        """Test analysis with non-existent image"""
        with self.assertRaises(ValueError):
            self.detector.analyze_image("nonexistent.jpg")


class TestEmotionClassifier(unittest.TestCase):
    """Test cases for EmotionClassifier class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = EmotionClassifier(method="opencv")
    
    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(self.classifier.method, "opencv")
        self.assertIn("happy", self.classifier.emotion_labels)
    
    def test_classify_emotion_empty_image(self):
        """Test emotion classification on empty image"""
        empty_image = np.zeros((50, 50, 3), dtype=np.uint8)
        emotion = self.classifier.classify_emotion(empty_image)
        self.assertIn(emotion, self.classifier.emotion_labels)
    
    def test_analyze_faces_no_faces(self):
        """Test emotion analysis with no faces"""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.classifier.analyze_faces(image, [])
        
        self.assertEqual(result["dominant_expression"], "neutral")
        self.assertEqual(result["smile_present"], 0)
        self.assertEqual(len(result["emotions"]), 0)


class TestNSFWDetector(unittest.TestCase):
    """Test cases for NSFWDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = NSFWDetector(method="rule_based", threshold=0.5)
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.method, "rule_based")
        self.assertEqual(self.detector.threshold, 0.5)
    
    def test_detect_nsfw_empty_image(self):
        """Test NSFW detection on empty image"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        score = self.detector.detect_nsfw(empty_image)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_detect_nsfw_bright_image(self):
        """Test NSFW detection on bright image"""
        bright_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        score = self.detector.detect_nsfw(bright_image)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestHumanSafetyEmotionDetector(unittest.TestCase):
    """Test cases for main HumanSafetyEmotionDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = HumanSafetyEmotionDetector()
    
    def test_initialization(self):
        """Test main detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.human_detector)
        self.assertIsNotNone(self.detector.emotion_classifier)
        self.assertIsNotNone(self.detector.nsfw_detector)
    
    def test_analyze_image_nonexistent(self):
        """Test analysis with non-existent image"""
        with self.assertRaises(FileNotFoundError):
            self.detector.analyze_image("nonexistent.jpg")
    
    def test_analyze_image_from_array(self):
        """Test analysis from numpy array"""
        # Create test image array
        image_array = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(image_array, "Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        result = self.detector.analyze_image_from_array(image_array)
        
        # Check required fields
        self.assertIn("human_present", result)
        self.assertIn("num_people", result)
        self.assertIn("model_expression", result)
        self.assertIn("smile_present", result)
        self.assertIn("nsfw_flag", result)
        self.assertIn("nsfw_score", result)
        
        # Check data types
        self.assertIsInstance(result["human_present"], int)
        self.assertIsInstance(result["num_people"], int)
        self.assertIsInstance(result["smile_present"], int)
        self.assertIsInstance(result["nsfw_flag"], int)
        self.assertIsInstance(result["nsfw_score"], (int, float))
    
    def test_batch_analyze(self):
        """Test batch analysis functionality"""
        # Create temporary test images
        test_images = []
        temp_files = []
        
        try:
            for i in range(3):
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.putText(image, f"Test{i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                temp_file = f"temp_test_{i}.jpg"
                cv2.imwrite(temp_file, image)
                test_images.append(temp_file)
                temp_files.append(temp_file)
            
            results = self.detector.batch_analyze(test_images)
            
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIn("human_present", result)
                self.assertIn("nsfw_score", result)
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def test_get_summary_stats(self):
        """Test summary statistics generation"""
        # Create mock results
        mock_results = [
            {"human_present": 1, "smile_present": 1, "nsfw_flag": 0, "nsfw_score": 0.1, "model_expression": "happy"},
            {"human_present": 0, "smile_present": 0, "nsfw_flag": 1, "nsfw_score": 0.8, "model_expression": "neutral"},
            {"human_present": 1, "smile_present": 0, "nsfw_flag": 0, "nsfw_score": 0.2, "model_expression": "sad"}
        ]
        
        summary = self.detector.get_summary_stats(mock_results)
        
        self.assertEqual(summary["total_images"], 3)
        self.assertEqual(summary["images_with_humans"], 2)
        self.assertEqual(summary["images_with_smiles"], 1)
        self.assertEqual(summary["images_with_nsfw"], 1)
        self.assertAlmostEqual(summary["human_detection_rate"], 2/3, places=2)
        self.assertAlmostEqual(summary["smile_rate"], 1/3, places=2)
        self.assertAlmostEqual(summary["nsfw_rate"], 1/3, places=2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = HumanSafetyEmotionDetector()
    
    def test_complete_workflow(self):
        """Test complete workflow with real image file"""
        # Create a test image and save it
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(test_image, "Integration Test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        test_file = "integration_test.jpg"
        cv2.imwrite(test_file, test_image)
        
        try:
            # Analyze the image
            result = self.detector.analyze_image(test_file)
            
            # Verify all required fields are present
            required_fields = [
                "human_present", "num_people", "model_expression", 
                "smile_present", "nsfw_flag", "nsfw_score"
            ]
            
            for field in required_fields:
                self.assertIn(field, result, f"Missing required field: {field}")
            
            # Verify data types and ranges
            self.assertIn(result["human_present"], [0, 1])
            self.assertGreaterEqual(result["num_people"], 0)
            self.assertIn(result["model_expression"], ["happy", "neutral", "sad", "angry", "surprised"])
            self.assertIn(result["smile_present"], [0, 1])
            self.assertIn(result["nsfw_flag"], [0, 1])
            self.assertGreaterEqual(result["nsfw_score"], 0.0)
            self.assertLessEqual(result["nsfw_score"], 1.0)
        
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)


def run_tests():
    """Run all test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestHumanDetector))
    test_suite.addTest(unittest.makeSuite(TestEmotionClassifier))
    test_suite.addTest(unittest.makeSuite(TestNSFWDetector))
    test_suite.addTest(unittest.makeSuite(TestHumanSafetyEmotionDetector))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Module 3 Test Suite")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n" + "=" * 50)
        print("All tests passed! ✅")
    else:
        print("\n" + "=" * 50)
        print("Some tests failed! ❌")
