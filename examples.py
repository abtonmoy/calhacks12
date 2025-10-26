"""
Example Usage and Test Cases for Module 3: Human/Safety/Emotion Detection
"""

import cv2
import numpy as np
import os
from module3_human_safety import HumanSafetyEmotionDetector


def create_test_images():
    """Create various test images for demonstration"""
    test_images = {}
    
    # Test Image 1: Simple text image (no humans)
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(img1, "No Humans", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    test_images["no_humans.jpg"] = img1
    
    # Test Image 2: Bright image (potential NSFW)
    img2 = np.ones((300, 300, 3), dtype=np.uint8) * 200
    cv2.putText(img2, "Bright Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    test_images["bright_image.jpg"] = img2
    
    # Test Image 3: Dark image
    img3 = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(img3, "Dark Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    test_images["dark_image.jpg"] = img3
    
    # Test Image 4: High contrast image
    img4 = np.zeros((300, 300, 3), dtype=np.uint8)
    img4[100:200, 100:200] = [255, 255, 255]  # White square
    cv2.putText(img4, "High Contrast", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    test_images["high_contrast.jpg"] = img4
    
    return test_images


def example_basic_usage():
    """Demonstrate basic usage of the HumanSafetyEmotionDetector"""
    print("=== Basic Usage Example ===")
    
    # Initialize detector
    detector = HumanSafetyEmotionDetector()
    
    # Create test images
    test_images = create_test_images()
    
    # Save test images
    for filename, image in test_images.items():
        cv2.imwrite(filename, image)
    
    try:
        # Analyze each test image
        for filename in test_images.keys():
            print(f"\nAnalyzing: {filename}")
            result = detector.analyze_image(filename)
            
            # Print key results
            print(f"  Human Present: {result['human_present']}")
            print(f"  Number of People: {result['num_people']}")
            print(f"  Model Expression: {result['model_expression']}")
            print(f"  Smile Present: {result['smile_present']}")
            print(f"  NSFW Flag: {result['nsfw_flag']}")
            print(f"  NSFW Score: {result['nsfw_score']}")
    
    finally:
        # Clean up test images
        for filename in test_images.keys():
            if os.path.exists(filename):
                os.remove(filename)


def example_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n=== Batch Processing Example ===")
    
    detector = HumanSafetyEmotionDetector()
    test_images = create_test_images()
    
    # Save test images
    for filename, image in test_images.items():
        cv2.imwrite(filename, image)
    
    try:
        # Batch analyze all images
        image_paths = list(test_images.keys())
        batch_results = detector.batch_analyze(image_paths)
        
        print(f"Processed {len(batch_results)} images")
        
        # Get summary statistics
        summary = detector.get_summary_stats(batch_results)
        
        print("\nSummary Statistics:")
        print(f"  Total Images: {summary['total_images']}")
        print(f"  Images with Humans: {summary['images_with_humans']}")
        print(f"  Images with Smiles: {summary['images_with_smiles']}")
        print(f"  Images with NSFW: {summary['images_with_nsfw']}")
        print(f"  Human Detection Rate: {summary['human_detection_rate']:.2%}")
        print(f"  Smile Rate: {summary['smile_rate']:.2%}")
        print(f"  NSFW Rate: {summary['nsfw_rate']:.2%}")
        print(f"  Average NSFW Score: {summary['average_nsfw_score']}")
        print(f"  Emotion Distribution: {summary['emotion_distribution']}")
    
    finally:
        # Clean up test images
        for filename in test_images.keys():
            if os.path.exists(filename):
                os.remove(filename)


def example_custom_configuration():
    """Demonstrate custom configuration options"""
    print("\n=== Custom Configuration Example ===")
    
    # Initialize with custom settings
    detector = HumanSafetyEmotionDetector(
        human_detection_method="opencv",
        emotion_method="opencv",
        nsfw_method="rule_based",
        nsfw_threshold=0.3  # Lower threshold for more sensitive detection
    )
    
    # Create a test image
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(test_image, "Custom Config Test", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite("custom_test.jpg", test_image)
    
    try:
        result = detector.analyze_image("custom_test.jpg")
        print("Custom Configuration Results:")
        print(f"  NSFW Threshold: {result.get('nsfw_score', 0)} (threshold: 0.3)")
        print(f"  NSFW Flag: {result['nsfw_flag']}")
    
    finally:
        if os.path.exists("custom_test.jpg"):
            os.remove("custom_test.jpg")


def example_error_handling():
    """Demonstrate error handling capabilities"""
    print("\n=== Error Handling Example ===")
    
    detector = HumanSafetyEmotionDetector()
    
    # Test with non-existent file
    try:
        result = detector.analyze_image("non_existent_file.jpg")
    except FileNotFoundError as e:
        print(f"Caught expected error: {e}")
    
    # Test with invalid image
    try:
        # Create a file that's not an image
        with open("not_an_image.txt", "w") as f:
            f.write("This is not an image")
        
        result = detector.analyze_image("not_an_image.txt")
    except Exception as e:
        print(f"Caught expected error: {e}")
    finally:
        if os.path.exists("not_an_image.txt"):
            os.remove("not_an_image.txt")


def example_array_input():
    """Demonstrate analyzing images from numpy arrays"""
    print("\n=== Array Input Example ===")
    
    detector = HumanSafetyEmotionDetector()
    
    # Create image as numpy array
    image_array = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.putText(image_array, "Array Input", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Analyze directly from array
    result = detector.analyze_image_from_array(image_array)
    
    print("Array Input Results:")
    print(f"  Human Present: {result['human_present']}")
    print(f"  Model Expression: {result['model_expression']}")
    print(f"  NSFW Score: {result['nsfw_score']}")


def run_all_examples():
    """Run all example functions"""
    print("Module 3: Human/Safety/Emotion Detection - Examples")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_batch_processing()
        example_custom_configuration()
        example_error_handling()
        example_array_input()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    run_all_examples()
