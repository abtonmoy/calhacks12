#!/usr/bin/env python3
"""
Image Orchestration Module
Processes images through: Human Detection → OCR → Claude API
No deduplication needed for direct image processing
"""

import os
import json
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import easyocr
import numpy as np
from dotenv import load_dotenv

# Import existing modules
from human_detection_agent.human_detector import SimpleHumanDetector
from extract_signals.main import get_signals


class ImageFeatureExtractor:
    """Handles Human Detection, OCR, and Claude API signal extraction for images"""
    
    def __init__(self):
        """Initialize OCR reader and human detector"""
        load_dotenv()
        
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=True)
        print("[+] OCR ready")
        
        print("Initializing human detector...")
        self.human_detector = SimpleHumanDetector()
        print("[+] Human detector ready")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image file as numpy array
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array (H, W, 3) RGB image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def extract_text(self, image_array: np.ndarray) -> str:
        """
        Extract all text from an image array
        
        Args:
            image_array: numpy array (H, W, 3) RGB image
            
        Returns:
            Extracted text string
        """
        results = self.reader.readtext(image_array)
        text = ' '.join([text for (bbox, text, conf) in results])
        return text
    
    def detect_humans(self, image_path: str) -> Dict[str, Any]:
        """
        Detect humans in the image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with human detection results
        """
        try:
            result = self.human_detector.detect_faces(image_path)
            return result
        except Exception as e:
            return {
                "human_present": 0,
                "num_people": 0,
                "error": str(e)
            }
    
    def get_signals(self, text: str) -> Dict[str, Any]:
        """
        Send text to Claude API and get marketing signals
        
        Args:
            text: Extracted text from image
            
        Returns:
            Dictionary with marketing signals
        """
        if not text.strip():
            return {
                "brand_name_text": "",
                "product_name": "",
                "industry": "",
                "promo_present": False,
                "promo_text": "",
                "promo_deadline": "",
                "price_value": "",
                "cta_present": False,
                "cta_type": "",
                "text_density": "low",
                "brand_text_contrast": "unknown",
                "error": "No text extracted"
            }
        
        try:
            return get_signals(text)
        except Exception as e:
            return {
                "error": str(e),
                "text_extracted": text
            }
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Complete feature extraction for one image: Human Detection → OCR → Claude
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Complete feature dictionary
        """
        # Load image
        image_array = self.load_image(image_path)
        
        # Human detection
        human_detection = self.detect_humans(image_path)
        
        # Extract text via OCR
        extracted_text = self.extract_text(image_array)
        
        # Get marketing signals from Claude API
        signals = self.get_signals(extracted_text)
        
        # Combine all features - flat structure with specific fields
        result = {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "timestamp": datetime.now().isoformat(),
            
            # Human detection results
            "human_present": human_detection.get("human_present", 0),
            "num_people": human_detection.get("num_people", 0),
            
            # Text extraction results
            "extracted_text": extracted_text,
            
            # Marketing signals from Claude (flat structure)
            "brand_name_text": signals.get("brand_name_text", ""),
            "product_name": signals.get("product_name", ""),
            "industry": signals.get("industry", ""),
            "promo_present": signals.get("promo_present", False),
            "promo_text": signals.get("promo_text", ""),
            "promo_deadline": signals.get("promo_deadline", ""),
            "price_value": signals.get("price_value", ""),
            "cta_present": signals.get("cta_present", False),
            "cta_type": signals.get("cta_type", ""),
            "text_density": signals.get("text_density", "low"),
            "brand_text_contrast": signals.get("brand_text_contrast", "unknown")
        }
        
        return result


class ImagePipelineOrchestrator:
    """
    Image Pipeline Orchestrator
    Processes images through: Human Detection → OCR → Claude API
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the image orchestrator
        
        Args:
            output_dir: Directory to save final outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("INITIALIZING IMAGE PIPELINE ORCHESTRATOR")
        print("="*70)
        
        # Initialize Feature Extraction Module
        print("\nInitializing Feature Extraction Module...")
        self.feature_extractor = ImageFeatureExtractor()
        
        print("\n" + "="*70)
        print("IMAGE ORCHESTRATOR READY")
        print("="*70 + "\n")
    
    def process(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Main orchestration method - runs complete image pipeline
        
        Args:
            image_paths: List of image file paths to process
            
        Returns:
            Complete results dictionary with all data
        """
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print("STARTING IMAGE PIPELINE EXECUTION")
        print("="*70)
        print(f"Images to process: {len(image_paths)}")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # ============================================================
        # FEATURE EXTRACTION
        # ============================================================
        print("\n" + "="*70)
        print("FEATURE EXTRACTION: Human Detection → OCR → Claude API")
        print("="*70 + "\n")
        
        all_features = []
        images_processed = 0
        images_with_humans = 0
        images_with_text = 0
        images_with_promos = 0
        failed_images = 0
        
        for image_path in image_paths:
            image_name = Path(image_path).name
            print(f"[*] {image_name}: ", end="", flush=True)
            
            try:
                # Extract all features (Human Detection → OCR → Claude)
                features = self.feature_extractor.process_image(image_path)
                all_features.append(features)
                
                # Update stats
                images_processed += 1
                if features.get('human_present'):
                    images_with_humans += 1
                if features.get('extracted_text'):
                    images_with_text += 1
                if features.get('promo_present'):
                    images_with_promos += 1
                
                # Status
                humans = features.get('num_people', 0)
                text = features.get('extracted_text', '')
                has_promo = features.get('promo_present', False)
                brand = features.get('brand_name_text', '')
                
                status_parts = []
                
                if humans > 0:
                    status_parts.append(f"{humans} person{'s' if humans > 1 else ''}")
                
                if text:
                    status_parts.append(f"{len(text)} chars")
                    if brand:
                        status_parts.append(f"brand: {brand[:15]}")
                    if has_promo:
                        status_parts.append("PROMO")
                else:
                    status_parts.append("no text")
                
                print(f"✓ ({', '.join(status_parts)})")
                
            except Exception as e:
                failed_images += 1
                print(f"✗ ERROR: {str(e)}")
                # Still add placeholder
                all_features.append({
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                })
        
        print(f"\n[+] Feature extraction complete:")
        print(f"  Images processed: {images_processed}/{len(image_paths)}")
        print(f"  Images with humans: {images_with_humans}")
        print(f"  Images with text: {images_with_text}")
        print(f"  Images with promos: {images_with_promos}")
        if failed_images > 0:
            print(f"  Failed images: {failed_images}")
        
        # ============================================================
        # COMPILE FINAL RESULTS
        # ============================================================
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        final_results = {
            "pipeline_metadata": {
                "execution_timestamp": start_time.isoformat(),
                "processing_time_seconds": processing_time,
                "total_images_processed": len(image_paths),
                "successful_images": images_processed,
                "failed_images": failed_images,
                "pipeline_version": "1.0-images",
                "pipeline_type": "image_only"
            },
            "feature_extraction_summary": {
                "total_images_analyzed": images_processed,
                "images_with_humans": images_with_humans,
                "images_with_text": images_with_text,
                "images_with_promos": images_with_promos,
                "human_detection_rate": f"{(images_with_humans/images_processed*100):.1f}%" if images_processed > 0 else "0%",
                "text_extraction_rate": f"{(images_with_text/images_processed*100):.1f}%" if images_processed > 0 else "0%",
                "promo_detection_rate": f"{(images_with_promos/images_processed*100):.1f}%" if images_processed > 0 else "0%"
            },
            "extracted_features": all_features
        }
        
        # ============================================================
        # SAVE OUTPUTS
        # ============================================================
        print("\n" + "="*70)
        print("SAVING OUTPUTS")
        print("="*70)
        
        self._save_outputs(final_results)
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print("\n" + "="*70)
        print("IMAGE PIPELINE EXECUTION COMPLETE")
        print("="*70)
        print(f"Total time: {processing_time:.2f} seconds")
        print(f"Images processed: {images_processed}/{len(image_paths)}")
        print(f"Output directory: {self.output_dir}")
        print("="*70 + "\n")
        
        return final_results
    
    def _save_outputs(self, results: Dict[str, Any]):
        """Save results to single JSON file with specific fields"""
        
        # Single output file with all data
        output_path = self.output_dir / "results.json"
        with open(output_path, 'w') as f:
            json.dump(results['extracted_features'], f, indent=2)
        print(f"[+] Saved: {output_path}")
    
    def _generate_insights(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate aggregated marketing insights"""
        
        brands = set()
        products = set()
        industries = set()
        promo_types = []
        cta_types = []
        total_people = 0
        
        for feature in features:
            if 'error' in feature:
                continue
            
            # Marketing signals
            if feature.get('brand_name_text'):
                brands.add(feature['brand_name_text'])
            if feature.get('product_name'):
                products.add(feature['product_name'])
            if feature.get('industry'):
                industries.add(feature['industry'])
            if feature.get('promo_present') and feature.get('promo_text'):
                promo_types.append(feature['promo_text'])
            if feature.get('cta_present') and feature.get('cta_type'):
                cta_types.append(feature['cta_type'])
            
            # Human detection
            total_people += feature.get('num_people', 0)
        
        return {
            "unique_brands": list(brands),
            "unique_products": list(products),
            "industries_detected": list(industries),
            "total_promos": len(promo_types),
            "promo_examples": promo_types[:10],
            "total_ctas": len(cta_types),
            "cta_examples": cta_types[:10],
            "total_people_detected": total_people
        }
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """Load the most recent results from disk"""
        output_path = self.output_dir / "results.json"
        if output_path.exists():
            with open(output_path, 'r') as f:
                return json.load(f)
        return None


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    """Main entry point for image processing"""
    import sys
    
    # Initialize orchestrator
    orchestrator = ImagePipelineOrchestrator(output_dir="./output")
    
    # Check if single file or directory processing
    if len(sys.argv) > 1:
        # Single file mode
        image_path = sys.argv[1]
        
        if not os.path.exists(image_path):
            print(f"\n[!] File not found: {image_path}")
            print("Usage: python image_orchestrator.py <image_path>")
            print("   or: python image_orchestrator.py  (processes data/images/)")
            return
        
        # Check if it's a valid image
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        if Path(image_path).suffix.lower() not in valid_extensions:
            print(f"\n[!] Not a valid image file: {image_path}")
            print(f"Supported formats: {', '.join(valid_extensions)}")
            return
        
        print(f"Processing single image: {image_path}")
        image_files = [image_path]
    else:
        # Directory mode (default)
        image_dir = Path("data/images")
        
        if not image_dir.exists():
            print(f"\n[!] Directory not found: {image_dir}")
            print("Please create 'data/images' directory and add images")
            print("\nOr specify a single file: python image_orchestrator.py <image_path>")
            return
        
        image_files = [
            str(f) for f in image_dir.glob("*.*")
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        ]
        
        if not image_files:
            print(f"\n[!] No images found in {image_dir}")
            print("Please add some images to process")
            print("\nOr specify a single file: python image_orchestrator.py <image_path>")
            return
        
        print(f"Found {len(image_files)} images in {image_dir}")
    
    # Run the complete pipeline
    results = orchestrator.process(image_files)
    
    # Results are automatically saved, display quick stats
    print("\n[*] Quick Stats:")
    print(f"   Human detection: {results['feature_extraction_summary']['human_detection_rate']} success")
    print(f"   Text extraction: {results['feature_extraction_summary']['text_extraction_rate']} success")
    print(f"   Promos detected: {results['feature_extraction_summary']['images_with_promos']}")
    
    return results


if __name__ == "__main__":
    results = main()