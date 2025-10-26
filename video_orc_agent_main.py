"""
Enhanced Vision Orchestrator
Uses Claude Vision API with embedding-derived context for richer analysis
Single video/image processing with temporal and similarity context
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import argparse
import sys
from io import BytesIO
from collections import Counter

import anthropic
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import cv2

from deduplication_agent.pipeline import VisualDeduplicationPipeline
from human_detection_agent.human_detector import SimpleHumanDetector


class EmbeddingContextAnalyzer:
    """Analyzes embeddings to generate meaningful context for Claude"""
    
    def __init__(self, dedup_pipeline: VisualDeduplicationPipeline):
        self.dedup_pipeline = dedup_pipeline
        self.similarity_agent = dedup_pipeline.similarity_agent
    
    def classify_frame_type(self, embedding: np.ndarray) -> str:
        """
        Classify frame type based on embedding characteristics
        Simple heuristic based on embedding distribution
        """
        # Calculate embedding statistics
        mean_val = np.mean(embedding)
        std_val = np.std(embedding)
        max_val = np.max(embedding)
        
        # Simple heuristic classification
        if std_val > 0.15:
            return "visually_complex"
        elif max_val > 0.5:
            return "high_contrast"
        elif mean_val > 0.1:
            return "bright_scene"
        else:
            return "standard"
    
    def get_brand_similarities(self, embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar frames in database and extract brand patterns
        """
        try:
            # Query for similar frames
            results = self.similarity_agent.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=top_k
            )
            
            similar_frames = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    similarity = 1.0 - distance
                    metadata = results['metadatas'][0][i]
                    
                    similar_frames.append({
                        'similarity': round(similarity, 3),
                        'source': metadata.get('source_path', 'unknown'),
                        'frame_index': metadata.get('frame_index', 0)
                    })
            
            return similar_frames
        except Exception as e:
            print(f"[!] Warning: Could not retrieve similar frames: {e}")
            return []
    
    def calculate_visual_change(self, prev_embedding: Optional[np.ndarray], 
                               curr_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Calculate visual change between frames
        """
        if prev_embedding is None:
            return {
                'change_detected': False,
                'change_magnitude': 0.0,
                'change_type': 'first_frame'
            }
        
        # Calculate cosine similarity
        similarity = np.dot(prev_embedding, curr_embedding) / (
            np.linalg.norm(prev_embedding) * np.linalg.norm(curr_embedding)
        )
        
        change_magnitude = 1.0 - similarity
        
        # Classify change type
        if change_magnitude > 0.3:
            change_type = 'major_scene_change'
        elif change_magnitude > 0.15:
            change_type = 'moderate_transition'
        else:
            change_type = 'minor_change'
        
        return {
            'change_detected': bool(change_magnitude > 0.15),  # Convert to Python bool
            'change_magnitude': round(float(change_magnitude), 3),  # Convert to Python float
            'change_type': change_type
        }
    
    def get_narrative_position(self, frame_index: int, total_frames: int) -> str:
        """Determine position in narrative arc"""
        position_ratio = frame_index / total_frames if total_frames > 0 else 0
        
        if position_ratio < 0.2:
            return "opening"
        elif position_ratio < 0.4:
            return "early_middle"
        elif position_ratio < 0.6:
            return "middle"
        elif position_ratio < 0.8:
            return "late_middle"
        else:
            return "conclusion"


class EnhancedVisionOrchestrator:
    """
    Enhanced orchestrator using Claude Vision with embedding-derived context
    """
    
    def __init__(
        self,
        db_path: str = "./chroma_visual_db",
        frame_interval: float = 0.3,
        similarity_threshold: float = 0.9,
        frames_storage_path: str = "./frames_storage"
    ):
        """Initialize the enhanced orchestrator"""
        load_dotenv()
        
        print("="*70)
        print("INITIALIZING ENHANCED VISION ORCHESTRATOR")
        print("="*70)
        
        # Initialize Claude API
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.claude_client = anthropic.Anthropic(api_key=api_key)
        print("\n[+] Claude Vision API ready")
        
        # Initialize Human Detector
        try:
            self.human_detector = SimpleHumanDetector()
            print("[+] Human detection agent ready")
        except Exception as e:
            print(f"[!] Warning: Could not initialize human detector: {e}")
            self.human_detector = None
        
        # Initialize Deduplication Pipeline
        print("\n[+] Initializing Deduplication Pipeline...")
        self.dedup_pipeline = VisualDeduplicationPipeline(
            db_path=db_path,
            frame_interval=frame_interval,
            similarity_threshold=similarity_threshold,
            frames_storage_path=frames_storage_path
        )
        
        # Initialize context analyzer
        self.context_analyzer = EmbeddingContextAnalyzer(self.dedup_pipeline)
        
        print("\n" + "="*70)
        print("ORCHESTRATOR READY")
        print("="*70 + "\n")
    
    def _encode_image(self, image_array: np.ndarray) -> str:
        """Convert numpy array to base64 string for Claude API"""
        pil_img = Image.fromarray(image_array)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def _save_temp_image(self, image_array: np.ndarray, temp_path: str) -> str:
        """Save numpy array as temporary image file for human detector"""
        pil_img = Image.fromarray(image_array)
        pil_img.save(temp_path, "JPEG", quality=95)
        return temp_path
    
    def _detect_humans_in_frame(self, frame_array: np.ndarray) -> bool:
        """Detect if humans are present in frame using the human detector module"""
        if self.human_detector is None:
            return False
        
        try:
            # Save frame to temporary file
            temp_path = "temp_frame_for_detection.jpg"
            self._save_temp_image(frame_array, temp_path)
            
            # Use the human detector
            result = self.human_detector.detect_faces(temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Return simple boolean
            return result.get("human_present", 0) == 1
        except Exception as e:
            print(f"[!] Warning: Human detection error: {e}")
            return False
    
    def _build_enhanced_prompt(
        self,
        frame_index: int,
        total_frames: int,
        embedding: np.ndarray,
        prev_embedding: Optional[np.ndarray],
        frame_type: str
    ) -> str:
        """Build prompt with embedding-derived context"""
        
        # Get context from embeddings
        similar_frames = self.context_analyzer.get_brand_similarities(embedding)
        visual_change = self.context_analyzer.calculate_visual_change(prev_embedding, embedding)
        narrative_position = self.context_analyzer.get_narrative_position(frame_index, total_frames)
        
        # Build context section
        context_parts = []
        
        # Frame type
        context_parts.append(f"Frame type: {frame_type}")
        
        # Narrative position
        context_parts.append(f"Narrative position: {narrative_position} (frame {frame_index + 1}/{total_frames})")
        
        # Visual change
        if visual_change['change_detected']:
            context_parts.append(
                f"Scene transition: {visual_change['change_type']} "
                f"(magnitude: {visual_change['change_magnitude']})"
            )
        else:
            context_parts.append("Scene: Continuation of previous shot")
        
        # Similar frames
        if similar_frames:
            context_parts.append("\nSimilar frames in database:")
            for sf in similar_frames[:2]:  # Top 2 similar
                context_parts.append(
                    f"  - {sf['similarity']*100:.1f}% similar to frame from {Path(sf['source']).name}"
                )
        
        context_text = "\n".join(context_parts)
        
        # Build full prompt with improved promo detection
        prompt = f"""Analyze this video frame for marketing signals, paying special attention to promotional offers and codes.

FRAME CONTEXT:
{context_text}

IMPORTANT INSTRUCTIONS FOR PROMO DETECTION:
- Look carefully for promo codes (e.g., "SAVE20", "FREESHIP", "GET50OFF", alphanumeric codes)
- Check for discount percentages or dollar amounts (e.g., "50% OFF", "$10 OFF", "BOGO")
- Look for time-limited offers and deadlines (e.g., "Ends 12/31", "Limited Time", "Today Only", "Offer expires...")
- Promo codes are often in ALL CAPS, may be in a box/banner, or highlighted differently
- If you see ANY promotional offer text, set promo_present to true
- If there's no promo code visible, set promo_code to null (not empty string)
- If there's no deadline mentioned, set promo_deadline to null

Return a JSON object with these fields:

{{
    "brand_name_text": "extracted brand name or null",
    "product_name": "product being advertised or null",
    "industry": "industry/category or null",
    "promo_present": true/false,
    "promo_text": "full promotional message if present, otherwise null",
    "promo_code": "specific promo code if visible (e.g., SAVE20), otherwise null",
    "promo_deadline": "deadline/expiration date if mentioned (be specific), otherwise null",
    "discount_type": "percentage/flat_amount/bogo/free_shipping/other or null",
    "price_value": "specific price, discount amount, or percentage (e.g., '50%', '$10 off') or null",
    "cta_present": true/false,
    "cta_type": "type of call to action (e.g., shop_now/download_app/visit_website/call_now) or null",
    "cta_text": "exact call-to-action text if visible, otherwise null",
    "text_density": "low/medium/high",
    "brand_text_contrast": "low/medium/high",
    "visual_elements": ["list of key visual elements seen"],
    "color_scheme": "dominant colors",
    "aesthetic_style": "description of visual style",
    "logos_detected": ["any brand logos visible"],
    "narrative_function": "purpose of this frame (intro/product_showcase/promo_highlight/cta/conclusion/etc)"
}}

Return ONLY valid JSON, no markdown formatting or extra text."""
        
        return prompt
    
    def _analyze_frame_with_vision(
        self,
        frame_array: np.ndarray,
        frame_metadata: Dict[str, Any],
        embedding: np.ndarray,
        prev_embedding: Optional[np.ndarray],
        total_frames: int
    ) -> Dict[str, Any]:
        """
        Analyze frame using Claude Vision with embedding context
        """
        frame_index = frame_metadata.get('frame_index', 0)
        
        # Classify frame type from embedding
        frame_type = self.context_analyzer.classify_frame_type(embedding)
        
        # Detect humans in frame (simple boolean)
        human_present = self._detect_humans_in_frame(frame_array)
        
        # Build enhanced prompt
        prompt = self._build_enhanced_prompt(
            frame_index,
            total_frames,
            embedding,
            prev_embedding,
            frame_type
        )
        
        # Encode image
        image_b64 = self._encode_image(frame_array)
        
        try:
            # Call Claude Vision API
            message = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            response = message.content[0].text.strip()
            
            # Remove markdown if present
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
            
            signals = json.loads(response)
            
            # Add embedding context to results
            similar_frames = self.context_analyzer.get_brand_similarities(embedding)
            visual_change = self.context_analyzer.calculate_visual_change(prev_embedding, embedding)
            
            return {
                "frame_id": frame_metadata.get("id"),
                "source_path": frame_metadata.get("source_path"),
                "frame_index": frame_index,
                "timestamp": frame_metadata.get("timestamp"),
                "frame_path": frame_metadata.get("frame_path"),
                "frame_type": frame_type,
                "human_present": human_present,
                "embedding_context": {
                    "similar_frames": similar_frames,
                    "visual_change": visual_change,
                    "narrative_position": self.context_analyzer.get_narrative_position(
                        frame_index, total_frames
                    )
                },
                "marketing_signals": signals,
                "api_tokens_used": message.usage.input_tokens + message.usage.output_tokens
            }
            
        except Exception as e:
            return {
                "frame_id": frame_metadata.get("id"),
                "source_path": frame_metadata.get("source_path"),
                "frame_index": frame_index,
                "human_present": human_present,
                "error": str(e)
            }
    
    def _consolidate_video_analysis(self, all_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate all frame analyses into a single video-level summary
        """
        # Extract all marketing signals from frames
        all_signals = [f.get('marketing_signals', {}) for f in all_features if 'error' not in f]
        
        # Extract human detection data (simple boolean)
        frames_with_humans = sum(1 for f in all_features if f.get('human_present', False))
        human_present_in_video = frames_with_humans > 0
        
        if not all_signals:
            return {
                "brand_name_text": None,
                "product_name": None,
                "industry": None,
                "promo_present": False,
                "promo_text": None,
                "promo_code": None,
                "promo_deadline": None,
                "discount_type": None,
                "price_value": None,
                "cta_present": False,
                "cta_type": None,
                "cta_text": None,
                "text_density": "low",
                "brand_text_contrast": "low",
                "visual_elements": [],
                "color_scheme": None,
                "aesthetic_style": None,
                "logos_detected": [],
                "narrative_arc": "unknown",
                "human_present": human_present_in_video
            }
        
        # Collect all values for each field
        brands = [s.get('brand_name_text') for s in all_signals if s.get('brand_name_text')]
        products = [s.get('product_name') for s in all_signals if s.get('product_name')]
        industries = [s.get('industry') for s in all_signals if s.get('industry')]
        promo_texts = [s.get('promo_text') for s in all_signals if s.get('promo_text')]
        promo_codes = [s.get('promo_code') for s in all_signals if s.get('promo_code')]
        promo_deadlines = [s.get('promo_deadline') for s in all_signals if s.get('promo_deadline')]
        discount_types = [s.get('discount_type') for s in all_signals if s.get('discount_type')]
        price_values = [s.get('price_value') for s in all_signals if s.get('price_value')]
        cta_types = [s.get('cta_type') for s in all_signals if s.get('cta_type')]
        cta_texts = [s.get('cta_text') for s in all_signals if s.get('cta_text')]
        text_densities = [s.get('text_density') for s in all_signals if s.get('text_density')]
        contrasts = [s.get('brand_text_contrast') for s in all_signals if s.get('brand_text_contrast')]
        color_schemes = [s.get('color_scheme') for s in all_signals if s.get('color_scheme')]
        aesthetics = [s.get('aesthetic_style') for s in all_signals if s.get('aesthetic_style')]
        
        # Collect all visual elements and logos
        all_visual_elements = []
        all_logos = []
        for s in all_signals:
            if s.get('visual_elements'):
                all_visual_elements.extend(s['visual_elements'])
            if s.get('logos_detected'):
                all_logos.extend(s['logos_detected'])
        
        # Count occurrences to find most common values
        brand_counter = Counter(brands)
        product_counter = Counter(products)
        industry_counter = Counter(industries)
        
        # Check if any frame has promo or CTA
        promo_present = any(s.get('promo_present') for s in all_signals)
        cta_present = any(s.get('cta_present') for s in all_signals)
        
        # Determine dominant text density (most common)
        text_density_counter = Counter(text_densities)
        dominant_text_density = text_density_counter.most_common(1)[0][0] if text_density_counter else "low"
        
        # Determine dominant contrast
        contrast_counter = Counter(contrasts)
        dominant_contrast = contrast_counter.most_common(1)[0][0] if contrast_counter else "low"
        
        # Get unique visual elements (top 10 most frequent)
        visual_element_counter = Counter(all_visual_elements)
        top_visual_elements = [elem for elem, count in visual_element_counter.most_common(10)]
        
        # Get unique logos
        unique_logos = list(set(logo for logo in all_logos if logo))
        
        # Determine narrative arc
        narrative_functions = [s.get('narrative_function') for s in all_signals if s.get('narrative_function')]
        if len(narrative_functions) >= 3:
            narrative_arc = f"{narrative_functions[0]} → {narrative_functions[len(narrative_functions)//2]} → {narrative_functions[-1]}"
        else:
            narrative_arc = " → ".join(narrative_functions) if narrative_functions else "unknown"
        
        return {
            "brand_name_text": brand_counter.most_common(1)[0][0] if brand_counter else None,
            "product_name": product_counter.most_common(1)[0][0] if product_counter else None,
            "industry": industry_counter.most_common(1)[0][0] if industry_counter else None,
            "promo_present": promo_present,
            "promo_text": Counter(promo_texts).most_common(1)[0][0] if promo_texts else None,
            "promo_code": Counter(promo_codes).most_common(1)[0][0] if promo_codes else None,
            "promo_deadline": Counter(promo_deadlines).most_common(1)[0][0] if promo_deadlines else None,
            "discount_type": Counter(discount_types).most_common(1)[0][0] if discount_types else None,
            "price_value": Counter(price_values).most_common(1)[0][0] if price_values else None,
            "cta_present": cta_present,
            "cta_type": Counter(cta_types).most_common(1)[0][0] if cta_types else None,
            "cta_text": Counter(cta_texts).most_common(1)[0][0] if cta_texts else None,
            "text_density": dominant_text_density,
            "brand_text_contrast": dominant_contrast,
            "visual_elements": top_visual_elements,
            "color_scheme": Counter(color_schemes).most_common(1)[0][0] if color_schemes else None,
            "aesthetic_style": Counter(aesthetics).most_common(1)[0][0] if aesthetics else None,
            "logos_detected": unique_logos,
            "narrative_arc": narrative_arc,
            "human_present": human_present_in_video,
            
            # Additional metadata about consolidation
            "_consolidation_meta": {
                "total_frames_analyzed": len(all_signals),
                "frames_with_brand": len(brands),
                "frames_with_product": len(products),
                "frames_with_promo": sum(1 for s in all_signals if s.get('promo_present')),
                "frames_with_cta": sum(1 for s in all_signals if s.get('cta_present')),
                "frames_with_humans": frames_with_humans,
                "unique_brands_detected": len(brand_counter),
                "unique_products_detected": len(product_counter)
            }
        }
    
    def process(self, file_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single video/image with enhanced vision analysis
        """
        start_time = datetime.now()
        
        # Validate file
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}
        
        file_name = Path(file_path).name
        
        print("\n" + "="*70)
        print(f"PROCESSING: {file_name}")
        print("="*70)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # ============================================================
        # STAGE 1: DEDUPLICATION
        # ============================================================
        print("\n" + "="*70)
        print("STAGE 1: DEDUPLICATION & EMBEDDING EXTRACTION")
        print("="*70)
        
        dedup_result = self.dedup_pipeline.process_file(file_path)
        
        if not dedup_result.get('success'):
            return {
                "success": False,
                "error": dedup_result.get('error', 'Deduplication failed')
            }
        
        unique_frames = dedup_result.get('unique_frames', 0)
        total_frames = dedup_result.get('total_frames', 0)
        
        print(f"\n[+] Extracted {unique_frames} unique frames from {total_frames} total")
        
        # ============================================================
        # STAGE 2: ENHANCED VISION ANALYSIS
        # ============================================================
        print("\n" + "="*70)
        print("STAGE 2: ENHANCED VISION ANALYSIS")
        print("="*70)
        print(f"Analyzing {unique_frames} frames with Claude Vision + embedding context\n")
        
        # Get frames and compute embeddings
        frame_arrays, frame_metadata = self.dedup_pipeline.get_frame_arrays_by_source(file_path)
        
        all_features = []
        total_tokens = 0
        prev_embedding = None
        
        for idx, (frame_array, metadata) in enumerate(zip(frame_arrays, frame_metadata)):
            frame_num = metadata['frame_index']
            print(f"Frame {frame_num}: ", end="", flush=True)
            
            # Compute embedding for context
            embedding = self.dedup_pipeline.similarity_agent.compute_embedding(frame_array)
            
            # Analyze with vision + context
            features = self._analyze_frame_with_vision(
                frame_array,
                metadata,
                embedding,
                prev_embedding,
                unique_frames
            )
            
            all_features.append(features)
            
            if 'api_tokens_used' in features:
                total_tokens += features['api_tokens_used']
            
            # Print status
            if 'error' in features:
                print(f"[-] ERROR: {features['error']}")
            else:
                signals = features.get('marketing_signals', {})
                brand = signals.get('brand_name_text', '')
                promo = signals.get('promo_present', False)
                promo_code = signals.get('promo_code', '')
                visuals = len(signals.get('visual_elements', []))
                human_present = features.get('human_present', False)
                
                status = f"[+] "
                if brand:
                    status += f"Brand: {brand[:15]}"
                if promo:
                    status += f", PROMO"
                    if promo_code:
                        status += f" ({promo_code})"
                if human_present:
                    status += f", Human detected"
                status += f", {visuals} visual elements"
                
                print(status)
            
            # Update previous embedding
            prev_embedding = embedding
        
        # Calculate stats
        successful = len([f for f in all_features if 'error' not in f])
        frames_with_brands = len([f for f in all_features 
                                  if f.get('marketing_signals', {}).get('brand_name_text')])
        frames_with_promos = len([f for f in all_features 
                                  if f.get('marketing_signals', {}).get('promo_present')])
        frames_with_promo_codes = len([f for f in all_features 
                                       if f.get('marketing_signals', {}).get('promo_code')])
        frames_with_humans = sum(1 for f in all_features if f.get('human_present', False))
        
        print(f"\n[+] Analysis complete:")
        print(f"  Frames analyzed: {successful}/{unique_frames}")
        print(f"  Brands detected: {frames_with_brands}")
        print(f"  Promos detected: {frames_with_promos}")
        print(f"  Promo codes detected: {frames_with_promo_codes}")
        print(f"  Frames with humans: {frames_with_humans}")
        print(f"  Total API tokens: {total_tokens}")
        print(f"  Estimated cost: ${total_tokens * 0.000003:.4f}")
        
        # ============================================================
        # COMPILE RESULTS
        # ============================================================
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Generate consolidated video-level analysis
        consolidated_analysis = self._consolidate_video_analysis(all_features)
        
        results = {
            "success": True,
            "file_info": {
                "file_path": file_path,
                "file_name": file_name,
                "file_type": dedup_result.get('file_type'),
                "processed_at": start_time.isoformat(),
                "processing_time_seconds": processing_time
            },
            "deduplication": {
                "total_frames_extracted": total_frames,
                "unique_frames_stored": unique_frames,
                "duplicate_frames_skipped": dedup_result.get('duplicate_frames', 0),
                "skip_ratio": f"{dedup_result.get('skip_ratio', 0)*100:.1f}%"
            },
            "analysis": {
                "frames_analyzed": successful,
                "frames_with_brands": frames_with_brands,
                "frames_with_promos": frames_with_promos,
                "frames_with_promo_codes": frames_with_promo_codes,
                "frames_with_humans": frames_with_humans,
                "total_tokens_used": total_tokens,
                "estimated_cost_usd": round(total_tokens * 0.000003, 4)
            },
            
            # Consolidated video-level analysis in the requested format
            "consolidated_video_analysis": consolidated_analysis,
            
            # Keep frame-level details for reference
            "frames": all_features,
            
            # Keep marketing insights
            "marketing_insights": self._generate_insights(all_features)
        }
        
        # Save output
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n[+] Results saved to: {output_path}")
        
        # Print consolidated summary
        print("\n" + "="*70)
        print("CONSOLIDATED VIDEO ANALYSIS")
        print("="*70)
        print(f"Brand: {consolidated_analysis.get('brand_name_text', 'N/A')}")
        print(f"Product: {consolidated_analysis.get('product_name', 'N/A')}")
        print(f"Industry: {consolidated_analysis.get('industry', 'N/A')}")
        print(f"Promo Present: {consolidated_analysis.get('promo_present')}")
        if consolidated_analysis.get('promo_present'):
            print(f"  Promo Text: {consolidated_analysis.get('promo_text', 'N/A')}")
            if consolidated_analysis.get('promo_code'):
                print(f"  Promo Code: {consolidated_analysis.get('promo_code')}")
            if consolidated_analysis.get('promo_deadline'):
                print(f"  Deadline: {consolidated_analysis.get('promo_deadline')}")
            if consolidated_analysis.get('discount_type'):
                print(f"  Discount Type: {consolidated_analysis.get('discount_type')}")
            if consolidated_analysis.get('price_value'):
                print(f"  Price/Discount: {consolidated_analysis.get('price_value')}")
        print(f"CTA Present: {consolidated_analysis.get('cta_present')}")
        if consolidated_analysis.get('cta_present'):
            if consolidated_analysis.get('cta_type'):
                print(f"  CTA Type: {consolidated_analysis.get('cta_type')}")
            if consolidated_analysis.get('cta_text'):
                print(f"  CTA Text: {consolidated_analysis.get('cta_text')}")
        
        # Print human presence info (simplified)
        human_present = consolidated_analysis.get('human_present', False)
        print(f"Humans Detected: {'Yes' if human_present else 'No'}")
        if human_present:
            frames_with_humans = consolidated_analysis.get('_consolidation_meta', {}).get('frames_with_humans', 0)
            print(f"  Frames with humans: {frames_with_humans}/{unique_frames}")
        
        print(f"Aesthetic: {consolidated_analysis.get('aesthetic_style', 'N/A')}")
        print(f"Color Scheme: {consolidated_analysis.get('color_scheme', 'N/A')}")
        visual_elems = consolidated_analysis.get('visual_elements', [])[:5]
        if visual_elems:
            print(f"Top Visual Elements: {', '.join(visual_elems)}")
        logos = consolidated_analysis.get('logos_detected', [])
        if logos:
            print(f"Logos Detected: {', '.join(logos)}")
        print(f"Narrative Arc: {consolidated_analysis.get('narrative_arc', 'N/A')}")
        print("="*70)
        
        return results
    
    def _generate_insights(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate aggregated insights"""
        brands = set()
        products = set()
        industries = set()
        aesthetics = set()
        promo_codes = set()
        all_visual_elements = []
        
        for feature in features:
            if 'error' in feature:
                continue
            
            signals = feature.get('marketing_signals', {})
            
            if signals.get('brand_name_text'):
                brands.add(signals['brand_name_text'])
            if signals.get('product_name'):
                products.add(signals['product_name'])
            if signals.get('industry'):
                industries.add(signals['industry'])
            if signals.get('aesthetic_style'):
                aesthetics.add(signals['aesthetic_style'])
            if signals.get('promo_code'):
                promo_codes.add(signals['promo_code'])
            if signals.get('visual_elements'):
                all_visual_elements.extend(signals['visual_elements'])
        
        # Count visual elements
        visual_element_counts = Counter(all_visual_elements)
        
        return {
            "brands_detected": list(brands),
            "products_detected": list(products),
            "industries": list(industries),
            "aesthetic_styles": list(aesthetics),
            "promo_codes_detected": list(promo_codes),
            "top_visual_elements": [
                {"element": elem, "count": count} 
                for elem, count in visual_element_counts.most_common(10)
            ]
        }


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced vision analysis with embedding context"
    )
    parser.add_argument("file", type=str, help="Path to video or image file")
    parser.add_argument("-o", "--output", type=str, default=None, 
                       help="Output JSON file")
    parser.add_argument("--frame-interval", type=float, default=0.3,
                       help="Frame extraction interval (seconds)")
    parser.add_argument("--similarity-threshold", type=float, default=0.9,
                       help="Deduplication threshold")
    
    args = parser.parse_args()
    
    # Get file path from args
    file_path = args.file
    
    if not os.path.exists(file_path):
        print(f"[-] Error: File not found: {file_path}")
        sys.exit(1)
    
    if args.output is None:
        file_name = Path(file_path).stem
        args.output = f"output/{file_name}_enhanced_results.json"
    
    # Initialize orchestrator
    orchestrator = EnhancedVisionOrchestrator(
        frame_interval=args.frame_interval,
        similarity_threshold=args.similarity_threshold
    )
    
    # Process file
    results = orchestrator.process(file_path, args.output)
    
    if not results.get('success'):
        print(f"[-] Failed: {results.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()