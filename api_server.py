#!/usr/bin/env python3
"""
Flask API Server for Human Detection and Audio Processing
Handles file uploads from React frontend and returns JSON results
"""

import os
import sys
import json
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'findHuman_agent'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'audio_processing'))

from human_detection_agent.human_detector import SimpleHumanDetector
from audio_processing_agent.audio_processor import analyze_video_audio
from video_orc_agent_main import EnhancedVisionOrchestrator
from img_orc_agent import ImageFeatureExtractor

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend on localhost:3000

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('audio_temp', exist_ok=True)

# Initialize detector
detector = SimpleHumanDetector()

# Lazy initialization of orchestrators (expensive to load)
video_orchestrator = None
image_extractor = None

def get_video_orchestrator():
    """Lazy-load video orchestrator to avoid loading on startup"""
    global video_orchestrator
    if video_orchestrator is None:
        print("Loading video orchestrator (first time only)...")
        video_orchestrator = EnhancedVisionOrchestrator()
    return video_orchestrator

def get_image_extractor():
    """Lazy-load image feature extractor to avoid loading on startup"""
    global image_extractor
    if image_extractor is None:
        print("Loading image feature extractor (first time only)...")
        image_extractor = ImageFeatureExtractor()
    return image_extractor


def allowed_file(filename, allowed_extensions):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'API server is running'
    })


@app.route('/api/detect-human', methods=['POST'])
def detect_human():
    """Detect humans in uploaded image"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'human_present': 0,
                'num_people': 0
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'human_present': 0,
                'num_people': 0
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({
                'error': 'Invalid file type. Only images (png, jpg, jpeg, gif, bmp) are allowed',
                'human_present': 0,
                'num_people': 0
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Detect humans
        result = detector.detect_faces(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result), 200
        
    except Exception as e:
        error_response = {
            'error': str(e),
            'human_present': 0,
            'num_people': 0
        }
        return jsonify(error_response), 500


@app.route('/api/analyze-audio', methods=['POST'])
def analyze_audio():
    """Analyze audio from uploaded video"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'duration_sec': 0,
                'gender_estimation': 'unknown',
                'mean_pitch': 0,
                'spectral_bandwidth': 0,
                'emotion': 'unknown'
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'duration_sec': 0,
                'gender_estimation': 'unknown',
                'mean_pitch': 0,
                'spectral_bandwidth': 0,
                'emotion': 'unknown'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({
                'error': 'Invalid file type. Only videos (mp4, avi, mov, mkv, webm) are allowed',
                'duration_sec': 0,
                'gender_estimation': 'unknown',
                'mean_pitch': 0,
                'spectral_bandwidth': 0,
                'emotion': 'unknown'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Analyze audio
        result = analyze_video_audio(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Clean up audio temp files
        try:
            audio_files = os.listdir('audio_temp')
            for audio_file in audio_files:
                os.remove(os.path.join('audio_temp', audio_file))
        except:
            pass
        
        return jsonify(result), 200
        
    except Exception as e:
        error_response = {
            'error': str(e),
            'duration_sec': 0,
            'gender_estimation': 'unknown',
            'mean_pitch': 0,
            'spectral_bandwidth': 0,
            'emotion': 'unknown'
        }
        return jsonify(error_response), 500


@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """Analyze video using the video orchestration agent"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'success': False
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({
                'error': 'Invalid file type. Only videos (mp4, avi, mov, mkv, webm) are allowed',
                'success': False
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get orchestrator (lazy-loaded, cached)
        orchestrator = get_video_orchestrator()
        
        # Process video (returns full analysis)
        result = orchestrator.process(filepath)
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Return the result
        return jsonify(result), 200
        
    except Exception as e:
        error_response = {
            'error': str(e),
            'success': False
        }
        return jsonify(error_response), 500


@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze image using the image orchestration agent"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'success': False
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({
                'error': 'Invalid file type. Only images (png, jpg, jpeg, gif, bmp) are allowed',
                'success': False
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get image extractor (lazy-loaded, cached)
        extractor = get_image_extractor()
        
        # Process image (returns full analysis)
        result = extractor.process_image(filepath)
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Return the result
        return jsonify(result), 200
        
    except Exception as e:
        error_response = {
            'error': str(e),
            'success': False
        }
        return jsonify(error_response), 500


if __name__ == '__main__':
    print("Starting API server...")
    print("API endpoints:")
    print("  - POST /api/detect-human (upload image - simple detection)")
    print("  - POST /api/analyze-image (upload image - full analysis)")
    print("  - POST /api/analyze-audio (upload video - audio only)")
    print("  - POST /api/analyze-video (upload video - full analysis)")
    print("  - GET  /api/health")
    print("\nNote: Orchestrators will load on first request (may take 30-60s)")
    print("Server running on http://localhost:5001")
    print("React frontend should connect to http://localhost:5001/api")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

