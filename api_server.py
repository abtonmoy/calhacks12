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

from human_detector import SimpleHumanDetector
import audio_processor

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
        result = audio_processor.analyze_video_audio(filepath)
        
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


if __name__ == '__main__':
    print("Starting API server...")
    print("API endpoints:")
    print("  - POST /api/detect-human (upload image)")
    print("  - POST /api/analyze-audio (upload video)")
    print("  - GET  /api/health")
    print("\nServer running on http://localhost:5001")
    print("React frontend should connect to http://localhost:5001/api")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

