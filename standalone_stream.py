#!/usr/bin/env python3
"""
Standalone Camera Streaming Script
This script provides camera streaming functionality without Flask API or Firebase dependencies.
It runs a simple HTTP server to serve the camera stream and a basic HTML viewer.
"""

import cv2
import threading
import time
import logging
from picamera2 import Picamera2
from http.server import HTTPServer, BaseHTTPRequestHandler
import io
import json
from datetime import datetime
import face_recognition
import numpy as np
import pickle
import os
import signal
import sys
import subprocess
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
camera = None
streaming_active = False
known_face_encodings = []
known_face_names = []
face_detection_enabled = True
motion_detection_enabled = True
background_subtractor = None
last_motion_time = 0
motion_cooldown = 10  # seconds
last_face_report = {}
face_cooldown = 10  # seconds

# H.264 encoding variables
h264_encoder = None
h264_process = None
temp_fifo = None

# Device Information
DEVICE_INFO = {
    "device_name": "Raspberry Pi v5",
    "camera": "Raspberry Pi Camera Module 3 12MP",
    "model": "RPI-001"
}

def load_face_encodings():
    """Load pre-trained face encodings if available."""
    global known_face_encodings, known_face_names
    try:
        if os.path.exists("encodings.pickle"):
            with open("encodings.pickle", "rb") as f:
                data = pickle.loads(f.read())
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
            logger.info(f"Loaded {len(known_face_names)} known faces")
            return True
        else:
            logger.info("No face encodings file found - face recognition disabled")
            return False
    except Exception as e:
        logger.error(f"Failed to load face encodings: {str(e)}")
        return False

def initialize_detection():
    """Initialize face detection and motion detection."""
    global background_subtractor
    try:
        # Initialize background subtractor for motion detection
        background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Load face encodings
        load_face_encodings()
        
        logger.info("Detection systems initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize detection: {str(e)}")
        return False

def cleanup_camera():
    """Clean up any existing camera processes."""
    global camera, streaming_active, h264_process, temp_fifo
    
    try:
        # Stop H.264 encoder
        if h264_process is not None:
            h264_process.terminate()
            h264_process = None
        
        # Clean up temp fifo
        if temp_fifo is not None and os.path.exists(temp_fifo):
            os.unlink(temp_fifo)
            temp_fifo = None
        
        # Stop camera
        if camera is not None:
            camera.stop()
            camera.close()
            camera = None
        streaming_active = False
        logger.info("Camera cleanup completed")
    except Exception as e:
        logger.warning(f"Camera cleanup warning: {str(e)}")

def initialize_h264_encoder():
    """Initialize H.264 encoder for mobile streaming."""
    global h264_encoder, h264_process, temp_fifo
    
    try:
        # Create temporary FIFO for H.264 stream
        temp_fifo = tempfile.mktemp(suffix='.h264')
        os.mkfifo(temp_fifo)
        
        # Start H.264 encoder process
        h264_process = subprocess.Popen([
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'yuv420p',
            '-s', '640x480',
            '-r', '30',
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-crf', '23',
            '-maxrate', '2M',
            '-bufsize', '4M',
            '-g', '30',
            '-keyint_min', '30',
            '-sc_threshold', '0',
            '-f', 'h264',
            '-'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        logger.info("H.264 encoder initialized for mobile streaming")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize H.264 encoder: {str(e)}")
        return False

def initialize_camera():
    """Initialize camera for streaming with proper cleanup."""
    global camera, streaming_active
    
    try:
        # First, try to cleanup any existing camera
        cleanup_camera()
        
        # Wait a moment for cleanup
        time.sleep(1)
        
        # Initialize new camera with H.264 support
        camera = Picamera2()
        
        # Configure camera for H.264 streaming
        camera_config = camera.create_video_configuration(
            main={"size": (640, 480), "format": "YUV420"},
            lores={"size": (320, 240), "format": "YUV420"}
        )
        camera.configure(camera_config)
        camera.start()
        streaming_active = True
        
        # Initialize detection systems
        initialize_detection()
        
        # Initialize H.264 encoder for mobile streaming
        initialize_h264_encoder()
        
        logger.info("Camera initialized for H.264 streaming with detection")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize camera: {str(e)}")
        logger.error("This might be because another process is using the camera")
        logger.error("Try stopping the main app.py first, then run this script")
        cleanup_camera()
        return False

def detect_motion(frame):
    """Detect motion in the frame (without Firebase upload)."""
    global background_subtractor, last_motion_time, motion_cooldown
    
    if not motion_detection_enabled or background_subtractor is None:
        return False
    
    try:
        # Apply background subtraction
        fg_mask = background_subtractor.apply(frame)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for significant motion
        motion_detected = False
        current_time = time.time()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Motion threshold
                motion_detected = True
                if current_time - last_motion_time >= motion_cooldown:
                    logger.info("Motion detected (not uploaded to Firebase)")
                    last_motion_time = current_time
                break
        
        return motion_detected
    except Exception as e:
        logger.error(f"Error in motion detection: {str(e)}")
        return False

def detect_faces(frame):
    """Detect and recognize faces in the frame (without Firebase upload)."""
    global known_face_encodings, known_face_names, last_face_report, face_cooldown
    
    if not face_detection_enabled or not known_face_encodings:
        return frame
    
    try:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        
        if not face_locations:
            return frame
            
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model='small')
        
        current_time = time.time()
        
        for i, face_encoding in enumerate(face_encodings):
            # Calculate face distances to all known faces
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # Scale back up face locations
            top, right, bottom, left = face_locations[i]
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5
            
            # Process face detection result
            if best_distance <= 0.6:  # Distance threshold
                name = known_face_names[best_match_index]
                confidence = 1.0 - best_distance
                
                # Check cooldown for this person
                if name not in last_face_report or current_time - last_face_report[name] >= face_cooldown:
                    logger.info(f"Known face detected: {name} (not uploaded to Firebase)")
                    last_face_report[name] = current_time
                
                # Draw green box for known faces
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                name = "Unknown"
                
                # Check cooldown for unknown faces
                if "unknown" not in last_face_report or current_time - last_face_report["unknown"] >= face_cooldown:
                    logger.info("Unknown face detected (not uploaded to Firebase)")
                    last_face_report["unknown"] = current_time
                
                # Draw red box for unknown faces
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        return frame
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return frame

def generate_frames():
    """Generate camera frames for MJPEG streaming with detection overlays."""
    global camera, streaming_active
    
    while streaming_active:
        try:
            if camera is not None:
                # Capture frame from camera
                frame = camera.capture_array()
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Detect motion
                motion_detected = detect_motion(frame_bgr)
                
                # Detect faces
                frame_bgr = detect_faces(frame_bgr)
                
                # Add motion indicator
                if motion_detected:
                    cv2.putText(frame_bgr, "MOTION DETECTED", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Add status information
                cv2.putText(frame_bgr, "Face Detection: ON", (10, frame_bgr.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame_bgr, "Motion Detection: ON", (10, frame_bgr.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame_bgr, "Standalone Mode", (10, frame_bgr.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    logger.warning("Failed to encode frame")
            else:
                # Send a placeholder frame if camera is not available
                placeholder = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
                cv2.putText(placeholder, "Camera Not Available", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"Error generating frame: {str(e)}")
            time.sleep(0.1)

def generate_h264_frames():
    """Generate H.264 frames for mobile streaming."""
    global camera, streaming_active, h264_process
    
    while streaming_active:
        try:
            if camera is not None and h264_process is not None:
                # Capture frame from camera
                frame = camera.capture_array()
                
                # Convert to YUV420 format for H.264 encoding
                frame_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
                
                # Send frame to H.264 encoder
                try:
                    h264_process.stdin.write(frame_yuv.tobytes())
                    h264_process.stdin.flush()
                except:
                    # Reinitialize encoder if it fails
                    initialize_h264_encoder()
                    continue
                
                # Read H.264 data from encoder
                try:
                    h264_data = h264_process.stdout.read(4096)
                    if h264_data:
                        yield h264_data
                except:
                    continue
            else:
                # Send placeholder if camera not available
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Not Available", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Convert placeholder to YUV and encode
                placeholder_yuv = cv2.cvtColor(placeholder, cv2.COLOR_BGR2YUV_I420)
                try:
                    h264_process.stdin.write(placeholder_yuv.tobytes())
                    h264_process.stdin.flush()
                    h264_data = h264_process.stdout.read(4096)
                    if h264_data:
                        yield h264_data
                except:
                    pass
            
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"Error generating H.264 frame: {str(e)}")
            time.sleep(0.1)

def generate_mp4_frames():
    """Generate MP4 frames for React Native/Android Studio."""
    global camera, streaming_active
    
    while streaming_active:
        try:
            if camera is not None:
                # Capture frame from camera
                frame = camera.capture_array()
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Detect motion and faces
                motion_detected = detect_motion(frame_bgr)
                frame_bgr = detect_faces(frame_bgr)
                
                # Add overlays
                if motion_detected:
                    cv2.putText(frame_bgr, "MOTION DETECTED", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.putText(frame_bgr, "Mobile Stream", (10, frame_bgr.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Encode as MP4-compatible format
                ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield frame_bytes
            else:
                # Send placeholder
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Not Available", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield frame_bytes
            
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"Error generating MP4 frame: {str(e)}")
            time.sleep(0.1)

def generate_rtsp_frames():
    """Generate RTSP-like frames for mobile apps."""
    global camera, streaming_active
    
    while streaming_active:
        try:
            if camera is not None:
                # Capture frame from camera
                frame = camera.capture_array()
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Detect motion and faces
                motion_detected = detect_motion(frame_bgr)
                frame_bgr = detect_faces(frame_bgr)
                
                # Add overlays
                if motion_detected:
                    cv2.putText(frame_bgr, "MOTION DETECTED", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.putText(frame_bgr, "RTSP Stream", (10, frame_bgr.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Encode for RTSP streaming
                ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield frame_bytes
            else:
                # Send placeholder
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Not Available", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield frame_bytes
            
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"Error generating RTSP frame: {str(e)}")
            time.sleep(0.1)

class StreamHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the camera stream."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/stream':
            # MJPEG stream for desktop browsers
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            try:
                for frame in generate_frames():
                    self.wfile.write(frame)
            except Exception as e:
                logger.error(f"Stream error: {str(e)}")
                
        elif self.path == '/stream/h264':
            # H.264 stream for mobile devices
            self.send_response(200)
            self.send_header('Content-Type', 'video/h264')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            try:
                for frame in generate_h264_frames():
                    self.wfile.write(frame)
            except Exception as e:
                logger.error(f"H.264 stream error: {str(e)}")
                
        elif self.path == '/stream/mp4':
            # MP4 stream for React Native/Android Studio
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            try:
                for frame in generate_mp4_frames():
                    self.wfile.write(frame)
            except Exception as e:
                logger.error(f"MP4 stream error: {str(e)}")
                
        elif self.path == '/stream/rtsp':
            # RTSP-like stream for mobile apps
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-rtsp')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                for frame in generate_rtsp_frames():
                    self.wfile.write(frame)
            except Exception as e:
                logger.error(f"RTSP stream error: {str(e)}")
                
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            status = {
                "streaming": streaming_active,
                "camera_initialized": camera is not None,
                "face_detection": face_detection_enabled,
                "motion_detection": motion_detection_enabled,
                "known_faces_count": len(known_face_names) if known_face_names else 0,
                "mode": "standalone",
                "timestamp": datetime.now().isoformat(),
                "streams": {
                    "mjpeg": "/stream",
                    "h264": "/stream/h264",
                    "mp4": "/stream/mp4",
                    "rtsp": "/stream/rtsp"
                }
            }
            
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/api/streams':
            # API endpoint for mobile app integration
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()
            
            streams_info = {
                "available_streams": {
                    "mjpeg": {
                        "url": "/stream",
                        "type": "image/jpeg",
                        "description": "MJPEG stream for desktop browsers",
                        "mobile_support": False
                    },
                    "h264": {
                        "url": "/stream/h264",
                        "type": "video/h264",
                        "description": "H.264 stream for mobile devices",
                        "mobile_support": True
                    },
                    "mp4": {
                        "url": "/stream/mp4",
                        "type": "video/mp4",
                        "description": "MP4 stream for React Native/Android Studio",
                        "mobile_support": True
                    },
                    "rtsp": {
                        "url": "/stream/rtsp",
                        "type": "application/x-rtsp",
                        "description": "RTSP-like stream for mobile apps",
                        "mobile_support": True
                    }
                },
                "recommended_for_mobile": "/stream/mp4",
                "recommended_for_react_native": "/stream/mp4",
                "recommended_for_android_studio": "/stream/rtsp"
            }
            
            self.wfile.write(json.dumps(streams_info).encode())
            
        elif self.path == '/api/network':
            # Network information for mobile app connection
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            import socket
            try:
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
            except:
                local_ip = "unknown"
            
            network_info = {
                "server_ip": local_ip,
                "port": 8080,
                "base_url": f"http://{local_ip}:8080",
                "stream_urls": {
                    "mjpeg": f"http://{local_ip}:8080/stream",
                    "h264": f"http://{local_ip}:8080/stream/h264",
                    "mp4": f"http://{local_ip}:8080/stream/mp4",
                    "rtsp": f"http://{local_ip}:8080/stream/rtsp"
                },
                "connection_help": {
                    "react_native": f"Use: http://{local_ip}:8080/stream/mp4",
                    "android_studio": f"Use: http://{local_ip}:8080/stream/rtsp",
                    "mobile_browser": f"Use: http://{local_ip}:8080/stream/h264"
                }
            }
            
            self.wfile.write(json.dumps(network_info).encode())
            
        elif self.path == '/':
            # Serve the HTML viewer
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raspberry Pi Camera Stream - Standalone Mode</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .stream-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .stream-image {
            max-width: 100%;
            height: auto;
            border: 2px solid #333;
            border-radius: 8px;
        }
        .status {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .status-label {
            font-weight: bold;
        }
        .status-value {
            color: #4CAF50;
        }
        .controls {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .btn:disabled {
            background-color: #666;
            cursor: not-allowed;
        }
        .info {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .warning {
            background-color: #ff9800;
            color: black;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 Raspberry Pi Camera Stream</h1>
            <h2>Standalone Mode</h2>
        </div>
        
        <div class="warning">
            <strong>⚠️ Standalone Mode:</strong> This camera stream is running independently without the main Flask API or Firebase connectivity. Face detection and motion detection are active but data is not being uploaded to Firebase.
        </div>
        
        <div class="stream-container">
            <video id="streamVideo" class="stream-image" autoplay muted playsinline controls style="display: none;">
                <source src="/stream/h264" type="video/h264">
                Your browser does not support H.264 video.
            </video>
            <img src="/stream" alt="Camera Stream" class="stream-image" id="streamImage">
        </div>
        
        <div class="status" id="status">
            <h3>📊 System Status</h3>
            <div class="status-item">
                <span class="status-label">Streaming:</span>
                <span class="status-value" id="streamingStatus">Loading...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Camera:</span>
                <span class="status-value" id="cameraStatus">Loading...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Face Detection:</span>
                <span class="status-value" id="faceDetectionStatus">Loading...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Motion Detection:</span>
                <span class="status-value" id="motionDetectionStatus">Loading...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Known Faces:</span>
                <span class="status-value" id="knownFacesCount">Loading...</span>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="refreshStatus()">🔄 Refresh Status</button>
            <button class="btn" onclick="reloadStream()">🔄 Reload Stream</button>
            <button class="btn" onclick="toggleStreamMode()">📱 Toggle Mobile Mode</button>
        </div>
        
        <div class="info">
            <h3>ℹ️ Information</h3>
            <p><strong>Device:</strong> Raspberry Pi v5</p>
            <p><strong>Camera:</strong> Raspberry Pi Camera Module 3 12MP</p>
            <p><strong>Resolution:</strong> 640x480</p>
            <p><strong>Frame Rate:</strong> ~30 FPS</p>
            <p><strong>Mode:</strong> Standalone (No Firebase/API)</p>
        </div>
    </div>

    <script>
        let isMobileMode = false;
        
        function isMobileDevice() {
            return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        }
        
        function refreshStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('streamingStatus').textContent = data.streaming ? 'Active' : 'Inactive';
                    document.getElementById('cameraStatus').textContent = data.camera_initialized ? 'Initialized' : 'Not Initialized';
                    document.getElementById('faceDetectionStatus').textContent = data.face_detection ? 'Enabled' : 'Disabled';
                    document.getElementById('motionDetectionStatus').textContent = data.motion_detection ? 'Enabled' : 'Disabled';
                    document.getElementById('knownFacesCount').textContent = data.known_faces_count;
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }
        
        function reloadStream() {
            if (isMobileMode) {
                const video = document.getElementById('streamVideo');
                video.src = '/stream/h264?' + new Date().getTime();
            } else {
                const img = document.getElementById('streamImage');
                img.src = '/stream?' + new Date().getTime();
            }
        }
        
        function toggleStreamMode() {
            const video = document.getElementById('streamVideo');
            const img = document.getElementById('streamImage');
            
            isMobileMode = !isMobileMode;
            
            if (isMobileMode) {
                img.style.display = 'none';
                video.style.display = 'block';
                video.src = '/stream/h264?' + new Date().getTime();
                video.play();
            } else {
                video.style.display = 'none';
                img.style.display = 'block';
                img.src = '/stream?' + new Date().getTime();
            }
        }
        
        // Auto-detect mobile and set appropriate stream
        function initializeStream() {
            if (isMobileDevice()) {
                isMobileMode = true;
                toggleStreamMode();
            }
        }
        
        // Auto-refresh status every 5 seconds
        setInterval(refreshStatus, 5000);
        
        // Initialize stream and status
        initializeStream();
        refreshStatus();
    </script>
</body>
</html>
            """
            
            self.wfile.write(html_content.encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        """Override to reduce log noise."""
        pass

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    cleanup_camera()
    sys.exit(0)

def start_server(host='0.0.0.0', port=8080):
    """Start the standalone streaming server."""
    global streaming_active, camera
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize camera
        if not initialize_camera():
            logger.error("Failed to initialize camera")
            logger.error("Make sure no other process is using the camera")
            logger.error("Try stopping app.py first: pkill -f app.py")
            return False
        
        # Start streaming
        streaming_active = True
        
        # Create and start HTTP server
        server = HTTPServer((host, port), StreamHandler)
        logger.info(f"🎥 Standalone camera streaming server started")
        logger.info(f"📡 Server running on http://{host}:{port}")
        logger.info(f"📹 Stream URL: http://{host}:{port}/stream")
        logger.info(f"📊 Status URL: http://{host}:{port}/status")
        logger.info(f"🌐 Viewer: http://{host}:{port}/")
        logger.info("⚠️  Running in standalone mode - no Firebase/API connectivity")
        logger.info("Press Ctrl+C to stop the server")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            # Cleanup
            streaming_active = False
            cleanup_camera()
            server.shutdown()
            logger.info("Server stopped")
            
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        cleanup_camera()
        return False

if __name__ == '__main__':
    print("🎥 Raspberry Pi Standalone Camera Stream")
    print("=" * 50)
    print("This script provides camera streaming without Flask API or Firebase")
    print("=" * 50)
    
    # Start the server
    start_server()
