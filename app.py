from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
from datetime import datetime
import traceback
import cv2
import threading
import time
from picamera2 import Picamera2
import io
import face_recognition
import numpy as np
import pickle
from datetime import datetime
import logging

from firebase_service import get_firebase_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global Firebase service instance
firebase_service = None

# Global camera instance for streaming
camera = None
camera_lock = threading.Lock()
frame_buffer = None
streaming_active = False

# Face detection and motion detection variables
known_face_encodings = []
known_face_names = []
face_detection_enabled = True
motion_detection_enabled = True
background_subtractor = None
last_motion_time = 0
motion_cooldown = 10  # seconds
last_face_report = {}
face_cooldown = 10  # seconds

# Device Information
DEVICE_INFO = {
    "device_name": "Raspberry Pi v5",
    "camera": "Raspberry Pi Camera Module 3 12MP",
    "model": "RPI-001"
}

def initialize_firebase():
    """Initialize Firebase service on first request."""
    global firebase_service
    try:
        firebase_service = get_firebase_service()
        logger.info("Firebase service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase service: {str(e)}")
        firebase_service = None

def load_face_encodings():
    """Load pre-trained face encodings."""
    global known_face_encodings, known_face_names
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.loads(f.read())
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]
        logger.info(f"Loaded {len(known_face_names)} known faces")
        return True
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

def initialize_camera():
    """Initialize camera for streaming."""
    global camera, streaming_active
    try:
        if camera is None:
            camera = Picamera2()
            camera_config = camera.create_preview_configuration()
            camera_config["main"]["size"] = (640, 480)  # Lower resolution for streaming
            camera_config["main"]["format"] = "RGB888"
            camera.configure(camera_config)
            camera.start()
            streaming_active = True
            
            # Initialize detection systems
            initialize_detection()
            
            logger.info("Camera initialized for streaming with detection")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize camera: {str(e)}")
        return False

def detect_motion(frame):
    """Detect motion in the frame."""
    global background_subtractor, last_motion_time, motion_cooldown, firebase_service
    
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
                    # Send motion detection to Firebase
                    if firebase_service:
                        try:
                            motion_frame = frame.copy()
                            motion_base64 = firebase_service.encode_image_to_base64(motion_frame)
                            if motion_base64:
                                firebase_service.save_motion_detection(
                                    timestamp=datetime.now(),
                                    location="raspberry_pi_camera",
                                    confidence=min(1.0, area / (frame.shape[0] * frame.shape[1])),
                                    captured_photo=motion_base64,
                                    device_info=DEVICE_INFO
                                )
                                last_motion_time = current_time
                                logger.info("Motion detection sent to Firebase")
                        except Exception as e:
                            logger.error(f"Error sending motion detection: {str(e)}")
                break
        
        return motion_detected
    except Exception as e:
        logger.error(f"Error in motion detection: {str(e)}")
        return False

def detect_faces(frame):
    """Detect and recognize faces in the frame."""
    global known_face_encodings, known_face_names, last_face_report, face_cooldown, firebase_service
    
    if not face_detection_enabled or not known_face_encodings:
        return frame
    
    try:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model='small')
        
        face_names = []
        current_time = time.time()
        
        for i, face_encoding in enumerate(face_encodings):
            # Calculate face distances to all known faces
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # Scale back up face locations
            top, right, bottom, left = face_locations[i]
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Extract face region
            face_image = frame[top:bottom, left:right]
            
            # Process face detection result
            if best_distance <= 0.6:  # Distance threshold
                name = known_face_names[best_match_index]
                confidence = 1.0 - best_distance
                
                # Check cooldown for this person
                if name not in last_face_report or current_time - last_face_report[name] >= face_cooldown:
                    # Send known face to Firebase
                    if firebase_service and face_image.size > 0:
                        try:
                            face_base64 = firebase_service.encode_image_to_base64(face_image)
                            if face_base64:
                                firebase_service.save_known_face(
                                    face_image_base64=face_base64,
                                    name=name,
                                    timestamp=datetime.now(),
                                    location="raspberry_pi_camera",
                                    confidence=confidence,
                                    device_info=DEVICE_INFO
                                )
                                last_face_report[name] = current_time
                                logger.info(f"Known face ({name}) sent to Firebase")
                        except Exception as e:
                            logger.error(f"Error sending known face: {str(e)}")
                
                # Draw green box for known faces
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                name = "Unknown"
                
                # Check cooldown for unknown faces
                if "unknown" not in last_face_report or current_time - last_face_report["unknown"] >= face_cooldown:
                    # Send unknown face to Firebase
                    if firebase_service and face_image.size > 0:
                        try:
                            face_base64 = firebase_service.encode_image_to_base64(face_image)
                            if face_base64:
                                firebase_service.save_unknown_face(
                                    face_image_base64=face_base64,
                                    timestamp=datetime.now(),
                                    location="raspberry_pi_camera",
                                    confidence=0.8,
                                    device_info=DEVICE_INFO
                                )
                                last_face_report["unknown"] = current_time
                                logger.info("Unknown face sent to Firebase")
                        except Exception as e:
                            logger.error(f"Error sending unknown face: {str(e)}")
                
                # Draw red box for unknown faces
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            
            face_names.append(name)
        
        return frame
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return frame

def generate_frames():
    """Generate camera frames for streaming with detection overlays."""
    global camera, frame_buffer, streaming_active
    
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
                cv2.putText(frame_bgr, "Raspberry Pi v5", (10, frame_bgr.shape[0] - 20), 
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

# Initialize Firebase when the application starts
with app.app_context():
    initialize_firebase()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        firebase_connected = firebase_service is not None and firebase_service.db is not None
        return jsonify({
            "status": "healthy" if firebase_connected else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "firebase_connected": firebase_connected,
            "version": "1.0.0"
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/motion-detection', methods=['POST'])
def report_motion_detection():
    """
    Report motion detection event to Firebase.
    This endpoint is called when motion is detected by the camera.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "No JSON data provided"
            }), 400
        
        # Extract parameters with defaults
        timestamp_str = data.get('timestamp')
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        location = data.get('location', 'default')
        confidence = float(data.get('confidence', 1.0))
        captured_photo = data.get('captured_photo')  # Optional base64 encoded photo
        
        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            return jsonify({
                "success": False,
                "message": "Confidence must be between 0.0 and 1.0"
            }), 400
        
        # Save motion detection to Firebase
        success = firebase_service.save_motion_detection(
            timestamp=timestamp,
            location=location,
            confidence=confidence,
            captured_photo=captured_photo
        )
        
        if success:
            return jsonify({
                "success": True,
                "message": "Motion detection event saved successfully",
                "data": {
                    "timestamp": timestamp.isoformat(),
                    "location": location,
                    "confidence": confidence
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to save motion detection"
            }), 500
            
    except ValueError as e:
        logger.error(f"Validation error in motion detection: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Validation error: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Error in motion detection endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/unknown-face', methods=['POST'])
def report_unknown_face():
    """
    Report unknown face detection event to Firebase.
    This endpoint is called when an unknown face is detected.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "No JSON data provided"
            }), 400
        
        # Extract required parameters
        face_image_base64 = data.get('face_image_base64')
        if not face_image_base64:
            return jsonify({
                "success": False,
                "message": "face_image_base64 is required"
            }), 400
        
        # Extract optional parameters with defaults
        timestamp_str = data.get('timestamp')
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        location = data.get('location', 'default')
        confidence = float(data.get('confidence', 0.0))
        
        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            return jsonify({
                "success": False,
                "message": "Confidence must be between 0.0 and 1.0"
            }), 400
        
        # Save unknown face to Firebase
        success = firebase_service.save_unknown_face(
            face_image_base64=face_image_base64,
            timestamp=timestamp,
            location=location,
            confidence=confidence
        )
        
        if success:
            return jsonify({
                "success": True,
                "message": "Unknown face event saved successfully",
                "data": {
                    "timestamp": timestamp.isoformat(),
                    "location": location,
                    "confidence": confidence
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to save unknown face"
            }), 500
            
    except ValueError as e:
        logger.error(f"Validation error in unknown face: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Validation error: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Error in unknown face endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/motion-logs', methods=['GET'])
def get_motion_logs():
    """
    Retrieve motion detection events from Firebase.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        # Get query parameters
        limit = int(request.args.get('limit', 100))
        processed_only = request.args.get('processed_only', 'false').lower() == 'true'
        
        # Validate limit
        if limit <= 0 or limit > 1000:
            return jsonify({
                "success": False,
                "message": "Limit must be between 1 and 1000"
            }), 400
        
        detections = firebase_service.get_motion_detections(limit=limit, processed_only=processed_only)
        
        return jsonify({
            "success": True,
            "count": len(detections),
            "data": detections
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error in get motion logs: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Validation error: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Error retrieving motion logs: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/face-detections', methods=['GET'])
def get_face_detections():
    """
    Retrieve face detection events from Firebase.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        # Get query parameters
        limit = int(request.args.get('limit', 100))
        face_type = request.args.get('type', None)  # 'known_face', 'unknown_face', or None for all
        
        # Validate limit
        if limit <= 0 or limit > 1000:
            return jsonify({
                "success": False,
                "message": "Limit must be between 1 and 1000"
            }), 400
        
        # Validate face_type
        if face_type and face_type not in ['known_face', 'unknown_face']:
            return jsonify({
                "success": False,
                "message": "Face type must be 'known_face' or 'unknown_face'"
            }), 400
        
        faces = firebase_service.get_face_detections(limit=limit, face_type=face_type)
        
        return jsonify({
            "success": True,
            "count": len(faces),
            "data": faces
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error in get face detections: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Validation error: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Error retrieving face detections: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/unknown-faces', methods=['GET'])
def get_unknown_faces():
    """
    Retrieve unknown face events from Firebase.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        # Get query parameters
        limit = int(request.args.get('limit', 100))
        status = request.args.get('status', 'pending_review')
        
        # Validate limit
        if limit <= 0 or limit > 1000:
            return jsonify({
                "success": False,
                "message": "Limit must be between 1 and 1000"
            }), 400
        
        faces = firebase_service.get_unknown_faces(limit=limit, status=status)
        
        return jsonify({
            "success": True,
            "count": len(faces),
            "data": faces
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error in get unknown faces: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Validation error: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Error retrieving unknown faces: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/motion-detections/<doc_id>/process', methods=['POST'])
def mark_motion_processed(doc_id):
    """
    Mark a motion detection event as processed.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        if not doc_id:
            return jsonify({
                "success": False,
                "message": "Document ID is required"
            }), 400
        
        success = firebase_service.mark_motion_processed(doc_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Motion detection {doc_id} marked as processed"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to mark motion as processed"
            }), 500
            
    except Exception as e:
        logger.error(f"Error marking motion as processed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/unknown-faces/<doc_id>/process', methods=['POST'])
def mark_face_processed(doc_id):
    """
    Mark an unknown face event as processed.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        if not doc_id:
            return jsonify({
                "success": False,
                "message": "Document ID is required"
            }), 400
        
        # Get JSON data from request
        data = request.get_json() or {}
        status = data.get('status', 'reviewed')
        
        # Validate status
        valid_statuses = ['pending_review', 'reviewed', 'approved', 'rejected']
        if status not in valid_statuses:
            return jsonify({
                "success": False,
                "message": f"Status must be one of: {', '.join(valid_statuses)}"
            }), 400
        
        success = firebase_service.mark_face_processed(doc_id, status)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Unknown face {doc_id} marked as {status}"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to mark face as processed"
            }), 500
            
    except Exception as e:
        logger.error(f"Error marking face as processed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/stream')
def video_stream():
    """Stream camera feed."""
    global streaming_active
    
    # Initialize camera if not already done
    if not streaming_active:
        if not initialize_camera():
            return "Camera initialization failed", 500
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/start', methods=['POST'])
def start_stream():
    """Start camera streaming."""
    global streaming_active
    
    try:
        if not streaming_active:
            if initialize_camera():
                return jsonify({
                    "success": True,
                    "message": "Camera streaming started",
                    "stream_url": "/stream"
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to initialize camera"
                }), 500
        else:
            return jsonify({
                "success": True,
                "message": "Camera streaming already active",
                "stream_url": "/stream"
            }), 200
    except Exception as e:
        logger.error(f"Error starting stream: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error starting stream: {str(e)}"
        }), 500

@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Stop camera streaming."""
    global camera, streaming_active
    
    try:
        streaming_active = False
        if camera is not None:
            camera.stop()
            camera = None
            logger.info("Camera streaming stopped")
        
        return jsonify({
            "success": True,
            "message": "Camera streaming stopped"
        }), 200
    except Exception as e:
        logger.error(f"Error stopping stream: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error stopping stream: {str(e)}"
        }), 500

@app.route('/stream/status', methods=['GET'])
def stream_status():
    """Get streaming status."""
    global streaming_active, camera, face_detection_enabled, motion_detection_enabled
    
    return jsonify({
        "streaming": streaming_active,
        "camera_initialized": camera is not None,
        "stream_url": "/stream" if streaming_active else None,
        "face_detection": face_detection_enabled,
        "motion_detection": motion_detection_enabled,
        "known_faces_count": len(known_face_names) if known_face_names else 0
    }), 200

@app.route('/detection/face/toggle', methods=['POST'])
def toggle_face_detection():
    """Toggle face detection on/off."""
    global face_detection_enabled
    
    try:
        data = request.get_json() or {}
        enable = data.get('enable', not face_detection_enabled)
        face_detection_enabled = enable
        
        return jsonify({
            "success": True,
            "face_detection_enabled": face_detection_enabled,
            "message": f"Face detection {'enabled' if enable else 'disabled'}"
        }), 200
    except Exception as e:
        logger.error(f"Error toggling face detection: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500

@app.route('/detection/motion/toggle', methods=['POST'])
def toggle_motion_detection():
    """Toggle motion detection on/off."""
    global motion_detection_enabled
    
    try:
        data = request.get_json() or {}
        enable = data.get('enable', not motion_detection_enabled)
        motion_detection_enabled = enable
        
        return jsonify({
            "success": True,
            "motion_detection_enabled": motion_detection_enabled,
            "message": f"Motion detection {'enabled' if enable else 'disabled'}"
        }), 200
    except Exception as e:
        logger.error(f"Error toggling motion detection: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500

@app.route('/viewer')
def camera_viewer():
    """Serve camera viewer HTML page."""
    try:
        with open('camera_viewer.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Camera viewer page not found", 404

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        "message": "Face Recognition & Motion Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "motion_detection": "/motion-detection",
            "unknown_face": "/unknown-face",
            "motion_logs": "/motion-logs",
            "face_detections": "/face-detections",
            "unknown_faces": "/unknown-faces",
            "stream": "/stream",
            "stream_start": "/stream/start",
            "stream_stop": "/stream/stop",
            "stream_status": "/stream/status",
            "camera_viewer": "/viewer"
        }
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "message": "Endpoint not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "success": False,
        "message": "Method not allowed"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "success": False,
        "message": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Initialize Firebase on startup
    try:
        firebase_service = get_firebase_service()
        logger.info("Firebase service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase service: {str(e)}")
        firebase_service = None
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
