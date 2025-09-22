import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
import requests
import json
from datetime import datetime
import logging
from firebase_service import get_firebase_service
from utils import encode_image_to_base64, decode_base64_to_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DISTANCE_THRESHOLD_LEVEL = 4  # 1-10 scale: 1=slow/accurate, 10=fast/lenient
MOTION_THRESHOLD = 15000  # Motion detection sensitivity - HARD (15000 = very strict, only huge changes)
MOTION_AREA_THRESHOLD = 50  # Minimum area for motion detection (HARD MODE - only large movements)

# Convert threshold level to actual distance threshold
# Level 1 = 0.2 (very strict), Level 10 = 0.8 (very lenient)
DISTANCE_THRESHOLD = 0.2 + (DISTANCE_THRESHOLD_LEVEL - 1) * 0.067  # Maps 1-10 to 0.2-0.8

def set_distance_threshold_level(level):
    """
    Set the distance threshold level (1-10).
    
    Args:
        level (int): Threshold level from 1 to 10
                   1 = slow but more accurate (strict matching)
                   10 = fast but less accurate (lenient matching)
    """
    global DISTANCE_THRESHOLD_LEVEL, DISTANCE_THRESHOLD
    if 1 <= level <= 10:
        DISTANCE_THRESHOLD_LEVEL = level
        DISTANCE_THRESHOLD = 0.2 + (level - 1) * 0.067
        logger.info(f"Distance threshold set to level {level} (actual threshold: {DISTANCE_THRESHOLD:.3f})")
    else:
        logger.error(f"Invalid threshold level {level}. Must be between 1 and 10.")

def get_threshold_info():
    """Get current threshold information."""
    return {
        "level": DISTANCE_THRESHOLD_LEVEL,
        "actual_threshold": DISTANCE_THRESHOLD,
        "description": f"Level {DISTANCE_THRESHOLD_LEVEL}: {'Strict' if DISTANCE_THRESHOLD_LEVEL <= 3 else 'Moderate' if DISTANCE_THRESHOLD_LEVEL <= 7 else 'Lenient'}"
    }

# API Configuration
API_BASE_URL = "http://localhost:5000"  # Change this to your Flask server URL
MOTION_COOLDOWN = 10  # Seconds between motion detection reports
UNKNOWN_FACE_COOLDOWN = 10  # Seconds between unknown face reports
KNOWN_FACE_COOLDOWN = 10  # Seconds between known face reports

# Performance optimization settings
FACE_DETECTION_INTERVAL = 3  # Process face detection every N frames (3 = every 3rd frame)
FRAME_SKIP_COUNT = 0  # Counter for frame skipping

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
camera_config["main"]["size"] = (1280, 720)  # More standard resolution
camera_config["main"]["format"] = "RGB888"    # Standard RGB format
picam2.configure(camera_config)
picam2.start()
time.sleep(2)  # Give the camera time to warm up


def check_api_health():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

# Initialize Firebase service
firebase_service = None
api_available = False
firebase_available = False

def check_services_availability():
    """Check if app.py and Firebase services are available."""
    global api_available, firebase_available, firebase_service
    
    # Check if API is running
    api_available = check_api_health()
    
    # Only initialize Firebase if API is available
    if api_available and firebase_service is None:
        try:
            firebase_service = get_firebase_service()
            firebase_available = True
            logger.info("Firebase service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase service: {str(e)}")
            firebase_available = False
    elif not api_available:
        # Disable Firebase if API is not available
        firebase_service = None
        firebase_available = False
        logger.warning("API not available, Firebase service disabled")

# Initial service check
check_services_availability()

# Initialize our variables
cv_scaler = 4  # this has to be a whole number
last_motion_time = 0  # Initialize the last motion detection time

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# Motion detection variables
background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
motion_detected = False


# Cooldown tracking
last_motion_report = 0
last_unknown_face_report = 0
last_known_face_report = {}  # Dictionary to track cooldown per person

def send_motion_detection():
    """Send motion detection to FastAPI backend."""
    global last_motion_report
    
    # Check if API is available
    if not api_available:
        return
    
    current_time = time.time()
    if current_time - last_motion_report < MOTION_COOLDOWN:
        return
    
    try:
        # Send to FastAPI
        motion_data = {
            "timestamp": datetime.now().isoformat(),
            "location": "raspberry_pi_hardware",
            "confidence": 1.0
        }
        
        response = requests.post(
            f"{API_BASE_URL}/motion-detection",
            json=motion_data,
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info("Motion detection sent to API successfully")
            last_motion_report = current_time
        else:
            logger.warning(f"Failed to send motion detection: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error sending motion detection: {str(e)}")
    except Exception as e:
        logger.error(f"Error sending motion detection: {str(e)}")

def send_unknown_face(face_image):
    """Send unknown face detection to FastAPI backend and store in Firebase."""
    global last_unknown_face_report
    
    # Check if Firebase is available
    if not firebase_available or not firebase_service:
        return
    
    current_time = time.time()
    if current_time - last_unknown_face_report < UNKNOWN_FACE_COOLDOWN:
        return
    
    try:
        # Encode face image to base64
        face_image_base64 = firebase_service.encode_image_to_base64(face_image)
        if not face_image_base64:
            logger.error("Failed to encode face image")
            return
        
        # Store directly in Firebase
        success = firebase_service.save_unknown_face(
            face_image_base64=face_image_base64,
            timestamp=datetime.now(),
            location="raspberry_pi_hardware",
            confidence=0.8
        )
        
        if success:
            logger.info("Unknown face saved to Firebase successfully")
            last_unknown_face_report = current_time
        else:
            logger.warning("Failed to save unknown face to Firebase")
            
    except Exception as e:
        logger.error(f"Error saving unknown face: {str(e)}")

def send_known_face(face_image, name, confidence):
    """Send known face detection to Firebase with cooldown."""
    global last_known_face_report
    
    # Check if Firebase is available
    if not firebase_available or not firebase_service:
        return
    
    current_time = time.time()
    
    # Check cooldown for this specific person
    if name in last_known_face_report:
        if current_time - last_known_face_report[name] < KNOWN_FACE_COOLDOWN:
            return
    
    try:
        # Encode face image to base64
        face_image_base64 = firebase_service.encode_image_to_base64(face_image)
        if not face_image_base64:
            logger.error("Failed to encode face image")
            return
        
        # Store directly in Firebase
        success = firebase_service.save_known_face(
            face_image_base64=face_image_base64,
            name=name,
            timestamp=datetime.now(),
            location="raspberry_pi_hardware",
            confidence=confidence
        )
        
        if success:
            logger.info(f"Known face ({name}) saved to Firebase successfully")
            last_known_face_report[name] = current_time
        else:
            logger.warning(f"Failed to save known face ({name}) to Firebase")
            
    except Exception as e:
        logger.error(f"Error saving known face: {str(e)}")

def detect_motion(frame):
    global motion_detected, last_motion_time
    
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
        if area > MOTION_AREA_THRESHOLD:
            motion_detected = True
            if current_time - last_motion_time >= MOTION_COOLDOWN:
                # Send motion detection to Firebase
                if firebase_available and firebase_service:
                    try:
                        motion_frame = frame.copy()
                        motion_base64 = encode_image_to_base64(motion_frame)
                        success = firebase_service.save_motion_detection(
                            timestamp=datetime.now(),
                            location="camera_1",
                            confidence=area / (frame.shape[0] * frame.shape[1]),
                            captured_photo=motion_base64
                        )
                        if success:
                            last_motion_time = current_time
                            logger.info("Motion detection with captured photo sent to Firebase")
                        else:
                            logger.warning("Failed to send motion detection to Firebase")
                    except Exception as e:
                        logger.error(f"Error sending motion detection: {str(e)}")
            break
    
    return motion_detected


def draw_motion_text(frame):
    """Draw motion detection text on the frame."""
    global motion_detected
    
    if motion_detected:
        # Motion text position (upper right, below FPS)
        text = "MOTION"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Position in upper right corner, below FPS counter
        text_x = frame.shape[1] - 120  # Right side
        text_y = 60  # Below FPS counter
        
        # Draw the text (smaller, no background for cleaner look)
        cv2.putText(frame, text, (text_x, text_y), 
                   font, font_scale, (0, 255, 255), thickness)  # Yellow text

def process_frame(frame):
    global face_locations, face_encodings, face_names, FRAME_SKIP_COUNT
    
    # Skip face detection on some frames for performance
    FRAME_SKIP_COUNT += 1
    if FRAME_SKIP_COUNT < FACE_DETECTION_INTERVAL:
        # Return frame with previous face locations for display
        return frame
    
    FRAME_SKIP_COUNT = 0  # Reset counter
    
    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    try:
        face_locations = face_recognition.face_locations(rgb_resized_frame, model='hog')  # Use faster HOG model
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='small')  # Use smaller model
        else:
            face_encodings = []
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        face_locations = []
        face_encodings = []
    
    face_names = []
    
    for i, face_encoding in enumerate(face_encodings):
        # Calculate face distances to all known faces
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Use the configured distance threshold
        distance_threshold = DISTANCE_THRESHOLD
        
        # Find the best match (smallest distance)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        
        # Extract face region from original frame for storage
        if i < len(face_locations):
            top, right, bottom, left = face_locations[i]
            # Scale back up face locations
            top *= cv_scaler
            right *= cv_scaler
            bottom *= cv_scaler
            left *= cv_scaler
            
            # Extract face image
            face_image = frame[top:bottom, left:right]
        
        # Process the face detection result
        if best_distance <= distance_threshold:
            name = known_face_names[best_match_index]
            # Draw a green box for known faces
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Put name text above the face box
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # Store known face in Firebase if face image exists
            if face_image.size > 0:
                confidence = 1.0 - best_distance  # Convert distance to confidence
                send_known_face(face_image, name, confidence)
        else:
            name = "Unknown"
            # Draw a red box for unknown faces
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Put "Unknown" text above the face box
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            
            # Store unknown face in Firebase if face image exists
            if face_image.size > 0:
                send_unknown_face(face_image)
        
        face_names.append(name)
    
    return frame

def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Choose color based on recognition status
        if name == "Unknown":
            box_color = (0, 0, 255)  # Red for unknown
            text_color = (255, 255, 255)  # White text
        else:
            box_color = (0, 255, 0)  # Green for known
            text_color = (0, 0, 0)  # Black text
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, text_color, 1)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

# Main loop
print("[INFO] Starting face recognition with Firebase integration...")
print(f"[INFO] API Base URL: {API_BASE_URL}")
print(f"[INFO] API service: {'Available' if api_available else 'Not available'}")
print(f"[INFO] Firebase service: {'Available' if firebase_available else 'Not available'}")

# Display threshold information
threshold_info = get_threshold_info()
print(f"[INFO] Distance Threshold: {threshold_info['description']} (Level {threshold_info['level']}, Actual: {threshold_info['actual_threshold']:.3f})")
print("[INFO] Threshold Guide: 1=Strict/Slow, 5=Moderate, 10=Lenient/Fast")
print(f"[INFO] Motion Sensitivity: HARD (Area Threshold: {MOTION_AREA_THRESHOLD})")
print("[INFO] Note: API and Firebase services will be disabled if app.py is not running")

# Service check counter
service_check_counter = 0
SERVICE_CHECK_INTERVAL = 30  # Check services every 30 frames

while True:
    # Capture a frame from camera
    frame = picam2.capture_array()
    
    # Periodically check service availability
    service_check_counter += 1
    if service_check_counter >= SERVICE_CHECK_INTERVAL:
        check_services_availability()
        service_check_counter = 0
    
    # Detect motion in the frame
    detect_motion(frame)
    
    # Process the frame with the function
    processed_frame = process_frame(frame)
    
    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)
    
    # Draw motion text if motion is detected
    draw_motion_text(display_frame)
    
    # Calculate and update FPS
    current_fps = calculate_fps()
    
    # Attach FPS counter to the text and boxes
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add API status indicator
    api_status = "API: OK" if api_available else "API: OFFLINE"
    cv2.putText(display_frame, api_status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if api_available else (0, 0, 255), 2)
    
    # Add Firebase status indicator
    firebase_status = "Firebase: OK" if firebase_available else "Firebase: OFFLINE"
    cv2.putText(display_frame, firebase_status, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if firebase_available else (0, 0, 255), 2)
    
    # Add threshold level indicator
    threshold_info = get_threshold_info()
    threshold_text = f"Threshold: {threshold_info['level']} ({threshold_info['description'].split(': ')[1]})"
    cv2.putText(display_frame, threshold_text, (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Add motion sensitivity indicator
    motion_text = f"Motion: HARD (Area: {MOTION_AREA_THRESHOLD})"
    cv2.putText(display_frame, motion_text, (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Display everything over the video feed.
    cv2.imshow('Video', display_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    elif key >= ord("1") and key <= ord("9"):  # Keys 1-9
        new_level = key - ord("0")
        set_distance_threshold_level(new_level)
    elif key == ord("0"):  # Key 0 = level 10
        set_distance_threshold_level(10)
    elif key == ord("h"):  # Help key
        print("\n=== KEYBOARD CONTROLS ===")
        print("1-9: Set threshold level 1-9")
        print("0: Set threshold level 10")
        print("h: Show this help")
        print("q: Quit")
        print("========================")

# By breaking the loop we run this code here which closes everything
cv2.destroyAllWindows()
picam2.stop()
print("[INFO] Face recognition stopped.")
