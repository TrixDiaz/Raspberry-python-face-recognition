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
from gpiozero import LED
from firebase_service import get_firebase_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DISTANCE_THRESHOLD = 0.4  # Lower = more strict, Higher = more lenient (0.3-0.6 recommended)
MOTION_THRESHOLD = 5000  # Motion detection sensitivity (higher = less sensitive)
MOTION_AREA_THRESHOLD = 1000  # Minimum area for motion detection

# API Configuration
API_BASE_URL = "http://localhost:5000"  # Change this to your Flask server URL
MOTION_COOLDOWN = 5  # Seconds between motion detection reports
UNKNOWN_FACE_COOLDOWN = 10  # Seconds between unknown face reports

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

# Initialize GPIO
output = LED(14)

# Initialize Firebase service
firebase_service = None
try:
    firebase_service = get_firebase_service()
    logger.info("Firebase service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firebase service: {str(e)}")

# Initialize our variables
cv_scaler = 4  # this has to be a whole number

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# Motion detection variables
background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
motion_detected = False
bell_icon_alpha = 0.0
bell_fade_speed = 0.05

# List of names that will trigger the GPIO pin
authorized_names = ["trix"]  # Replace with names you wish to authorise THIS IS CASE-SENSITIVE

# Cooldown tracking
last_motion_report = 0
last_unknown_face_report = 0

def send_motion_detection():
    """Send motion detection to FastAPI backend."""
    global last_motion_report
    
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
    """Send unknown face detection to FastAPI backend."""
    global last_unknown_face_report
    
    current_time = time.time()
    if current_time - last_unknown_face_report < UNKNOWN_FACE_COOLDOWN:
        return
    
    try:
        # Encode face image to base64
        face_image_base64 = firebase_service.encode_image_to_base64(face_image)
        if not face_image_base64:
            logger.error("Failed to encode face image")
            return
        
        # Send to FastAPI
        face_data = {
            "face_image_base64": face_image_base64,
            "timestamp": datetime.now().isoformat(),
            "location": "raspberry_pi_hardware",
            "confidence": 0.8
        }
        
        response = requests.post(
            f"{API_BASE_URL}/unknown-face",
            json=face_data,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("Unknown face sent to API successfully")
            last_unknown_face_report = current_time
        else:
            logger.warning(f"Failed to send unknown face: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error sending unknown face: {str(e)}")
    except Exception as e:
        logger.error(f"Error sending unknown face: {str(e)}")

def detect_motion(frame):
    global motion_detected, bell_icon_alpha
    
    # Apply background subtraction
    fg_mask = background_subtractor.apply(frame)
    
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for significant motion
    motion_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MOTION_AREA_THRESHOLD:
            motion_detected = True
            # Send motion detection to API
            send_motion_detection()
            break
    
    # Update bell icon alpha
    if motion_detected:
        bell_icon_alpha = min(1.0, bell_icon_alpha + bell_fade_speed)
    else:
        bell_icon_alpha = max(0.0, bell_icon_alpha - bell_fade_speed)
    
    return motion_detected

def draw_bell_icon(frame):
    global bell_icon_alpha
    
    if bell_icon_alpha > 0:
        # Bell icon position (upper right corner)
        icon_size = 40
        icon_x = frame.shape[1] - icon_size - 20
        icon_y = 20
        
        # Create bell icon (simple circle with line)
        overlay = frame.copy()
        
        # Bell body (circle)
        cv2.circle(overlay, (icon_x + icon_size//2, icon_y + icon_size//2), 
                  icon_size//2 - 2, (0, 255, 255), -1)  # Yellow bell
        cv2.circle(overlay, (icon_x + icon_size//2, icon_y + icon_size//2), 
                  icon_size//2 - 2, (0, 0, 0), 2)  # Black border
        
        # Bell clapper (small circle)
        cv2.circle(overlay, (icon_x + icon_size//2, icon_y + icon_size//2 + 5), 
                  3, (0, 0, 0), -1)
        
        # Bell handle (line)
        cv2.line(overlay, (icon_x + icon_size//2, icon_y + 2), 
                (icon_x + icon_size//2, icon_y + icon_size//2 - 8), (0, 0, 0), 2)
        
        # Apply alpha blending
        cv2.addWeighted(overlay, bell_icon_alpha, frame, 1 - bell_icon_alpha, 0, frame)
        
        # Add "MOTION" text
        if motion_detected:
            cv2.putText(frame, "MOTION", (icon_x - 10, icon_y + icon_size + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def process_frame(frame):
    global face_locations, face_encodings, face_names
    
    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    authorized_face_detected = False
    
    for i, face_encoding in enumerate(face_encodings):
        # Calculate face distances to all known faces
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Use the configured distance threshold
        distance_threshold = DISTANCE_THRESHOLD
        
        # Find the best match (smallest distance)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        
        # Only assign a name if the distance is below the threshold
        if best_distance <= distance_threshold:
            name = known_face_names[best_match_index]
            # Check if the detected face is in our authorized list
            if name in authorized_names:
                authorized_face_detected = True
        else:
            name = "Unknown"
            # Send unknown face to API
            if i < len(face_locations):
                # Extract face region from original frame
                top, right, bottom, left = face_locations[i]
                # Scale back up face locations
                top *= cv_scaler
                right *= cv_scaler
                bottom *= cv_scaler
                left *= cv_scaler
                
                # Extract face image
                face_image = frame[top:bottom, left:right]
                if face_image.size > 0:
                    send_unknown_face(face_image)
        
        face_names.append(name)
    
    # Control the GPIO pin based on face detection
    if authorized_face_detected:
        output.on()  # Turn on Pin
    else:
        output.off()  # Turn off Pin
    
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
        elif name in authorized_names:
            box_color = (0, 255, 0)  # Green for authorized
            text_color = (0, 0, 0)  # Black text
        else:
            box_color = (0, 165, 255)  # Orange for known but not authorized
            text_color = (255, 255, 255)  # White text
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, text_color, 1)
        
        # Add an indicator if the person is authorized
        if name in authorized_names:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), font, 0.6, (0, 255, 0), 1)
        elif name != "Unknown":
            cv2.putText(frame, "Not Authorized", (left + 6, bottom + 23), font, 0.6, (0, 165, 255), 1)
    
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

def check_api_health():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

# Main loop
print("[INFO] Starting face recognition with API integration and GPIO control...")
print(f"[INFO] API Base URL: {API_BASE_URL}")
print(f"[INFO] Firebase service: {'Available' if firebase_service else 'Not available'}")
print(f"[INFO] Authorized names: {authorized_names}")

while True:
    # Capture a frame from camera
    frame = picam2.capture_array()
    
    # Detect motion in the frame
    detect_motion(frame)
    
    # Process the frame with the function
    processed_frame = process_frame(frame)
    
    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)
    
    # Draw bell icon if motion is detected
    draw_bell_icon(display_frame)
    
    # Calculate and update FPS
    current_fps = calculate_fps()
    
    # Attach FPS counter to the text and boxes
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add API status indicator
    api_status = "API: OK" if check_api_health() else "API: OFFLINE"
    cv2.putText(display_frame, api_status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_api_health() else (0, 0, 255), 2)
    
    # Add Firebase status indicator
    firebase_status = "Firebase: OK" if firebase_service else "Firebase: OFFLINE"
    cv2.putText(display_frame, firebase_status, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if firebase_service else (0, 0, 255), 2)
    
    # Add GPIO status indicator
    gpio_status = "GPIO: ON" if output.is_lit else "GPIO: OFF"
    cv2.putText(display_frame, gpio_status, (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if output.is_lit else (0, 0, 255), 2)
    
    # Display everything over the video feed.
    cv2.imshow('Video', display_frame)
    
    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# By breaking the loop we run this code here which closes everything
cv2.destroyAllWindows()
picam2.stop()
output.off()  # Make sure to turn off the GPIO pin when exiting
print("[INFO] Face recognition stopped.")
