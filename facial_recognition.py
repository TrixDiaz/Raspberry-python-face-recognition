import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle

# Configuration
DISTANCE_THRESHOLD = 0.4  # Lower = more strict, Higher = more lenient (0.3-0.6 recommended)
MOTION_THRESHOLD = 5000  # Motion detection sensitivity (higher = less sensitive)
MOTION_AREA_THRESHOLD = 1000  # Minimum area for motion detection

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1366, 768)}))
picam2.start()

# Initialize our variables
cv_scaler = 10 # this has to be a whole number

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
    for face_encoding in face_encodings:
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
        else:
            name = "Unknown"
        
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
    
    # Display everything over the video feed.
    cv2.imshow('Video', display_frame)
    
    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# By breaking the loop we run this code here which closes everything
cv2.destroyAllWindows()
picam2.stop()