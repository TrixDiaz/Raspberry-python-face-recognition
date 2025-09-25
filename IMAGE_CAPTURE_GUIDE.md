# Image Capture and Firebase Integration Guide

## Overview

This system now captures images automatically when motion or face detection occurs and stores them in Firebase Firestore database. Both motion detection and face detection (known and unknown faces) will capture and send images to the database.

## Features Implemented

### 1. Motion Detection with Image Capture

- **When**: Motion is detected above the threshold
- **What**: Captures the full camera frame at the moment of motion
- **Storage**: Saves to Firebase with timestamp, location, confidence, and captured image
- **Cooldown**: 10 seconds between motion detections to prevent spam

### 2. Face Detection with Image Capture

- **Known Faces**: Captures face region when a known person is recognized
- **Unknown Faces**: Captures face region when an unknown person is detected
- **Storage**: Saves to Firebase with face image, timestamp, location, and confidence
- **Cooldown**: 10 seconds per person to prevent duplicate entries

## How It Works

### Motion Detection Flow

1. Camera continuously monitors for motion using background subtraction
2. When motion is detected above threshold:
   - Captures the current frame
   - Encodes image to base64
   - Saves to Firebase with metadata
   - Logs success/failure

### Face Detection Flow

1. Camera continuously scans for faces
2. When a face is detected:
   - Extracts face region from frame
   - Attempts to match with known faces
   - If known: saves with person's name
   - If unknown: saves as unknown face
   - Encodes face image to base64
   - Saves to Firebase with metadata

## Firebase Collections

### Motion Logs Collection (`motion_logs`)

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "location": "raspberry_pi_camera",
  "confidence": 0.95,
  "type": "motion_detection",
  "captured_photo": "base64_encoded_image_data",
  "processed": false,
  "device_info": {
    "device_name": "Raspberry Pi v5",
    "camera": "Raspberry Pi Camera Module 3 12MP",
    "model": "RPI-001"
  }
}
```

### Face Detections Collection (`face_detections`)

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "location": "raspberry_pi_camera",
  "confidence": 0.95,
  "type": "known_face", // or "unknown_face"
  "name": "John_Doe", // only for known faces
  "face_image": "base64_encoded_face_data",
  "processed": false,
  "status": "recognized", // or "pending_review" for unknown faces
  "device_info": {
    "device_name": "Raspberry Pi v5",
    "camera": "Raspberry Pi Camera Module 3 12MP",
    "model": "RPI-001"
  }
}
```

## Configuration

### Motion Detection Settings

- **Motion Threshold**: 15000 (area threshold for motion detection)
- **Motion Area Threshold**: 50 (minimum area for motion detection)
- **Cooldown**: 10 seconds between motion reports

### Face Detection Settings

- **Distance Threshold**: Configurable (1-10 scale)
- **Face Detection Interval**: Every 3rd frame for performance
- **Cooldown**: 10 seconds per person

## Testing

Run the test script to verify functionality:

```bash
python test_image_capture.py
```

This will test:

- Firebase connection
- Image encoding
- Motion detection save
- Unknown face save
- Known face save

## API Endpoints

### Motion Detection

- **POST** `/motion-detection` - Report motion with optional image
- **GET** `/motion-logs` - Retrieve motion detection logs

### Face Detection

- **POST** `/unknown-face` - Report unknown face with image
- **GET** `/face-detections` - Retrieve face detection logs
- **GET** `/unknown-faces` - Retrieve unknown face logs

## Usage

### Starting the System

1. Start the Flask API server:

   ```bash
   python app.py
   ```

2. Start the face recognition service:
   ```bash
   python face_detection_service.py
   ```

### Monitoring

- The system displays real-time status on the camera feed
- API status: Shows if the API server is running
- Firebase status: Shows if Firebase is connected
- Threshold level: Shows current face recognition threshold
- Motion sensitivity: Shows current motion detection settings

### Keyboard Controls (when running face_detection_service.py)

- **1-9**: Set face recognition threshold (1=strict, 9=lenient)
- **0**: Set threshold to 10 (most lenient)
- **h**: Show help
- **q**: Quit

## Troubleshooting

### Common Issues

1. **Firebase Connection Failed**

   - Check `firebase-config.json` exists and is valid
   - Verify Firebase project credentials
   - Check internet connection

2. **Images Not Saving**

   - Check Firebase service initialization
   - Verify image encoding is working
   - Check Firebase permissions

3. **High CPU Usage**
   - Reduce face detection interval
   - Lower camera resolution
   - Increase motion threshold

### Logs

- Check console output for detailed logs
- Motion detection: "Motion detection with captured photo sent to Firebase successfully"
- Face detection: "Known/Unknown face with captured image saved to Firebase successfully"

## Performance Optimization

### For Better Performance

- Use lower camera resolution
- Increase face detection interval
- Use higher motion threshold
- Enable frame skipping

### For Better Accuracy

- Use higher camera resolution
- Lower motion threshold
- Use stricter face recognition threshold
- Ensure good lighting conditions

## Security Considerations

- Images are stored as base64 in Firebase
- Consider implementing image compression for large images
- Set up proper Firebase security rules
- Consider data retention policies
- Monitor storage usage in Firebase

## Next Steps

1. **Image Compression**: Implement image compression to reduce storage
2. **Cloud Storage**: Move images to Firebase Storage instead of Firestore
3. **Real-time Alerts**: Set up real-time notifications for detections
4. **Analytics**: Add detection analytics and reporting
5. **Mobile App**: Create mobile app to view detections and images
