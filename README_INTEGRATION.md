# Face Recognition with Firebase Integration

This project integrates your existing face recognition system with Firebase for data storage and a Flask API for communication.

## Features

- **Motion Detection**: Automatically detects motion and sends alerts to Firebase
- **Unknown Face Detection**: Captures and stores unknown faces in Firebase
- **Real-time Processing**: Non-blocking data transmission to avoid network errors
- **GPIO Control**: Hardware version includes GPIO pin control for authorized users
- **Flask API**: RESTful API for managing detection events

## Files Overview

### Core Files

- `firebase_service.py` - Firebase integration service
- `app.py` - Flask API server
- `start_server.py` - Server startup script

### Face Recognition Scripts

- `facial_recognition_integrated.py` - Basic face recognition with API integration
- `facial_recognition_hardware_integrated.py` - Hardware version with GPIO control

### Original Files (unchanged)

- `facial_recognition.py` - Original basic version
- `facial_recognition_hardware.py` - Original hardware version
- `image_capture.py` - Image capture utility
- `model_training.py` - Model training script

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure your `firebase-config.json` file is in the project directory

3. Make sure you have your trained model files:
   - `encodings.pickle` (from model training)

## Usage

### Starting the Flask Server

1. Start the Flask API server:

```bash
python start_server.py
```

The server will start on `http://localhost:5000` by default.

### Running Face Recognition

#### Basic Version (without GPIO)

```bash
python facial_recognition_integrated.py
```

#### Hardware Version (with GPIO control)

```bash
python facial_recognition_hardware_integrated.py
```

## API Endpoints

### Health Check

- `GET /health` - Check server and Firebase connection status

### Motion Detection

- `POST /motion-detection` - Report motion detection event
- `GET /motion-detections` - Retrieve motion detection history
- `POST /motion-detections/{id}/process` - Mark motion event as processed

### Unknown Face Detection

- `POST /unknown-face` - Report unknown face detection
- `GET /unknown-faces` - Retrieve unknown face history
- `POST /unknown-faces/{id}/process` - Mark face event as processed

## Configuration

### API Configuration

Edit the following variables in the face recognition scripts:

```python
API_BASE_URL = "http://localhost:5000"  # Flask server URL
MOTION_COOLDOWN = 5  # Seconds between motion reports
UNKNOWN_FACE_COOLDOWN = 10  # Seconds between unknown face reports
```

### Face Recognition Configuration

```python
DISTANCE_THRESHOLD = 0.4  # Face recognition sensitivity
MOTION_AREA_THRESHOLD = 1000  # Motion detection sensitivity
```

### Hardware Configuration (hardware version only)

```python
authorized_names = ["trix"]  # Names that trigger GPIO pin
```

## Firebase Data Structure

### Motion Detections Collection

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "location": "raspberry_pi",
  "confidence": 1.0,
  "type": "motion_detection",
  "processed": false
}
```

### Unknown Faces Collection

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "location": "raspberry_pi",
  "confidence": 0.8,
  "type": "unknown_face",
  "face_image": "base64_encoded_image",
  "processed": false,
  "status": "pending_review"
}
```

## Error Handling

The system includes comprehensive error handling:

- Network timeouts and connection errors
- Firebase service unavailability
- Invalid data validation
- Graceful degradation when services are offline

## Status Indicators

The face recognition interface shows:

- **API Status**: Connection to Flask server
- **Firebase Status**: Connection to Firebase
- **GPIO Status**: Hardware pin state (hardware version only)
- **FPS Counter**: Real-time performance metrics

## Troubleshooting

### Common Issues

1. **Firebase Connection Failed**

   - Check `firebase-config.json` file exists and is valid
   - Verify Firebase project permissions

2. **API Connection Failed**

   - Ensure Flask server is running on correct port
   - Check network connectivity

3. **No Face Recognition**

   - Verify `encodings.pickle` file exists
   - Check camera permissions and hardware

4. **GPIO Not Working** (hardware version)
   - Verify GPIO permissions
   - Check pin configuration

### Logs

Check the console output for detailed error messages and status information.

## Security Notes

- The Flask server runs with CORS enabled for development
- Firebase credentials should be kept secure
- Consider implementing authentication for production use
- GPIO pins should be properly configured for your hardware setup

## Performance

- Motion detection has a 5-second cooldown to prevent spam
- Unknown face detection has a 10-second cooldown
- Face images are compressed before transmission
- Non-blocking API calls prevent UI freezing
