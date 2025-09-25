# Raspberry Pi Face Recognition & Motion Detection System

A comprehensive face recognition and motion detection system designed for Raspberry Pi, featuring Firebase integration for data storage and a Flask API for real-time monitoring and management.

## 🚀 Features

- **Real-time Face Recognition**: Detect and identify known faces using computer vision
- **Motion Detection**: Advanced motion detection with configurable sensitivity
- **Firebase Integration**: Cloud storage for face detections and motion logs
- **REST API**: Complete API for data retrieval and management
- **Standalone Streaming**: Independent camera streaming without API dependencies
- **Dataset Management**: Automated dataset synchronization and model training
- **Configurable Thresholds**: Adjustable face recognition and motion detection sensitivity

## 🎯 Two Operating Modes

### 1. **Full System Mode** (Main App)

- Complete Flask API with Firebase integration
- Real-time data upload to cloud storage
- Full REST API endpoints for data management
- Camera streaming with cloud connectivity

### 2. **Standalone Mode** (Independent Streaming)

- Camera streaming without Flask API
- Face detection and motion detection (no cloud upload)
- Lightweight HTTP server for streaming
- Works independently when main app is not running

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [API Endpoints](#api-endpoints)
- [Data Structures](#data-structures)
- [Setup Instructions](#setup-instructions)
- [Configuration](#configuration)
- [Usage](#usage)
- [Firebase Collections](#firebase-collections)
- [Face Recognition Values](#face-recognition-values)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Option 1: Full System Mode (Recommended for Production)

**Start the complete system with Firebase integration:**

```bash
# 1. Start the main Flask API
python app.py

# 2. In another terminal, start face detection
python face_detection_service.py
```

**Access:**

- **API**: `http://[YOUR_PI_IP]:5000`
- **Camera Stream**: `http://[YOUR_PI_IP]:5000/stream`
- **Health Check**: `http://[YOUR_PI_IP]:5000/health`

### Option 2: Standalone Mode (Quick Camera Streaming)

**Start independent camera streaming without API/Firebase:**

```bash
# Safe launcher (handles camera conflicts automatically)
python run_standalone_safe.py
```

**Access:**

- **Web Viewer**: `http://[YOUR_PI_IP]:8080`
- **MJPEG Stream**: `http://[YOUR_PI_IP]:8080/stream` (Desktop browsers)
- **H.264 Stream**: `http://[YOUR_PI_IP]:8080/stream/h264` (Mobile devices)
- **Status API**: `http://[YOUR_PI_IP]:8080/status`

### Option 3: Manual Standalone (If camera conflicts occur)

```bash
# Stop main app first
pkill -f app.py

# Run camera cleanup
python camera_cleanup.py

# Start standalone streaming
python start_standalone_stream.py
```

## 🏗️ System Architecture

The system consists of several key components:

1. **Face Detection Service** (`face_detection_service.py`) - Main camera processing and face recognition
2. **Flask API** (`app.py`) - REST API for data management
3. **Firebase Service** (`firebase_service.py`) - Cloud database integration
4. **Dataset Manager** (`dataset_manager.py`) - Local dataset synchronization
5. **Model Training** (`model_training.py`) - Face recognition model training
6. **Sync Service** (`sync_dataset.py`) - Bidirectional data synchronization

## 🔌 API Endpoints

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "firebase_connected": true,
  "version": "1.0.0"
}
```

### Motion Detection

```http
POST /motion-detection
```

**Request Body:**

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "location": "camera_1",
  "confidence": 0.95
}
```

**Response:**

```json
{
  "success": true,
  "message": "Motion detection event saved successfully",
  "data": {
    "timestamp": "2024-01-15T10:30:00.000Z",
    "location": "camera_1",
    "confidence": 0.95
  }
}
```

### Unknown Face Detection

```http
POST /unknown-face
```

**Request Body:**

```json
{
  "face_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "location": "camera_1",
  "confidence": 0.8
}
```

**Response:**

```json
{
  "success": true,
  "message": "Unknown face event saved successfully",
  "data": {
    "timestamp": "2024-01-15T10:30:00.000Z",
    "location": "camera_1",
    "confidence": 0.8
  }
}
```

### Get Motion Logs

```http
GET /motion-logs?limit=100&processed_only=false
```

**Query Parameters:**

- `limit` (optional): Number of records to retrieve (1-1000, default: 100)
- `processed_only` (optional): Filter processed events (true/false, default: false)

**Response:**

```json
{
  "success": true,
  "count": 25,
  "data": [
    {
      "timestamp": "2024-01-15T10:30:00.000Z",
      "location": "camera_1",
      "confidence": 0.95,
      "type": "motion_detection",
      "processed": false
    }
  ]
}
```

### Get Face Detections

```http
GET /face-detections?limit=100&type=known_face
```

**Query Parameters:**

- `limit` (optional): Number of records to retrieve (1-1000, default: 100)
- `type` (optional): Filter by type (`known_face`, `unknown_face`, or omit for all)

**Response:**

```json
{
  "success": true,
  "count": 15,
  "data": [
    {
      "timestamp": "2024-01-15T10:30:00.000Z",
      "location": "camera_1",
      "confidence": 0.92,
      "type": "known_face",
      "name": "John Doe",
      "face_image": "data:image/jpeg;base64,/9j/4AAQ...",
      "processed": false,
      "status": "recognized"
    }
  ]
}
```

### Get Unknown Faces

```http
GET /unknown-faces?limit=100&status=pending_review
```

**Query Parameters:**

- `limit` (optional): Number of records to retrieve (1-1000, default: 100)
- `status` (optional): Filter by status (`pending_review`, `reviewed`, `approved`, `rejected`, default: `pending_review`)

### Mark Motion as Processed

```http
POST /motion-detections/{doc_id}/process
```

**Response:**

```json
{
  "success": true,
  "message": "Motion detection {doc_id} marked as processed"
}
```

### Mark Face as Processed

```http
POST /unknown-faces/{doc_id}/process
```

**Request Body:**

```json
{
  "status": "reviewed"
}
```

**Valid status values:** `pending_review`, `reviewed`, `approved`, `rejected`

## 📊 Data Structures

### Motion Detection Document

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "location": "camera_1",
  "confidence": 0.95,
  "type": "motion_detection",
  "processed": false
}
```

### Known Face Detection Document

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "location": "camera_1",
  "confidence": 0.92,
  "type": "known_face",
  "name": "John Doe",
  "face_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "processed": false,
  "status": "recognized"
}
```

### Unknown Face Detection Document

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "location": "camera_1",
  "confidence": 0.8,
  "type": "unknown_face",
  "face_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "processed": false,
  "status": "pending_review"
}
```

## 🔧 Setup Instructions

### Prerequisites

- Raspberry Pi with camera module
- Python 3.7+
- Firebase project with Firestore enabled

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Firebase Configuration

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Firestore Database
3. Generate a service account key:
   - Go to Project Settings > Service Accounts
   - Click "Generate new private key"
   - Save the JSON file as `firebase-config.json`

### 3. Camera Setup

Ensure your Raspberry Pi camera is enabled:

```bash
sudo raspi-config
# Navigate to Interfacing Options > Camera > Enable
```

### 4. Initial Dataset Setup

1. Create a `dataset` folder in your project directory
2. Add person folders with images:
   ```
   dataset/
   ├── john_doe/
   │   ├── john_doe_001.jpg
   │   ├── john_doe_002.jpg
   │   └── ...
   ├── jane_smith/
   │   ├── jane_smith_001.jpg
   │   └── ...
   ```

### 5. Train Initial Model

```bash
python model_training.py
```

## ⚙️ Configuration

### Face Recognition Thresholds

The system uses configurable distance thresholds for face matching:

- **Level 1**: Very strict (0.2 threshold) - Slow but accurate
- **Level 5**: Moderate (0.47 threshold) - Balanced
- **Level 10**: Lenient (0.8 threshold) - Fast but less accurate

**Runtime Controls:**

- Press `1-9` keys to set threshold levels 1-9
- Press `0` key to set threshold level 10
- Press `h` key to show help
- Press `q` key to quit

### Motion Detection Settings

```python
# In face_detection_service.py
MOTION_THRESHOLD = 15000  # Motion detection sensitivity
MOTION_AREA_THRESHOLD = 50  # Minimum area for motion detection
```

### Performance Optimization

```python
# Frame processing settings
FACE_DETECTION_INTERVAL = 3  # Process every 3rd frame
cv_scaler = 4  # Scale factor for face detection (higher = faster)
```

## 🚀 Usage

### Full System Mode (Production)

**Start the complete system:**

```bash
# 1. Start the Flask API server
python app.py

# 2. In another terminal, start face detection service
python face_detection_service.py
```

**Features:**

- Complete REST API at `http://[YOUR_PI_IP]:5000`
- Camera streaming at `http://[YOUR_PI_IP]:5000/stream`
- Firebase integration for data storage
- Real-time face recognition and motion detection
- Full data management capabilities

### Standalone Mode (Quick Streaming)

**Start independent camera streaming:**

```bash
# Recommended: Safe launcher with automatic conflict resolution
python run_standalone_safe.py

# Alternative: Manual startup
python start_standalone_stream.py

# Direct execution
python standalone_stream.py
```

**Features:**

- Lightweight HTTP server at `http://[YOUR_PI_IP]:8080`
- Camera streaming without Flask API
- **Mobile Support**: H.264 encoding for mobile devices
- **Desktop Support**: MJPEG encoding for desktop browsers
- Face detection and motion detection (no cloud upload)
- Works independently when main app is not running
- Built-in web viewer interface with automatic mobile detection

### Dataset Management

**Sync dataset with Firebase:**

```bash
python sync_dataset.py
```

**Full sync and retrain:**

```bash
python dataset_manager.py
```

### Camera Conflict Resolution

**If you get "pipeline handler in use" errors:**

```bash
# Quick fix
python run_standalone_safe.py

# Manual fix
pkill -f app.py
python camera_cleanup.py
python start_standalone_stream.py
```

## 📱 Mobile Streaming Support

The standalone mode includes enhanced mobile support with H.264 encoding:

### **Stream Types:**

- **MJPEG Stream**: `/stream` - For desktop browsers (higher bandwidth)
- **H.264 Stream**: `/stream/h264` - For mobile devices (better compression)

### **Mobile Features:**

- Automatic mobile device detection
- H.264 video encoding for better mobile performance
- Lower bandwidth usage on mobile networks
- Touch-friendly web interface
- Responsive design for all screen sizes

### **Requirements for H.264:**

```bash
# Install FFmpeg for H.264 encoding
sudo apt install ffmpeg
```

### **Mobile Access:**

- **Web Viewer**: `http://[YOUR_PI_IP]:8080` (auto-detects mobile)
- **Direct H.264**: `http://[YOUR_PI_IP]:8080/stream/h264`
- **Direct MJPEG**: `http://[YOUR_PI_IP]:8080/stream`

## 📊 Mode Comparison

| Feature              | Full System Mode | Standalone Mode |
| -------------------- | ---------------- | --------------- |
| **Camera Streaming** | ✅               | ✅              |
| **Face Detection**   | ✅               | ✅              |
| **Motion Detection** | ✅               | ✅              |
| **Firebase Upload**  | ✅               | ❌              |
| **REST API**         | ✅               | Limited         |
| **Web Interface**    | ❌               | ✅              |
| **Dependencies**     | High             | Low             |
| **Startup Time**     | Slow             | Fast            |
| **Resource Usage**   | High             | Low             |
| **Use Case**         | Production       | Quick Demo      |

## 🗄️ Firebase Collections

### motion_logs

Stores motion detection events.

**Fields:**

- `timestamp`: DateTime when motion was detected
- `location`: String identifier for camera location
- `confidence`: Float (0.0-1.0) confidence level
- `type`: String "motion_detection"
- `processed`: Boolean indicating if event has been processed

### face_detections

Stores both known and unknown face detection events.

**Fields:**

- `timestamp`: DateTime when face was detected
- `location`: String identifier for camera location
- `confidence`: Float (0.0-1.0) confidence level
- `type`: String "known_face" or "unknown_face"
- `face_image`: Base64 encoded image data
- `processed`: Boolean indicating if event has been processed
- `status`: String status (varies by type)
- `name`: String name (only for known faces)

### faces

Collection storing face data with sample images and metadata.

**Fields:**

- `name`: String name of the person (e.g., "josh")
- `sample_image`: Base64 encoded image data (data:image/jpeg;base64,...)
- `image_path`: String path to the image file (e.g., "faces_kmnK4Er3qWzNToprTsmq.jpg")
- `timestamp`: DateTime when face was captured/detected
- `processed`: Boolean indicating if the face data has been processed
- `source`: String indicating data source (e.g., "local_capture")

**Example Document:**

```json
{
  "name": "josh",
  "sample_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAA...",
  "image_path": "faces_kmnK4Er3qWzNToprTsmq.jpg",
  "timestamp": "September 21, 2025 at 5:55:52 AM UTC+8",
  "processed": false,
  "source": "local_capture"
}
```

## 👤 Face Recognition Values

### Confidence Scores

- **1.0**: Perfect match (very rare)
- **0.9-0.99**: Excellent match
- **0.8-0.89**: Good match
- **0.7-0.79**: Fair match
- **0.6-0.69**: Poor match
- **Below 0.6**: Likely not a match

### Distance Thresholds

The system uses Euclidean distance for face matching:

- **Distance 0.0-0.2**: Very strict matching
- **Distance 0.2-0.4**: Strict matching
- **Distance 0.4-0.6**: Moderate matching
- **Distance 0.6-0.8**: Lenient matching
- **Distance >0.8**: Very lenient matching

### Face Detection Models

- **HOG (Histogram of Oriented Gradients)**: Faster, less accurate
- **CNN (Convolutional Neural Network)**: Slower, more accurate

### Image Processing

- **Input Resolution**: 1280x720
- **Face Detection Scale**: 1/4 (320x180)
- **Face Encoding Model**: Small model for speed
- **Image Format**: RGB888

## 🔍 Troubleshooting

### Common Issues

1. **Camera not detected**

   ```bash
   # Check camera connection
   vcgencmd get_camera
   # Should return: supported=1 detected=1
   ```

2. **"Pipeline handler in use" error (Standalone Mode)**

   ```bash
   # Quick fix - automatic conflict resolution
   python run_standalone_safe.py

   # Manual fix
   pkill -f app.py
   python camera_cleanup.py
   python start_standalone_stream.py
   ```

3. **Firebase connection issues (Full System Mode)**

   - Verify `firebase-config.json` is present and valid
   - Check Firestore rules allow read/write access
   - Ensure service account has proper permissions

4. **Low face recognition accuracy**

   - Increase dataset size (more images per person)
   - Adjust distance threshold (try levels 1-3 for stricter matching)
   - Ensure good lighting and image quality

5. **Performance issues**

   - Increase `FACE_DETECTION_INTERVAL` to process fewer frames
   - Increase `cv_scaler` to reduce processing resolution
   - Use HOG model instead of CNN for face detection

6. **Standalone mode not starting**

   ```bash
   # Check for camera conflicts
   python camera_cleanup.py

   # Verify camera availability
   python -c "from picamera2 import Picamera2; print('Camera OK')"
   ```

### Logging

All services use Python logging. Check console output for detailed error messages and status information.

### API Testing

Test API endpoints using curl:

```bash
# Health check
curl http://localhost:5000/health

# Get motion logs
curl "http://localhost:5000/motion-logs?limit=10"
```

## 📝 License

This project is open source. Please ensure you comply with all applicable laws and regulations when using face recognition technology.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review Firebase and camera setup
3. Check system logs for detailed error messages
