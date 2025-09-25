# Standalone Camera Streaming Guide

## Overview

This guide explains how to use the standalone camera streaming functionality that works independently of the main Flask API and Firebase services. This is useful when you want camera streaming to work even when the main `app.py` is not running.

## Features

- ✅ **Independent Operation**: Works without Flask API or Firebase
- ✅ **Camera Streaming**: Live camera feed with MJPEG streaming
- ✅ **Face Detection**: Recognizes known and unknown faces (no Firebase upload)
- ✅ **Motion Detection**: Detects motion in the camera feed (no Firebase upload)
- ✅ **Web Interface**: Built-in HTML viewer with status information
- ✅ **Status API**: Simple HTTP endpoints for monitoring
- ✅ **Lightweight**: Minimal dependencies, fast startup

## Quick Start

### 1. Start Standalone Streaming (Recommended)

```bash
python run_standalone_safe.py
```

This script automatically handles camera conflicts and starts streaming safely.

### 2. Alternative: Manual Start

```bash
python start_standalone_stream.py
```

### 3. Access the Camera Stream

Open your web browser and go to:

```
http://[YOUR_PI_IP_ADDRESS]:8080
```

The camera feed will appear with face detection and motion detection overlays.

## Files

- `standalone_stream.py` - Main standalone streaming server
- `run_standalone_safe.py` - **Recommended**: Safe launcher with camera conflict handling
- `start_standalone_stream.py` - Manual startup script
- `camera_cleanup.py` - Camera process cleanup utility
- `test_standalone_stream.py` - Test script for functionality
- `STANDALONE_STREAMING_GUIDE.md` - This guide

## Usage

### Method 1: Safe Launcher (Recommended)

```bash
python run_standalone_safe.py
```

This script will:

- Automatically kill any existing camera processes
- Check camera availability
- Start the standalone streaming server safely
- Handle camera conflicts automatically

### Method 2: Manual Startup

```bash
python start_standalone_stream.py
```

This script will:

- Check all dependencies
- Verify camera functionality
- Start the standalone streaming server
- Display connection information

### Method 3: Direct Execution

```bash
python standalone_stream.py
```

### Method 4: Camera Cleanup

```bash
python camera_cleanup.py
```

### Method 5: Testing

```bash
python test_standalone_stream.py
```

## Access Methods

### Web Viewer

- **URL**: `http://[PI_IP]:8080/`
- **Features**: Full web interface with status information
- **Compatible**: All modern browsers, mobile devices

### Direct Stream

- **URL**: `http://[PI_IP]:8080/stream`
- **Format**: MJPEG stream
- **Use Case**: Embed in other applications

### Status API

- **URL**: `http://[PI_IP]:8080/status`
- **Format**: JSON response
- **Use Case**: Monitoring and automation

## Configuration

### Port Configuration

- **Default Port**: 8080
- **Protocol**: HTTP
- **Host**: 0.0.0.0 (all interfaces)

### Camera Settings

- **Resolution**: 640x480
- **Format**: RGB888
- **Frame Rate**: ~30 FPS
- **Quality**: 85% JPEG compression

### Detection Settings

- **Face Detection**: Enabled by default
- **Motion Detection**: Enabled by default
- **Face Recognition**: Enabled if `encodings.pickle` exists

## Network Access

### Finding Your Pi's IP Address

```bash
hostname -I
```

### Access from Different Networks

- **Local Network**: `http://192.168.1.XXX:8080`
- **Same Device**: `http://localhost:8080`
- **Mobile Devices**: Use the Pi's IP address

## Features Comparison

| Feature          | Main App (app.py) | Standalone Stream |
| ---------------- | ----------------- | ----------------- |
| Camera Streaming | ✅                | ✅                |
| Face Detection   | ✅                | ✅                |
| Motion Detection | ✅                | ✅                |
| Firebase Upload  | ✅                | ❌                |
| API Endpoints    | ✅                | Limited           |
| Web Interface    | ❌                | ✅                |
| Dependencies     | High              | Low               |
| Startup Time     | Slow              | Fast              |

## Troubleshooting

### Common Issues

#### Camera Not Initializing / Pipeline Handler in Use

This error occurs when another process is already using the camera. Solutions:

**Quick Fix (Recommended):**

```bash
python run_standalone_safe.py
```

**Manual Fix:**

```bash
# Stop the main app.py
pkill -f app.py

# Run camera cleanup
python camera_cleanup.py

# Start standalone streaming
python start_standalone_stream.py
```

**Check Camera Hardware:**

```bash
# Check camera hardware
libcamera-hello --list-cameras

# Check camera permissions
sudo usermod -a -G video $USER
```

#### Port Already in Use

```bash
# Check what's using port 8080
sudo netstat -tlnp | grep :8080

# Kill process if needed
sudo kill -9 [PID]
```

#### Dependencies Missing

```bash
# Install required packages
pip install opencv-python picamera2 face-recognition numpy
```

### Debug Commands

```bash
# Test camera functionality
python -c "from picamera2 import Picamera2; print('Camera OK')"

# Test standalone server
python test_standalone_stream.py

# Check server status
curl http://localhost:8080/status
```

## API Endpoints

### GET /

Returns the HTML viewer interface.

### GET /stream

Returns the MJPEG camera stream.

### GET /status

Returns JSON status information:

```json
{
  "streaming": true,
  "camera_initialized": true,
  "face_detection": true,
  "motion_detection": true,
  "known_faces_count": 5,
  "mode": "standalone",
  "timestamp": "2024-01-01T12:00:00"
}
```

## Integration

### Embedding in Other Applications

```html
<img src="http://[PI_IP]:8080/stream" alt="Camera Stream" />
```

### Monitoring with Scripts

```bash
# Check if server is running
curl -s http://localhost:8080/status | grep -o '"streaming":true'
```

## Security Notes

- ⚠️ **No Authentication**: The standalone server has no authentication
- ⚠️ **Local Network Only**: Ensure firewall rules are configured
- ⚠️ **No HTTPS**: Uses HTTP only (not suitable for production)

## Performance

- **CPU Usage**: Low (optimized for Raspberry Pi)
- **Memory Usage**: Minimal
- **Network**: ~1-2 Mbps for 640x480 stream
- **Latency**: <100ms typical

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running, or:

```bash
# Find and kill the process
ps aux | grep standalone_stream.py
kill [PID]
```

Your standalone camera streaming is ready! 🎥✨
