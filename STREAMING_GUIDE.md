# Camera Streaming Guide

## Overview

Your Raspberry Pi camera system supports live streaming via IP address. Access the camera feed from any device on your network using a web browser.

## Quick Start

### 1. Start the Flask API Server

```bash
python app.py
```

### 2. Access the Camera Stream

Open your web browser and go to:

```
http://[YOUR_PI_IP_ADDRESS]:5000/viewer
```

### 3. Start Streaming

- Click "Start Stream" button
- The camera feed will appear
- Use "Stop Stream" to stop the camera

## Access Methods

### Web Browser (Recommended)

- **URL**: `http://[PI_IP]:5000/viewer`
- **Features**: Web interface with controls
- **Compatibility**: Works on any device with a web browser

### Direct Stream URL

- **URL**: `http://[PI_IP]:5000/stream`
- **Use Case**: Embed in other applications
- **Format**: MJPEG stream

### Mobile Devices

- **iOS**: Safari, Chrome, Firefox
- **Android**: Chrome, Firefox, Samsung Internet
- **Tablets**: All major browsers supported

## Network Configuration

### Finding Your Pi's IP Address

```bash
hostname -I
```

### Port Configuration

- **Default Port**: 5000
- **Protocol**: HTTP
- **Firewall**: Ensure port 5000 is open

### Access from Different Networks

- **Local Network**: `http://192.168.1.XXX:5000/viewer`
- **Same Device**: `http://localhost:5000/viewer`

## Device Information

### Hardware Details

- **Device**: Raspberry Pi v5
- **Camera**: Raspberry Pi Camera Module 3 12MP
- **Model**: RPI-001
- **Streaming Resolution**: 640x480
- **Frame Rate**: ~30 FPS

## Troubleshooting

### Common Issues

#### Camera Not Initializing

- Check if camera is connected properly
- Verify camera permissions
- Restart the Flask application

#### Stream Not Loading

- Ensure Flask app is running
- Check network connection
- Verify IP address and port
- Try refreshing the page

#### Poor Stream Quality

- Check network bandwidth
- Reduce other network usage
- Restart the stream

### Debug Commands

```bash
# Check Flask app status
curl http://localhost:5000/health

# Test stream status
curl http://localhost:5000/stream/status

# Check camera hardware
libcamera-hello --list-cameras
```

## Testing

### Manual Testing

1. Start Flask app: `python app.py`
2. Open browser: `http://localhost:5000/viewer`
3. Click "Start Stream"
4. Verify camera feed appears

Your Raspberry Pi camera is ready for live streaming! 🎥✨
