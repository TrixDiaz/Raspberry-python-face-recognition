# Camera Streaming Guide

## Overview

Your Raspberry Pi camera system supports live streaming via IP address with face detection and motion detection overlays.

## Quick Start

### 1. Start the Flask API Server

```bash
python app.py
```

### 2. Access the Camera Stream

Open your web browser and go to:

```
http://[YOUR_PI_IP_ADDRESS]:5000/stream
```

The camera feed will appear with face detection and motion detection overlays.

## Access Methods

### Direct Stream URL

- **URL**: `http://[PI_IP]:5000/stream`
- **Format**: MJPEG stream with detection overlays
- **Features**: Real-time face detection and motion detection

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
2. Open browser: `http://localhost:5000/stream`
3. Verify camera feed appears with detection overlays

Your Raspberry Pi camera is ready for live streaming! 🎥✨
