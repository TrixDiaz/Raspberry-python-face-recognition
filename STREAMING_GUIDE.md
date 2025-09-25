# Camera Streaming Guide

## Overview

Your Raspberry Pi camera system now supports live streaming via IP address. You can access the camera feed from any device on your network using a web browser.

## Features

### 🎥 Live Camera Streaming

- **Resolution**: 640x480 (optimized for streaming)
- **Frame Rate**: ~30 FPS
- **Format**: MJPEG (Motion JPEG)
- **Access**: Via web browser or direct stream URL

### 🌐 Web Interface

- **Camera Viewer**: Beautiful web interface to view and control the camera
- **Device Information**: Shows Raspberry Pi v5 and Camera Module 3 12MP details
- **Stream Controls**: Start/stop streaming with one click
- **Status Monitoring**: Real-time streaming status

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

## API Endpoints

### Streaming Endpoints

| Endpoint         | Method | Description                  |
| ---------------- | ------ | ---------------------------- |
| `/stream`        | GET    | Direct camera stream (MJPEG) |
| `/stream/start`  | POST   | Start camera streaming       |
| `/stream/stop`   | POST   | Stop camera streaming        |
| `/stream/status` | GET    | Get streaming status         |
| `/viewer`        | GET    | Web interface for camera     |

### Example API Usage

#### Start Streaming

```bash
curl -X POST http://localhost:5000/stream/start
```

#### Check Status

```bash
curl http://localhost:5000/stream/status
```

#### Stop Streaming

```bash
curl -X POST http://localhost:5000/stream/stop
```

## Access Methods

### 1. Web Browser (Recommended)

- **URL**: `http://[PI_IP]:5000/viewer`
- **Features**: Full web interface with controls
- **Compatibility**: Works on any device with a web browser

### 2. Direct Stream URL

- **URL**: `http://[PI_IP]:5000/stream`
- **Use Case**: Embed in other applications
- **Format**: MJPEG stream

### 3. Mobile Devices

- **iOS**: Safari, Chrome, Firefox
- **Android**: Chrome, Firefox, Samsung Internet
- **Tablets**: All major browsers supported

## Network Configuration

### Finding Your Pi's IP Address

```bash
# Method 1: Using hostname
hostname -I

# Method 2: Using ifconfig
ifconfig wlan0 | grep inet

# Method 3: Using ip command
ip addr show wlan0
```

### Port Configuration

- **Default Port**: 5000
- **Protocol**: HTTP
- **Firewall**: Ensure port 5000 is open

### Access from Different Networks

- **Local Network**: `http://192.168.1.XXX:5000/viewer`
- **Same Device**: `http://localhost:5000/viewer`
- **Remote Access**: Configure port forwarding on your router

## Device Information

### Hardware Details

- **Device**: Raspberry Pi v5
- **Camera**: Raspberry Pi Camera Module 3 12MP
- **Model**: RPI-001
- **Streaming Resolution**: 640x480
- **Frame Rate**: ~30 FPS

### Performance Optimization

- **Lower Resolution**: 640x480 for better performance
- **JPEG Quality**: 85% for good quality/size balance
- **Frame Rate**: Limited to ~30 FPS for stability

## Troubleshooting

### Common Issues

#### 1. Camera Not Initializing

**Symptoms**: "Camera initialization failed" error
**Solutions**:

- Check if camera is connected properly
- Verify camera permissions
- Restart the Flask application
- Check if another process is using the camera

#### 2. Stream Not Loading

**Symptoms**: Black screen or "Camera Not Available"
**Solutions**:

- Ensure Flask app is running
- Check network connection
- Verify IP address and port
- Try refreshing the page

#### 3. Poor Stream Quality

**Symptoms**: Choppy or low-quality video
**Solutions**:

- Check network bandwidth
- Reduce other network usage
- Restart the stream
- Check Pi's CPU usage

#### 4. Connection Timeout

**Symptoms**: Stream stops after a while
**Solutions**:

- The web interface auto-refreshes every 30 seconds
- Manually refresh the page
- Check if the Flask app is still running

### Debug Commands

#### Check Flask App Status

```bash
curl http://localhost:5000/health
```

#### Test Stream Status

```bash
curl http://localhost:5000/stream/status
```

#### Check Camera Hardware

```bash
# List video devices
ls /dev/video*

# Test camera with libcamera
libcamera-hello --list-cameras
```

## Security Considerations

### Network Security

- **Local Network Only**: By default, only accessible on local network
- **Firewall**: Configure firewall rules as needed
- **Authentication**: Consider adding authentication for production use

### Privacy

- **Camera Access**: Only authorized users should access the stream
- **Recording**: Stream is not recorded by default
- **Logs**: Check logs for unauthorized access attempts

## Advanced Usage

### Embedding in Other Applications

```html
<img src="http://[PI_IP]:5000/stream" alt="Camera Stream" />
```

### Using with VLC Media Player

1. Open VLC Media Player
2. Go to Media → Open Network Stream
3. Enter: `http://[PI_IP]:5000/stream`
4. Click Play

### Mobile App Integration

- Use the stream URL in mobile apps
- Implement custom controls using the API endpoints
- Add authentication as needed

## Performance Monitoring

### System Resources

- **CPU Usage**: Monitor with `htop` or `top`
- **Memory Usage**: Check with `free -h`
- **Network Usage**: Monitor with `iftop`

### Stream Quality

- **Frame Rate**: Should be around 30 FPS
- **Latency**: Typically 1-3 seconds
- **Bandwidth**: ~1-2 Mbps for 640x480 stream

## Testing

### Run Streaming Tests

```bash
python test_streaming.py
```

This will test:

- API connection
- Stream status
- Start/stop functionality
- Stream URL access
- Web interface

### Manual Testing

1. Start Flask app: `python app.py`
2. Open browser: `http://localhost:5000/viewer`
3. Click "Start Stream"
4. Verify camera feed appears
5. Test controls (start/stop/status)

## Next Steps

### Enhancements

1. **Authentication**: Add login system
2. **Recording**: Add video recording capability
3. **Multiple Cameras**: Support multiple camera streams
4. **Mobile App**: Create dedicated mobile app
5. **Cloud Streaming**: Stream to cloud services

### Integration

1. **Home Assistant**: Integrate with home automation
2. **Security Systems**: Connect to security monitoring
3. **IoT Platforms**: Connect to IoT platforms
4. **WebRTC**: Implement WebRTC for better performance

## Support

### Logs

- Check Flask application logs for errors
- Monitor system logs: `journalctl -u your-service`
- Check camera logs: `dmesg | grep camera`

### Common Commands

```bash
# Restart Flask app
pkill -f "python app.py"
python app.py &

# Check camera status
libcamera-hello --list-cameras

# Test network connectivity
ping [PI_IP_ADDRESS]

# Check port availability
netstat -tlnp | grep 5000
```

Your Raspberry Pi camera is now ready for live streaming! 🎥✨
