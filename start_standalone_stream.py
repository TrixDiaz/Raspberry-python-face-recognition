#!/usr/bin/env python3
"""
Startup script for standalone camera streaming.
This script provides an easy way to start the standalone camera stream.
"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are available."""
    required_modules = [
        'cv2', 'picamera2', 'face_recognition', 'numpy'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module} is available")
        except ImportError:
            missing_modules.append(module)
            logger.error(f"❌ {module} is missing")
    
    if missing_modules:
        logger.error(f"Missing required modules: {', '.join(missing_modules)}")
        logger.error("Please install missing dependencies:")
        logger.error("pip install opencv-python picamera2 face-recognition numpy")
        return False
    
    return True

def check_camera():
    """Check if camera is available."""
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        camera_config = camera.create_preview_configuration()
        camera.configure(camera_config)
        camera.start()
        camera.stop()
        camera.close()
        camera = None
        logger.info("✅ Camera is available and working")
        return True
    except Exception as e:
        logger.error(f"❌ Camera check failed: {str(e)}")
        logger.error("This might be because another process is using the camera")
        logger.error("Try running: python camera_cleanup.py")
        return False

def check_face_encodings():
    """Check if face encodings file exists."""
    if os.path.exists("encodings.pickle"):
        logger.info("✅ Face encodings file found - face recognition enabled")
        return True
    else:
        logger.warning("⚠️  Face encodings file not found - face recognition will be disabled")
        logger.warning("To enable face recognition, run the training script first")
        return False

def start_standalone_stream():
    """Start the standalone streaming server."""
    try:
        logger.info("🚀 Starting standalone camera streaming server...")
        logger.info("=" * 60)
        
        # Check dependencies
        if not check_dependencies():
            return False
        
        # Check camera
        if not check_camera():
            logger.error("Camera is not available. This might be because:")
            logger.error("1. Another process is using the camera (like app.py)")
            logger.error("2. Camera hardware issue")
            logger.error("3. Camera permissions issue")
            logger.error("")
            logger.error("Try these solutions:")
            logger.error("1. Stop the main app.py: pkill -f app.py")
            logger.error("2. Run camera cleanup: python camera_cleanup.py")
            logger.error("3. Check camera hardware: libcamera-hello --list-cameras")
            return False
        
        # Check face encodings
        check_face_encodings()
        
        logger.info("=" * 60)
        logger.info("🎥 Starting standalone camera stream...")
        logger.info("📡 Server will run on http://0.0.0.0:8080")
        logger.info("🌐 Open your browser and go to: http://[YOUR_PI_IP]:8080")
        logger.info("📹 Direct stream URL: http://[YOUR_PI_IP]:8080/stream")
        logger.info("📊 Status URL: http://[YOUR_PI_IP]:8080/status")
        logger.info("⚠️  Running in standalone mode - no Firebase/API connectivity")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 60)
        
        # Start the standalone streaming server
        subprocess.run([sys.executable, "standalone_stream.py"])
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down standalone streaming server...")
        logger.info("✅ Server stopped")
    except Exception as e:
        logger.error(f"❌ Error starting standalone server: {str(e)}")
        return False

def main():
    """Main function."""
    print("🎥 Raspberry Pi Standalone Camera Stream")
    print("=" * 50)
    print("This script starts a standalone camera streaming server")
    print("without Flask API or Firebase dependencies.")
    print("=" * 50)
    
    # Get the current IP address
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"📍 Your Pi's IP address: {local_ip}")
        print(f"🌐 Access the stream at: http://{local_ip}:8080")
        print("=" * 50)
    except:
        print("📍 Access the stream at: http://[YOUR_PI_IP]:8080")
        print("=" * 50)
    
    # Start the standalone stream
    start_standalone_stream()

if __name__ == "__main__":
    main()
