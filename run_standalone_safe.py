#!/usr/bin/env python3
"""
Safe Standalone Streaming Launcher
This script automatically handles camera conflicts and starts standalone streaming safely.
"""

import subprocess
import sys
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def kill_existing_processes():
    """Kill any existing camera-related processes."""
    logger.info("🔍 Checking for existing camera processes...")
    
    try:
        # Kill app.py if running
        result = subprocess.run(['pkill', '-f', 'app.py'], capture_output=True)
        if result.returncode == 0:
            logger.info("✅ Stopped app.py process")
        else:
            logger.info("ℹ️  No app.py process found")
        
        # Kill any other camera processes
        result = subprocess.run(['pkill', '-f', 'picamera'], capture_output=True)
        if result.returncode == 0:
            logger.info("✅ Stopped picamera processes")
        
        # Wait for cleanup
        time.sleep(2)
        return True
        
    except Exception as e:
        logger.error(f"Error killing processes: {str(e)}")
        return False

def check_camera_availability():
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
        logger.info("✅ Camera is available")
        return True
    except Exception as e:
        logger.error(f"❌ Camera not available: {str(e)}")
        return False

def start_standalone_stream():
    """Start the standalone streaming server."""
    try:
        logger.info("🚀 Starting standalone streaming server...")
        
        # Run the standalone stream
        process = subprocess.run([sys.executable, "standalone_stream.py"])
        return process.returncode == 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Stopping standalone streaming...")
        return True
    except Exception as e:
        logger.error(f"❌ Error starting standalone stream: {str(e)}")
        return False

def main():
    """Main function."""
    print("🎥 Safe Standalone Camera Stream Launcher")
    print("=" * 50)
    print("This script automatically handles camera conflicts")
    print("and starts standalone streaming safely.")
    print("=" * 50)
    
    # Step 1: Kill existing processes
    logger.info("Step 1: Cleaning up existing camera processes...")
    kill_existing_processes()
    
    # Step 2: Check camera availability
    logger.info("Step 2: Checking camera availability...")
    if not check_camera_availability():
        logger.error("❌ Camera is not available after cleanup")
        logger.error("Please check:")
        logger.error("1. Camera hardware connection")
        logger.error("2. Camera permissions: sudo usermod -a -G video $USER")
        logger.error("3. Try rebooting the Raspberry Pi")
        return False
    
    # Step 3: Start standalone streaming
    logger.info("Step 3: Starting standalone streaming...")
    print("\n🎥 Starting standalone camera stream...")
    print("📡 Server will run on http://0.0.0.0:8080")
    print("🌐 Open your browser and go to: http://[YOUR_PI_IP]:8080")
    print("⚠️  Running in standalone mode - no Firebase/API connectivity")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    return start_standalone_stream()

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Standalone streaming completed successfully")
        else:
            print("\n❌ Standalone streaming failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Standalone streaming stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
