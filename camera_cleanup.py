#!/usr/bin/env python3
"""
Camera Cleanup Script
This script helps clean up any existing camera processes that might be blocking
the standalone streaming functionality.
"""

import subprocess
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_camera_processes():
    """Find processes that might be using the camera."""
    try:
        # Find Python processes that might be using the camera
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout.split('\n')
        
        camera_processes = []
        for process in processes:
            if any(keyword in process.lower() for keyword in ['app.py', 'picamera', 'camera', 'stream']):
                if 'python' in process.lower():
                    parts = process.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        command = ' '.join(parts[10:])
                        camera_processes.append((pid, command))
        
        return camera_processes
    except Exception as e:
        logger.error(f"Error finding processes: {str(e)}")
        return []

def kill_process(pid):
    """Kill a process by PID."""
    try:
        subprocess.run(['kill', '-TERM', pid], check=True)
        logger.info(f"Sent TERM signal to process {pid}")
        time.sleep(1)
        
        # Check if process is still running
        result = subprocess.run(['ps', '-p', pid], capture_output=True)
        if result.returncode == 0:
            # Process still running, force kill
            subprocess.run(['kill', '-KILL', pid], check=True)
            logger.info(f"Force killed process {pid}")
        else:
            logger.info(f"Process {pid} terminated successfully")
        return True
    except subprocess.CalledProcessError:
        logger.warning(f"Could not kill process {pid}")
        return False
    except Exception as e:
        logger.error(f"Error killing process {pid}: {str(e)}")
        return False

def cleanup_camera_processes():
    """Clean up camera-related processes."""
    logger.info("🔍 Looking for camera-related processes...")
    
    processes = find_camera_processes()
    
    if not processes:
        logger.info("✅ No camera-related processes found")
        return True
    
    logger.info(f"Found {len(processes)} camera-related processes:")
    for pid, command in processes:
        logger.info(f"  PID {pid}: {command}")
    
    # Ask for confirmation
    print("\n⚠️  Found camera-related processes that might be blocking the camera.")
    print("These processes will be terminated to allow standalone streaming.")
    print("\nProcesses to terminate:")
    for pid, command in processes:
        print(f"  PID {pid}: {command}")
    
    response = input("\nDo you want to terminate these processes? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        logger.info("Terminating camera processes...")
        success_count = 0
        
        for pid, command in processes:
            if kill_process(pid):
                success_count += 1
        
        logger.info(f"✅ Terminated {success_count}/{len(processes)} processes")
        
        # Wait a moment for cleanup
        time.sleep(2)
        
        return success_count > 0
    else:
        logger.info("❌ Process termination cancelled")
        return False

def check_camera_availability():
    """Check if camera is available after cleanup."""
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        camera_config = camera.create_preview_configuration()
        camera.configure(camera_config)
        camera.start()
        camera.stop()
        camera.close()
        camera = None
        logger.info("✅ Camera is now available")
        return True
    except Exception as e:
        logger.error(f"❌ Camera still not available: {str(e)}")
        return False

def main():
    """Main cleanup function."""
    print("🧹 Camera Cleanup Script")
    print("=" * 40)
    print("This script helps clean up camera processes that might")
    print("be blocking the standalone streaming functionality.")
    print("=" * 40)
    
    # Find and clean up processes
    if cleanup_camera_processes():
        logger.info("🔍 Checking camera availability...")
        if check_camera_availability():
            logger.info("🎉 Camera cleanup successful! You can now run standalone streaming.")
            print("\n✅ Camera is now available for standalone streaming!")
            print("Run: python start_standalone_stream.py")
        else:
            logger.error("❌ Camera still not available after cleanup")
            print("\n❌ Camera cleanup failed. Camera may still be in use.")
            print("Try rebooting the Raspberry Pi if the issue persists.")
    else:
        logger.info("ℹ️  No processes were terminated")
        print("\nℹ️  No camera processes were terminated.")
        print("You can try running the standalone stream directly.")

if __name__ == "__main__":
    main()
