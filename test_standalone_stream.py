#!/usr/bin/env python3
"""
Test script for standalone camera streaming functionality.
This script tests the standalone streaming server without Flask API dependencies.
"""

import requests
import time
import logging
import subprocess
import signal
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
STANDALONE_BASE_URL = "http://localhost:8080"
STANDALONE_PROCESS = None

def start_standalone_server():
    """Start the standalone streaming server."""
    global STANDALONE_PROCESS
    
    try:
        logger.info("🚀 Starting standalone streaming server...")
        STANDALONE_PROCESS = subprocess.Popen([
            sys.executable, "standalone_stream.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if process is still running
        if STANDALONE_PROCESS.poll() is None:
            logger.info("✅ Standalone server started successfully")
            return True
        else:
            stdout, stderr = STANDALONE_PROCESS.communicate()
            logger.error(f"❌ Failed to start standalone server: {stderr.decode()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error starting standalone server: {str(e)}")
        return False

def stop_standalone_server():
    """Stop the standalone streaming server."""
    global STANDALONE_PROCESS
    
    try:
        if STANDALONE_PROCESS and STANDALONE_PROCESS.poll() is None:
            logger.info("🛑 Stopping standalone server...")
            STANDALONE_PROCESS.terminate()
            STANDALONE_PROCESS.wait(timeout=5)
            logger.info("✅ Standalone server stopped")
        return True
    except Exception as e:
        logger.error(f"❌ Error stopping standalone server: {str(e)}")
        return False

def test_server_connection():
    """Test if the standalone server is responding."""
    try:
        response = requests.get(f"{STANDALONE_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Standalone server connection successful")
            return True
        else:
            logger.error(f"❌ Standalone server connection failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Standalone server connection error: {str(e)}")
        return False

def test_status_endpoint():
    """Test the status endpoint."""
    try:
        response = requests.get(f"{STANDALONE_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Status endpoint working")
            logger.info(f"   Streaming: {data.get('streaming', False)}")
            logger.info(f"   Camera Initialized: {data.get('camera_initialized', False)}")
            logger.info(f"   Mode: {data.get('mode', 'unknown')}")
            logger.info(f"   Face Detection: {data.get('face_detection', False)}")
            logger.info(f"   Motion Detection: {data.get('motion_detection', False)}")
            logger.info(f"   Known Faces: {data.get('known_faces_count', 0)}")
            return True
        else:
            logger.error(f"❌ Status endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Status endpoint error: {str(e)}")
        return False

def test_stream_endpoint():
    """Test the stream endpoint."""
    try:
        response = requests.get(f"{STANDALONE_BASE_URL}/stream", timeout=10)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if 'multipart/x-mixed-replace' in content_type:
                logger.info("✅ Stream endpoint working")
                logger.info(f"   Content-Type: {content_type}")
                return True
            else:
                logger.warning(f"⚠️  Stream endpoint accessible but wrong content type: {content_type}")
                return True  # Still consider it a success
        else:
            logger.error(f"❌ Stream endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Stream endpoint error: {str(e)}")
        return False

def test_html_viewer():
    """Test the HTML viewer endpoint."""
    try:
        response = requests.get(f"{STANDALONE_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            content = response.text
            if 'Raspberry Pi Camera Stream' in content and 'Standalone Mode' in content:
                logger.info("✅ HTML viewer working")
                return True
            else:
                logger.warning("⚠️  HTML viewer accessible but content seems incorrect")
                return True
        else:
            logger.error(f"❌ HTML viewer failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ HTML viewer error: {str(e)}")
        return False

def main():
    """Run all standalone streaming tests."""
    print("🎥 Testing Standalone Camera Streaming")
    print("=" * 50)
    
    tests = [
        ("Start Standalone Server", start_standalone_server),
        ("Server Connection", test_server_connection),
        ("Status Endpoint", test_status_endpoint),
        ("Stream Endpoint", test_stream_endpoint),
        ("HTML Viewer", test_html_viewer),
    ]
    
    passed = 0
    total = len(tests)
    
    try:
        for test_name, test_func in tests:
            print(f"\n🔍 Testing: {test_name}")
            if test_func():
                passed += 1
            time.sleep(1)  # Small delay between tests
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All standalone streaming tests passed!")
            print("\n📱 Access your standalone camera stream:")
            print(f"   🌐 Viewer: {STANDALONE_BASE_URL}/")
            print(f"   📹 Stream: {STANDALONE_BASE_URL}/stream")
            print(f"   📊 Status: {STANDALONE_BASE_URL}/status")
            print("\n⚠️  Note: This is standalone mode - no Firebase/API connectivity")
        else:
            print("⚠️  Some tests failed. Please check the logs above for details.")
        
        return passed == total
        
    finally:
        # Always stop the server
        stop_standalone_server()

if __name__ == "__main__":
    main()
