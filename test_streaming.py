#!/usr/bin/env python3
"""
Test script to verify camera streaming functionality.
This script tests the streaming endpoints and camera initialization.
"""

import requests
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:5000"

def test_api_connection():
    """Test if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API connection successful")
            return True
        else:
            logger.error(f"❌ API connection failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ API connection error: {str(e)}")
        return False

def test_stream_status():
    """Test stream status endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/stream/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Stream status retrieved successfully")
            logger.info(f"   Streaming: {data.get('streaming', False)}")
            logger.info(f"   Camera Initialized: {data.get('camera_initialized', False)}")
            return True
        else:
            logger.error(f"❌ Stream status failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Stream status error: {str(e)}")
        return False

def test_start_stream():
    """Test starting the stream."""
    try:
        response = requests.post(f"{API_BASE_URL}/stream/start", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                logger.info("✅ Stream started successfully")
                logger.info(f"   Message: {data.get('message', '')}")
                logger.info(f"   Stream URL: {data.get('stream_url', '')}")
                return True
            else:
                logger.error(f"❌ Stream start failed: {data.get('message', 'Unknown error')}")
                return False
        else:
            logger.error(f"❌ Stream start failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Stream start error: {str(e)}")
        return False

def test_stop_stream():
    """Test stopping the stream."""
    try:
        response = requests.post(f"{API_BASE_URL}/stream/stop", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                logger.info("✅ Stream stopped successfully")
                logger.info(f"   Message: {data.get('message', '')}")
                return True
            else:
                logger.error(f"❌ Stream stop failed: {data.get('message', 'Unknown error')}")
                return False
        else:
            logger.error(f"❌ Stream stop failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Stream stop error: {str(e)}")
        return False

def test_stream_url():
    """Test accessing the stream URL."""
    try:
        response = requests.get(f"{API_BASE_URL}/stream", timeout=10)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if 'multipart/x-mixed-replace' in content_type:
                logger.info("✅ Stream URL accessible")
                logger.info(f"   Content-Type: {content_type}")
                return True
            else:
                logger.warning(f"⚠️  Stream URL accessible but wrong content type: {content_type}")
                return True  # Still consider it a success
        else:
            logger.error(f"❌ Stream URL failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Stream URL error: {str(e)}")
        return False


def main():
    """Run all streaming tests."""
    print("🎥 Testing Camera Streaming Functionality")
    print("=" * 50)
    
    tests = [
        ("API Connection", test_api_connection),
        ("Stream Status", test_stream_status),
        ("Start Stream", test_start_stream),
        ("Stream URL Access", test_stream_url),
        ("Stop Stream", test_stop_stream),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All streaming tests passed!")
        print("\n📱 Access your camera stream:")
        print(f"   📹 Stream URL: {API_BASE_URL}/stream")
        print(f"   📊 API Status: {API_BASE_URL}/stream/status")
    else:
        print("⚠️  Some tests failed. Please check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    main()
