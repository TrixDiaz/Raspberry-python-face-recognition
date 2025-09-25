#!/usr/bin/env python3
"""
Test script for mobile streaming endpoints.
This script tests all the mobile streaming endpoints to ensure they work properly.
"""

import requests
import time
import logging
import subprocess
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://localhost:8080"

def test_server_connection():
    """Test if the server is running."""
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Server connection successful")
            return True
        else:
            logger.error(f"❌ Server connection failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Server connection error: {str(e)}")
        return False

def test_api_endpoints():
    """Test API endpoints for mobile integration."""
    endpoints = [
        ("/status", "Status endpoint"),
        ("/api/streams", "Streams API"),
        ("/api/network", "Network API")
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {description} working")
                if endpoint == "/api/network":
                    data = response.json()
                    logger.info(f"   Server IP: {data.get('server_ip', 'unknown')}")
                    logger.info(f"   Base URL: {data.get('base_url', 'unknown')}")
            else:
                logger.error(f"❌ {description} failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ {description} error: {str(e)}")

def test_stream_endpoints():
    """Test streaming endpoints."""
    stream_endpoints = [
        ("/stream", "MJPEG stream"),
        ("/stream/h264", "H.264 stream"),
        ("/stream/mp4", "MP4 stream"),
        ("/stream/rtsp", "RTSP stream")
    ]
    
    for endpoint, description in stream_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                logger.info(f"✅ {description} working")
                logger.info(f"   Content-Type: {content_type}")
                logger.info(f"   Response size: {len(response.content)} bytes")
            else:
                logger.error(f"❌ {description} failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ {description} error: {str(e)}")

def test_cors_headers():
    """Test CORS headers for mobile app integration."""
    try:
        response = requests.get(f"{BASE_URL}/api/streams", timeout=5)
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
        }
        
        logger.info("✅ CORS headers check:")
        for header, value in cors_headers.items():
            if value:
                logger.info(f"   {header}: {value}")
            else:
                logger.warning(f"   {header}: Not set")
                
    except Exception as e:
        logger.error(f"❌ CORS headers test error: {str(e)}")

def test_mobile_specific_features():
    """Test mobile-specific features."""
    try:
        # Test network API
        response = requests.get(f"{BASE_URL}/api/network", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Mobile integration features:")
            logger.info(f"   Server IP: {data.get('server_ip', 'unknown')}")
            logger.info(f"   Port: {data.get('port', 'unknown')}")
            
            stream_urls = data.get('stream_urls', {})
            for stream_type, url in stream_urls.items():
                logger.info(f"   {stream_type.upper()}: {url}")
                
            connection_help = data.get('connection_help', {})
            for platform, instruction in connection_help.items():
                logger.info(f"   {platform}: {instruction}")
        else:
            logger.error(f"❌ Network API failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Mobile features test error: {str(e)}")

def generate_react_native_example():
    """Generate React Native example code."""
    try:
        response = requests.get(f"{BASE_URL}/api/network", timeout=5)
        if response.status_code == 200:
            data = response.json()
            server_ip = data.get('server_ip', '192.168.1.100')
            
            example_code = f'''
// React Native Camera Stream Example
import React from 'react';
import {{ View, Text }} from 'react-native';
import Video from 'react-native-video';

const CameraStream = () => {{
  const streamUrl = "http://{server_ip}:8080/stream/mp4";
  
  return (
    <View style={{{{ flex: 1, backgroundColor: '#000' }}}}>
      <Video
        source={{{{ uri: streamUrl }}}}
        style={{{{ width: '100%', height: 300 }}}}
        controls={{true}}
        resizeMode="contain"
      />
    </View>
  );
}};

export default CameraStream;
'''
            
            logger.info("✅ React Native example code generated:")
            print(example_code)
            
    except Exception as e:
        logger.error(f"❌ Example generation error: {str(e)}")

def main():
    """Run all mobile streaming tests."""
    print("📱 Testing Mobile Streaming Endpoints")
    print("=" * 50)
    
    tests = [
        ("Server Connection", test_server_connection),
        ("API Endpoints", test_api_endpoints),
        ("Stream Endpoints", test_stream_endpoints),
        ("CORS Headers", test_cors_headers),
        ("Mobile Features", test_mobile_specific_features),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All mobile streaming tests passed!")
        print("\n📱 Mobile Integration Ready:")
        print("   • React Native: Use /stream/mp4 endpoint")
        print("   • Android Studio: Use /stream/rtsp endpoint")
        print("   • Mobile Browser: Use /stream/h264 endpoint")
        print("\n🔧 Get connection info:")
        print(f"   curl {BASE_URL}/api/network")
        
        # Generate example code
        print("\n📝 Generating React Native example...")
        generate_react_native_example()
    else:
        print("⚠️  Some tests failed. Please check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    main()
