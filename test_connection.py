#!/usr/bin/env python3
"""
Simple connection test for the Flask server.
"""

import requests
import time
import sys

def test_server_connection():
    """Test if the Flask server is running."""
    try:
        print("🔍 Testing server connection...")
        
        # Test health endpoint
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            data = response.json()
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Firebase Connected: {data.get('firebase_connected', False)}")
            return True
        else:
            print(f"❌ Server responded with status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        print("   Start the server with: python start_server.py")
        return False
    except Exception as e:
        print(f"❌ Error testing connection: {str(e)}")
        return False

def test_stream_endpoint():
    """Test the stream endpoint."""
    try:
        print("🔍 Testing stream endpoint...")
        
        response = requests.get("http://localhost:5000/stream", timeout=10)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if 'multipart/x-mixed-replace' in content_type:
                print("✅ Stream endpoint is working!")
                return True
            else:
                print(f"⚠️ Stream endpoint responded but wrong content type: {content_type}")
                return True
        else:
            print(f"❌ Stream endpoint failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing stream: {str(e)}")
        return False

def main():
    """Run connection tests."""
    print("🧪 Testing Flask Server Connection")
    print("=" * 40)
    
    # Test server connection
    if not test_server_connection():
        print("\n💡 To start the server, run:")
        print("   python start_server.py")
        sys.exit(1)
    
    # Test stream endpoint
    test_stream_endpoint()
    
    print("\n🎉 Server is working!")
    print("📱 Access your camera stream at: http://localhost:5000/stream")
    print("🌐 Or use your Pi's IP: http://[PI_IP]:5000/stream")

if __name__ == "__main__":
    main()
