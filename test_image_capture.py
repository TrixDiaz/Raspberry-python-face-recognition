#!/usr/bin/env python3
"""
Test script to verify image capture and Firebase integration functionality.
This script tests both motion detection and face detection image capture.
"""

import cv2
import numpy as np
from datetime import datetime
import logging
from firebase_service import get_firebase_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_firebase_connection():
    """Test Firebase connection and basic functionality."""
    try:
        firebase_service = get_firebase_service()
        if firebase_service and firebase_service.db:
            logger.info("✅ Firebase connection successful")
            return True
        else:
            logger.error("❌ Firebase connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ Firebase connection error: {str(e)}")
        return False

def test_image_encoding():
    """Test image encoding functionality."""
    try:
        firebase_service = get_firebase_service()
        
        # Create a test image (100x100 red square)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [0, 0, 255]  # Red color
        
        # Test encoding
        encoded = firebase_service.encode_image_to_base64(test_image)
        if encoded:
            logger.info("✅ Image encoding successful")
            logger.info(f"   Encoded length: {len(encoded)} characters")
            return True
        else:
            logger.error("❌ Image encoding failed")
            return False
    except Exception as e:
        logger.error(f"❌ Image encoding error: {str(e)}")
        return False

def test_motion_detection_save():
    """Test motion detection with image save to Firebase."""
    try:
        firebase_service = get_firebase_service()
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:, :] = [50, 50, 50]  # Dark gray
        
        # Encode image
        encoded_image = firebase_service.encode_image_to_base64(test_image)
        if not encoded_image:
            logger.error("❌ Failed to encode test image")
            return False
        
        # Device information
        device_info = {
            "device_name": "Raspberry Pi v5",
            "camera": "Raspberry Pi Camera Module 3 12MP",
            "model": "RPI-001"
        }
        
        # Save motion detection
        success = firebase_service.save_motion_detection(
            timestamp=datetime.now(),
            location="test_camera",
            confidence=0.95,
            captured_photo=encoded_image,
            device_info=device_info
        )
        
        if success:
            logger.info("✅ Motion detection with image saved successfully")
            return True
        else:
            logger.error("❌ Failed to save motion detection")
            return False
    except Exception as e:
        logger.error(f"❌ Motion detection save error: {str(e)}")
        return False

def test_unknown_face_save():
    """Test unknown face detection with image save to Firebase."""
    try:
        firebase_service = get_firebase_service()
        
        # Create a test face image
        test_face = np.zeros((150, 150, 3), dtype=np.uint8)
        test_face[:, :] = [100, 100, 100]  # Gray face
        
        # Encode image
        encoded_face = firebase_service.encode_image_to_base64(test_face)
        if not encoded_face:
            logger.error("❌ Failed to encode test face image")
            return False
        
        # Device information
        device_info = {
            "device_name": "Raspberry Pi v5",
            "camera": "Raspberry Pi Camera Module 3 12MP",
            "model": "RPI-001"
        }
        
        # Save unknown face
        success = firebase_service.save_unknown_face(
            face_image_base64=encoded_face,
            timestamp=datetime.now(),
            location="test_camera",
            confidence=0.8,
            device_info=device_info
        )
        
        if success:
            logger.info("✅ Unknown face with image saved successfully")
            return True
        else:
            logger.error("❌ Failed to save unknown face")
            return False
    except Exception as e:
        logger.error(f"❌ Unknown face save error: {str(e)}")
        return False

def test_known_face_save():
    """Test known face detection with image save to Firebase."""
    try:
        firebase_service = get_firebase_service()
        
        # Create a test face image
        test_face = np.zeros((150, 150, 3), dtype=np.uint8)
        test_face[:, :] = [0, 255, 0]  # Green face
        
        # Encode image
        encoded_face = firebase_service.encode_image_to_base64(test_face)
        if not encoded_face:
            logger.error("❌ Failed to encode test face image")
            return False
        
        # Device information
        device_info = {
            "device_name": "Raspberry Pi v5",
            "camera": "Raspberry Pi Camera Module 3 12MP",
            "model": "RPI-001"
        }
        
        # Save known face
        success = firebase_service.save_known_face(
            face_image_base64=encoded_face,
            name="Test_Person",
            timestamp=datetime.now(),
            location="test_camera",
            confidence=0.95,
            device_info=device_info
        )
        
        if success:
            logger.info("✅ Known face with image saved successfully")
            return True
        else:
            logger.error("❌ Failed to save known face")
            return False
    except Exception as e:
        logger.error(f"❌ Known face save error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Image Capture and Firebase Integration")
    print("=" * 50)
    
    tests = [
        ("Firebase Connection", test_firebase_connection),
        ("Image Encoding", test_image_encoding),
        ("Motion Detection Save", test_motion_detection_save),
        ("Unknown Face Save", test_unknown_face_save),
        ("Known Face Save", test_known_face_save)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Image capture and Firebase integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    main()
