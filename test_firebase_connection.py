#!/usr/bin/env python3
"""
Test script to verify Firebase connection and face retrieval.
Run this script to test if the ALTS credentials issue is fixed and all faces are retrieved.
"""

import os
import sys
import logging

# Set environment variables to avoid ALTS credentials issues on Raspberry Pi
os.environ['GRPC_DNS_RESOLVER'] = 'native'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_firebase_connection():
    """Test Firebase connection and face retrieval."""
    try:
        print("="*60)
        print("TESTING FIREBASE CONNECTION AND FACE RETRIEVAL")
        print("="*60)
        
        # Import after setting environment variables
        from dataset_manager import DatasetManager
        
        # Initialize dataset manager
        print("Initializing DatasetManager...")
        dataset_manager = DatasetManager()
        
        if not dataset_manager.firebase_service or not dataset_manager.firebase_service.db:
            print("❌ Firebase service not available")
            return False
        
        print("✅ Firebase service initialized successfully")
        
        # Test face retrieval
        print("\nTesting face retrieval...")
        faces_data = dataset_manager.get_all_known_faces_from_detections()
        
        if not faces_data:
            print("⚠️  No faces found with primary method, trying alternative...")
            faces_data = dataset_manager.get_all_known_faces_alternative()
        
        if faces_data:
            print(f"✅ Successfully retrieved {len(faces_data)} unique faces:")
            for name, images in faces_data.items():
                print(f"   - {name}: {len(images)} images")
        else:
            print("❌ No faces found in Firebase")
            return False
        
        # Test a small sync operation
        print("\nTesting small sync operation...")
        sync_result = dataset_manager.sync_dataset_with_firebase()
        
        if sync_result.get('success'):
            print("✅ Sync operation completed successfully")
            print(f"   - Total faces: {sync_result.get('total_faces', 0)}")
            print(f"   - New faces added: {sync_result.get('new_faces_added', 0)}")
            print(f"   - Existing faces updated: {sync_result.get('existing_updated', 0)}")
            print(f"   - Total images downloaded: {sync_result.get('total_images_downloaded', 0)}")
        else:
            print(f"❌ Sync operation failed: {sync_result.get('message', 'Unknown error')}")
            return False
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("The ALTS credentials issue should be fixed and all faces should be retrieved.")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_firebase_connection()
    sys.exit(0 if success else 1)
