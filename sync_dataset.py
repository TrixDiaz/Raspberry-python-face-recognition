#!/usr/bin/env python3
"""
Simple script to sync dataset with Firebase and retrain the model.
This script can be run manually or scheduled to run periodically.
"""

import sys
import logging
from dataset_manager import DatasetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to sync dataset and retrain model."""
    try:
        print("="*60)
        print("FACE RECOGNITION DATASET SYNC")
        print("="*60)
        
        # Initialize dataset manager
        dataset_manager = DatasetManager()
        
        # Perform full sync and retrain
        result = dataset_manager.full_sync_and_retrain()
        
        # Display results
        print("\nSYNC RESULTS:")
        print("-" * 40)
        print(f"✓ Success: {result['success']}")
        print(f"📊 Total faces in Firebase: {result.get('total_faces', 0)}")
        print(f"🔄 Existing faces updated: {result.get('existing_updated', 0)}")
        print(f"➕ New faces added: {result.get('new_faces_added', 0)}")
        print(f"📸 Total images downloaded: {result.get('total_images_downloaded', 0)}")
        print(f"🤖 Model retrained: {result.get('model_retrained', False)}")
        
        if result.get('errors'):
            print(f"\n❌ Errors ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"   • {error}")
        else:
            print("\n✅ No errors encountered!")
        
        print("="*60)
        
        # Return appropriate exit code
        if result['success'] and not result.get('errors'):
            print("🎉 Dataset sync completed successfully!")
            return 0
        else:
            print("⚠️  Dataset sync completed with some issues.")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error during dataset sync: {str(e)}")
        print(f"\n💥 Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)