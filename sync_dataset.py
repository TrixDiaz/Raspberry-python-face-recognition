#!/usr/bin/env python3
"""
Simple script to sync dataset with Firebase by fetching all documents
and storing sample images in the local dataset.
"""

import sys
import os
import logging
import base64
import cv2
import numpy as np
from datetime import datetime
from firebase_service import get_firebase_service
from utils import decode_base64_to_image

# Set environment variables to avoid ALTS credentials issues on Raspberry Pi
os.environ['GRPC_DNS_RESOLVER'] = 'native'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleDatasetSync:
    def __init__(self, dataset_path="dataset"):
        """Initialize the simple dataset sync."""
        self.dataset_path = dataset_path
        self.firebase_service = get_firebase_service()
        self.ensure_dataset_directory()
    
    def ensure_dataset_directory(self):
        """Ensure the dataset directory exists."""
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            logger.info(f"Created dataset directory: {self.dataset_path}")
    
    def get_all_collections(self):
        """Get all collections from Firebase."""
        try:
            if not self.firebase_service or not self.firebase_service.db:
                logger.error("Firebase service not available")
                return []
            
            # Get all collections
            collections = self.firebase_service.db.collections()
            collection_names = [col.id for col in collections]
            logger.info(f"Found collections: {collection_names}")
            return collection_names
            
        except Exception as e:
            logger.error(f"Failed to get collections: {str(e)}")
            return []
    
    def fetch_all_documents_with_sample_images(self):
        """Fetch all documents from all collections that have sample_image field."""
        if not self.firebase_service or not self.firebase_service.db:
            logger.error("Firebase service not available")
            return {}
        
        try:
            logger.info("Starting to fetch all documents with sample images...")
            
            all_documents = {}
            total_docs = 0
            docs_with_images = 0
            
            # Get all collections
            collections = self.get_all_collections()
            
            for collection_name in collections:
                logger.info(f"Processing collection: {collection_name}")
                collection_ref = self.firebase_service.db.collection(collection_name)
                
                # Get all documents from this collection
                docs = collection_ref.stream()
                collection_docs = []
                
                for doc in docs:
                    total_docs += 1
                    doc_data = doc.to_dict()
                    
                    # Check if document has sample_image field
                    sample_image = doc_data.get('sample_image')
                    if sample_image:
                        docs_with_images += 1
                        
                        # Extract name from document (try different possible fields)
                        name = (doc_data.get('name') or 
                               doc_data.get('person_name') or 
                               doc_data.get('label') or 
                               doc_data.get('id') or 
                               f"person_{doc.id}")
                        
                        collection_docs.append({
                            'id': doc.id,
                            'name': name,
                            'sample_image': sample_image,
                            'timestamp': doc_data.get('timestamp'),
                            'collection': collection_name
                        })
                        
                        logger.debug(f"Found document with sample image: {name} in {collection_name}")
                
                if collection_docs:
                    all_documents[collection_name] = collection_docs
                    logger.info(f"Collection {collection_name}: {len(collection_docs)} documents with sample images")
            
            logger.info(f"Total documents processed: {total_docs}")
            logger.info(f"Documents with sample images: {docs_with_images}")
            logger.info(f"Collections with sample images: {list(all_documents.keys())}")
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to fetch documents: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def download_image_from_base64(self, image_base64, filename):
        """Download and save a base64 encoded image to the dataset."""
        try:
            # Decode base64 to image
            image_array = decode_base64_to_image(image_base64)
            if image_array is None:
                logger.error(f"Failed to decode base64 image for {filename}")
                return False
            
            # Save image
            cv2.imwrite(filename, image_array)
            logger.info(f"Saved image: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save image {filename}: {str(e)}")
            return False
    
    def create_placeholder_image(self, filename, person_name):
        """Create a placeholder image when actual images are not available."""
        try:
            # Create a simple placeholder image
            img = np.ones((200, 200, 3), dtype=np.uint8) * 128
            
            # Add text to the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"No Image\n{person_name}"
            text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
            text_x = (200 - text_size[0]) // 2
            text_y = (200 + text_size[1]) // 2
            
            cv2.putText(img, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2)
            
            # Save the placeholder image
            cv2.imwrite(filename, img)
            logger.info(f"Created placeholder image: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create placeholder image {filename}: {str(e)}")
            return False
    
    def sync_documents_to_dataset(self, all_documents):
        """Sync all documents with sample images to the dataset."""
        logger.info("Starting to sync documents to dataset...")
        
        sync_summary = {
            "success": True,
            "total_documents": 0,
            "documents_with_images": 0,
            "images_downloaded": 0,
            "failed_downloads": 0,
            "persons_processed": 0,
            "errors": []
        }
        
        # Track processed persons to avoid duplicates
        processed_persons = set()
        
        for collection_name, documents in all_documents.items():
            logger.info(f"Processing collection: {collection_name} ({len(documents)} documents)")
            
            for doc in documents:
                try:
                    sync_summary["total_documents"] += 1
                    sync_summary["documents_with_images"] += 1
                    
                    person_name = doc['name']
                    sample_image = doc['sample_image']
                    
                    # Create person directory
                    person_dir = os.path.join(self.dataset_path, person_name)
                    if not os.path.exists(person_dir):
                        os.makedirs(person_dir)
                        logger.info(f"Created directory for {person_name}")
                    
                    # Create unique filename based on collection and document ID
                    filename = os.path.join(person_dir, f"{collection_name}_{doc['id']}.jpg")
                    
                    # Download image
                    if self.download_image_from_base64(sample_image, filename):
                        sync_summary["images_downloaded"] += 1
                        
                        # Track unique persons processed
                        if person_name not in processed_persons:
                            processed_persons.add(person_name)
                            sync_summary["persons_processed"] += 1
                    else:
                        sync_summary["failed_downloads"] += 1
                        
                        # Create placeholder if download failed
                        placeholder_filename = os.path.join(person_dir, f"placeholder_{doc['id']}.jpg")
                        if self.create_placeholder_image(placeholder_filename, person_name):
                            sync_summary["images_downloaded"] += 1
                            
                            if person_name not in processed_persons:
                                processed_persons.add(person_name)
                                sync_summary["persons_processed"] += 1
                    
                    logger.debug(f"Processed document {doc['id']} for {person_name}")
                    
                except Exception as e:
                    error_msg = f"Error processing document {doc.get('id', 'unknown')}: {str(e)}"
                    logger.error(error_msg)
                    sync_summary["errors"].append(error_msg)
                    sync_summary["failed_downloads"] += 1
        
        logger.info(f"Sync completed. Summary: {sync_summary}")
        return sync_summary
    
    def retrain_model(self):
        """Trigger model retraining after dataset update."""
        try:
            # Import and run model training
            from model_training import main as train_model
            train_model()
            logger.info("Model retraining completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retrain model: {str(e)}")
            return False
    
    def full_sync_and_retrain(self):
        """Perform full dataset sync and model retraining."""
        logger.info("Starting full sync operation...")
        
        # Fetch all documents with sample images
        all_documents = self.fetch_all_documents_with_sample_images()
        
        if not all_documents:
            logger.warning("No documents with sample images found")
            return {
                "success": False, 
                "message": "No documents with sample images found",
                "model_retrained": False
            }
        
        # Sync documents to dataset
        sync_result = self.sync_documents_to_dataset(all_documents)
        
        # Retrain model if images were downloaded
        if sync_result["images_downloaded"] > 0:
            logger.info("Images downloaded, retraining model...")
            retrain_success = self.retrain_model()
            sync_result["model_retrained"] = retrain_success
        else:
            sync_result["model_retrained"] = False
            logger.info("No images downloaded, skipping model retraining")
        
        sync_result["success"] = sync_result["images_downloaded"] > 0
        return sync_result

def main():
    """Main function to sync dataset and retrain model."""
    try:
        print("="*60)
        print("FACE RECOGNITION DATASET SYNC")
        print("="*60)
        
        # Initialize simple dataset sync
        dataset_sync = SimpleDatasetSync()
        
        # Perform full sync and retrain
        result = dataset_sync.full_sync_and_retrain()
        
        # Display results
        print("\nSYNC RESULTS:")
        print("-" * 40)
        print(f"✓ Success: {result['success']}")
        print(f"📊 Total documents: {result.get('total_documents', 0)}")
        print(f"📸 Documents with images: {result.get('documents_with_images', 0)}")
        print(f"⬇️  Images downloaded: {result.get('images_downloaded', 0)}")
        print(f"👥 Persons processed: {result.get('persons_processed', 0)}")
        print(f"❌ Failed downloads: {result.get('failed_downloads', 0)}")
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