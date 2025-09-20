#!/usr/bin/env python3
"""
Bidirectional sync script for Firebase and local dataset.
- Uploads local captured images to Firebase
- Downloads Firebase documents with sample images to local dataset
- Automatically retrains the face recognition model
"""

import sys
import os
import logging
import base64
import cv2
import numpy as np
from datetime import datetime
from firebase_service import get_firebase_service
from utils import decode_base64_to_image, encode_image_to_base64
import glob

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
                        # Handle different image formats
                        if isinstance(sample_image, list):
                            # If it's an array of images, use the first one
                            if len(sample_image) > 0:
                                sample_image = sample_image[0]
                                logger.info(f"Found image array with {len(sample_image)} images, using first one")
                            else:
                                continue  # Skip if array is empty
                        
                        # Check if it's a data URL and extract base64 part
                        if isinstance(sample_image, str) and sample_image.startswith('data:image'):
                            # Extract base64 part from data URL
                            sample_image = sample_image.split(',')[1] if ',' in sample_image else sample_image
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
                            'sample_images': doc_data.get('sample_images', [sample_image]),  # Support multiple images
                            'timestamp': doc_data.get('timestamp'),
                            'collection': collection_name
                        })
                        
                        logger.info(f"Found document with sample image: {name} in {collection_name} (ID: {doc.id})")
                    else:
                        # Log documents without sample images for debugging
                        logger.debug(f"Document {doc.id} in {collection_name} has no sample_image field")
                
                if collection_docs:
                    all_documents[collection_name] = collection_docs
                    logger.info(f"Collection {collection_name}: {len(collection_docs)} documents with sample images")
            
            logger.info(f"Total documents processed: {total_docs}")
            logger.info(f"Documents with sample images: {docs_with_images}")
            logger.info(f"Collections with sample images: {list(all_documents.keys())}")
            
            # Log detailed summary of what was found
            for collection_name, docs in all_documents.items():
                logger.info(f"Collection '{collection_name}' has {len(docs)} documents with sample images:")
                for doc in docs:
                    logger.info(f"  - {doc['name']} (ID: {doc['id']})")
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to fetch documents: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def download_image_from_base64(self, image_base64, filename):
        """Download and save a base64 encoded image to the dataset."""
        try:
            logger.debug(f"Starting to decode base64 image (length: {len(image_base64)})")
            
            # Decode base64 to image
            image_array = decode_base64_to_image(image_base64)
            if image_array is None:
                logger.error(f"Failed to decode base64 image for {filename} - decode_base64_to_image returned None")
                return False
            
            logger.debug(f"Successfully decoded image, shape: {image_array.shape}")
            
            # Save image
            success = cv2.imwrite(filename, image_array)
            if success:
                logger.info(f"Saved image: {filename}")
                return True
            else:
                logger.error(f"cv2.imwrite failed to save image: {filename}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to save image {filename}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
    
    def get_local_dataset_images(self):
        """Get all images from the local dataset directory."""
        try:
            logger.info("Scanning local dataset directory for images...")
            
            local_images = {}
            total_images = 0
            
            if not os.path.exists(self.dataset_path):
                logger.warning(f"Dataset directory does not exist: {self.dataset_path}")
                return local_images
            
            # Get all person directories
            for person_name in os.listdir(self.dataset_path):
                person_dir = os.path.join(self.dataset_path, person_name)
                
                if os.path.isdir(person_dir):
                    # Find all image files in the person directory
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                    person_images = []
                    
                    for extension in image_extensions:
                        pattern = os.path.join(person_dir, extension)
                        person_images.extend(glob.glob(pattern))
                        # Also check uppercase extensions
                        pattern_upper = os.path.join(person_dir, extension.upper())
                        person_images.extend(glob.glob(pattern_upper))
                    
                    if person_images:
                        local_images[person_name] = person_images
                        total_images += len(person_images)
                        logger.info(f"Found {len(person_images)} images for {person_name}")
            
            logger.info(f"Total local images found: {total_images} for {len(local_images)} persons")
            return local_images
            
        except Exception as e:
            logger.error(f"Failed to scan local dataset: {str(e)}")
            return {}
    
    def upload_image_to_firebase(self, image_path, person_name, collection_name="faces"):
        """Upload a local image to Firebase."""
        try:
            # Read the image
            image_array = cv2.imread(image_path)
            if image_array is None:
                logger.error(f"Failed to read image: {image_path}")
                return False
            
            # Convert to base64
            image_base64 = encode_image_to_base64(image_array)
            if not image_base64:
                logger.error(f"Failed to encode image to base64: {image_path}")
                return False
            
            # Create document data
            doc_data = {
                "name": person_name,
                "sample_image": f"data:image/jpeg;base64,{image_base64}",
                "timestamp": datetime.now(),
                "source": "local_capture",
                "image_path": os.path.basename(image_path),
                "processed": False
            }
            
            # Upload to Firebase
            if not self.firebase_service or not self.firebase_service.db:
                logger.error("Firebase service not available")
                return False
            
            collection_ref = self.firebase_service.db.collection(collection_name)
            doc_ref = collection_ref.add(doc_data)
            
            logger.info(f"Uploaded image {os.path.basename(image_path)} for {person_name} (ID: {doc_ref[1].id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload image {image_path}: {str(e)}")
            return False
    
    def upload_local_images_to_firebase(self, local_images, collection_name="faces"):
        """Upload all local images to Firebase."""
        logger.info("Starting to upload local images to Firebase...")
        
        upload_summary = {
            "success": True,
            "total_local_images": 0,
            "images_uploaded": 0,
            "failed_uploads": 0,
            "persons_processed": 0,
            "errors": []
        }
        
        for person_name, image_paths in local_images.items():
            try:
                logger.info(f"Processing {person_name}: {len(image_paths)} images")
                upload_summary["persons_processed"] += 1
                
                for image_path in image_paths:
                    upload_summary["total_local_images"] += 1
                    
                    if self.upload_image_to_firebase(image_path, person_name, collection_name):
                        upload_summary["images_uploaded"] += 1
                    else:
                        upload_summary["failed_uploads"] += 1
                        
            except Exception as e:
                error_msg = f"Error processing {person_name}: {str(e)}"
                logger.error(error_msg)
                upload_summary["errors"].append(error_msg)
        
        logger.info(f"Local upload completed. Summary: {upload_summary}")
        return upload_summary
    
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
            
            for i, doc in enumerate(documents):
                try:
                    sync_summary["total_documents"] += 1
                    sync_summary["documents_with_images"] += 1
                    
                    person_name = doc['name']
                    sample_image = doc['sample_image']
                    
                    logger.info(f"Processing document {i+1}/{len(documents)} for {person_name} (ID: {doc['id']})")
                    
                    # Create person directory
                    person_dir = os.path.join(self.dataset_path, person_name)
                    if not os.path.exists(person_dir):
                        os.makedirs(person_dir)
                        logger.info(f"Created directory for {person_name}")
                    
                    # Create unique filename based on collection and document ID
                    filename = os.path.join(person_dir, f"{collection_name}_{doc['id']}.jpg")
                    
                    logger.info(f"Attempting to download image to: {filename}")
                    
                    # Download primary image
                    if self.download_image_from_base64(sample_image, filename):
                        sync_summary["images_downloaded"] += 1
                        logger.info(f"Successfully downloaded image for {person_name}")
                        
                        # Track unique persons processed
                        if person_name not in processed_persons:
                            processed_persons.add(person_name)
                            sync_summary["persons_processed"] += 1
                            logger.info(f"New person added: {person_name}")
                        
                        # Handle additional images if available
                        additional_images = doc.get('sample_images', [])
                        if len(additional_images) > 1:
                            logger.info(f"Found {len(additional_images)} additional images for {person_name}")
                            for img_idx, additional_image in enumerate(additional_images[1:], 1):
                                additional_filename = os.path.join(person_dir, f"{collection_name}_{doc['id']}_img{img_idx}.jpg")
                                if self.download_image_from_base64(additional_image, additional_filename):
                                    sync_summary["images_downloaded"] += 1
                                    logger.info(f"Downloaded additional image {img_idx} for {person_name}")
                                else:
                                    logger.warning(f"Failed to download additional image {img_idx} for {person_name}")
                    else:
                        sync_summary["failed_downloads"] += 1
                        logger.warning(f"Failed to download image for {person_name}, creating placeholder")
                        
                        # Create placeholder if download failed
                        placeholder_filename = os.path.join(person_dir, f"placeholder_{doc['id']}.jpg")
                        if self.create_placeholder_image(placeholder_filename, person_name):
                            sync_summary["images_downloaded"] += 1
                            logger.info(f"Created placeholder for {person_name}")
                            
                            if person_name not in processed_persons:
                                processed_persons.add(person_name)
                                sync_summary["persons_processed"] += 1
                    
                    logger.info(f"Completed processing document {doc['id']} for {person_name}")
                    
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
        """Perform full dataset sync and model retraining (bidirectional)."""
        logger.info("Starting full bidirectional sync operation...")
        
        # Step 1: Upload local images to Firebase
        logger.info("=" * 50)
        logger.info("STEP 1: Uploading local images to Firebase")
        logger.info("=" * 50)
        
        local_images = self.get_local_dataset_images()
        upload_result = {"images_uploaded": 0, "failed_uploads": 0}
        
        if local_images:
            upload_result = self.upload_local_images_to_firebase(local_images)
            logger.info(f"Local upload completed: {upload_result['images_uploaded']} uploaded, {upload_result['failed_uploads']} failed")
        else:
            logger.info("No local images found to upload")
        
        # Step 2: Download Firebase documents to local dataset
        logger.info("=" * 50)
        logger.info("STEP 2: Downloading Firebase documents to local dataset")
        logger.info("=" * 50)
        
        all_documents = self.fetch_all_documents_with_sample_images()
        
        if not all_documents:
            logger.warning("No documents with sample images found in Firebase")
            sync_result = {
                "success": upload_result["images_uploaded"] > 0,
                "message": "No documents with sample images found in Firebase",
                "model_retrained": False,
                "local_images_uploaded": upload_result["images_uploaded"],
                "local_upload_failed": upload_result["failed_uploads"]
            }
        else:
            # Sync documents to dataset
            sync_result = self.sync_documents_to_dataset(all_documents)
            sync_result["local_images_uploaded"] = upload_result["images_uploaded"]
            sync_result["local_upload_failed"] = upload_result["failed_uploads"]
        
        # Step 3: Retrain model if any changes were made
        total_changes = sync_result.get("images_downloaded", 0) + upload_result["images_uploaded"]
        
        if total_changes > 0:
            logger.info("=" * 50)
            logger.info("STEP 3: Retraining model")
            logger.info("=" * 50)
            logger.info(f"Total changes detected: {total_changes} (downloaded: {sync_result.get('images_downloaded', 0)}, uploaded: {upload_result['images_uploaded']})")
            retrain_success = self.retrain_model()
            sync_result["model_retrained"] = retrain_success
        else:
            sync_result["model_retrained"] = False
            logger.info("No changes detected, skipping model retraining")
        
        sync_result["success"] = total_changes > 0
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
        print(f"📤 Local images uploaded: {result.get('local_images_uploaded', 0)}")
        print(f"❌ Local upload failed: {result.get('local_upload_failed', 0)}")
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