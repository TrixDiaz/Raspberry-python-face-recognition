import os
import base64
import cv2
import numpy as np
from datetime import datetime
import logging
from firebase_service import get_firebase_service
from utils import decode_base64_to_image
from google.cloud import firestore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, dataset_path="dataset"):
        """
        Initialize the Dataset Manager.
        
        Args:
            dataset_path: Path to the local dataset directory
        """
        self.dataset_path = dataset_path
        self.firebase_service = None
        self.initialize_firebase()
        self.ensure_dataset_directory()
    
    def initialize_firebase(self):
        """Initialize Firebase service."""
        try:
            self.firebase_service = get_firebase_service()
            logger.info("Firebase service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase service: {str(e)}")
            self.firebase_service = None
    
    def ensure_dataset_directory(self):
        """Ensure the dataset directory exists."""
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            logger.info(f"Created dataset directory: {self.dataset_path}")
    
    def get_all_known_faces_from_detections(self):
        """
        Get all faces from faces collection.
        
        Returns:
            dict: Dictionary with face names as keys and list of images as values
        """
        if not self.firebase_service or not self.firebase_service.db:
            logger.error("Firebase service not available")
            return {}
        
        try:
            logger.info("Starting to retrieve all faces from Firebase faces collection...")
            
            # Get all documents from faces collection
            faces_ref = self.firebase_service.db.collection('faces')
            all_docs = faces_ref.stream()
            
            faces_data = {}
            total_docs = 0
            
            for doc in all_docs:
                total_docs += 1
                detection_data = doc.to_dict()
                name = detection_data.get('name')
                face_image = detection_data.get('face_image')
                
                logger.debug(f"Processing document for {name}")
                
                if name and face_image:
                    # Check if image data is valid (base64 or data URL)
                    is_valid_image = (
                        face_image.startswith('data:image') or 
                        len(face_image) > 100 or
                        face_image.startswith('/9j/')  # JPEG base64 starts with this
                    )
                    
                    if is_valid_image:
                        if name not in faces_data:
                            faces_data[name] = []
                        
                        # Add timestamp and image data
                        timestamp = detection_data.get('timestamp')
                        faces_data[name].append({
                            'image': face_image,
                            'timestamp': timestamp,
                            'confidence': detection_data.get('confidence', 1.0),
                            'doc_id': doc.id
                        })
                        logger.debug(f"Added image for {name}")
                    else:
                        logger.warning(f"Invalid image data for {name}: {len(face_image) if face_image else 0} chars")
                else:
                    logger.warning(f"Missing name or image data in document: name={name}, has_image={bool(face_image)}")
            
            logger.info(f"Retrieved {total_docs} total documents from faces collection")
            logger.info(f"Found {len(faces_data)} unique faces: {list(faces_data.keys())}")
            
            # Log details for each person found
            for name, images in faces_data.items():
                logger.info(f"  - {name}: {len(images)} images")
            
            # Sort images by timestamp for each person (most recent first)
            for name in faces_data:
                try:
                    faces_data[name].sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)
                except Exception as e:
                    logger.warning(f"Could not sort images for {name}: {str(e)}")
            
            return faces_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve faces from faces collection: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def get_all_known_faces_alternative(self):
        """
        Alternative method to get all faces from faces collection.
        This is a fallback method in case the main method fails.
        
        Returns:
            dict: Dictionary with face names as keys and list of images as values
        """
        if not self.firebase_service or not self.firebase_service.db:
            logger.error("Firebase service not available")
            return {}
        
        try:
            logger.info("Trying alternative method to retrieve faces from faces collection...")
            
            # Try using a different query approach
            faces_ref = self.firebase_service.db.collection('faces')
            
            # Get all documents and filter manually
            all_docs = list(faces_ref.stream())
            logger.info(f"Retrieved {len(all_docs)} total documents from faces collection")
            
            faces_data = {}
            
            for doc in all_docs:
                detection_data = doc.to_dict()
                name = detection_data.get('name')
                face_image = detection_data.get('face_image')
                
                if name and face_image:
                    if name not in faces_data:
                        faces_data[name] = []
                    
                    faces_data[name].append({
                        'image': face_image,
                        'timestamp': detection_data.get('timestamp'),
                        'confidence': detection_data.get('confidence', 1.0),
                        'doc_id': doc.id
                    })
            
            logger.info(f"Alternative method found {len(faces_data)} unique faces: {list(faces_data.keys())}")
            return faces_data
            
        except Exception as e:
            logger.error(f"Alternative method also failed: {str(e)}")
            return {}
    
    def download_image_from_base64(self, image_base64, filename):
        """
        Download and save a base64 encoded image to the dataset.
        
        Args:
            image_base64: Base64 encoded image string
            filename: Name of the file to save
            
        Returns:
            bool: True if successful, False otherwise
        """
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
    
    def create_placeholder_image(self, filename, face_name):
        """
        Create a placeholder image when actual images are not available.
        
        Args:
            filename: Target filename for the placeholder
            face_name: Name of the person (for text on image)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a simple placeholder image
            img = np.ones((200, 200, 3), dtype=np.uint8) * 128
            
            # Add text to the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"No Image\n{face_name}"
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
    
    def get_existing_dataset_faces(self):
        """
        Get list of faces that already exist in the local dataset.
        
        Returns:
            list: List of face names in the dataset
        """
        try:
            if not os.path.exists(self.dataset_path):
                return []
            
            existing_faces = []
            for item in os.listdir(self.dataset_path):
                item_path = os.path.join(self.dataset_path, item)
                if os.path.isdir(item_path):
                    existing_faces.append(item)
            
            return existing_faces
            
        except Exception as e:
            logger.error(f"Failed to get existing dataset faces: {str(e)}")
            return []
    
    def sync_face_to_dataset(self, face_name, face_images, max_images=20):
        """
        Sync a single face to the dataset.
        
        Args:
            face_name: Name of the person
            face_images: List of image data dictionaries
            max_images: Maximum number of images to download per person
            
        Returns:
            dict: Result summary
        """
        try:
            # Create person directory
            person_dir = os.path.join(self.dataset_path, face_name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                logger.info(f"Created directory for {face_name}")
            
            downloaded_count = 0
            failed_count = 0
            
            # Download up to max_images images
            for i, image_data in enumerate(face_images[:max_images]):
                try:
                    filename = os.path.join(person_dir, f"{face_name}_{i+1:03d}.jpg")
                    
                    if self.download_image_from_base64(image_data['image'], filename):
                        downloaded_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process image {i+1} for {face_name}: {str(e)}")
                    failed_count += 1
            
            # If no images were downloaded successfully, create a placeholder
            if downloaded_count == 0:
                placeholder_filename = os.path.join(person_dir, f"{face_name}_placeholder.jpg")
                if self.create_placeholder_image(placeholder_filename, face_name):
                    downloaded_count = 1
                    logger.info(f"Created placeholder for {face_name}")
            
            result = {
                'success': downloaded_count > 0,
                'downloaded': downloaded_count,
                'failed': failed_count,
                'total_available': len(face_images)
            }
            
            logger.info(f"Synced {face_name}: {downloaded_count} images downloaded, {failed_count} failed")
            return result
            
        except Exception as e:
            logger.error(f"Failed to sync face {face_name}: {str(e)}")
            return {
                'success': False,
                'downloaded': 0,
                'failed': len(face_images) if face_images else 0,
                'total_available': len(face_images) if face_images else 0,
                'error': str(e)
            }
    
    def sync_dataset_with_firebase(self):
        """
        Main function to sync the dataset with Firebase face_detections collection.
        
        Returns:
            dict: Summary of sync operation
        """
        logger.info("Starting dataset sync with Firebase...")
        
        if not self.firebase_service or not self.firebase_service.db:
            logger.error("Firebase service not available")
            return {"success": False, "message": "Firebase service not available"}
        
        # Get all known faces from face_detections
        faces_data = self.get_all_known_faces_from_detections()
        
        # If no faces found, try alternative method
        if not faces_data:
            logger.warning("No known faces found with primary method, trying alternative...")
            faces_data = self.get_all_known_faces_alternative()
        
        if not faces_data:
            logger.warning("No known faces found in Firebase face_detections with any method")
            return {"success": False, "message": "No known faces found in Firebase"}
        
        # Get existing faces in dataset
        existing_dataset_faces = self.get_existing_dataset_faces()
        
        sync_summary = {
            "success": True,
            "total_faces": len(faces_data),
            "existing_updated": 0,
            "new_faces_added": 0,
            "total_images_downloaded": 0,
            "errors": []
        }
        
        # Process each face from Firebase
        for face_name, face_images in faces_data.items():
            try:
                logger.info(f"Processing face: {face_name} ({len(face_images)} images available)")
                
                # Sync the face to dataset
                result = self.sync_face_to_dataset(face_name, face_images)
                
                if result['success']:
                    if face_name in existing_dataset_faces:
                        sync_summary["existing_updated"] += 1
                        logger.info(f"Updated existing face: {face_name}")
                    else:
                        sync_summary["new_faces_added"] += 1
                        logger.info(f"Added new face: {face_name}")
                    
                    sync_summary["total_images_downloaded"] += result['downloaded']
                else:
                    error_msg = f"Failed to sync {face_name}: {result.get('error', 'Unknown error')}"
                    sync_summary["errors"].append(error_msg)
                    logger.error(error_msg)
                        
            except Exception as e:
                error_msg = f"Error processing {face_name}: {str(e)}"
                logger.error(error_msg)
                sync_summary["errors"].append(error_msg)
        
        logger.info(f"Dataset sync completed. Summary: {sync_summary}")
        return sync_summary
    
    def retrain_model(self):
        """
        Trigger model retraining after dataset update.
        
        Returns:
            bool: True if successful, False otherwise
        """
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
        """
        Perform full dataset sync and model retraining.
        
        Returns:
            dict: Complete operation summary
        """
        logger.info("Starting full sync and retrain operation...")
        
        # Sync dataset
        sync_result = self.sync_dataset_with_firebase()
        
        if not sync_result["success"]:
            return sync_result
        
        # Retrain model if dataset was updated
        if sync_result["existing_updated"] > 0 or sync_result["new_faces_added"] > 0:
            logger.info("Dataset updated, retraining model...")
            retrain_success = self.retrain_model()
            sync_result["model_retrained"] = retrain_success
        else:
            sync_result["model_retrained"] = False
            logger.info("No dataset changes, skipping model retraining")
        
        return sync_result


def main():
    """Main function to run dataset sync."""
    dataset_manager = DatasetManager()
    
    # Perform full sync and retrain
    result = dataset_manager.full_sync_and_retrain()
    
    print("\n" + "="*50)
    print("DATASET SYNC SUMMARY")
    print("="*50)
    print(f"Success: {result['success']}")
    print(f"Total faces in Firebase: {result.get('total_faces', 0)}")
    print(f"Existing faces updated: {result.get('existing_updated', 0)}")
    print(f"New faces added: {result.get('new_faces_added', 0)}")
    print(f"Total images downloaded: {result.get('total_images_downloaded', 0)}")
    print(f"Model retrained: {result.get('model_retrained', False)}")
    
    if result.get('errors'):
        print(f"\nErrors ({len(result['errors'])}):")
        for error in result['errors']:
            print(f"  - {error}")
    
    print("="*50)


if __name__ == "__main__":
    main()