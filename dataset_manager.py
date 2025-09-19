import os
import base64
import cv2
import numpy as np
from datetime import datetime
import logging
from firebase_service import get_firebase_service
from utils import decode_base64_to_image

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
    
    def get_faces_from_firebase(self):
        """
        Retrieve all faces from Firebase faces collection.
        
        Returns:
            dict: Dictionary with face names as keys and face data as values
        """
        if not self.firebase_service or not self.firebase_service.db:
            logger.error("Firebase service not available")
            return {}
        
        try:
            faces_ref = self.firebase_service.db.collection('faces')
            docs = faces_ref.stream()
            
            faces_data = {}
            for doc in docs:
                face_data = doc.to_dict()
                face_name = doc.id  # Document ID is the face name
                faces_data[face_name] = face_data
                logger.info(f"Retrieved face data for: {face_name}")
            
            return faces_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve faces from Firebase: {str(e)}")
            return {}
    
    def get_face_images_from_detections(self, face_name):
        """
        Retrieve actual face images from face_detections collection for a specific person.
        
        Args:
            face_name: Name of the person to get images for
            
        Returns:
            list: List of base64 encoded images
        """
        if not self.firebase_service or not self.firebase_service.db:
            logger.error("Firebase service not available")
            return []
        
        try:
            # Query face_detections collection for known faces with this name
            detections_ref = self.firebase_service.db.collection('face_detections')
            query = detections_ref.where('type', '==', 'known_face').where('name', '==', face_name)
            docs = query.stream()
            
            images = []
            for doc in docs:
                detection_data = doc.to_dict()
                face_image = detection_data.get('face_image')
                if face_image:
                    images.append(face_image)
            
            logger.info(f"Retrieved {len(images)} actual images for {face_name} from face_detections")
            return images
            
        except Exception as e:
            logger.error(f"Failed to retrieve images for {face_name} from face_detections: {str(e)}")
            return []
    
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
    
    def copy_image_from_filename(self, source_filename, target_filename):
        """
        Copy an image file from source to target location.
        
        Args:
            source_filename: Source image filename
            target_filename: Target image filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import shutil
            
            # Check if source file exists
            if not os.path.exists(source_filename):
                logger.warning(f"Source image not found: {source_filename}")
                return False
            
            # Copy the file
            shutil.copy2(source_filename, target_filename)
            logger.info(f"Copied image: {source_filename} -> {target_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy image {source_filename}: {str(e)}")
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
            import numpy as np
            
            # Create a 200x200 image with a light gray background
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
    
    def update_existing_face_dataset(self, face_name, face_data):
        """
        Update dataset for an existing face by downloading all images and updating timestamp.
        
        Args:
            face_name: Name of the face
            face_data: Face data from Firebase
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create person directory
            person_dir = os.path.join(self.dataset_path, face_name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                logger.info(f"Created directory for {face_name}")
            
            # First, try to get actual images from face_detections collection
            actual_images = self.get_face_images_from_detections(face_name)
            downloaded_count = 0
            
            if actual_images:
                # Use actual images from face_detections
                logger.info(f"Using {len(actual_images)} actual images from face_detections for {face_name}")
                for i, image_base64 in enumerate(actual_images):
                    filename = os.path.join(person_dir, f"{face_name}_{i+1:03d}.jpg")
                    if self.download_image_from_base64(image_base64, filename):
                        downloaded_count += 1
            else:
                # Fallback to image_urls from faces collection
                logger.info(f"No actual images found in face_detections, trying image_urls for {face_name}")
                image_urls = face_data.get('image_url', [])
                if not image_urls:
                    logger.warning(f"No images found for {face_name}")
                    return False
                
                for i, image_url in enumerate(image_urls):
                    filename = os.path.join(person_dir, f"{face_name}_{i+1:03d}.jpg")
                    
                    # Check if image_url is base64 data or a filename
                    if image_url.startswith('data:image') or len(image_url) > 100:
                        # This looks like base64 data
                        if self.download_image_from_base64(image_url, filename):
                            downloaded_count += 1
                    else:
                        # This is a filename, try to copy from various possible locations
                        possible_paths = [
                            image_url,  # Direct path
                            os.path.join("images", image_url),  # In images folder
                            os.path.join("uploads", image_url),  # In uploads folder
                            os.path.join("dataset", image_url),  # In dataset folder
                        ]
                        
                        copied = False
                        for source_path in possible_paths:
                            if self.copy_image_from_filename(source_path, filename):
                                downloaded_count += 1
                                copied = True
                                break
                        
                        if not copied:
                            logger.warning(f"Could not find or copy image: {image_url}")
                            # Create a placeholder image as fallback
                            if self.create_placeholder_image(filename, face_name):
                                downloaded_count += 1
            
            # If no images were downloaded, create at least one placeholder
            if downloaded_count == 0:
                logger.warning(f"No images found for {face_name}, creating placeholder")
                placeholder_filename = os.path.join(person_dir, f"{face_name}_001.jpg")
                if self.create_placeholder_image(placeholder_filename, face_name):
                    downloaded_count = 1
            
            # Update timestamp in Firebase
            self.update_face_timestamp(face_name)
            
            logger.info(f"Updated dataset for {face_name}: {downloaded_count} images")
            return downloaded_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update dataset for {face_name}: {str(e)}")
            return False
    
    def update_face_timestamp(self, face_name):
        """
        Update the timestamp for a face in Firebase to current time.
        
        Args:
            face_name: Name of the face to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.firebase_service or not self.firebase_service.db:
            logger.error("Firebase service not available")
            return False
        
        try:
            face_ref = self.firebase_service.db.collection('faces').document(face_name)
            face_ref.update({
                'updated_at': datetime.now(),
                'last_sync': datetime.now()
            })
            logger.info(f"Updated timestamp for {face_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update timestamp for {face_name}: {str(e)}")
            return False
    
    def add_new_face_to_dataset(self, face_name, face_data):
        """
        Add a new face to the dataset.
        
        Args:
            face_name: Name of the new face
            face_data: Face data from Firebase
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create person directory
            person_dir = os.path.join(self.dataset_path, face_name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                logger.info(f"Created directory for new face: {face_name}")
            
            # First, try to get actual images from face_detections collection
            actual_images = self.get_face_images_from_detections(face_name)
            downloaded_count = 0
            
            if actual_images:
                # Use actual images from face_detections
                logger.info(f"Using {len(actual_images)} actual images from face_detections for {face_name}")
                for i, image_base64 in enumerate(actual_images):
                    filename = os.path.join(person_dir, f"{face_name}_{i+1:03d}.jpg")
                    if self.download_image_from_base64(image_base64, filename):
                        downloaded_count += 1
            else:
                # Fallback to image_urls from faces collection
                logger.info(f"No actual images found in face_detections, trying image_urls for {face_name}")
                image_urls = face_data.get('image_url', [])
                if not image_urls:
                    logger.warning(f"No images found for new face {face_name}")
                    return False
                
                for i, image_url in enumerate(image_urls):
                    filename = os.path.join(person_dir, f"{face_name}_{i+1:03d}.jpg")
                    
                    # Check if image_url is base64 data or a filename
                    if image_url.startswith('data:image') or len(image_url) > 100:
                        # This looks like base64 data
                        if self.download_image_from_base64(image_url, filename):
                            downloaded_count += 1
                    else:
                        # This is a filename, try to copy from various possible locations
                        possible_paths = [
                            image_url,  # Direct path
                            os.path.join("images", image_url),  # In images folder
                            os.path.join("uploads", image_url),  # In uploads folder
                            os.path.join("dataset", image_url),  # In dataset folder
                        ]
                        
                        copied = False
                        for source_path in possible_paths:
                            if self.copy_image_from_filename(source_path, filename):
                                downloaded_count += 1
                                copied = True
                                break
                        
                        if not copied:
                            logger.warning(f"Could not find or copy image: {image_url}")
                            # Create a placeholder image as fallback
                            if self.create_placeholder_image(filename, face_name):
                                downloaded_count += 1
            
            # If no images were downloaded, create at least one placeholder
            if downloaded_count == 0:
                logger.warning(f"No images found for {face_name}, creating placeholder")
                placeholder_filename = os.path.join(person_dir, f"{face_name}_001.jpg")
                if self.create_placeholder_image(placeholder_filename, face_name):
                    downloaded_count = 1
            
            # Add sync timestamp to Firebase
            self.add_sync_timestamp(face_name)
            
            logger.info(f"Added new face {face_name} to dataset: {downloaded_count} images")
            return downloaded_count > 0
            
        except Exception as e:
            logger.error(f"Failed to add new face {face_name} to dataset: {str(e)}")
            return False
    
    def add_sync_timestamp(self, face_name):
        """
        Add sync timestamp to a face in Firebase.
        
        Args:
            face_name: Name of the face
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.firebase_service or not self.firebase_service.db:
            logger.error("Firebase service not available")
            return False
        
        try:
            face_ref = self.firebase_service.db.collection('faces').document(face_name)
            face_ref.update({
                'last_sync': datetime.now(),
                'synced_to_dataset': True
            })
            logger.info(f"Added sync timestamp for {face_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add sync timestamp for {face_name}: {str(e)}")
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
    
    def sync_dataset_with_firebase(self):
        """
        Main function to sync the dataset with Firebase faces collection.
        - For existing faces: download all images and update timestamp
        - For new faces: add to dataset and mark as synced
        
        Returns:
            dict: Summary of sync operation
        """
        logger.info("Starting dataset sync with Firebase...")
        
        if not self.firebase_service or not self.firebase_service.db:
            logger.error("Firebase service not available")
            return {"success": False, "message": "Firebase service not available"}
        
        # Get all faces from Firebase
        firebase_faces = self.get_faces_from_firebase()
        if not firebase_faces:
            logger.warning("No faces found in Firebase")
            return {"success": False, "message": "No faces found in Firebase"}
        
        # Get existing faces in dataset
        existing_dataset_faces = self.get_existing_dataset_faces()
        
        sync_summary = {
            "success": True,
            "total_faces": len(firebase_faces),
            "existing_updated": 0,
            "new_faces_added": 0,
            "errors": []
        }
        
        # Process each face from Firebase
        for face_name, face_data in firebase_faces.items():
            try:
                if face_name in existing_dataset_faces:
                    # Update existing face
                    if self.update_existing_face_dataset(face_name, face_data):
                        sync_summary["existing_updated"] += 1
                        logger.info(f"Updated existing face: {face_name}")
                    else:
                        sync_summary["errors"].append(f"Failed to update {face_name}")
                else:
                    # Add new face
                    if self.add_new_face_to_dataset(face_name, face_data):
                        sync_summary["new_faces_added"] += 1
                        logger.info(f"Added new face: {face_name}")
                    else:
                        sync_summary["errors"].append(f"Failed to add {face_name}")
                        
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
    print(f"Total faces in Firebase: {result['total_faces']}")
    print(f"Existing faces updated: {result['existing_updated']}")
    print(f"New faces added: {result['new_faces_added']}")
    print(f"Model retrained: {result.get('model_retrained', False)}")
    
    if result['errors']:
        print(f"\nErrors ({len(result['errors'])}):")
        for error in result['errors']:
            print(f"  - {error}")
    
    print("="*50)


if __name__ == "__main__":
    main()
