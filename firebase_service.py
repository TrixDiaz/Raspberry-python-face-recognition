import firebase_admin
from firebase_admin import credentials, firestore
import json
import base64
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirebaseService:
    def __init__(self, config_path="firebase-config.json"):
        """Initialize Firebase service with the provided configuration."""
        self.db = None
        self.initialize_firebase(config_path)
    
    def initialize_firebase(self, config_path):
        """Initialize Firebase Admin SDK."""
        try:
            # Load the service account key
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create credentials object
            cred = credentials.Certificate(config)
            
            # Initialize the app (only if not already initialized)
            if not firebase_admin._apps:
                # Set environment variables to avoid ALTS credentials issues
                import os
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config_path
                os.environ['GRPC_DNS_RESOLVER'] = 'native'
                
                firebase_admin.initialize_app(cred, {
                    'projectId': config.get('project_id', 'auth-b83c4')
                })
            
            # Get Firestore client with explicit project ID
            self.db = firestore.client(project=config.get('project_id', 'auth-b83c4'))
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
            raise
    
    def save_motion_detection(self, timestamp=None, location="default", confidence=1.0):
        """
        Save motion detection event to Firebase.
        
        Args:
            timestamp: When the motion was detected (defaults to now)
            location: Location identifier
            confidence: Confidence level of motion detection
        """
        try:
            if not self.db:
                logger.error("Firebase not initialized")
                return False
            
            if timestamp is None:
                timestamp = datetime.now()
            
            motion_data = {
                "timestamp": timestamp,
                "location": location,
                "confidence": confidence,
                "type": "motion_detection",
                "processed": False
            }
            
            # Add to motion_logs collection (renamed from motion_detections)
            doc_ref = self.db.collection('motion_logs').add(motion_data)
            logger.info(f"Motion detection saved with ID: {doc_ref[1].id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save motion detection: {str(e)}")
            return False
    
    def save_unknown_face(self, face_image_base64, timestamp=None, location="default", confidence=0.0):
        """
        Save unknown face detection event to Firebase.
        
        Args:
            face_image_base64: Base64 encoded face image
            timestamp: When the face was detected (defaults to now)
            location: Location identifier
            confidence: Confidence level of face detection
        """
        try:
            if not self.db:
                logger.error("Firebase not initialized")
                return False
            
            if timestamp is None:
                timestamp = datetime.now()
            
            face_data = {
                "timestamp": timestamp,
                "location": location,
                "confidence": confidence,
                "type": "unknown_face",
                "face_image": face_image_base64,
                "processed": False,
                "status": "pending_review"
            }
            
            # Add to face_detections collection
            doc_ref = self.db.collection('face_detections').add(face_data)
            logger.info(f"Unknown face saved with ID: {doc_ref[1].id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save unknown face: {str(e)}")
            return False
    
    def save_known_face(self, face_image_base64, name, timestamp=None, location="default", confidence=1.0):
        """
        Save known face detection event to Firebase.
        
        Args:
            face_image_base64: Base64 encoded face image
            name: Name of the recognized person
            timestamp: When the face was detected (defaults to now)
            location: Location identifier
            confidence: Confidence level of face detection
        """
        try:
            if not self.db:
                logger.error("Firebase not initialized")
                return False
            
            if timestamp is None:
                timestamp = datetime.now()
            
            face_data = {
                "timestamp": timestamp,
                "location": location,
                "confidence": confidence,
                "type": "known_face",
                "name": name,
                "face_image": face_image_base64,
                "processed": False,
                "status": "recognized"
            }
            
            # Add to face_detections collection
            doc_ref = self.db.collection('face_detections').add(face_data)
            logger.info(f"Known face ({name}) saved with ID: {doc_ref[1].id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save known face: {str(e)}")
            return False
    
    def get_motion_detections(self, limit=100, processed_only=False):
        """
        Retrieve motion detection events from Firebase.
        
        Args:
            limit: Maximum number of records to retrieve
            processed_only: If True, only return processed events
        """
        try:
            if not self.db:
                logger.error("Firebase not initialized")
                return []
            
            query = self.db.collection('motion_logs').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            
            if processed_only:
                query = query.where('processed', '==', True)
            
            docs = query.stream()
            return [doc.to_dict() for doc in docs]
            
        except Exception as e:
            logger.error(f"Failed to retrieve motion detections: {str(e)}")
            return []
    
    def get_unknown_faces(self, limit=100, status="pending_review"):
        """
        Retrieve unknown face events from Firebase.
        
        Args:
            limit: Maximum number of records to retrieve
            status: Status filter (pending_review, reviewed, etc.)
        """
        try:
            if not self.db:
                logger.error("Firebase not initialized")
                return []
            
            query = self.db.collection('face_detections').where('type', '==', 'unknown_face').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            
            if status:
                query = query.where('status', '==', status)
            
            docs = query.stream()
            return [doc.to_dict() for doc in docs]
            
        except Exception as e:
            logger.error(f"Failed to retrieve unknown faces: {str(e)}")
            return []
    
    def get_face_detections(self, limit=100, face_type=None):
        """
        Retrieve face detection events from Firebase.
        
        Args:
            limit: Maximum number of records to retrieve
            face_type: Filter by face type ('known_face', 'unknown_face', or None for all)
        """
        try:
            if not self.db:
                logger.error("Firebase not initialized")
                return []
            
            query = self.db.collection('face_detections').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            
            if face_type:
                query = query.where('type', '==', face_type)
            
            docs = query.stream()
            return [doc.to_dict() for doc in docs]
            
        except Exception as e:
            logger.error(f"Failed to retrieve face detections: {str(e)}")
            return []
    
    def mark_motion_processed(self, doc_id):
        """Mark a motion detection event as processed."""
        try:
            if not self.db:
                logger.error("Firebase not initialized")
                return False
            
            self.db.collection('motion_logs').document(doc_id).update({'processed': True})
            logger.info(f"Motion detection {doc_id} marked as processed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark motion as processed: {str(e)}")
            return False
    
    def mark_face_processed(self, doc_id, status="reviewed"):
        """Mark an unknown face event as processed."""
        try:
            if not self.db:
                logger.error("Firebase not initialized")
                return False
            
            self.db.collection('face_detections').document(doc_id).update({
                'processed': True,
                'status': status
            })
            logger.info(f"Face detection {doc_id} marked as {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark face as processed: {str(e)}")
            return False
    
    def encode_image_to_base64(self, image_array):
        """
        Convert OpenCV image array to base64 string.
        
        Args:
            image_array: OpenCV image array (numpy array)
        """
        try:
            import cv2
            # Encode image as JPEG
            _, buffer = cv2.imencode('.jpg', image_array)
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            logger.error(f"Failed to encode image: {str(e)}")
            return None

# Global instance
firebase_service = None

def get_firebase_service():
    """Get or create Firebase service instance."""
    global firebase_service
    if firebase_service is None:
        firebase_service = FirebaseService()
    return firebase_service
