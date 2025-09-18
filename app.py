from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import traceback

from firebase_service import get_firebase_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global Firebase service instance
firebase_service = None

@app.before_first_request
def initialize_firebase():
    """Initialize Firebase service on first request."""
    global firebase_service
    try:
        firebase_service = get_firebase_service()
        logger.info("Firebase service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase service: {str(e)}")
        firebase_service = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        firebase_connected = firebase_service is not None and firebase_service.db is not None
        return jsonify({
            "status": "healthy" if firebase_connected else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "firebase_connected": firebase_connected,
            "version": "1.0.0"
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/motion-detection', methods=['POST'])
def report_motion_detection():
    """
    Report motion detection event to Firebase.
    This endpoint is called when motion is detected by the camera.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "No JSON data provided"
            }), 400
        
        # Extract parameters with defaults
        timestamp_str = data.get('timestamp')
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        location = data.get('location', 'default')
        confidence = float(data.get('confidence', 1.0))
        
        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            return jsonify({
                "success": False,
                "message": "Confidence must be between 0.0 and 1.0"
            }), 400
        
        # Save motion detection to Firebase
        success = firebase_service.save_motion_detection(
            timestamp=timestamp,
            location=location,
            confidence=confidence
        )
        
        if success:
            return jsonify({
                "success": True,
                "message": "Motion detection event saved successfully",
                "data": {
                    "timestamp": timestamp.isoformat(),
                    "location": location,
                    "confidence": confidence
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to save motion detection"
            }), 500
            
    except ValueError as e:
        logger.error(f"Validation error in motion detection: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Validation error: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Error in motion detection endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/unknown-face', methods=['POST'])
def report_unknown_face():
    """
    Report unknown face detection event to Firebase.
    This endpoint is called when an unknown face is detected.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "No JSON data provided"
            }), 400
        
        # Extract required parameters
        face_image_base64 = data.get('face_image_base64')
        if not face_image_base64:
            return jsonify({
                "success": False,
                "message": "face_image_base64 is required"
            }), 400
        
        # Extract optional parameters with defaults
        timestamp_str = data.get('timestamp')
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        location = data.get('location', 'default')
        confidence = float(data.get('confidence', 0.0))
        
        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            return jsonify({
                "success": False,
                "message": "Confidence must be between 0.0 and 1.0"
            }), 400
        
        # Save unknown face to Firebase
        success = firebase_service.save_unknown_face(
            face_image_base64=face_image_base64,
            timestamp=timestamp,
            location=location,
            confidence=confidence
        )
        
        if success:
            return jsonify({
                "success": True,
                "message": "Unknown face event saved successfully",
                "data": {
                    "timestamp": timestamp.isoformat(),
                    "location": location,
                    "confidence": confidence
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to save unknown face"
            }), 500
            
    except ValueError as e:
        logger.error(f"Validation error in unknown face: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Validation error: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Error in unknown face endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/motion-detections', methods=['GET'])
def get_motion_detections():
    """
    Retrieve motion detection events from Firebase.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        # Get query parameters
        limit = int(request.args.get('limit', 100))
        processed_only = request.args.get('processed_only', 'false').lower() == 'true'
        
        # Validate limit
        if limit <= 0 or limit > 1000:
            return jsonify({
                "success": False,
                "message": "Limit must be between 1 and 1000"
            }), 400
        
        detections = firebase_service.get_motion_detections(limit=limit, processed_only=processed_only)
        
        return jsonify({
            "success": True,
            "count": len(detections),
            "data": detections
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error in get motion detections: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Validation error: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Error retrieving motion detections: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/unknown-faces', methods=['GET'])
def get_unknown_faces():
    """
    Retrieve unknown face events from Firebase.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        # Get query parameters
        limit = int(request.args.get('limit', 100))
        status = request.args.get('status', 'pending_review')
        
        # Validate limit
        if limit <= 0 or limit > 1000:
            return jsonify({
                "success": False,
                "message": "Limit must be between 1 and 1000"
            }), 400
        
        faces = firebase_service.get_unknown_faces(limit=limit, status=status)
        
        return jsonify({
            "success": True,
            "count": len(faces),
            "data": faces
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error in get unknown faces: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Validation error: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Error retrieving unknown faces: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/motion-detections/<doc_id>/process', methods=['POST'])
def mark_motion_processed(doc_id):
    """
    Mark a motion detection event as processed.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        if not doc_id:
            return jsonify({
                "success": False,
                "message": "Document ID is required"
            }), 400
        
        success = firebase_service.mark_motion_processed(doc_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Motion detection {doc_id} marked as processed"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to mark motion as processed"
            }), 500
            
    except Exception as e:
        logger.error(f"Error marking motion as processed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/unknown-faces/<doc_id>/process', methods=['POST'])
def mark_face_processed(doc_id):
    """
    Mark an unknown face event as processed.
    """
    try:
        if not firebase_service:
            return jsonify({
                "success": False,
                "message": "Firebase service not available"
            }), 503
        
        if not doc_id:
            return jsonify({
                "success": False,
                "message": "Document ID is required"
            }), 400
        
        # Get JSON data from request
        data = request.get_json() or {}
        status = data.get('status', 'reviewed')
        
        # Validate status
        valid_statuses = ['pending_review', 'reviewed', 'approved', 'rejected']
        if status not in valid_statuses:
            return jsonify({
                "success": False,
                "message": f"Status must be one of: {', '.join(valid_statuses)}"
            }), 400
        
        success = firebase_service.mark_face_processed(doc_id, status)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Unknown face {doc_id} marked as {status}"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to mark face as processed"
            }), 500
            
    except Exception as e:
        logger.error(f"Error marking face as processed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        "message": "Face Recognition & Motion Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "motion_detection": "/motion-detection",
            "unknown_face": "/unknown-face",
            "motion_detections": "/motion-detections",
            "unknown_faces": "/unknown-faces"
        }
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "message": "Endpoint not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "success": False,
        "message": "Method not allowed"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "success": False,
        "message": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Initialize Firebase on startup
    try:
        firebase_service = get_firebase_service()
        logger.info("Firebase service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase service: {str(e)}")
        firebase_service = None
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
