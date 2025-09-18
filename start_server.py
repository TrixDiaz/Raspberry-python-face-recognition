#!/usr/bin/env python3
"""
Startup script for the Flask server.
This script initializes the Flask application and starts the server.
"""

import os
import sys
import logging
from app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_firebase_config():
    """Check if Firebase configuration file exists."""
    config_path = "firebase-config.json"
    if not os.path.exists(config_path):
        logger.error(f"Firebase configuration file not found: {config_path}")
        logger.error("Please ensure firebase-config.json is in the current directory")
        return False
    return True

def main():
    """Main function to start the server."""
    logger.info("Starting Face Recognition & Motion Detection API Server...")
    
    # Check Firebase configuration
    if not check_firebase_config():
        logger.error("Cannot start server without Firebase configuration")
        sys.exit(1)
    
    # Get configuration from environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Server configuration:")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Debug: {debug}")
    
    try:
        # Start the Flask server
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
