import base64
import cv2
import numpy as np

def encode_image_to_base64(image):
    """
    Convert an image (numpy array) to base64 string
    """
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def decode_base64_to_image(base64_string):
    """
    Convert a base64 string back to an image (numpy array)
    """
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)