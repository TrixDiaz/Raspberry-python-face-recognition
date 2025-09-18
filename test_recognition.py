#!/usr/bin/env python3
"""
Test script to validate face recognition improvements.
This script helps you test the distance threshold settings.
"""

import face_recognition
import cv2
import numpy as np
import pickle
import os

def test_distance_threshold():
    """Test different distance thresholds to find the optimal setting."""
    
    # Load the encodings
    if not os.path.exists("encodings.pickle"):
        print("Error: encodings.pickle not found. Please run model_training.py first.")
        return
    
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
    
    print(f"Loaded {len(known_face_encodings)} face encodings for {len(set(known_face_names))} unique people:")
    for name in set(known_face_names):
        count = known_face_names.count(name)
        print(f"  - {name}: {count} encodings")
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6]
    
    print("\nDistance threshold recommendations:")
    print("0.3 - Very strict (may miss some valid matches)")
    print("0.4 - Balanced (recommended)")
    print("0.5 - Lenient (may accept some false matches)")
    print("0.6 - Very lenient (likely to have false matches)")
    
    print(f"\nCurrent configuration uses threshold: 0.4")
    print("If you're getting too many false matches, try lowering to 0.3")
    print("If you're missing valid matches, try raising to 0.5")

def analyze_encoding_quality():
    """Analyze the quality of face encodings in the dataset."""
    
    if not os.path.exists("encodings.pickle"):
        print("Error: encodings.pickle not found. Please run model_training.py first.")
        return
    
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
    
    # Calculate average distances within each person's encodings
    unique_names = list(set(known_face_names))
    
    print("\nFace encoding quality analysis:")
    for name in unique_names:
        # Get all encodings for this person
        person_encodings = [enc for enc, n in zip(known_face_encodings, known_face_names) if n == name]
        
        if len(person_encodings) > 1:
            # Calculate average distance between this person's encodings
            distances = []
            for i in range(len(person_encodings)):
                for j in range(i + 1, len(person_encodings)):
                    dist = face_recognition.face_distance([person_encodings[i]], person_encodings[j])[0]
                    distances.append(dist)
            
            avg_distance = np.mean(distances)
            print(f"  {name}: {len(person_encodings)} encodings, avg internal distance: {avg_distance:.3f}")
            
            if avg_distance > 0.4:
                print(f"    ⚠️  High internal distance - consider capturing more photos")
            elif avg_distance < 0.2:
                print(f"    ✅ Good consistency")
        else:
            print(f"  {name}: {len(person_encodings)} encoding (only one photo)")

if __name__ == "__main__":
    print("Face Recognition Test Script")
    print("=" * 40)
    
    test_distance_threshold()
    analyze_encoding_quality()
    
    print("\n" + "=" * 40)
    print("Recommendations:")
    print("1. If unknown faces are being recognized as known people:")
    print("   - Lower DISTANCE_THRESHOLD to 0.3")
    print("   - Capture more diverse photos of known people")
    print("   - Ensure good lighting and clear face visibility")
    print()
    print("2. If known people are being marked as unknown:")
    print("   - Raise DISTANCE_THRESHOLD to 0.5")
    print("   - Check if photos have good quality and lighting")
    print("   - Consider retraining with more photos")
    print()
    print("3. For best results:")
    print("   - Use 10-20 photos per person with different angles/lighting")
    print("   - Ensure faces are clearly visible and well-lit")
    print("   - Test with the person in different conditions")
