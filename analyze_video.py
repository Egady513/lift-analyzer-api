import cv2
import torch
import numpy as np
import json
import os
from screens.processing_screen import analyze_pose, calculate_angles  # Import key functions

def process_video(video_path):
    """Analyze weightlifting form from video"""
    # Call functions from processing_screen.py
    pose_data = analyze_pose(video_path)
    angles = calculate_angles(pose_data)
    
    # Generate form feedback based on angles
    feedback = generate_feedback(angles)
    
    return {
        "frame_count": len(pose_data),
        "pose_data_path": "results.json",
        "angles": angles,
        "feedback": feedback
    }

def generate_feedback(angles):
    """Generate form feedback based on joint angles"""
    feedback = []
    
    # Example logic - would be based on your specific criteria
    if angles["knee"] < 80:
        feedback.append("Your knees could be bending too much at the bottom position.")
    if angles["back"] < 140:
        feedback.append("Your back appears to be rounding - try to maintain a straighter back angle.")
    
    return ". ".join(feedback) if feedback else "Your form looks good overall."

def load_pose_model():
    # This function is not used in the new process_video function
    # It's kept for potential future use
    pass

def calculate_points(keypoints):
    # This function is not used in the new process_video function
    # It's kept for potential future use
    pass

def calculate_bar_path(points):
    # This function is not used in the new process_video function
    # It's kept for potential future use
    pass

def calculate_angles(points):
    # This function is not used in the new process_video function
    # It's kept for potential future use
    pass

def process_video_old(video_path):
    """Process video and extract pose data"""
    # Load YOLOv7 model
    model = load_pose_model()
    
    # Initialize data structures
    frames = []
    frame_data = {
        "landmarks": [],
        "angles": [],
        "bar_path": [],
        "phase_markers": []
    }
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame with YOLOv7
        results = model(frame)
        
        # Extract keypoints
        if len(results.xyxy[0]) > 0:
            keypoints = results.keypoints[0].cpu().numpy()
            
            # Calculate points from keypoints
            points = calculate_points(keypoints)
            
            # Store frame data
            frame_data["landmarks"].append(points)
            
            # Calculate angles
            angles = calculate_angles(points)
            frame_data["angles"].append(angles)
            
            # Calculate bar path
            bar_pos = calculate_bar_path(points)
            if bar_pos:
                frame_data["bar_path"].append(bar_pos)
            
            frames.append(frame)
    
    cap.release()
    
    # Save data to file
    output_path = os.path.splitext(video_path)[0] + "_pose_data.json"
    with open(output_path, 'w') as f:
        json.dump(frame_data, f)
    
    return {
        "pose_data_path": output_path,
        "frame_count": len(frames)
    } 