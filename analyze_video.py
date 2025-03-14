import cv2
import torch
import numpy as np
import json
import os

def process_video(video_path):
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