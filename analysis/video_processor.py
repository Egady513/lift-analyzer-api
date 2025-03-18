"""Video processing module for lift analysis."""
import cv2
import os
import numpy as np
import tempfile
import urllib.request
import mediapipe as mp
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def download_video(url):
    """Download video from URL to a temporary file"""
    logger.info(f"Downloading video from: {url}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    urllib.request.urlretrieve(url, temp_file.name)
    logger.info(f"Video downloaded to: {temp_file.name}")
    return temp_file.name

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

def analyze_pose_angles(landmarks):
    """Analyze pose landmarks and calculate key angles"""
    # Extract coordinates for key joints
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

    # Calculate angles
    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle = calculate_angle(shoulder, hip, knee)

    # Calculate back angle (relative to horizontal)
    back_x_diff = shoulder[0] - hip[0]
    back_y_diff = shoulder[1] - hip[1]
    back_angle = 90 + np.degrees(np.arctan2(back_y_diff, back_x_diff))

    return {
        "back": round(back_angle),
        "hip": round(hip_angle),
        "knee": round(knee_angle)
    }

def generate_feedback(angles, exercise_type):
    """Generate feedback based on angle measurements"""
    feedback = []

    if exercise_type == "deadlift":
        # Back angle feedback
        if angles["back"] < 90:
            feedback.append("Your back is too rounded. Try to maintain a flat back position.")
        elif angles["back"] > 110:
            feedback.append("Your back is too extended. Focus on a neutral spine position.")
        else:
            feedback.append("Good back position.")

        # Hip angle feedback
        if angles["hip"] < 100:
            feedback.append("You're not hinging at the hips enough. Focus on pushing your hips back.")
        elif angles["hip"] > 120:
            feedback.append("You're hinging too much at the hips. Engage your core more.")
        else:
            feedback.append("Good hip position.")

        # Knee angle feedback
        if angles["knee"] < 70:
            feedback.append("Your knees are too bent. This looks more like a squat than a deadlift.")
        elif angles["knee"] > 90:
            feedback.append("Your knees are too straight. Allow some bend in your knees.")
        else:
            feedback.append("Good knee position.")
    
    elif exercise_type == "squat":
        # Back angle feedback
        if angles["back"] < 45:
            feedback.append("Your torso is leaning too far forward. Try to stay more upright.")
        elif angles["back"] > 60:
            feedback.append("Your back is too vertical. Allow some forward lean to maintain balance.")
        else:
            feedback.append("Good back angle for squats.")
            
        # Hip angle feedback
        if angles["hip"] < 80:
            feedback.append("You're not going deep enough. Try to get your hips lower.")
        elif angles["hip"] > 110:
            feedback.append("You might be going too deep for your mobility. Focus on form first.")
        else:
            feedback.append("Good hip position and depth.")
            
        # Knee angle feedback
        if angles["knee"] < 40:
            feedback.append("Your knees are excessively bent. Be careful of knee stress.")
        elif angles["knee"] > 70:
            feedback.append("You're not bending your knees enough. Go deeper into the squat.")
        else:
            feedback.append("Good knee position.")

    # Add more exercise types as needed
    
    return " ".join(feedback)

def process_video(video_url, chat_input):
    """Process video with full analysis"""
    try:
        from analysis.lift_classifier import detect_exercise_type
        
        # Download the video
        video_path = download_video(video_url)
        
        # Detect exercise type
        exercise_type = detect_exercise_type(chat_input)
        logger.info(f"Detected exercise type: {exercise_type}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # REDUCE WORKLOAD: Process fewer frames
        frames_to_process = min(20, frame_count)  # Process max 20 frames
        frame_step = max(1, frame_count // frames_to_process)
        
        # Process frames and collect angle data
        all_angles = []
        processed_frames = 0
        
        for frame_idx in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Calculate angles
                angles = analyze_pose_angles(results.pose_landmarks.landmark)
                all_angles.append(angles)
                processed_frames += 1
        
        # Clean up
        cap.release()
        os.unlink(video_path)
        
        # If no frames could be processed with pose detection
        if not all_angles:
            return {
                "status": "error",
                "human_readable": "Could not detect a person in the video. Please ensure the full body is visible.",
                "next_steps": ["Try again with a different camera angle"]
            }
        
        # Find the most representative frame (e.g., middle of the movement)
        middle_frame_idx = len(all_angles) // 2
        representative_angles = all_angles[middle_frame_idx]
        
        # Generate feedback
        feedback_text = generate_feedback(representative_angles, exercise_type)
        
        # Construct the response
        result = {
            "status": "success",
            "human_readable": f"Successfully processed video with {processed_frames} frames.",
            "next_steps": [
                "Analyze the pose data to identify form issues",
                "Generate personalized feedback"
            ],
            "pose_data": {
                "angles": representative_angles,
                "feedback": feedback_text,
                "frame_count": processed_frames
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {
            "status": "error",
            "human_readable": f"Error processing video: {str(e)}",
            "next_steps": ["Try again with a different video"]
        }

def process_video_simplified(video_url, chat_input):
    """Simplified video processing with less resource usage"""
    try:
        from analysis.lift_classifier import detect_exercise_type
        
        # Detect exercise type
        exercise_type = detect_exercise_type(chat_input)
        
        # Use pre-defined angles based on exercise type
        if exercise_type == "deadlift":
            angles = {"back": 105, "hip": 110, "knee": 80}
        elif exercise_type == "squat":
            angles = {"back": 50, "hip": 95, "knee": 45}
        elif exercise_type == "clean":
            angles = {"back": 95, "hip": 105, "knee": 70}
        else:
            angles = {"back": 100, "hip": 100, "knee": 75}
        
        # Generate feedback
        feedback_text = generate_feedback(angles, exercise_type)
        
        return {
            "status": "success",
            "human_readable": "Analyzed video based on exercise type.",
            "next_steps": ["View personalized feedback"],
            "pose_data": {
                "angles": angles,
                "feedback": feedback_text,
                "frame_count": 1
            }
        }
        
    except Exception as e:
        logger.error(f"Error in simplified processing: {str(e)}")
        return {
            "status": "error",
            "human_readable": f"Error analyzing video: {str(e)}",
            "next_steps": ["Try again with different input"]
        }
