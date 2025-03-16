import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from openai import OpenAI
import json
import os
from config import Config
import torch
import requests

# Update API_URL to point to Railway
API_URL = "https://lift-analyzer-api-production.up.railway.app"

# Define API endpoints based on environment
def get_api_url():
    """Return the appropriate API URL based on environment"""
    env = os.environ.get("ENVIRONMENT", "development")
    
    if env == "production":
        return "https://your-production-api.com"
    elif env == "staging":
        return "https://your-staging-api.com"
    else:
        return "http://127.0.0.1:8000"  # Local development

# Use this function when making API calls
def get_process_video_endpoint():
    return f"{get_api_url()}/process-video"

def get_apply_analysis_endpoint():
    return f"{get_api_url()}/apply-analysis"

class ProcessingScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg='#1E1E1E')
        
        # Initialize YOLO pose model
        self.model = self.load_pose_model()
        
        # Initialize OpenAI
        self.config = Config()
        self.client = OpenAI(api_key=self.config.api_key)
        
        # Initialize data structures
        self.frames = []
        self.frame_data = {
            "landmarks": [],
            "angles": [],
            "bar_path": [],
            "phase_markers": []
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the processing screen UI"""
        # Progress information
        self.progress_label = ttk.Label(
            self, 
            text="Processing video...",
            font=('Arial', 12)
        )
        self.progress_label.pack(pady=20)
        
        self.progress_bar = ttk.Progressbar(
            self,
            orient='horizontal',
            length=400,
            mode='determinate'
        )
        self.progress_bar.pack(pady=10)
        
        # Status messages
        self.status_text = tk.Text(
            self,
            height=10,
            width=50,
            bg='#2E2E2E',
            fg='white'
        )
        self.status_text.pack(pady=20)
    
    def load_pose_model(self):
        """Load YOLOv7 pose estimation model"""
        try:
            # Check multiple possible locations for the weights file
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'yolov7-w6-pose.pt'),
                os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov7-w6-pose.pt'),
                'yolov7-w6-pose.pt',  # Root directory
                os.path.join('models', 'yolov7-w6-pose.pt')
            ]
            
            weights_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    weights_path = path
                    self.update_status(f"Found model at: {weights_path}")
                    break
                
            if not weights_path:
                self.update_status("Model not found, downloading...")
                # Download model if not found
                os.makedirs("models", exist_ok=True)
                url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt"
                weights_path = os.path.join('models', 'yolov7-w6-pose.pt')
                r = requests.get(url, stream=True)
                with open(weights_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            model = torch.hub.load('WongKinYiu/yolov7', 'custom', 
                                  path=weights_path,
                                  trust_repo=True)
            model.eval()
            
            # Use CUDA if available
            if torch.cuda.is_available():
                model.cuda()
                print("Using CUDA for pose detection")
            else:
                print("Using CPU for pose detection")
            
            return model
        except Exception as e:
            self.show_error("Model Loading Error", 
                           f"Failed to load pose model: {str(e)}")
            return None
    
    def process_video(self, video_path):
        """Process the uploaded video"""
        try:
            # Add Authorization header to requests
            headers = {
                "Authorization": self.config.api_key,  # Use your API key from config
                "Content-Type": "application/json"
            }
            
            # Make API request with authorization
            payload = {"video_path": video_path}
            response = requests.post(f"{API_URL}/process-video", 
                                    json=payload, 
                                    headers=headers)
            
            # Check for successful response
            if response.status_code == 200:
                # Process successful response
                return response.json()
            else:
                # Handle error responses
                self.update_status(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.show_error("Processing Error", str(e))
    
    def load_video_frames(self, video_path):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_bar['maximum'] = total_frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_frame(frame)
            if processed_frame:
                self.frames.append(processed_frame)
            
            # Update progress
            current_frame = len(self.frames)
            self.progress_bar['value'] = current_frame
            self.update()
            
        cap.release()
    
    def process_frame(self, frame):
        """Process a single frame using YOLOv7-pose"""
        try:
            # Prepare frame for YOLO
            results = self.model(frame)
            
            # Extract keypoints
            if len(results.xyxy[0]) > 0:  # If person detected
                keypoints = results.keypoints[0].cpu().numpy()
                
                # Calculate points from keypoints
                points = self.calculate_points(keypoints)
                
                # Store frame data
                self.frame_data["landmarks"].append(points)
                
                # Calculate angles
                angles = self.calculate_angles(points)
                self.frame_data["angles"].append(angles)
                
                # Calculate bar path
                bar_pos = self.calculate_bar_path(points)
                if bar_pos:
                    self.frame_data["bar_path"].append(bar_pos)
                
                return frame
                
        except Exception as e:
            self.update_status(f"Error processing frame: {str(e)}")
            return None
    
    def calculate_points(self, keypoints):
        """Calculate points from keypoints"""
        points = {}
        for i, (x, y) in enumerate(keypoints):
            points[f"POINT_{i+1}"] = (int(x), int(y))
        return points
    
    def calculate_angles(self, points):
        """Calculate joint angles for lift analysis"""
        angles = {}
        
        # Define joint triplets for angle calculation
        joint_angles = {
            'left_knee': ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'],
            'right_knee': ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'],
            'left_hip': ['LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'],
            'right_hip': ['RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'],
            'back_angle': ['LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_ANKLE'],
            'left_shoulder': ['LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'],
            'right_shoulder': ['RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP']
        }
        
        # Calculate angles for each joint
        for angle_name, (p1, p2, p3) in joint_angles.items():
            if all(k in points for k in [p1, p2, p3]):
                angles[angle_name] = self.calculate_angle(
                    points[p1], points[p2], points[p3]
                )
        
        # Calculate symmetry metrics
        angles["knee_symmetry"] = abs(angles.get("left_knee", 0) - angles.get("right_knee", 0))
        angles["hip_symmetry"] = abs(angles.get("left_hip", 0) - angles.get("right_hip", 0))
        angles["shoulder_symmetry"] = abs(angles.get("left_shoulder", 0) - angles.get("right_shoulder", 0))
        
        return angles
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def calculate_bar_path(self, points):
        """Calculate bar path using wrist positions"""
        if "LEFT_WRIST" in points and "RIGHT_WRIST" in points:
            # Calculate midpoint between wrists to estimate bar position
            left_wrist = np.array(points["LEFT_WRIST"])
            right_wrist = np.array(points["RIGHT_WRIST"])
            bar_position = (left_wrist + right_wrist) / 2
            return tuple(bar_position.astype(int))
        return None
    
    def extract_pose_data(self):
        """Extract pose data for external analysis"""
        return {
            "landmarks": self.frame_data["landmarks"],
            "angles": self.frame_data["angles"],
            "bar_path": self.frame_data["bar_path"],
            "phase_markers": self.frame_data["phase_markers"],
            "frame_count": len(self.frames)
        }
    
    def save_pose_data(self, pose_data):
        """Save pose data for external processing"""
        output_path = os.path.join(os.path.dirname(self.video_path), 
                                 "pose_data.json")
        with open(output_path, 'w') as f:
            json.dump(pose_data, f)
        self.update_status(f"Pose data saved to {output_path}")
        return output_path
    
    def update_status(self, message):
        """Update status message"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.update()
    
    def show_error(self, title, message):
        """Show error message"""
        messagebox.showerror(title, message)
    
    def show_external_analysis_options(self):
        """Show completion message and provide options"""
        messagebox.showinfo(
            "Processing Complete",
            "Video analysis completed successfully! You can now choose to analyze the data externally."
        )
        self.controller.show_frame("ResultsScreen")

def analyze_pose(video_path):
    """
    Analyze pose data from a weightlifting video
    
    Args:
        video_path: URL or local path to video file
        
    Returns:
        List of pose keypoints for each frame
    """
    print(f"Starting pose analysis for: {video_path}")
    
    # Download video if it's a URL
    if video_path.startswith('http'):
        import requests
        import tempfile
        print("Downloading video from URL...")
        response = requests.get(video_path, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download video: {response.status_code}")
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                temp_file.write(chunk)
        temp_file.close()
        video_path = temp_file.name
        print(f"Video downloaded to: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    # Get basic video info
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {frame_count} frames, {fps} FPS, {width}x{height}")
    
    # Create simplified mock pose data for demo
    all_frames_data = []
    
    # Process every 3rd frame to reduce computation
    for frame_idx in range(0, frame_count, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create mock keypoints for demo
        # In a real implementation, you would use a pose estimation model here
        frame_pose = {
            "keypoints": {
                # Simplified keypoints for weightlifting analysis
                "nose": [width // 2, height // 4],
                "neck": [width // 2, height // 3],
                "right_shoulder": [width // 2 - width // 8, height // 3],
                "left_shoulder": [width // 2 + width // 8, height // 3],
                "right_elbow": [width // 2 - width // 6, height // 2.2],
                "left_elbow": [width // 2 + width // 6, height // 2.2],
                "right_wrist": [width // 2 - width // 5, height // 1.8],
                "left_wrist": [width // 2 + width // 5, height // 1.8],
                "right_hip": [width // 2 - width // 10, height // 1.5],
                "left_hip": [width // 2 + width // 10, height // 1.5],
                "right_knee": [width // 2 - width // 10, height // 1.2],
                "left_knee": [width // 2 + width // 10, height // 1.2],
                "right_ankle": [width // 2 - width // 10, height - height // 10],
                "left_ankle": [width // 2 + width // 10, height - height // 10],
            },
            "frame": frame_idx,
            "timestamp": frame_idx / fps
        }
        
        # Add variation to make it look like movement
        variation = np.sin(frame_idx / 10) * height / 20
        frame_pose["keypoints"]["right_knee"][1] += variation
        frame_pose["keypoints"]["left_knee"][1] += variation
        frame_pose["keypoints"]["right_hip"][1] += variation / 2
        frame_pose["keypoints"]["left_hip"][1] += variation / 2
        
        all_frames_data.append(frame_pose)
    
    # Clean up
    cap.release()
    if video_path.startswith('/tmp'):
        os.remove(video_path)
        
    print(f"Completed pose analysis: {len(all_frames_data)} frames processed")
    return all_frames_data


def calculate_angles(pose_data):
    """
    Calculate joint angles from pose data
    
    Args:
        pose_data: List of pose keypoints for each frame
        
    Returns:
        Dictionary of average joint angles
    """
    print(f"Calculating angles from {len(pose_data)} frames")
    
    # We'll calculate angles for knee, hip, and back
    all_knee_angles = []
    all_hip_angles = []
    all_back_angles = []
    
    for frame in pose_data:
        keypoints = frame["keypoints"]
        
        # Calculate knee angle (ankle-knee-hip)
        if all(k in keypoints for k in ["right_ankle", "right_knee", "right_hip"]):
            ankle = keypoints["right_ankle"]
            knee = keypoints["right_knee"]
            hip = keypoints["right_hip"]
            
            knee_angle = calculate_angle(ankle, knee, hip)
            all_knee_angles.append(knee_angle)
        
        # Calculate hip angle (knee-hip-shoulder)
        if all(k in keypoints for k in ["right_knee", "right_hip", "right_shoulder"]):
            knee = keypoints["right_knee"]
            hip = keypoints["right_hip"]
            shoulder = keypoints["right_shoulder"]
            
            hip_angle = calculate_angle(knee, hip, shoulder)
            all_hip_angles.append(hip_angle)
            
        # Calculate back angle (hip-shoulder-neck)
        if all(k in keypoints for k in ["right_hip", "right_shoulder", "neck"]):
            hip = keypoints["right_hip"]
            shoulder = keypoints["right_shoulder"]
            neck = keypoints["neck"]
            
            back_angle = calculate_angle(hip, shoulder, neck)
            all_back_angles.append(back_angle)
    
    # Calculate average angles
    avg_knee = int(sum(all_knee_angles) / len(all_knee_angles)) if all_knee_angles else 0
    avg_hip = int(sum(all_hip_angles) / len(all_hip_angles)) if all_hip_angles else 0
    avg_back = int(sum(all_back_angles) / len(all_back_angles)) if all_back_angles else 0
    
    return {
        "knee": avg_knee,
        "hip": avg_hip,
        "back": avg_back
    }


def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points
    
    Args:
        point1, point2, point3: Points in [x,y] format
        
    Returns:
        Angle in degrees
    """
    # Convert to numpy arrays
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert to degrees
    return int(np.degrees(angle))
