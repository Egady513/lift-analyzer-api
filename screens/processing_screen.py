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
            self.update_status("Starting video processing...")
            
            # Load and process video frames
            self.load_video_frames(video_path)
            
            # Extract pose data only, no GPT analysis
            pose_data = self.extract_pose_data()
            
            # Save pose data for external analysis
            self.save_pose_data(pose_data)
            
            # Show completion and provide options
            self.show_external_analysis_options()
            
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
