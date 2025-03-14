import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import json

class ResultsScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Initialize data structures
        self.current_phase = None
        self.current_frame_index = 0
        self.analysis_data = None
        
        # Setup UI components
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the results screen UI layout"""
        # Main container with grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)  # Video section gets more space
        
        # Left Panel - Navigation and Overview
        self.setup_left_panel()
        
        # Center Panel - Video Display and Controls
        self.setup_center_panel()
        
        # Right Panel - Feedback and Recommendations
        self.setup_right_panel()
        
    def setup_left_panel(self):
        """Setup the left panel with phase navigation and overview"""
        left_panel = ttk.Frame(self)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Phase Navigation
        ttk.Label(left_panel, text="Lift Phases", font=('Arial', 12, 'bold')).pack(pady=5)
        self.phase_listbox = tk.Listbox(left_panel, height=10)
        self.phase_listbox.pack(fill=tk.X, pady=5)
        self.phase_listbox.bind('<<ListboxSelect>>', self.on_phase_select)
        
        # Overall Analysis
        ttk.Label(left_panel, text="Overall Analysis", font=('Arial', 12, 'bold')).pack(pady=5)
        self.overview_text = tk.Text(left_panel, height=10, wrap=tk.WORD)
        self.overview_text.pack(fill=tk.X, pady=5)
        
        # Key Metrics
        ttk.Label(left_panel, text="Key Metrics", font=('Arial', 12, 'bold')).pack(pady=5)
        self.metrics_frame = ttk.Frame(left_panel)
        self.metrics_frame.pack(fill=tk.X, pady=5)
        
    def setup_center_panel(self):
        """Setup the center panel with enhanced video display and controls"""
        center_panel = ttk.Frame(self)
        center_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Video Display
        self.video_frame = ttk.Frame(center_panel)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.video_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Playback Controls
        controls = ttk.Frame(center_panel)
        controls.pack(fill=tk.X, pady=10)
        
        # Frame Navigation
        nav_frame = ttk.Frame(controls)
        nav_frame.pack(fill=tk.X)
        
        # Add playback controls
        self.play_button = ttk.Button(nav_frame, text="▶", width=3, command=self.toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(nav_frame, text="⏮", width=3, command=self.first_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="◀", width=3, command=self.prev_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="▶", width=3, command=self.next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="⏭", width=3, command=self.last_frame).pack(side=tk.LEFT, padx=2)
        
        # Frame Counter and Slider
        self.frame_label = ttk.Label(nav_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=20)
        
        # Speed Control
        speed_frame = ttk.Frame(controls)
        speed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.StringVar(value="1.0x")
        speeds = ["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"]
        speed_menu = ttk.OptionMenu(speed_frame, self.speed_var, "1.0x", *speeds, 
                                   command=self.update_playback_speed)
        speed_menu.pack(side=tk.LEFT, padx=5)
        
        # Frame Slider
        self.frame_slider = ttk.Scale(controls, from_=0, to=100, orient=tk.HORIZONTAL, 
                                     command=self.on_slider_change)
        self.frame_slider.pack(fill=tk.X, padx=5, pady=5)
        
        # Bind keyboard shortcuts
        self.bind_shortcuts()
        
    def setup_right_panel(self):
        """Setup the right panel with feedback and recommendations"""
        right_panel = ttk.Frame(self)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        
        # Current Frame Feedback
        ttk.Label(right_panel, text="Form Feedback", font=('Arial', 12, 'bold')).pack(pady=5)
        self.feedback_text = tk.Text(right_panel, height=8, wrap=tk.WORD)
        self.feedback_text.pack(fill=tk.X, pady=5)
        
        # Coaching Cues
        ttk.Label(right_panel, text="Coaching Cues", font=('Arial', 12, 'bold')).pack(pady=5)
        self.cues_frame = ttk.Frame(right_panel)
        self.cues_frame.pack(fill=tk.X, pady=5)
        
        # Internal Cues
        ttk.Label(self.cues_frame, text="Internal:").pack(anchor=tk.W)
        self.internal_cues = tk.Text(self.cues_frame, height=3, wrap=tk.WORD)
        self.internal_cues.pack(fill=tk.X, pady=2)
        
        # External Cues
        ttk.Label(self.cues_frame, text="External:").pack(anchor=tk.W)
        self.external_cues = tk.Text(self.cues_frame, height=3, wrap=tk.WORD)
        self.external_cues.pack(fill=tk.X, pady=2)
        
        # Recommendations
        ttk.Label(right_panel, text="Recommendations", font=('Arial', 12, 'bold')).pack(pady=5)
        self.recommendations = tk.Text(right_panel, height=8, wrap=tk.WORD)
        self.recommendations.pack(fill=tk.X, pady=5)
        
    def display_analysis(self, analysis_data):
        """Display the analysis data"""
        self.analysis_data = analysis_data
        
        # Clear previous data
        self.clear_displays()
        
        # Populate phase navigation
        self.populate_phases()
        
        # Display overall analysis
        self.display_overall_analysis()
        
        # Show first frame
        self.show_current_frame()
        
    def clear_displays(self):
        """Clear all display elements"""
        self.phase_listbox.delete(0, tk.END)
        self.overview_text.delete(1.0, tk.END)
        self.feedback_text.delete(1.0, tk.END)
        self.internal_cues.delete(1.0, tk.END)
        self.external_cues.delete(1.0, tk.END)
        self.recommendations.delete(1.0, tk.END)
        
    def populate_phases(self):
        """Populate the phase navigation listbox"""
        if not self.analysis_data:
            return
            
        phases = self.analysis_data["frames"]["phase_frames"].keys()
        for phase in phases:
            self.phase_listbox.insert(tk.END, phase)
            
    def display_overall_analysis(self):
        """Display the overall analysis summary"""
        if not self.analysis_data:
            return
            
        overall = self.analysis_data["feedback"]["overall"]
        
        # Display strengths and concerns
        self.overview_text.insert(tk.END, "STRENGTHS:\n")
        for strength in overall["strengths"]:
            self.overview_text.insert(tk.END, f"• {strength}\n")
            
        self.overview_text.insert(tk.END, "\nPRIMARY CONCERNS:\n")
        for concern in overall["primary_concerns"]:
            self.overview_text.insert(tk.END, f"• {concern}\n")
            
    def show_current_frame(self):
        """Display the current frame with annotations"""
        if not self.analysis_data or not self.current_phase:
            return
            
        phase_frames = self.analysis_data["frames"]["phase_frames"][self.current_phase]
        if not phase_frames:
            return
            
        # Get current frame data
        frame_data = phase_frames[self.current_frame_index]
        frame = frame_data["frame"].copy()  # Make copy to preserve original
        
        # Draw annotations on frame
        self.draw_annotations(frame, frame_data["annotations"])
        
        # Convert frame for display
        frame_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame_image))
        
        # Update canvas
        self.canvas.config(width=frame.shape[1], height=frame.shape[0])
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo  # Keep reference
        
        # Update frame counter and feedback
        self.frame_label.config(text=f"Frame: {self.current_frame_index + 1}/{len(phase_frames)}")
        self.update_feedback(frame_data)
        
    def draw_annotations(self, frame, annotations):
        """Draw visual annotations on the frame"""
        if not annotations:
            return frame
        
        for annotation in annotations:
            try:
                annotation_type = annotation["type"]
                location = annotation["location"]
                color = self.get_color(annotation["color"])
                message = annotation["message"]
                
                # Get landmark coordinates from stored pose data
                coordinates = self.get_landmark_coordinates(frame, location)
                if not coordinates:
                    continue
                
                # Draw based on annotation type
                if annotation_type == "arrow":
                    self.draw_arrow_annotation(frame, coordinates, annotation, color)
                elif annotation_type == "circle":
                    self.draw_circle_annotation(frame, coordinates, annotation, color)
                elif annotation_type == "highlight":
                    self.draw_highlight_annotation(frame, coordinates, annotation, color)
                
                # Add annotation message
                if message:
                    self.draw_annotation_message(frame, coordinates, message, color)
                
            except Exception as e:
                print(f"Error drawing annotation: {str(e)}")
                continue
        
        return frame

    def draw_arrow_annotation(self, frame, start_point, annotation, color):
        """Draw arrow annotation with proper positioning"""
        direction = annotation.get("direction", "up")
        length = annotation.get("length", 50)
        
        # Calculate end point based on direction and length
        end_point = self.calculate_arrow_endpoint(start_point, direction, length)
        
        # Draw arrow
        cv2.arrowedLine(
            frame, 
            start_point, 
            end_point, 
            color, 
            2,  # thickness
            cv2.LINE_AA, 
            tipLength=0.3
        )

    def draw_circle_annotation(self, frame, center, annotation, color):
        """Draw circle annotation with proper sizing"""
        radius = annotation.get("radius", 30)
        thickness = annotation.get("thickness", 2)
        
        cv2.circle(
            frame,
            center,
            radius,
            color,
            thickness,
            cv2.LINE_AA
        )

    def draw_highlight_annotation(self, frame, coordinates, annotation, color):
        """Draw highlight annotation with proper region"""
        region = self.calculate_highlight_region(frame, coordinates, annotation)
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            region[0],
            region[1],
            color,
            -1  # filled rectangle
        )
        
        # Blend overlay with original frame
        alpha = annotation.get("opacity", 0.3)
        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0,
            frame
        )

    def draw_annotation_message(self, frame, coordinates, message, color):
        """Draw annotation message with smart positioning and improved readability"""
        # Split message if too long
        message_lines = self.wrap_message(message, max_width=30)
        
        # Calculate optimal position for text block
        text_block_position = self.calculate_text_block_position(
            frame, 
            coordinates, 
            message_lines,
            padding=5
        )
        
        # Draw text block with background
        self.draw_text_block(
            frame,
            text_block_position,
            message_lines,
            color,
            padding=5,
            opacity=0.8
        )

    def wrap_message(self, message, max_width=30):
        """Wrap message into multiple lines if too long"""
        words = message.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines

    def calculate_text_block_position(self, frame, coordinates, text_lines, padding):
        """Calculate optimal position for text block to avoid overlaps"""
        h, w = frame.shape[:2]
        font_scale = 0.6
        line_height = 25  # Pixels between lines
        
        # Calculate text block dimensions
        max_line_width = 0
        for line in text_lines:
            (line_width, text_height), _ = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                2
            )
            max_line_width = max(max_line_width, line_width)
        
        block_width = max_line_width + (padding * 2)
        block_height = (len(text_lines) * line_height) + (padding * 2)
        
        # Try different positions in order of preference
        positions = [
            self.try_position_above,
            self.try_position_below,
            self.try_position_left,
            self.try_position_right,
            self.try_position_diagonal
        ]
        
        for position_func in positions:
            pos = position_func(
                frame_size=(w, h),
                coordinates=coordinates,
                block_size=(block_width, block_height),
                padding=padding
            )
            if pos:
                return pos
        
        # Fallback: return position above with possible overflow handling
        return (
            max(padding, min(w - block_width - padding, coordinates[0] - block_width//2)),
            max(block_height + padding, coordinates[1] - block_height - 10)
        )

    def try_position_above(self, frame_size, coordinates, block_size, padding):
        """Try to position text block above the point"""
        w, h = frame_size
        block_w, block_h = block_size
        x = coordinates[0] - block_w//2
        y = coordinates[1] - block_h - 10
        
        # Check if position is valid
        if (x >= padding and x + block_w <= w - padding and 
            y >= padding and y + block_h <= coordinates[1] - 5):
            return (x, y)
        return None

    def try_position_below(self, frame_size, coordinates, block_size, padding):
        """Try to position text block below the point"""
        w, h = frame_size
        block_w, block_h = block_size
        x = coordinates[0] - block_w//2
        y = coordinates[1] + 10
        
        if (x >= padding and x + block_w <= w - padding and 
            y >= coordinates[1] + 5 and y + block_h <= h - padding):
            return (x, y)
        return None

    def try_position_left(self, frame_size, coordinates, block_size, padding):
        """Try to position text block to the left of the point"""
        w, h = frame_size
        block_w, block_h = block_size
        x = coordinates[0] - block_w - 10
        y = coordinates[1] - block_h//2
        
        if (x >= padding and x + block_w <= coordinates[0] - 5 and 
            y >= padding and y + block_h <= h - padding):
            return (x, y)
        return None

    def try_position_right(self, frame_size, coordinates, block_size, padding):
        """Try to position text block to the right of the point"""
        w, h = frame_size
        block_w, block_h = block_size
        x = coordinates[0] + 10
        y = coordinates[1] - block_h//2
        
        if (x >= coordinates[0] + 5 and x + block_w <= w - padding and 
            y >= padding and y + block_h <= h - padding):
            return (x, y)
        return None

    def try_position_diagonal(self, frame_size, coordinates, block_size, padding):
        """Try to position text block diagonally from the point"""
        w, h = frame_size
        block_w, block_h = block_size
        x = coordinates[0] + 10
        y = coordinates[1] - block_h - 10
        
        if (x >= padding and x + block_w <= w - padding and 
            y >= padding and y + block_h <= h - padding):
            return (x, y)
        return None

    def draw_text_block(self, frame, position, text_lines, color, padding=5, opacity=0.8):
        """Draw text block with background and border"""
        x, y = position
        font_scale = 0.6
        line_height = 25
        
        # Calculate block dimensions
        max_line_width = 0
        for line in text_lines:
            (line_width, text_height), _ = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                2
            )
            max_line_width = max(max_line_width, line_width)
        
        block_width = max_line_width + (padding * 2)
        block_height = (len(text_lines) * line_height) + (padding * 2)
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + block_width, y + block_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        
        # Draw border
        cv2.rectangle(
            frame,
            (x, y),
            (x + block_width, y + block_height),
            color,
            1,
            cv2.LINE_AA
        )
        
        # Draw text lines
        for i, line in enumerate(text_lines):
            text_y = y + padding + ((i + 1) * line_height)
            cv2.putText(
                frame,
                line,
                (x + padding, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                2,
                cv2.LINE_AA
            )

    def get_landmark_coordinates(self, frame, location):
        """Get coordinates for body landmark from stored pose data"""
        try:
            # Get pose landmarks from frame data
            if not hasattr(frame, 'pose_landmarks'):
                return None
            
            landmarks = frame.pose_landmarks
            if not landmarks:
                return None
            
            # Map location string to MediaPipe pose landmark
            landmark_map = {
                "nose": 0,
                "left_shoulder": 11,
                "right_shoulder": 12,
                "left_elbow": 13,
                "right_elbow": 14,
                "left_wrist": 15,
                "right_wrist": 16,
                "left_hip": 23,
                "right_hip": 24,
                "left_knee": 25,
                "right_knee": 26,
                "left_ankle": 27,
                "right_ankle": 28,
                # Add more mappings as needed
            }
            
            if location.lower() not in landmark_map:
                return None
            
            landmark_index = landmark_map[location.lower()]
            landmark = landmarks.landmark[landmark_index]
            
            # Convert normalized coordinates to pixel coordinates
            h, w = frame.shape[:2]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            return (x, y)
            
        except Exception as e:
            print(f"Error getting landmark coordinates: {str(e)}")
            return None

    def calculate_arrow_endpoint(self, start_point, direction, length=50):
        """Calculate arrow endpoint based on direction and length"""
        direction_vectors = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
            "up_left": (-0.707, -0.707),
            "up_right": (0.707, -0.707),
            "down_left": (-0.707, 0.707),
            "down_right": (0.707, 0.707)
        }
        
        if direction not in direction_vectors:
            return start_point
        
        dx, dy = direction_vectors[direction]
        end_x = int(start_point[0] + dx * length)
        end_y = int(start_point[1] + dy * length)
        
        return (end_x, end_y)

    def calculate_highlight_region(self, frame, center, annotation):
        """Calculate highlight region based on annotation parameters"""
        width = annotation.get("width", 100)
        height = annotation.get("height", 100)
        
        x1 = max(0, center[0] - width//2)
        y1 = max(0, center[1] - height//2)
        x2 = min(frame.shape[1], center[0] + width//2)
        y2 = min(frame.shape[0], center[1] + height//2)
        
        return ((x1, y1), (x2, y2))

    def get_color(self, color_name):
        """Convert color name to BGR values"""
        colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255)
        }
        return colors.get(color_name.lower(), (255, 255, 255))

    def update_feedback(self, frame_data):
        """Update the feedback display for the current frame"""
        # Clear previous feedback
        self.feedback_text.delete(1.0, tk.END)
        self.internal_cues.delete(1.0, tk.END)
        self.external_cues.delete(1.0, tk.END)
        
        # Display annotations feedback
        for annotation in frame_data["feedback"]:
            feedback = annotation["feedback"]
            self.feedback_text.insert(tk.END, f"ISSUE: {feedback['issue']}\n")
            self.feedback_text.insert(tk.END, f"IMPACT: {feedback['impact']}\n")
            self.feedback_text.insert(tk.END, f"CORRECTION: {feedback['correction']}\n\n")
            
        # Display cues
        for cue in frame_data["cues"]["internal"]:
            self.internal_cues.insert(tk.END, f"• {cue}\n")
        for cue in frame_data["cues"]["external"]:
            self.external_cues.insert(tk.END, f"• {cue}\n")
            
        # Update recommendations
        self.update_recommendations(frame_data)
        
    def update_recommendations(self, frame_data):
        """Update the recommendations display"""
        self.recommendations.delete(1.0, tk.END)
        
        # Display related drills
        self.recommendations.insert(tk.END, "RECOMMENDED DRILLS:\n")
        for drill in frame_data["related_drills"]:
            self.recommendations.insert(tk.END, f"• {drill['name']}\n")
            self.recommendations.insert(tk.END, f"  Purpose: {drill['purpose']}\n")
            self.recommendations.insert(tk.END, f"  Sets/Reps: {drill['sets_reps']}\n\n")
            
    def on_phase_select(self, event):
        """Handle phase selection"""
        selection = self.phase_listbox.curselection()
        if not selection:
            return
            
        self.current_phase = self.phase_listbox.get(selection[0])
        self.current_frame_index = 0
        self.show_current_frame()
        
    def next_frame(self):
        """Show next frame in current phase"""
        if not self.current_phase:
            return
            
        phase_frames = self.analysis_data["frames"]["phase_frames"][self.current_phase]
        if self.current_frame_index < len(phase_frames) - 1:
            self.current_frame_index += 1
            self.show_current_frame()
            
    def prev_frame(self):
        """Show previous frame in current phase"""
        if not self.current_phase and self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.show_current_frame()

    def bind_shortcuts(self):
        """Bind keyboard shortcuts for navigation"""
        self.master.bind('<Left>', lambda e: self.prev_frame())
        self.master.bind('<Right>', lambda e: self.next_frame())
        self.master.bind('<space>', lambda e: self.toggle_playback())
        self.master.bind('<Home>', lambda e: self.first_frame())
        self.master.bind('<End>', lambda e: self.last_frame())
        self.master.bind('<Up>', lambda e: self.prev_phase())
        self.master.bind('<Down>', lambda e: self.next_phase())

    def toggle_playback(self):
        """Toggle video playback"""
        if hasattr(self, 'playing') and self.playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """Start automatic playback"""
        self.playing = True
        self.play_button.configure(text="⏸")
        self.play_next_frame()

    def stop_playback(self):
        """Stop automatic playback"""
        self.playing = False
        self.play_button.configure(text="▶")
        if hasattr(self, 'after_id'):
            self.after_cancel(self.after_id)

    def play_next_frame(self):
        """Play next frame with delay based on speed"""
        if not self.playing:
            return
        
        if self.current_frame_index < self.get_total_frames() - 1:
            self.next_frame()
            speed = float(self.speed_var.get().replace('x', ''))
            delay = int(1000 / (30 * speed))  # Assuming 30 FPS
            self.after_id = self.after(delay, self.play_next_frame)
        else:
            self.stop_playback()

    def update_playback_speed(self, *args):
        """Update playback speed"""
        if hasattr(self, 'playing') and self.playing:
            self.stop_playback()
            self.start_playback()

    def on_slider_change(self, value):
        """Handle frame slider change"""
        if not self.analysis_data or not self.current_phase:
            return
        
        frame_index = int(float(value))
        if frame_index != self.current_frame_index:
            self.current_frame_index = frame_index
            self.show_current_frame()

    def first_frame(self):
        """Go to first frame"""
        if self.current_phase:
            self.current_frame_index = 0
            self.show_current_frame()

    def last_frame(self):
        """Go to last frame"""
        if self.current_phase:
            phase_frames = self.analysis_data["frames"]["phase_frames"][self.current_phase]
            self.current_frame_index = len(phase_frames) - 1
            self.show_current_frame()

    def next_phase(self):
        """Go to next phase"""
        if not self.analysis_data:
            return
        
        phases = list(self.analysis_data["frames"]["phase_frames"].keys())
        if not phases:
            return
        
        current_index = phases.index(self.current_phase) if self.current_phase else -1
        if current_index < len(phases) - 1:
            self.current_phase = phases[current_index + 1]
            self.current_frame_index = 0
            self.phase_listbox.selection_clear(0, tk.END)
            self.phase_listbox.selection_set(current_index + 1)
            self.show_current_frame()

    def prev_phase(self):
        """Go to previous phase"""
        if not self.analysis_data or not self.current_phase:
            return
        
        phases = list(self.analysis_data["frames"]["phase_frames"].keys())
        current_index = phases.index(self.current_phase)
        if current_index > 0:
            self.current_phase = phases[current_index - 1]
            self.current_frame_index = 0
            self.phase_listbox.selection_clear(0, tk.END)
            self.phase_listbox.selection_set(current_index - 1)
            self.show_current_frame()

    def get_total_frames(self):
        """Get total number of frames in current phase"""
        if not self.analysis_data or not self.current_phase:
            return 0
        return len(self.analysis_data["frames"]["phase_frames"][self.current_phase])

    def update_slider(self):
        """Update slider position and range"""
        total_frames = self.get_total_frames()
        self.frame_slider.configure(to=total_frames - 1)
        self.frame_slider.set(self.current_frame_index) 