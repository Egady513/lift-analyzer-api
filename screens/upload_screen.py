import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from pathlib import Path
import os
from PIL import Image, ImageTk
import tkinter.messagebox
from config import Config

class UploadScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg='#1E1E1E')  # Darker background
        
        # Check for API key before proceeding
        self.check_api_key()
        
        # Verify FFmpeg is available
        if not self.verify_ffmpeg():
            tk.messagebox.showerror(
                "Error",
                "FFmpeg not found. Please ensure FFmpeg is installed in C:\\FFmpeg\\bin"
            )
        
        # Load header image
        image_path = os.path.join(os.path.dirname(__file__), "../assets/header.jpg")
        self.header_image = Image.open(image_path)
        # Resize image to fit screen width while maintaining aspect ratio
        basewidth = 1000  # Increased width for better quality
        wpercent = (basewidth/float(self.header_image.size[0]))
        hsize = int((float(self.header_image.size[1])*float(wpercent)))
        self.header_image = self.header_image.resize((basewidth,hsize), Image.Resampling.LANCZOS)
        self.header_photo = ImageTk.PhotoImage(self.header_image)
        
        self.setup_ui()
        self.video_path = None
        
    def setup_ui(self):
        # Create a canvas with scrollbar
        canvas = tk.Canvas(self, bg='#1E1E1E', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        # Main container with padding
        main_container = tk.Frame(canvas, bg='#1E1E1E', padx=40, pady=20)
        
        # Configure the canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create a window in the canvas for the main container
        canvas_window = canvas.create_window((0, 0), window=main_container, anchor="nw", width=canvas.winfo_width())
        
        # Update scroll region when window size changes
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def configure_window_size(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        # Bind events
        main_container.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_window_size)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Header Image - REDUCED SIZE
        header_frame = tk.Frame(main_container, bg='#1E1E1E', bd=0)
        header_frame.pack(pady=(0,20))
        
        # Resize header image to smaller size
        basewidth = 800  # Reduced from 1000
        wpercent = (basewidth/float(self.header_image.size[0]))
        hsize = int((float(self.header_image.size[1])*float(wpercent)))
        self.header_image = self.header_image.resize((basewidth,hsize), Image.Resampling.LANCZOS)
        self.header_photo = ImageTk.PhotoImage(self.header_image)
        
        header_label = tk.Label(
            header_frame,
            image=self.header_photo,
            bg='#1E1E1E',
            bd=0
        )
        header_label.pack()

        # App Description
        description = tk.Label(
            main_container,
            text="LiftIQ uses advanced AI to analyze your lifting form and provide\n" +
                 "real-time feedback to help improve your technique and prevent injury.",
            font=('Helvetica', 14),
            fg='#CCCCCC',
            bg='#1E1E1E',
            justify=tk.CENTER
        )
        description.pack(pady=(0,30))

        # Upload button
        self.upload_btn = tk.Button(
            main_container,
            text="Upload Your Lift",
            command=self.select_video,
            font=('Helvetica', 32, 'bold'),
            fg='white',
            bg='#2ECC71',
            activebackground='#27AE60',
            activeforeground='white',
            padx=50,
            pady=20,
            relief='flat',
            cursor='hand2'
        )
        self.upload_btn.pack(pady=20)

        # Frame for video preview
        self.preview_frame = tk.Frame(main_container, bg='#1E1E1E')
        self.preview_frame.pack(pady=20, expand=True, fill='both')

        # Selected file label
        self.file_label = tk.Label(
            main_container,
            text="",
            font=('Helvetica', 12),
            fg='#CCCCCC',
            bg='#1E1E1E',
            wraplength=500
        )
        self.file_label.pack(pady=10)

        # Analysis button (hidden initially)
        self.analyze_btn = tk.Button(
            main_container,
            text="Analyze My Video",
            command=self.process_video,
            state='disabled',
            font=('Helvetica', 16, 'bold'),
            bg='#2ECC71',
            fg='white',
            padx=40,
            pady=15,
            relief='flat',
            activebackground='#27AE60',
            cursor='hand2'
        )
        # Will be packed when video is selected

    def select_video(self):
        """Handle video file selection"""
        filetypes = [
            ('Video files', '*.mp4 *.mov *.avi')
        ]
        file_path = filedialog.askopenfilename(
            title="Select a Video",
            filetypes=filetypes,
            initialdir=os.path.expanduser("~")
        )
        
        if file_path:
            self.video_path = file_path
            self.file_label.configure(text=f"Selected: {Path(file_path).name}")
            
            # Generate and show video thumbnail
            self.show_video_thumbnail(file_path)
            
            # Show analyze button
            self.analyze_btn.pack(pady=20)
            self.analyze_btn.configure(state='normal')

    def show_video_thumbnail(self, video_path):
        """Generate and display video thumbnail"""
        # Clear previous thumbnail
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        # Generate thumbnail using OpenCV
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize maintaining aspect ratio
            height, width = frame.shape[:2]
            max_size = 400
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size/width))
            else:
                new_height = max_size
                new_width = int(width * (max_size/height))
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PhotoImage
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            
            # Display thumbnail
            thumbnail_label = tk.Label(
                self.preview_frame,
                image=photo,
                bg='#1E1E1E'
            )
            thumbnail_label.image = photo  # Keep reference
            thumbnail_label.pack(pady=10)
        
        cap.release()

    def verify_video_length(self):
        """Check if video is under 60 seconds"""
        if not self.video_path:
            return
            
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps
        cap.release()

        if duration > 60:
            self.warning_label.pack(pady=5)
            self.trim_btn.pack(pady=5)
            self.continue_btn.configure(state='disabled')
        else:
            self.warning_label.pack_forget()
            self.trim_btn.pack_forget()
            self.continue_btn.configure(state='normal')
            
    def trim_video(self):
        """Open video trimming interface"""
        trim_window = tk.Toplevel(self)
        trim_window.title("Trim Video")
        trim_window.configure(bg='#2B2B2B')
        trim_window.geometry("800x400")

        # Video info
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        # Time selection frame
        time_frame = tk.Frame(trim_window, bg='#2B2B2B')
        time_frame.pack(pady=20)

        # Start time selection
        tk.Label(
            time_frame,
            text="Start Time (seconds):",
            fg='white',
            bg='#2B2B2B',
            font=('Helvetica', 10)
        ).pack(side=tk.LEFT, padx=5)

        start_var = tk.StringVar(value="0.0")
        start_entry = tk.Entry(
            time_frame,
            textvariable=start_var,
            width=10
        )
        start_entry.pack(side=tk.LEFT, padx=5)

        # End time selection
        tk.Label(
            time_frame,
            text="End Time (seconds):",
            fg='white',
            bg='#2B2B2B',
            font=('Helvetica', 10)
        ).pack(side=tk.LEFT, padx=5)

        end_var = tk.StringVar(value=str(round(duration, 1)))
        end_entry = tk.Entry(
            time_frame,
            textvariable=end_var,
            width=10
        )
        end_entry.pack(side=tk.LEFT, padx=5)

        # Preview slider
        preview_frame = tk.Frame(trim_window, bg='#2B2B2B')
        preview_frame.pack(pady=20, fill=tk.X, padx=20)

        preview_slider = ttk.Scale(
            preview_frame,
            from_=0,
            to=duration,
            orient=tk.HORIZONTAL,
            length=700
        )
        preview_slider.pack(fill=tk.X)

        # Time labels
        time_label = tk.Label(
            preview_frame,
            text=f"0:00 / {int(duration//60)}:{int(duration%60):02d}",
            fg='white',
            bg='#2B2B2B'
        )
        time_label.pack(pady=5)

        def update_time_label(event):
            current = preview_slider.get()
            time_label.config(
                text=f"{int(current//60)}:{int(current%60):02d} / {int(duration//60)}:{int(duration%60):02d}"
            )

        preview_slider.bind("<Motion>", update_time_label)

        def trim_and_save():
            try:
                start_time = float(start_var.get())
                end_time = float(end_var.get())
                
                if start_time >= end_time or start_time < 0 or end_time > duration:
                    raise ValueError("Invalid time range")

                # Create output filename
                file_path = Path(self.video_path)
                output_path = file_path.parent / f"{file_path.stem}_trimmed{file_path.suffix}"
                
                # Use full path to FFmpeg
                ffmpeg_path = r"C:\FFmpeg\bin\ffmpeg.exe"
                
                # Use FFmpeg with full path
                os.system(f'"{ffmpeg_path}" -i "{self.video_path}" -ss {start_time} -t {end_time-start_time} -c copy "{output_path}"')
                
                # Update video path and verify length
                self.video_path = str(output_path)
                self.verify_video_length()
                
                # Close trim window
                trim_window.destroy()
                
            except ValueError as e:
                tk.messagebox.showerror(
                    "Error",
                    "Please enter valid start and end times"
                )

        # Trim button
        trim_btn = tk.Button(
            trim_window,
            text="Trim Video",
            command=trim_and_save,
            bg='#4A90E2',
            fg='white',
            font=('Helvetica', 12),
            padx=20,
            pady=10,
            relief='flat'
        )
        trim_btn.pack(pady=20)

    def compress_video(self, input_path):
        """Compress video using FFmpeg H.265 encoding"""
        try:
            ffmpeg_path = r"C:\FFmpeg\bin\ffmpeg.exe"
            output_path = str(Path(input_path).parent / f"{Path(input_path).stem}_compressed.mp4")
            
            # FFmpeg command for H.265 compression with good quality/size balance
            compress_command = f'"{ffmpeg_path}" -i "{input_path}" -c:v libx265 -crf 28 -preset medium -c:a aac -b:a 128k "{output_path}"'
            os.system(compress_command)
            
            return output_path
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to compress video: {str(e)}")
            return None

    def process_video(self):
        """Handle video processing and navigation to next screen"""
        if not self.video_path:
            return
        
        # Get the processing screen instance
        processing_screen = self.controller.frames["ProcessingScreen"]
        
        # Navigate to processing screen and start processing
        self.controller.show_frame("ProcessingScreen")
        processing_screen.start_processing(self.video_path)

    def verify_ffmpeg(self):
        """Verify FFmpeg is available"""
        ffmpeg_path = r"C:\FFmpeg\bin\ffmpeg.exe"
        return os.path.exists(ffmpeg_path)

    def check_api_key(self):
        """Check if API key is configured"""
        config = Config()
        if not config.api_key:
            self.show_api_key_dialog()
    
    def show_api_key_dialog(self):
        """Show dialog for entering OpenAI API key"""
        dialog = tk.Toplevel(self)
        dialog.title("OpenAI API Key Setup")
        dialog.geometry("500x250")
        dialog.configure(bg='#1E1E1E')
        
        # Make dialog modal
        dialog.transient(self)
        dialog.grab_set()
        
        # Instructions
        instructions = ttk.Label(
            dialog,
            text=(
                "Please enter your OpenAI API key for LiftIQ:\n\n"
                "1. Go to https://platform.openai.com/api-keys\n"
                "2. Click 'Create new secret key'\n"
                "3. Copy the key and paste it below"
            ),
            wraplength=450,
            justify=tk.LEFT
        )
        instructions.pack(pady=20, padx=20)
        
        # API Key Entry
        key_frame = ttk.Frame(dialog)
        key_frame.pack(fill=tk.X, padx=20)
        
        key_entry = ttk.Entry(key_frame, width=50)
        key_entry.pack(side=tk.LEFT, pady=10)
        
        # Paste button
        def paste_key():
            key_entry.delete(0, tk.END)
            key_entry.insert(0, dialog.clipboard_get())
        
        paste_btn = ttk.Button(
            key_frame,
            text="Paste",
            command=paste_key
        )
        paste_btn.pack(side=tk.LEFT, padx=5)
        
        # Save button
        def save_key():
            api_key = key_entry.get().strip()
            if not api_key:
                messagebox.showerror(
                    "Error",
                    "Please enter an API key"
                )
                return
            
            if not api_key.startswith("sk-"):
                messagebox.showerror(
                    "Error",
                    "Invalid API key format. Key should start with 'sk-'"
                )
                return
            
            try:
                config = Config()
                config.save_config(api_key)
                messagebox.showinfo(
                    "Success",
                    "API key saved successfully!"
                )
                dialog.destroy()
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to save API key: {str(e)}"
                )
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)
        
        save_btn = ttk.Button(
            btn_frame,
            text="Save API Key",
            command=save_key
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(
            btn_frame,
            text="Cancel",
            command=dialog.destroy
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Center dialog on screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')
