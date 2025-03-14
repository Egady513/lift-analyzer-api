import tkinter as tk
from screens.upload_screen import UploadScreen
from screens.processing_screen import ProcessingScreen

class LiftAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()

        # Configure main window
        self.title("LiftIQ - AI Lifting Analysis")
        self.geometry("1000x800")
        self.configure(bg='#1E1E1E')

        # Create a container to hold all screens
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        # Dictionary to store screens
        self.frames = {}

        # Initialize all screens
        for F in (UploadScreen, ProcessingScreen):
            frame = F(container, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Show initial screen
        self.show_frame("UploadScreen")

    def show_frame(self, screen_name):
        """Bring specified screen to the front"""
        frame = self.frames[screen_name]
        frame.tkraise()

if __name__ == "__main__":
    app = LiftAnalyzer()
    app.mainloop() 