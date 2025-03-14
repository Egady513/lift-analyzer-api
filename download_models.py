import os
import requests
import subprocess

def download_file(url, path):
    """Download a file from URL to path"""
    print(f"Downloading {url} to {path}")
    if os.path.exists(path):
        print(f"File already exists at {path}")
        return
        
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Download complete: {path}")

if __name__ == "__main__":
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download YOLOv7 pose model
    download_file(
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt",
        "yolov7-w6-pose.pt"
    )
    
    print("All models downloaded successfully!") 