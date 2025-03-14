import requests

url = "http://127.0.0.1:5000/process-video"
data = {"video_path": "C:\\Users\\eddie.gady\\OneDrive - Centric Consulting\\cursor\\lift_analyzer\\sample_video.mp4"}

response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}") 