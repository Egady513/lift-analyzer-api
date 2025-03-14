import urllib.request
import json
import socket
import sys

def test_api():
    print(f"Python version: {sys.version}")
    print(f"Testing connection to localhost...")
    
    # Check if we can resolve localhost
    try:
        print(f"Local IP: {socket.gethostbyname('localhost')}")
    except Exception as e:
        print(f"DNS resolution error: {e}")
    
    # Test the simple GET endpoint
    print("Attempting to connect to API...")
    try:
        with urllib.request.urlopen('http://127.0.0.1:5000/test') as response:
            print(f"Connection established. Status code: {response.status}")
            data = json.loads(response.read().decode())
            print(f"API Test Response: {data}")
            return True
    except Exception as e:
        print(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    print("Testing API connection...")
    result = test_api()
    print(f"API Connection {'Successful' if result else 'Failed'}") 