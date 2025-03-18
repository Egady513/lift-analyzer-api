import os
import sys
from flask import Flask

# Create a minimal Flask app
app = Flask(__name__)

@app.route('/')
def hello():
    # Return environment debug info
    env_vars = {k: v for k, v in os.environ.items()}
    return {
        "message": "Hello from minimal test server",
        "environment": env_vars,
        "python_version": sys.version,
        "port": 8000
    }

if __name__ == '__main__':
    # Print debugging info
    print("=" * 40)
    print("STARTING MINIMAL TEST SERVER")
    print(f"Python version: {sys.version}")
    print(f"Environment variables: {dict(os.environ)}")
    print("=" * 40)
    
    # Run with hardcoded port
    app.run(host='0.0.0.0', port=8000, debug=True) 