import os
import sys
from flask import Flask

# Create a minimal Flask app
app = Flask(__name__)

@app.route('/')
def hello():
    # Return environment debug info
    port = os.environ.get('PORT', 'not set')
    return {
        "message": "Hello from Railway test server",
        "port_env_var": port,
        "environment": {k: v for k, v in os.environ.items() if k in ['PORT', 'RAILWAY_SERVICE_NAME', 'RAILWAY_ENVIRONMENT']},
        "python_version": sys.version
    }

# Print debugging info during import
print("=" * 40)
print(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")
print(f"Python version: {sys.version}")
print("=" * 40)

# Note: No if __name__ == '__main__' block needed when using gunicorn 