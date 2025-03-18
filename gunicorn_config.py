import os

# Get the port from environment, default to 8000
port = int(os.environ.get("PORT", 8000))

# Gunicorn config
bind = f"0.0.0.0:{port}"
workers = 2
threads = 4
timeout = 120 