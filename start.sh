#!/bin/bash

# Get the PORT from environment variable or default to 8000
PORT="${PORT:-8000}"

# Echo for debugging
echo "Starting server on port: $PORT"

# Start Gunicorn
exec gunicorn api_server:app --bind "0.0.0.0:$PORT" 