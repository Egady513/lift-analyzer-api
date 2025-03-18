#!/bin/bash

# Print all environment variables for debugging
echo "===== ENVIRONMENT VARIABLES ====="
printenv
echo "================================="

# Set a fixed port regardless of what Railway provides
export PORT=8000
echo "Using fixed PORT=8000"

# Start gunicorn with the fixed port
exec gunicorn api_server:app --bind 0.0.0.0:8000
