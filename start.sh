#!/bin/bash
echo "Starting on port 8000"
exec gunicorn api_server:app --bind 0.0.0.0:8000 