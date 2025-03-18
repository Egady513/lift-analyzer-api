#!/bin/bash
# Print debugging information
echo "Environment variables:"
printenv | grep PORT

# Check if PORT is set
if [ -z "$PORT" ]; then
  echo "PORT is not set, using default 8000"
  PORT=8000
else
  echo "Using PORT=$PORT from environment"
fi

# Execute gunicorn with the determined port
exec gunicorn api_server:app --bind "0.0.0.0:$PORT" 