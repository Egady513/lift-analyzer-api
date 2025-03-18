#!/bin/bash

# Print all environment variables
echo "Environment variables:"
env | grep PORT

# Print PORT specifically
echo "PORT environment variable: ${PORT}"

# Default port to 8000 if PORT is not set or invalid
if [[ -z "${PORT}" ]] || ! [[ "${PORT}" =~ ^[0-9]+$ ]]; then
    export PORT=8000
    echo "Setting default PORT to 8000"
else
    echo "Using PORT=${PORT}"
fi

# Print final PORT value
echo "Final PORT value: ${PORT}"

# Start gunicorn with debug output
exec gunicorn --bind 0.0.0.0:${PORT} --log-level debug api_server:app 