FROM python:3.9-slim

# Install Flask and Gunicorn
RUN pip install flask gunicorn

WORKDIR /app
COPY test.py .

# Use shell form to properly expand $PORT environment variable
CMD gunicorn --bind 0.0.0.0:$PORT test:app
