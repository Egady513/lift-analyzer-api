FROM python:3.9-slim

# Install required packages
RUN pip install flask gunicorn

WORKDIR /app

# Copy just what we need for the minimal test
COPY test.py .
COPY Procfile .

# Railway will use the Procfile to determine how to start the app
CMD ["gunicorn", "test:app"]
