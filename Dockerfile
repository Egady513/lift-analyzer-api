FROM python:3.9-slim

# Install only Flask
RUN pip install flask

WORKDIR /app
COPY test.py .

# Run the test server directly with Python
CMD ["python", "test.py"]
