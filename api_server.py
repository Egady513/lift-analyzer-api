from flask import Flask, request, jsonify
import json
import os
from flask_cors import CORS
# import torch
# from analyze_video import process_video
import logging
import sys
import tempfile
import requests
import base64
import cv2
import mediapipe as mp
import numpy as np
from screens.processing_screen import process_video, analyze_pose_angles
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add basic authentication if needed
@app.before_request
def authenticate():
    # Simple auth example - replace with proper auth in production
    auth = request.headers.get('Authorization')
    if not auth or auth != os.environ.get('API_KEY', 'default-dev-key'):
        return jsonify({"error": "Unauthorized"}), 401

ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
SKIP_MODEL_LOAD = ENVIRONMENT == "production"

# Initialize vector database
def initialize_knowledge_base():
    """Initialize the vector database with your lifting documentation"""
    # Load documents from a directory
    loader = DirectoryLoader("./lifting_knowledge", glob="**/*.md")
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    
    return vector_store

# Create a global vector store
vector_store = initialize_knowledge_base()

# Add this function to api_server.py
def get_relevant_knowledge(exercise_type, angles):
    """Retrieve relevant knowledge for the specific exercise and angles"""
    query = f"technique advice for {exercise_type} with back angle {angles['back']}, hip angle {angles['hip']}, knee angle {angles['knee']}"
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

@app.route('/process-video', methods=['POST'])
def process_video_endpoint():
    data = request.json
    video_url = data.get('video_url')
    chat_input = data.get('chatInput', 'Please analyze this weightlifting video')
    
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    
    try:
        # Use simplified processing that doesn't require mediapipe
        result = process_video_simplified(video_url, chat_input)
        
        # Add relevant knowledge from the RAG system
        exercise_type = detect_exercise_type(chat_input)
        relevant_knowledge = get_relevant_knowledge(
            exercise_type, 
            result["pose_data"]["angles"]
        )
        
        # Add the knowledge to the result
        result["pose_data"]["relevant_knowledge"] = relevant_knowledge
        
        # Return the results as JSON
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error processing video: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/apply-analysis', methods=['POST'])
def apply_analysis():
    """Apply external analysis results to video"""
    data = request.json
    video_path = data.get('video_path')
    analysis = data.get('analysis')
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    try:
        # Create annotated video with analysis
        from annotate_video import create_annotated_video
        result_path = create_annotated_video(video_path, analysis)
        
        return jsonify({
            "status": "success", 
            "result_path": result_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    try:
        logger.info("Test endpoint called")
        return jsonify({
            "status": "success",
            "message": "API is working correctly!"
        })
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        "status": "success",
        "message": "API is available",
        "environment": os.environ.get("ENVIRONMENT", "unknown")
    })

@app.route('/process-video-base64', methods=['POST'])
def process_video_base64():
    data = request.json
    video_base64 = data.get('video_base64')
    filename = data.get('filename', 'video.mp4')
    
    if not video_base64:
        return jsonify({"error": "No video data provided"}), 400
    
    try:
        # Decode base64 to file
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, filename)
        
        with open(temp_video_path, 'wb') as f:
            f.write(base64.b64decode(video_base64))
        
        # Process video and return results
        # ...
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test-url', methods=['POST'])
def test_url():
    """Test if a URL can be downloaded"""
    data = request.json
    video_url = data.get('video_url')
    
    if not video_url:
        return jsonify({"error": "No URL provided"}), 400
        
    try:
        response = requests.head(video_url, timeout=10)
        return jsonify({
            "status": "success" if response.status_code < 400 else "error",
            "status_code": response.status_code,
            "content_type": response.headers.get('Content-Type', 'unknown'),
            "message": "URL is accessible" if response.status_code < 400 else "URL returned an error"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use fixed port 8000 instead of environment variable
    port = 8000
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host='0.0.0.0', port=port, debug=debug) 