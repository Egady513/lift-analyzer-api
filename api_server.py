from flask import Flask, request, jsonify
import json
import os
from flask_cors import CORS
from analyze_video import process_video
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
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

@app.route('/process-video', methods=['POST'])
def process_video_endpoint():
    """Process video from a given path"""
    data = request.json
    video_path = data.get('video_path')
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    try:
        pose_data = process_video(video_path)
        return jsonify({
            "status": "success",
            "pose_data": pose_data,
            "human_readable": f"Successfully processed video with {pose_data['frame_count']} frames. Pose data saved to {pose_data['pose_data_path']}.",
            "next_steps": [
                "Analyze the pose data to identify form issues",
                "Generate personalized feedback",
                "Create annotated video"
            ]
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "error": str(e),
            "human_readable": f"Failed to process video: {str(e)}"
        }), 500

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
def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return jsonify({
        "status": "success",
        "message": "API is working correctly!"
    })

if __name__ == '__main__':
    # Use production WSGI server if available
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host='0.0.0.0', port=port, debug=debug) 