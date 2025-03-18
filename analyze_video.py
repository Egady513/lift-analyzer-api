"""Main video analysis module that integrates all components."""
import logging
import os
from analysis.video_processor import process_video, process_video_simplified
from analysis.lift_classifier import detect_exercise_type

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if OpenCV is available
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available. Some functions will be limited.")

def analyze_video(video_url, chat_input=None):
    """Analyze a lifting video and return feedback.
    
    Args:
        video_url: URL to the video to analyze
        chat_input: Optional text describing the video/exercise
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info(f"Starting analysis of video: {video_url}")
    
    if not OPENCV_AVAILABLE:
        logger.warning("OpenCV not available, using simplified processing")
        return process_video_simplified(video_url, chat_input)
    
    try:
        # Process the video with full analysis
        result = process_video(video_url, chat_input)
        return result
    except Exception as e:
        logger.error(f"Error in video analysis: {str(e)}")
        # Fall back to simplified processing
        logger.info("Falling back to simplified processing")
        return process_video_simplified(video_url, chat_input)

# For direct testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
        chat_input = sys.argv[2] if len(sys.argv) > 2 else "deadlift analysis"
        result = analyze_video(video_url, chat_input)
        print(result)
    else:
        print("Usage: python analyze_video.py <video_url> [chat_input]") 