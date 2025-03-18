"""Exercise type classification from input text or video frames."""
import logging

# Configure logging
logger = logging.getLogger(__name__)

def detect_exercise_type(chat_input):
    """Detect the type of exercise from the chat input"""
    if not chat_input:
        logger.warning("No chat input provided, defaulting to deadlift")
        return "deadlift"
        
    chat_input = chat_input.lower()

    if any(word in chat_input for word in ["deadlift", "dead lift"]):
        return "deadlift"
    elif any(word in chat_input for word in ["squat", "squats"]):
        return "squat"
    elif any(word in chat_input for word in ["bench", "press", "chest"]):
        return "bench_press"
    elif any(word in chat_input for word in ["clean", "power clean"]):
        return "clean"

    # Default to deadlift if no specific exercise is mentioned
    logger.info(f"No specific exercise detected in: '{chat_input}', defaulting to deadlift")
    return "deadlift"

def detect_exercise_from_video(frames):
    """Attempt to detect exercise type from video frames
    
    This is a placeholder for future ML-based exercise detection
    """
    # This would use a trained model to classify the exercise type
    # For now, just return None to indicate we should use chat_input instead
    return None
