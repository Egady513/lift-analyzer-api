from flask import Flask, jsonify
import os
import logging
import sys

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    logger.info("Root endpoint accessed")
    return jsonify({"message": "Hello from Lift Analyzer API"})

@app.route('/ping', methods=['GET'])
def ping():
    logger.info("Ping endpoint accessed")
    return jsonify({"status": "success", "message": "Pong!"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port) 