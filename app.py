from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load your trained model
# Update this path to your actual model location
MODEL_PATH = 'model/pneumonia_model.h5'
model = load_model(MODEL_PATH)

# Model configuration (based on your training)
IMG_SIZE = 150  # Your model uses 150x150 images
IS_GRAYSCALE = True  # Your model was trained on grayscale images

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_bytes):
    """
    Preprocess image for model prediction
    Args:
        image_bytes: Raw image bytes
    Returns:
        Preprocessed image array ready for prediction
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Read image as grayscale
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Could not read image file")
    
    # Resize to model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    
    # Reshape for model input (batch_size, height, width, channels)
    img_array = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    return img_array

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the frontend
    Expects: multipart/form-data with 'image' field
    Returns: JSON with prediction and confidence
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Check file size
        if len(image_bytes) > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': f'File too large. Maximum {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)} MB'}), 400
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Format result
        # Assuming model outputs probability for pneumonia class
        # Adjust threshold if needed (0.5 is standard)
        if prediction > 0.5:
            result = "Pneumonia"
            confidence = float(prediction) * 100
        else:
            result = "Normal"
            confidence = float(1 - prediction) * 100
        
        # Return JSON response
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 2)
        })
        
    except Exception as e:
        # Log error for debugging (optional)
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# Optional: Add a health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

# ============================================
# PRODUCTION READY (for Render/Gunicorn)
# ============================================

if __name__ == '__main__':
    # For local development only
    # On Render, Gunicorn will run the app
    print("=" * 60)
    print("🚀 PneumoAI - Pneumonia Detection Application")
    print("=" * 60)
    print(f"📁 Model loaded from: {MODEL_PATH}")
    print(f"📐 Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"🎨 Color mode: {'Grayscale' if IS_GRAYSCALE else 'RGB'}")
    print("=" * 60)
    print("🌐 Running locally at: http://127.0.0.1:5000")
    print("=" * 60)
    
    # DEBUG MODE: False for production, True only for local testing
    app.run(debug=False, host='0.0.0.0', port=5000)