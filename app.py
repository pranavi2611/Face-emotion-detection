# app.py
import os
import pickle
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import speech_recognition as sr
from textblob import TextBlob
import cv2
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Configure paths
MODEL_PATH = "models/best_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Initialize components
model = None
le = None

logging.basicConfig(level=logging.INFO)

def load_or_create_label_encoder():
    """Load existing label encoder or create a new one"""
    global le
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    try:
        # Check if file exists and has content
        if os.path.exists(ENCODER_PATH) and os.path.getsize(ENCODER_PATH) > 0:
            with open(ENCODER_PATH, "rb") as f:
                le = pickle.load(f)
            logging.info("✓ Label encoder loaded successfully")
        else:
            raise FileNotFoundError("Label encoder file missing or empty")
    except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
        logging.warning(f"Creating new label encoder: {e}")
        # Create new label encoder
        le = LabelEncoder()
        le.fit(emotions)
        # Save it
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(le, f)
        logging.info("✓ New label encoder created and saved")

def load_models():
    """Load ML models with error handling"""
    global model, le
    
    try:
        # Load main model
        model = load_model(MODEL_PATH)
        logging.info("✓ Model loaded successfully")
        
        # Print model input shape for debugging
        input_shape = model.input_shape
        logging.info(f"Model expects input shape: {input_shape}")
        
        # Load or create label encoder
        load_or_create_label_encoder()
        
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise

# Load models at startup
try:
    load_models()
except Exception as e:
    logging.error(f"Failed to initialize models: {e}")

def preprocess_image(file, target_size=(48, 48)):
    """Preprocess image for model prediction based on model's expected input"""
    # Get model's expected input shape
    expected_channels = model.input_shape[-1]
    
    if expected_channels == 3:
        # Model expects RGB images (3 channels)
        img = load_img(file, color_mode="rgb", target_size=target_size)
        logging.info("Preprocessing as RGB image (3 channels)")
    elif expected_channels == 1:
        # Model expects grayscale images (1 channel)
        img = load_img(file, color_mode="grayscale", target_size=target_size)
        logging.info("Preprocessing as grayscale image (1 channel)")
    else:
        # Default to RGB
        img = load_img(file, color_mode="rgb", target_size=target_size)
        logging.info("Preprocessing as RGB image (default)")
    
    arr = img_to_array(img).astype("float32") / 255.0
    
    # Ensure the shape matches model expectations
    if expected_channels == 1 and len(arr.shape) == 3 and arr.shape[2] == 3:
        # Convert RGB to grayscale if needed
        arr = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140])
        arr = arr.reshape(target_size[0], target_size[1], 1)
    
    arr = arr.reshape(1, target_size[0], target_size[1], expected_channels)
    
    logging.info(f"Final preprocessed image shape: {arr.shape}")
    return arr

def preprocess_image_adaptive(file, target_size=(48, 48)):
    """Alternative adaptive preprocessing that handles both grayscale and RGB"""
    try:
        # First try to load as RGB
        img = load_img(file, color_mode="rgb", target_size=target_size)
        arr = img_to_array(img).astype("float32") / 255.0
        
        # If model expects grayscale but we have RGB, convert
        if model.input_shape[-1] == 1 and arr.shape[-1] == 3:
            # Convert RGB to grayscale using luminance formula
            gray = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140])
            arr = gray.reshape(target_size[0], target_size[1], 1)
        
        arr = arr.reshape(1, target_size[0], target_size[1], model.input_shape[-1])
        logging.info(f"Adaptive preprocessing - Final shape: {arr.shape}")
        return arr
        
    except Exception as e:
        logging.error(f"Adaptive preprocessing failed: {e}")
        raise

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/predict_image", methods=["POST"])
def predict_image():
    """Handle image-based emotion prediction"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        return jsonify({"error": "Invalid file type. Please upload an image."}), 400

    temp_path = "static/temp.jpg"
    try:
        file.save(temp_path)
        
        # Use adaptive preprocessing
        x = preprocess_image_adaptive(temp_path)
        pred = model.predict(x)
        label_idx = np.argmax(pred)
        label = le.inverse_transform([label_idx])[0]
        confidence = float(np.max(pred))
        
        logging.info(f"Image prediction: {label} (confidence: {confidence:.2f})")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({
            "emotion": label,
            "confidence": confidence
        })
        
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Could not process image: {str(e)}"}), 500

@app.route("/predict_voice", methods=["POST"])
def predict_voice():
    """Handle voice-based emotion prediction"""
    r = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("🎤 Listening... Speak now!")
            # Adjust for ambient noise with shorter timeout
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=7, phrase_time_limit=5)
            
    except sr.WaitTimeoutError:
        logging.warning("Listening timed out")
        return jsonify({
            "voice_text": "No speech detected. Please try again.", 
            "emotion": "neutral"
        }), 400
    except Exception as e:
        logging.error(f"Microphone error: {e}")
        return jsonify({
            "error": "Microphone not available. Please check your audio settings."
        }), 500

    try:
        text = r.recognize_google(audio)
        logging.info(f"Recognized text: {text}")

        # Enhanced sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # More nuanced emotion mapping
        if sentiment > 0.3:
            emotion = "happy"
        elif sentiment > 0.1:
            emotion = "neutral"
        elif sentiment < -0.3:
            emotion = "angry"
        elif sentiment < -0.1:
            emotion = "sad"
        else:
            # Consider subjectivity for neutral cases
            emotion = "surprise" if subjectivity > 0.6 else "neutral"
            
        logging.info(f"Text sentiment: {sentiment:.2f}, Predicted emotion: {emotion}")

        return jsonify({
            "voice_text": text, 
            "emotion": emotion,
            "sentiment_score": sentiment
        })
        
    except sr.UnknownValueError:
        logging.warning("Speech recognition could not understand audio")
        return jsonify({
            "voice_text": "Could not understand speech. Please try again.", 
            "emotion": "neutral"
        }), 400
    except sr.RequestError as e:
        logging.error(f"Speech service error: {e}")
        return jsonify({
            "error": "Speech recognition service unavailable. Please check your internet connection."
        }), 503
    except Exception as e:
        logging.error(f"Unexpected error in voice prediction: {e}")
        return jsonify({"error": "An error occurred during voice processing."}), 500

@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    """Handle real-time camera emotion prediction"""
    cap = cv2.VideoCapture(0)
    
    try:
        if not cap.isOpened():
            logging.error("Cannot open camera")
            return jsonify({"error": "Camera not available"}), 500
        
        # Set higher resolution for better face detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        ret, frame = cap.read()
        if not ret:
            return jsonify({"error": "Failed to capture frame from camera"}), 500

        # Convert based on model requirements
        if model.input_shape[-1] == 3:
            # Model expects RGB
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Model expects grayscale
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_resized = cv2.resize(processed_frame, (48, 48))
        
        # Reshape based on model input requirements
        if model.input_shape[-1] == 1:
            face_resized = face_resized.reshape(1, 48, 48, 1)
        else:
            face_resized = face_resized.reshape(1, 48, 48, 3)
            
        face_resized = face_resized.astype("float32") / 255.0

        # Make prediction
        pred = model.predict(face_resized)
        label_idx = np.argmax(pred)
        label = le.inverse_transform([label_idx])[0]
        confidence = float(np.max(pred))
        
        logging.info(f"Camera prediction: {label} (confidence: {confidence:.2f})")
        
        return jsonify({
            "emotion": label,
            "confidence": confidence
        })
        
    except Exception as e:
        logging.error(f"Error in camera prediction: {e}")
        return jsonify({"error": f"Failed to process camera input: {str(e)}"}), 500
    finally:
        cap.release()

@app.route("/model_info", methods=["GET"])
def model_info():
    """Endpoint to check model information"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    info = {
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "layers": len(model.layers),
        "expected_channels": model.input_shape[-1]
    }
    return jsonify(info)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    status = {
        "model_loaded": model is not None,
        "label_encoder_loaded": le is not None,
        "model_input_shape": model.input_shape if model else None,
        "status": "healthy" if (model is not None and le is not None) else "degraded"
    }
    return jsonify(status)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logging.info("Starting Emotion Detection Flask App...")
    app.run(debug=True, host='0.0.0.0', port=5000)
    