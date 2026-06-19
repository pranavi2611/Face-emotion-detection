# emotion_model.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import logging

# --- Configuration ---
MAX_WORDS = 10000 # Max vocabulary size, should match train.py
MAX_LEN = 50      # Max length of sequences
MODEL_PATH = 'emotion_model.h5'
TOKENIZER_PATH = 'tokenizer.json'
EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] # Must match train.py

def load_model_and_tokenizer():
    """
    Loads the saved pre-trained model and tokenizer from disk.
    """
    try:
        # 1. Load the trained model
        logging.info(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)

        # 2. Load the tokenizer
        logging.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except (IOError, FileNotFoundError) as e:
        logging.error(f"Error loading model or tokenizer: {e}. Please run train.py first.")
        return None, None

def predict_emotion(text: str, model, tokenizer) -> str:
    """
    Predicts the emotion from a given text string.

    Args:
        text (str): The input text.
        model: The trained Keras model.
        tokenizer: The Keras tokenizer.

    Returns:
        str: The predicted emotion label.
    """
    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    # Predict
    prediction = model.predict(padded_sequence, verbose=0)
    predicted_index = np.argmax(prediction)
    
    return EMOTION_LABELS[predicted_index]