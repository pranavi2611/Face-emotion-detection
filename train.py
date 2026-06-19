# train.py (updated to use local data)
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import logging

logging.basicConfig(level=logging.INFO)

def load_local_data():
    """Load data from local train/test folders"""
    images = []
    labels = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    for emotion in emotions:
        # Load from train folder
        train_path = f"train/{emotion}"
        if os.path.exists(train_path):
            for img_file in os.listdir(train_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(train_path, img_file)
                    img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(emotion)
        
        # Load from test folder  
        test_path = f"test/{emotion}"
        if os.path.exists(test_path):
            for img_file in os.listdir(test_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(test_path, img_file)
                    img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(emotion)
    
    return np.array(images), np.array(labels)

def train_model():
    try:
        logging.info("Loading local dataset...")
        images, labels = load_local_data()
        
        # Preprocess
        images = images.astype('float32') / 255.0
        
        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        labels_categorical = keras.utils.to_categorical(labels_encoded, 7)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels_categorical, test_size=0.2, random_state=42
        )
        
        # Build model
        model = keras.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train
        model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)
        
        # Save
        os.makedirs('models', exist_ok=True)
        model.save('models/best_model.h5')
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
            
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    train_model()