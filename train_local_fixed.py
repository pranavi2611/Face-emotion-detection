# train_local_fixed.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)

def load_images_from_directory():
    """Load images from your local train/test directories"""
    images = []
    labels = []
    
    # Emotion classes based on your folder structure
    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    total_loaded = 0
    # Load from both train and test directories
    for dataset_type in ['train', 'test']:
        for emotion in emotion_classes:
            folder_path = os.path.join(dataset_type, emotion)
            
            if not os.path.exists(folder_path):
                logging.warning(f"Folder not found: {folder_path}")
                continue
                
            emotion_count = 0
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(folder_path, img_file)
                    try:
                        # Load image as grayscale
                        img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                        img_array = img_to_array(img)
                        images.append(img_array)
                        labels.append(emotion)
                        emotion_count += 1
                        total_loaded += 1
                    except Exception as e:
                        logging.warning(f"Could not load {img_path}: {e}")
            
            logging.info(f"Loaded {emotion_count} images from {folder_path}")
    
    if len(images) == 0:
        raise ValueError("No images found! Check your directory structure.")
    
    logging.info(f"Successfully loaded {total_loaded} images total")
    
    # Show class distribution
    unique, counts = np.unique(labels, return_counts=True)
    logging.info("Class distribution:")
    for emotion, count in zip(unique, counts):
        logging.info(f"  {emotion}: {count} images")
    
    return np.array(images), np.array(labels), emotion_classes

def build_improved_model(input_shape, num_classes):
    """Build a better CNN model for emotion detection"""
    model = keras.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth conv block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()

def train_model():
    """Main training function using local dataset"""
    try:
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        logging.info("Loading images from local directories...")
        
        # Load your actual dataset
        images, labels, emotion_classes = load_images_from_directory()
        
        # Preprocess data
        images = images.astype('float32') / 255.0
        input_shape = images[0].shape
        
        logging.info(f"Input shape: {input_shape}")
        logging.info(f"Number of classes: {len(emotion_classes)}")
        
        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        labels_categorical = keras.utils.to_categorical(labels_encoded, len(emotion_classes))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels_categorical, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        # Further split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
        )
        
        logging.info(f"Training set: {X_train.shape}")
        logging.info(f"Validation set: {X_val.shape}")
        logging.info(f"Test set: {X_test.shape}")
        
        # Build model
        model = build_improved_model(input_shape, len(emotion_classes))
        
        # Compile with better optimizer settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        
        # Enhanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_accuracy',
                min_delta=0.01
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                monitor='val_loss'
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        ]
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        datagen.fit(X_train)
        
        # Train model
        logging.info("Starting training with your actual dataset...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=len(X_train) // 32
        )
        
        # Save final model
        model.save('models/emotion_model_final.h5')
        
        # Save label encoder
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
        
        # Evaluate model
        logging.info("Evaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        logging.info("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=emotion_classes))
        
        # Plot training history
        plot_training_history(history)
        
        logging.info("Training completed successfully!")
        logging.info(f"Best model saved to: models/best_model.h5")
        logging.info(f"Final model saved to: models/emotion_model_final.h5")
        logging.info(f"Label encoder saved to: models/label_encoder.pkl")
        
        return test_accuracy
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model()