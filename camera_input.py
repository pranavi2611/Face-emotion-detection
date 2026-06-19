# camera_input.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

MODEL_PATH = "models/best_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"

# Load model + encoder
model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

cap = cv2.VideoCapture(0)  # open default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # resize to 48x48
    face_resized = cv2.resize(gray, (48, 48))
    face_resized = face_resized.reshape(1, 48, 48, 1).astype("float32") / 255.0

    # predict
    pred = model.predict(face_resized)
    label_idx = np.argmax(pred)
    label = le.inverse_transform([label_idx])[0]

    # display on video
    cv2.putText(frame, f"Emotion: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
