# voice_input.py
import speech_recognition as sr
import logging
from emotion_model import load_model_and_tokenizer, predict_emotion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_voice_text(timeout: int = 5, phrase_time_limit: int = 10) -> str | None:
    """
    Captures audio from the microphone and converts it to text using Google's Speech Recognition.

    Args:
        timeout (int): Maximum number of seconds to wait for a phrase to start.
        phrase_time_limit (int): Maximum number of seconds a phrase can be.

    Returns:
        str | None: The recognized text, or None if an error occurred.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        logging.info("🎤 Say something (your emotion)...")
        r.adjust_for_ambient_noise(source)  # reduce background noise
        
        try:
            logging.info("Listening...")
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            logging.warning("⌛ Timed out waiting for phrase to start.")
            return None

    try:
        logging.info("Recognizing...")
        text = r.recognize_google(audio)
        logging.info("You said: %s", text)
        return text
    except sr.UnknownValueError:
        logging.error("❌ Could not understand audio.")
        return None
    except sr.RequestError as e:
        logging.error("❌ Could not request results from Google Speech Recognition service; %s", e)
        return None

if __name__ == "__main__":
    try:
        # 1. Load the model and tokenizer once at startup
        logging.info("Loading emotion detection model...")
        model, tokenizer = load_model_and_tokenizer()

        if model is None or tokenizer is None:
            logging.critical("Could not load model or tokenizer. Please ensure 'emotion_model.h5' and 'tokenizer.json' exist by running train.py first.")
            exit() # Exit the script if files are not found
        logging.info("✅ Model loaded. Ready to start.")

        while True:
            # 2. Get voice input from the user
            emotion_text = get_voice_text()

            # 3. Check for exit command
            if emotion_text and emotion_text.lower() in ["exit", "stop", "quit"]:
                logging.info("Exit command received. Shutting down.")
                break

            # 4. Predict the emotion if text was captured
            if emotion_text:
                print(f"\n✅ Voice Input Received: '{emotion_text}'")
                predicted_emotion = predict_emotion(emotion_text, model, tokenizer)
                print(f"🧠 Predicted Emotion: {predicted_emotion.upper()}\n")
            else:
                print("\n❌ No voice input received. Please try again.\n")

    except KeyboardInterrupt:
        print("\n👋 Application stopped by user. Goodbye!")
