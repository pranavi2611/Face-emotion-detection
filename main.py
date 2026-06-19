# main.py
from voice_input import get_voice_text
import subprocess

if __name__ == "__main__":
    # Step 1: capture voice
    voice_emotion = get_voice_text()
    print("Voice said:", voice_emotion)

    # Step 2: start live camera detection
    print("🎥 Starting camera for emotion detection...")
    subprocess.run(["python", "camera_input.py"])

    print("✅ Program finished")
