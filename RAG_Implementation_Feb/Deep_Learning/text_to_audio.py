from gtts import gTTS
import os

# Code to implement text to audio using a pre-trained model

def text_to_audio(text, output_file="output_audio.mp3", lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_file)
    print(f"Audio saved to {output_file}")

if __name__ == "__main__":
    sample_text = "Hello, how are you doing?"
    text_to_audio(sample_text)
