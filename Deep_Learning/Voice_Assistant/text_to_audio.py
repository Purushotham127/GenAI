from gtts import gTTS
import os
import tempfile
from pydub import AudioSegment

# Code to implement text to audio using a pre-trained model

def text_to_audio(text, output_file="output_audio.wav", lang="en"):
    # Save TTS output as temporary mp3
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tts = gTTS(text=text, lang=lang)
        tts.save(tmp_mp3.name)
        tmp_mp3_path = tmp_mp3.name

    # Convert mp3 to wav
    audio = AudioSegment.from_mp3(tmp_mp3_path)
    audio.export(output_file, format="wav")
    print(f"Audio saved to {output_file}")

    # Clean up temporary mp3 file
    os.remove(tmp_mp3_path)

if __name__ == "__main__":
    sample_text = "Hello, how are you doing?"
    text_to_audio(sample_text)
