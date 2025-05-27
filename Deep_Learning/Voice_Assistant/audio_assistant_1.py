import pyttsx3
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torchaudio

# Load Whisper tiny model for speech-to-text
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Load DistilGPT2 for text generation
generator = pipeline('text-generation', model='distilgpt2')

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

def speech_to_text(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    generated_ids = model.generate(inputs.input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

def generate_response(text):
    response = generator(text, max_length=50, num_return_sequences=1)[0]['generated_text']
    return response

def text_to_speech(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Example usage
audio_path = "output_audio.wav"  # Replace with your audio file path
import os

audio_path = "output_audio.wav"  # Replace with your audio file path
if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file '{audio_path}' not found. Please provide a valid file path.")

# transcription = speech_to_text(audio_path)
transcription = "Hello, who is michael jackson?"
print("You said:", transcription)
response = generate_response(transcription)
print("Assistant:", response)

text_to_speech(response)
