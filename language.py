import whisper
import os
import time

# Set the cache directory to current directory
os.environ["XDG_CACHE_HOME"] = os.getcwd()

# Load the model (it will download to ./models/ folder)
print("Loading model...")
model = whisper.load_model("base")

# Check if voice.wav exists
if not os.path.exists("voice.wav"):
    print("Error: voice.wav not found in current directory")
    exit()

print("Transcribing voice.wav...")

# Start timing
start_time = time.perf_counter()

# Transcribe the audio file
result = model.transcribe(
    "voice.wav",
    language=None,  # Auto-detect language (or set to "en", "zh", etc.)
    task="transcribe",  # or "translate" to translate to English
    verbose=True  # Show progress
)

# End timing
end_time = time.perf_counter()
processing_time = end_time - start_time

# Print results
print("\n" + "="*50)
print("FULL TRANSCRIPTION:")
print("="*50)
print(result["text"])

print("\n" + "="*50)
print("SEGMENTS WITH TIMESTAMPS:")
print("="*50)
for segment in result["segments"]:
    start = segment['start']
    end = segment['end']
    text = segment['text']
    print(f"[{start:6.2f}s -> {end:6.2f}s] {text}")

# Save transcription to file
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("\n" + "="*50)
print("Transcription saved to: transcription.txt")
print(f"Model downloaded to: {os.path.join(os.getcwd(), 'models')}")
print(f"Processing time: {processing_time:.2f} seconds")
print("="*50)