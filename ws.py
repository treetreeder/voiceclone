import whisper
import os
import time
import io
import numpy as np
import soundfile as sf

class WhisperTranscriber:
    def __init__(self, model_size="large", cache_dir="./models"):
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = os.path.abspath(cache_dir)
        
        print(f"📦 Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size, download_root=cache_dir)
        print("✅ Model ready.")

    def process(self, audio_input, task="transcribe", language=None):
        """
        :param audio_input: Can be a file path (str) OR raw audio bytes
        """
        start_time = time.perf_counter()

        # Handle raw bytes input
        if isinstance(audio_input, bytes):
            print("🚀 Processing audio from memory (bytes)...")
            # Wrap bytes in a file-like object and read with soundfile
            with io.BytesIO(audio_input) as audio_file:
                audio_array, samplerate = sf.read(audio_file)
                print(f"📊 Original Sample Rate: {samplerate} Hz | Channels: {audio_array.shape[1] if len(audio_array.shape) > 1 else 1} | Duration: {len(audio_array)/samplerate:.2f}s")
            if samplerate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=samplerate, target_sr=16000)
                            
            # Whisper expects 16k mono float32 audio. 
            # soundfile usually returns float64 or int; we ensure float32.
            # If stereo, convert to mono by averaging channels
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            audio_data = audio_array.astype(np.float32)
        else:
            # Handle file path input
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            print(f"🚀 Starting {task} for: {os.path.basename(audio_input)}...")
            audio_data = audio_input

        # Core Whisper execution
        result = self.model.transcribe(
            audio_data,
            task=task,
            language=language,
            verbose=False
        )
        
        execution_time = time.perf_counter() - start_time
        
        return {
            "text": result["text"].strip(),
            "segments": result["segments"],
            "language": result.get("language"),
            "duration": execution_time
        }