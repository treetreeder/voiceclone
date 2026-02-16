import time
import json
import os
import signal
import numpy as np
import torch
import soundfile as sf
import librosa
from qwen_tts import Qwen3TTSModel

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# Custom exception for the timeout
class GenerationTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise GenerationTimeout("TIMEOUT: Generation exceeded 10 seconds.")

def run_generation(model, text, language, voice_clone_prompt):
    """Run generation with GPU synchronization for accurate timing."""
    torch.cuda.synchronize()
    start = time.perf_counter()

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        x_vector_only_mode=True
    )

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    
    audio = wavs[0] if wavs else np.array([])
    duration = len(audio) / sr if sr > 0 else 0
    rtf = total_time / duration if duration > 0 else 0

    return {
        "total_time": total_time,
        "audio": audio,
        "sample_rate": sr,
        "audio_duration": duration,
        "rtf": rtf
    }

def main():
    total_start = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("🚀 Qwen3-TTS Worker Active (RTX 4090)")
    print(f"📂 Watch Directory: {base_dir}")
    print("=" * 60)

    # Load Model
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model.enable_streaming_optimizations(
        use_compile=True,
        compile_mode="max-autotune",
    )

    # Warmup
    print("\n🔥 Initializing Kernels (Warmup)...")
    warmup_sr = 24000
    dummy_prompt = model.create_voice_clone_prompt(
        ref_audio=(np.zeros(warmup_sr), warmup_sr), 
        ref_text="warmup"
    )
    run_generation(model, "Warmup generation.", "English", dummy_prompt)
    print("✅ System Ready.")

    counter = 1
    # Register the signal for the timeout
    signal.signal(signal.SIGALRM, timeout_handler)

    while True:
        choice = input(f"\n[{counter}] Enter JSON number (1-4) or 'EXIT': ").strip()
        
        if choice.upper() == "EXIT":
            print(f"\nShutting down. Total uptime: {time.time() - total_start:.2f}s")
            break
        
        json_path = os.path.join(base_dir, f"{choice}.json")
        
        if not os.path.exists(json_path):
            print(f"❌ File not found: {json_path}")
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            ref_audio_path = data["ref_audio_path"]
            if not os.path.isabs(ref_audio_path):
                ref_audio_path = os.path.join(base_dir, ref_audio_path)

            print(f"📖 Task {choice} | Lang: {data['language']}")

            # --- PRE-PROCESSING & TIMEOUT START ---
            signal.alarm(50) # Set 10s timer
            try:
                # 1. Handle m4a/clipping warnings by pre-loading with librosa
                if ref_audio_path.lower().endswith(('.m4a', '.mp3')):
                    y, sr = librosa.load(ref_audio_path, sr=24000)
                    y = librosa.util.normalize(y) * 0.95
                    ref_input = (y, sr)
                else:
                    ref_input = ref_audio_path

                # 2. Create prompt
                prompt = model.create_voice_clone_prompt(
                    ref_audio=ref_input, 
                    ref_text=data["ref_text"]
                )
                
                # 3. Generate
                res = run_generation(model, data["user_text"], data["language"], prompt)

                # 4. Save
                out_filename = os.path.join(base_dir, f"output_{choice}.wav")
                sf.write(out_filename, res["audio"], res["sample_rate"])
                
                print(f"✨ Success: output_{choice}.wav")
                print(f"📊 RTF: {res['rtf']:.4f} | Time: {res['total_time']:.3f}s")

            except GenerationTimeout as te:
                print(f"🚨 {te}")
                torch.cuda.empty_cache() # Clear VRAM in case of hung kernels
            
            finally:
                signal.alarm(0) # Disable the alarm
            # --- TIMEOUT END ---

            counter += 1

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()