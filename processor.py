import time
import os
import tempfile
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')


def log_time(start, operation):
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()


class TTSProcessor:
    """
    Loads Qwen3-TTS model once and keeps it in memory.
    Call `generate()` for each new job from the worker.
    """

    def __init__(self):
        self.model = None
        self.ready = False

    def load_model(self):
        """Load model, enable optimizations, and run warmup."""
        total_start = time.time()

        print("=" * 60)
        print("🚀 Initializing Qwen3-TTS on RTX 4090")
        print("=" * 60)

        start = time.time()
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        log_time(start, "Model loaded")

        # ============== Setup Optimizations ==============
        print("\nEnabling optimizations (max-autotune)...")
        self.model.enable_streaming_optimizations(
            decode_window_frames=1200,
            use_compile=True,
            use_cuda_graphs=False,
            compile_mode="max-autotune",
            use_fast_codebook=True,
            compile_codebook_predictor=True,
            compile_talker=True,
        )

        # ============== Warmup Runs ==============
        # Warmup
        print("\n🔥 Initializing Kernels (Warmup)...")
        warmup_sr = 24000
        dummy_prompt = self.model.create_voice_clone_prompt(
            ref_audio=(np.zeros(warmup_sr), warmup_sr), 
            ref_text="warmup"
        )
        self._run_generation("Warmup generation.", "English", dummy_prompt, label="warmup")
        print("✅ System Ready.")   
        
        self.ready = True
        log_time(total_start, "✅ TTS Processor is READY")

    def generate(self, ref_audio_path: str,ref_text:str, text: str, language: str = "auto"):
        """
        Generate cloned voice audio from reference audio bytes and text.

        Args:
            ref_audio_bytes: Raw audio file bytes (wav/mp3) downloaded from R2.
            text: The text to synthesize.
            language: Language hint (default: "auto").

        Returns:
            dict with audio numpy array, sample_rate, timing stats.
        """
        if not self.ready:
            raise RuntimeError("Model is not loaded yet. Call load_model() first.")

        start = time.time()  # <-- Add this line
        try:
            voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=ref_text,
            )
            log_time(start, "Voice clone prompt created")
        finally:
            # Clean up temp file
            if os.path.exists(ref_audio_path):
                os.remove(ref_audio_path)

        # Run generation
        result = self._run_generation(text, language, voice_clone_prompt, label="job")
        return result

    def _run_generation(self, text, language, voice_clone_prompt, label="generation"):
        """Internal: run non-streaming generation with GPU sync."""
        torch.cuda.synchronize()
        start = time.perf_counter()

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            x_vector_only_mode=True,
        )

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        audio = wavs[0] if wavs else np.array([])
        audio_duration = len(audio) / sr if sr > 0 else 0
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        return {
            "label": label,
            "total_time": total_time,
            "audio": audio,
            "sample_rate": sr,
            "audio_duration": audio_duration,
            "rtf": rtf,
            "text": text,
        }