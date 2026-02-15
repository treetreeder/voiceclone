import time
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

def run_generation(
    model,
    text: str,
    language: str,
    voice_clone_prompt,
    label: str = "generation",
):
    """Run non-streaming generation and return timing stats with GPU sync."""
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
    audio_duration = len(audio) / sr if sr > 0 else 0
    rtf = total_time / audio_duration if audio_duration > 0 else 0

    return {
        "label": label,
        "total_time": total_time,
        "audio": audio,
        "sample_rate": sr,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "text": text
    }

def main():
    total_start = time.time()

    print("=" * 60)
    print("🚀 Initializing Qwen3-TTS on RTX 4090")
    print("=" * 60)

    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    log_time(start, "Model loaded")

    # Reference audio setup
    ref_audio_path = "voice.wav"
    ref_text = (
        "我们党必须坚定地站在时代潮流的前头"
        "团结和带领全国各族人民，实现推进现代化建设、完成祖国统一、维护世界和平与促进共同发展这三大历史任务，"
        "在中国特色社会主义道路上实现中华民族的伟大复兴。这是历史和时代"
    )

    start = time.time()
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )
    log_time(start, "Voice clone prompt created")

    # ============== Setup Optimizations ==============
    print("\nEnabling optimizations (max-autotune)...")
    model.enable_streaming_optimizations(
        decode_window_frames=1200,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="max-autotune",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )

    # ============== Warmup Runs ==============
    warmup_texts = ["随着新年将至，各大社交媒体上，“未婚群体是否要给压岁钱”的话题再次“上了桌”。随着社会关系日趋多元，压岁钱的呈现形式也随之发生变化。但形式变化与否，其“关爱传递”核心内涵始终未变，和结婚与否更无直接关联。‌‌",
        "德国总理默茨当天在发言中“喊话”美国：美国没有实力独行，欧洲意识到应尽快摆脱对美过度依赖。默茨表示，德国坚定支持自由贸易、气候协定和世界卫生组织等，“只有团结协作才能应对全球性挑战”。‌‌",
        "俄乌冲突后，德国对北约承诺的担忧加剧，导致政治层面对美依赖加深，与经济利益形成撕裂。‌‌",
        "让我看看是不是真的有用"]
    print("\n🔥 Warming up & Compiling (Please wait)...")
    for i, warmup_text in enumerate(warmup_texts, 1):
        run_generation(model, warmup_text, "Chinese", voice_clone_prompt, label="warmup")
    print("✅ System Ready.")

    # ============== Interactive Loop ==============
    print("\n" + "—" * 60)
    print("Interactive Mode Active. Type 'EXIT' to quit.")
    print("—" * 60)

    counter = 1
    while True:
        user_text = input(f"\n[{counter}] Input Chinese text: ").strip()
        
        if user_text.upper() == "EXIT":
            print("\nShutting down. Total uptime: {:.2f}s".format(time.time() - total_start))
            break
        
        if not user_text:
            continue

        # Run Generation
        res = run_generation(
            model, 
            user_text, 
            "Chinese", 
            voice_clone_prompt,
            label=f"user_input_{counter}"
        )

        # Save Audio
        filename = f"output_interactive_{counter}.wav"
        sf.write(filename, res["audio"], res["sample_rate"])

        # Display Performance
        print(f"✨ Generated: {filename}")
        print(f"📊 Performance: Latency: {res['total_time']:.3f}s | Audio: {res['audio_duration']:.2f}s | RTF: {res['rtf']:.4f}")
        
        counter += 1

if __name__ == "__main__":
    main()