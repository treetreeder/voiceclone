from faster_whisper import WhisperModel
import time
import asyncio
import os
import io
import librosa
import boto3
import soundfile as sf
from dotenv import load_dotenv
from bullmq import Worker, Queue
import subprocess

load_dotenv()

# ============== R2 & Redis 初始化 ==============
r2_client = boto3.client(
    's3',
    endpoint_url=os.getenv("R2_ENDPOINT"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
)
redis_url = os.getenv("REDIS_URL")


# ==========================================
# 获取 GPU 信息
# ==========================================
def get_gpu_model():
    """自动检测 GPU 模型"""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            text=True,
            timeout=5
        ).strip().split('\n')[0]
        return output
    except Exception as e:
        print(f"⚠️ GPU Detection Error: {e}") # This will print the exact reason it failed
        return os.getenv("WORKER_GPU", "unknown")

GPU_MODEL = get_gpu_model()
print(f"🎛️  Detected GPU: {GPU_MODEL}")


# 初始化 TTS 下一阶段的队列
tts_queue = Queue("tts-queue", {"connection": redis_url})

# ==========================================
# 步骤 1：加载 faster-whisper 模型
# ==========================================
print("🔄 Loading Whisper model...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("✅ Model loaded successfully\n")


async def process_transcribe_task(job, job_token):
    data = job.data
    print(f"\n[Whisper Worker] 🚀 开始处理任务: {job.id}")
    
    voice_key = data['key']
    bucket = os.getenv("R2_BUCKET_NAME")
    
    try:
        # 1. 下载参考音频到内存
        print(f"📥 正在加载音频到内存: {voice_key}")
        audio_buffer = io.BytesIO()
        r2_client.download_fileobj(bucket, voice_key, audio_buffer)
        audio_buffer.seek(0)
        
        # 2. 加载到 numpy 数组
        audio_data, sample_rate = sf.read(audio_buffer)
        print(f"✅ 音频已加载: {len(audio_data)} samples at {sample_rate}Hz")
        
        # Convert to mono if stereo
        if len(audio_data.shape) == 2:
            audio_data = audio_data.mean(axis=1)
            print(f"ℹ️  转换为单声道: {len(audio_data)} samples")
        
        # Convert to float32
        audio_data = audio_data.astype('float32')
        
        # ===== CRITICAL: Resample to 16kHz =====
        if sample_rate != 16000:
            print(f"🔄 Resampling {sample_rate}Hz → 16kHz")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
            print(f"✅ Resampled: {len(audio_data)} samples at {sample_rate}Hz")
        
        # ===== DEBUG INFO =====
        import numpy as np
        print(f"🔍 Audio stats:")
        print(f"   Min: {np.min(audio_data):.4f}, Max: {np.max(audio_data):.4f}")
        print(f"   Mean: {np.mean(audio_data):.4f}, Std: {np.std(audio_data):.4f}")
        
        # Normalize if too quiet
        max_val = np.max(np.abs(audio_data))
        if max_val < 0.1:
            print(f"⚠️  Audio is very quiet, normalizing...")
            audio_data = audio_data / (max_val + 1e-8) * 0.9
            print(f"   After norm - Max: {np.max(np.abs(audio_data)):.4f}")
        
        start_time = time.perf_counter()
        prompt = "这是一段语音记录。请包含标点符号！"

        # Transcribe WITHOUT initial_prompt first (it can confuse the model)
        segments, info = model.transcribe(
            audio_data,
    
            beam_size=5,
            initial_prompt="Please add punctuation. 请添加标点符号。",
            condition_on_previous_text=False,
            vad_filter=False  # Disable VAD to test
        )

        full_text = ""
        for segment in segments:
            full_text += segment.text
        
        print(f"✅ 转录成功: {full_text if full_text else '(empty result)'}")
        print(f"🗣️  检测到语言: {info.language}")

        end_time = time.perf_counter()
        processing_time = end_time - start_time


        return {"transcript": full_text, "processingTime": round(processing_time, 2), "workerVersion": "whisper-v1", "workerGPU": GPU_MODEL}

    except Exception as e:
        print(f"❌ Whisper 任务 {job.id} 失败: {str(e)}")
        raise e


async def main():
    queue_name = os.getenv("QUEUE_NAME", "transcribe-queue")
    print(f"\n🤖 Whisper Worker 正在监听: {queue_name}...\n")
    
    worker = Worker(
        queue_name,
        process_transcribe_task,
        {"connection": redis_url, "concurrency": 1}
    )
    
    print("✅ Worker is active and waiting for tasks...\n")
    
    try:
        await asyncio.Event().wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n🛑 Shutting down worker...")
    finally:
        await worker.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass