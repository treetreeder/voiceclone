import asyncio
import os
import io
import boto3
import soundfile as sf
from dotenv import load_dotenv
from bullmq import Worker
from processor import TTSProcessor

load_dotenv()

# ============== R2 & Redis 初始化 ==============
r2_client = boto3.client(
    's3',
    endpoint_url=os.getenv("R2_ENDPOINT"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
)
redis_url = os.getenv("REDIS_URL")

# 加载 TTS 模型 (只在这个进程中加载)
qwen = TTSProcessor()
qwen.load_model()

async def process_tts_task(job, job_token):
    data = job.data
    print(f"\n[TTS Worker] 🚀 开始处理任务: {job.id}")
    print(f"数据: data={data}")
    
    voice_key = data['key']
    tts_text = data.get('tts', '')
    ref_text = data.get('transcript', '')
    tts_language = data.get('tts_language', 'auto')
    
    try:
        # 1. 下载参考音频到内存
        bucket = os.getenv("R2_BUCKET_NAME")
        print(f"📥 正在下载参考音频: {voice_key}")
        
        audio_bytes = io.BytesIO()
        r2_client.download_fileobj(bucket, voice_key, audio_bytes)
        audio_bytes.seek(0)
        
        # 1. Load audio to memory (BytesIO)
        ref_audio_array, ref_sr = sf.read(audio_bytes, dtype='float32')

        # 2. Convert stereo to mono if needed
        if len(ref_audio_array.shape) > 1:
            ref_audio_array = ref_audio_array.mean(axis=1)

        # 3. Pass as tuple (exactly like warmup code)
        result = qwen.generate(
            ref_audio=(ref_audio_array, ref_sr),  # ✅ Tuple, not dict
            ref_text=ref_text,
            text=tts_text,
            language=tts_language,
        )
        
      # 4. 保存输出音频到 R2
        output_filename = f"output_job_{job.id}.wav"
        
        # 写入音频到内存缓冲区
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, result["audio"], result["sample_rate"], format='WAV')
        audio_bytes.seek(0)
        
        # 上传到 R2
        bucket = os.getenv("R2_BUCKET_NAME")
        r2_key = f"{bucket}/{output_filename}"
        r2_client.upload_fileobj(audio_bytes, bucket, r2_key)
        
        print(f"✨ 语音生成完毕: {r2_key}")
        
        return {
            "status": "success",
            "output_file": output_filename,
            "latency": result.get("total_time", 0)
        }

    except Exception as e:
        print(f"❌ TTS 任务 {job.id} 失败: {str(e)}")
        raise e

async def main():
    print(f"\n🤖 TTS Worker 正在监听: clone-queue...")
    
    worker = Worker(
        "clone-queue",
        process_tts_task,
        {"connection": redis_url, "concurrency": 1}  # 如果显存够大，可以调高并发
    )
    
    try:
        await asyncio.Event().wait()
    finally:
        await worker.close()

if __name__ == "__main__":
    asyncio.run(main())