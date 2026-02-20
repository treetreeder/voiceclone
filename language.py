from faster_whisper import WhisperModel
import time
import asyncio
import os
import boto3
from dotenv import load_dotenv
from bullmq import Worker, Queue

load_dotenv()

# ============== R2 & Redis 初始化 ==============
r2_client = boto3.client(
    's3', endpoint_url=os.getenv("R2_ENDPOINT"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
)
redis_url = os.getenv("REDIS_URL")

# 初始化 TTS 下一阶段的队列
tts_queue = Queue("tts-queue", {"connection": redis_url})

# ==========================================
# 步骤 1：加载 faster-whisper 模型
# ==========================================
model = WhisperModel("large-v3", device="cuda", compute_type="float16")



async def process_transcribe_task(job, job_token):
    data = job.data
    print(f"\n[Whisper Worker] 🚀 开始处理任务: {job.id}")
    
    voice_key = data['key']
    bucket = os.getenv("R2_BUCKET_NAME")
    
    try:
        # 1. 下载参考音频
        temp_dir = "./temp_whisper"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, voice_key.split('/')[-1])
        
        print(f"📥 正在下载参考音频: {voice_key}")
        with open(temp_path, 'wb') as f:
            r2_client.download_fileobj(bucket, voice_key, f)
            
        start_time = time.perf_counter()
        prompt = "这是一段语音记录。请包含标点符号！This is a voice recording. Please include punctuation!"

        # 调用 transcribe() 时，它会瞬间自动分析出语言，并返回 info 和一段段的 segments
        segments, info = model.transcribe(
            temp_path,
            initial_prompt=prompt,
            beam_size=5,                      # 搜索束大小，默认 5 可以在准确率和速度间取得好平衡
            condition_on_previous_text=False, # 防止模型陷入死循环（防幻觉策略 1）
            
            # --- 核心终极防幻觉功能：VAD ---
            vad_filter=True,                  # 开启语音活动检测，直接剪掉所有静音
            vad_parameters=dict(min_silence_duration_ms=500) # 过滤掉所有超过 0.5 秒的静音
        )

        full_text = ""
        for segment in segments:
            full_text += segment.text
        print(f"✅ 转录成功: {full_text}")

        end_time = time.perf_counter()
        processing_time = end_time - start_time


        print("\n" + "="*50)
        print(f"处理时间: {processing_time:.2f} 秒")
        print("="*50)
        
        # # 3. 将任务推送到 TTS 队列 (携带刚转录出来的文本)
        # data['transcribed_text'] = full_text # 将结果塞进数据包
        
        # await tts_queue.add(
        #     name="tts-job",
        #     data=data,
        #     opts={"removeOnComplete": True, "removeOnFail": False}
        # )
        # print("➡️ 任务已成功移交至 tts-queue")
        
        return {"transcript": full_text}

    except Exception as e:
        print(f"❌ Whisper 任务 {job.id} 失败: {str(e)}")
        raise e

async def main():
    queue_name = os.getenv("QUEUE_NAME", "transcribe-queue")
    print(f"\n🤖 Whisper Worker 正在监听: {queue_name}...")
    
    worker = Worker(
        queue_name,
        process_transcribe_task,
        {"connection": redis_url, "concurrency": 1}
    )
    
    try:
        await asyncio.Event().wait()
    finally:
        await worker.close()

if __name__ == "__main__":
    asyncio.run(main())
