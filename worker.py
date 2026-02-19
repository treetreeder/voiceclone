import asyncio
import os
import io
import time
import boto3
import soundfile as sf
from dotenv import load_dotenv
from bullmq import Worker
from processor import TTSProcessor
from ws import WhisperTranscriber

load_dotenv()

# ============== R2 Configuration ==============
r2_endpoint = os.getenv("R2_ENDPOINT")
r2_access_key = os.getenv("R2_ACCESS_KEY_ID")
r2_secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
bucket = os.getenv("R2_BUCKET_NAME")

# Initialize R2 Client
try:
    r2_client = boto3.client(
        's3',
        endpoint_url=r2_endpoint,
        aws_access_key_id=r2_access_key,
        aws_secret_access_key=r2_secret_key,
    )
except Exception as e:
    print(f"⚠️ Failed to initialize R2 client: {e}")
    r2_client = None

# ============== Load TTS Model Once ==============
tts = TTSProcessor()
tts.load_model()
transcriber = WhisperTranscriber(model_size="turbo")


async def process_task(job, job_token):
    data = job.data
    print(f"\n--- 🚀 Processing Job {job.id} ---")
    print(f"Data: {data}")

    source_key = data['voice']['key']
    text_to_speak = data.get('inputTranscript', '')
    output_language = data.get('outputLanguage', 'auto')

    try:
        # 1. Download reference audio from R2 into memory
        print(f"📥 Downloading ref audio: {source_key}")
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)  # Creates if doesn't exist, does nothing if it does

        # Get filename from source key
        filename = source_key.split('/')[-1]
        temp_path = os.path.join(temp_dir, filename)

        # Download directly to file
        with open(temp_path, 'wb') as f:
            r2_client.download_fileobj(bucket, source_key, f)

        print(f"✅ Downloaded to {temp_path}")
        ref_audio_path = temp_path


        speachToText = transcriber.process(ref_audio_path)
        
        print(f"🎤 Transcribed Text: {speachToText['text']}")
        # 2. Run TTS generation
        result = {"text": speachToText['text']}
        result = tts.generate(
            ref_audio_path=ref_audio_path,
            ref_text=speachToText['text'],
            text=text_to_speak,
            language=output_language,
        )

        # 3. Save output audio
        output_filename = f"output_job_{job.id}.wav"
        sf.write(output_filename, result["audio"], result["sample_rate"])

        print(f"✨ Generated: {output_filename}")
        print(f"📊 Latency: {result['total_time']:.3f}s | Audio: {result['audio_duration']:.2f}s | RTF: {result['rtf']:.4f}")
        print(f"✅ Job {job.id} done.")

        return {
            "status": "success",
            "processed_voice_id": data['voice']['id'],
            "output_file": output_filename,
            "audio_duration": result["audio_duration"],
            "latency": result["total_time"],
        }

    except Exception as e:
        print(f"❌ Failed to process job {job.id} ({source_key}): {str(e)}")
        raise e


async def main():
    redis_url = os.getenv("REDIS_URL")
    queue_name = os.getenv("QUEUE_NAME", "clone-queue")

    shutdown_event = asyncio.Event()

    print(f"\n🤖 Worker connecting to: {queue_name}...")

    worker = Worker(
        queue_name,
        process_task,
        {
            "connection": redis_url,
            "concurrency": 1,
            "decode_responses": True,
        }
    )

    print(f"✅ Worker is active and waiting for tasks...\n")

    try:
        await shutdown_event.wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\nShutting down worker...")
    finally:
        await worker.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass