import whisper
import os
import time

os.environ["XDG_CACHE_HOME"] = os.getcwd()
audio_file = "voice.wav"

if not os.path.exists(audio_file):
    print(f"Error: {audio_file} not found")
    exit()

print("正在加载官方 Whisper 'large' 模型 (完美兼容 CUDA 13)...")
model = whisper.load_model("turbo")

start_time = time.perf_counter()

# ==========================================
# 步骤 1：安全检测语言 (Pad/Trim 防止形状报错)
# ==========================================
audio_data = whisper.load_audio(audio_file)
audio_padded = whisper.pad_or_trim(audio_data)
mel = whisper.log_mel_spectrogram(audio_padded, n_mels=model.dims.n_mels).to(model.device)

_, probs = model.detect_language(mel)
detected_lang = max(probs, key=probs.get)
print(f"检测到语言: {detected_lang}")

# ==========================================
# 步骤 2：设置标点符号提示词
# ==========================================
prompts = {
    "zh": "这是一段语音记录。请注意，输出必须包含适当的标点符号，例如逗号、句号、和问号！",
    "en": "This is a voice recording. Please ensure the output includes proper punctuation, such as commas, periods, and question marks!"
}
dynamic_prompt = prompts.get(detected_lang, "Please include punctuation. 请包含标点符号！")

# ==========================================
# 步骤 3：转录并开启【防幻觉参数】
# ==========================================
print("开始转录并过滤静音幻觉...")
result = model.transcribe(
    audio_file,
    language=detected_lang,
    initial_prompt=dynamic_prompt,
    task="transcribe",
    
    # 核心防幻觉参数（过滤掉大部分无声片段生成的废话）
    condition_on_previous_text=False, # 防止死循环重复
    logprob_threshold=-1.0,           # 低于此置信度的乱猜直接丢弃
    no_speech_threshold=0.6,          # 提高静音判断阈值，静音时不输出
    compression_ratio_threshold=2.4,  # 屏蔽重复的乱码文本
    
    verbose=True 
)

end_time = time.perf_counter()

print("\n" + "="*50)
print("完整转录内容:")
print("="*50)
print(result["text"])
print("\n" + "="*50)
print(f"处理时间: {end_time - start_time:.2f} 秒")
print("="*50)