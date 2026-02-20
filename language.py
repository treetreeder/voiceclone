from faster_whisper import WhisperModel
import time
import os

# 设置 HuggingFace 模型缓存目录（faster-whisper 从此处下载模型）
os.environ["HF_HOME"] = os.getcwd()

audio_file = "voice.wav"

if not os.path.exists(audio_file):
    print(f"错误：找不到文件 {audio_file}")
    exit()

# ==========================================
# 步骤 1：加载 faster-whisper 模型
# ==========================================
print("正在加载 faster-whisper 'large-v3' 模型...")
# device="cuda" 使用 GPU。如果没有 GPU，改为 "cpu"
# compute_type="float16" 能在保持精度的同时减少显存占用并提速；显存如果不够可以改用 "int8_float16"
model = WhisperModel("large-v3", device="cuda", compute_type="default")

print("开始处理音频...")
start_time = time.perf_counter()

# ==========================================
# 步骤 2：设置双语提示词并转录
# ==========================================
# faster-whisper 的模型非常聪明，一个高质量的双语提示词通常就能完美强制多语言输出标点
prompt = "这是一段语音记录。请包含标点符号！This is a voice recording. Please include punctuation!"

# 调用 transcribe() 时，它会瞬间自动分析出语言，并返回 info 和一段段的 segments
segments, info = model.transcribe(
    audio_file,
    initial_prompt=prompt,
    beam_size=5,                      # 搜索束大小，默认 5 可以在准确率和速度间取得好平衡
    condition_on_previous_text=False, # 防止模型陷入死循环（防幻觉策略 1）
    
    # --- 核心终极防幻觉功能：VAD ---
    vad_filter=True,                  # 开启语音活动检测，直接剪掉所有静音
    vad_parameters=dict(min_silence_duration_ms=500) # 过滤掉所有超过 0.5 秒的静音
)

# ==========================================
# 步骤 3：输出结果
# ==========================================
print("\n" + "="*50)
print(f"自动检测到的主要语言: {info.language} (置信度: {info.language_probability:.2f})")
print("="*50)
print("完整转录内容:\n")

# 注意：faster-whisper 返回的 segments 是一个生成器 (generator)
# 也就是说，它是一边转录一边输出的。必须通过 for 循环遍历它才会真正执行转录！
full_text = ""
for segment in segments:
    # 你可以实时看到每句话的开始/结束时间
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    full_text += segment.text

end_time = time.perf_counter()
processing_time = end_time - start_time

print("\n" + "="*50)
print(f"处理时间: {processing_time:.2f} 秒")
print("="*50)