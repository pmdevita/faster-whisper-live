from faster_whisper_live import LiveWhisper

model = LiveWhisper("small", compute_type="int8_float16")
with open("whispertest.m4a", "rb") as f:
    for segment in model.transcribe(f, vad_filter=True):
        print(segment)