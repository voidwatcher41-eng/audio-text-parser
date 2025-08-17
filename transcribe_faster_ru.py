
import os
from faster_whisper import WhisperModel

# Ensure folders exist
os.makedirs("audio", exist_ok=True)
os.makedirs("text", exist_ok=True)

# Defaults (RU language)
MODEL_NAME = os.environ.get("WHISPER_MODEL", "medium")     # tiny/base/small/medium/large-v3
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")          # cpu or cuda
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")  # int8 (CPU), float16 (GPU), float32
LANGUAGE = os.environ.get("WHISPER_LANG", "ru")           # force Russian by default

print(f"Loading Faster-Whisper: model={MODEL_NAME}, device={DEVICE}, compute={COMPUTE_TYPE}, language={LANGUAGE}")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

SUPPORTED = (".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".wma", ".aac", ".mkv")
files = [f for f in os.listdir("audio") if f.lower().endswith(SUPPORTED)]
if not files:
    print("Положите аудиофайлы в папку 'audio' и запустите снова.")
    raise SystemExit(0)

for fname in files:
    src = os.path.join("audio", fname)
    print(f"\nОбработка: {fname}")
    segments, info = model.transcribe(src, beam_size=5, language=LANGUAGE)
    out_path = os.path.join("text", fname + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}\n")
    print(f"✅ Сохранено: {out_path}")

print("\nГотово! Посмотрите результаты в папке 'text'.")
