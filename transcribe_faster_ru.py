
import os
from pathlib import Path
from faster_whisper import WhisperModel

# Resolve paths relative to this file to avoid surprises when the script is
# executed from a different working directory.
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
TEXT_DIR = BASE_DIR / "text"

# Ensure folders exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# Defaults (RU language)
MODEL_NAME = os.environ.get("WHISPER_MODEL", "medium")     # tiny/base/small/medium/large-v3
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")          # cpu or cuda
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")  # int8 (CPU), float16 (GPU), float32
LANGUAGE = os.environ.get("WHISPER_LANG", "ru")           # force Russian by default

print(f"Loading Faster-Whisper: model={MODEL_NAME}, device={DEVICE}, compute={COMPUTE_TYPE}, language={LANGUAGE}")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

SUPPORTED = (".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".wma", ".aac", ".mkv")
files = [p for p in AUDIO_DIR.iterdir() if p.suffix.lower() in SUPPORTED]
if not files:
    print("Положите аудиофайлы в папку 'audio' и запустите снова.")
    raise SystemExit(0)

for src in files:
    print(f"\nОбработка: {src.name}")
    segments, info = model.transcribe(str(src), beam_size=5, language=LANGUAGE)
    out_path = TEXT_DIR / f"{src.stem}.txt"
    with open(out_path, "w", encoding="utf-8") as out_file:
        for seg in segments:
            out_file.write(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}\n")
    print(f"✅ Сохранено: {out_path}")

print("\nГотово! Посмотрите результаты в папке 'text'.")
