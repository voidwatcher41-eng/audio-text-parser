import os
from flask import Flask, request, render_template_string
from faster_whisper import WhisperModel

# Ensure folders exist
os.makedirs("audio", exist_ok=True)
os.makedirs("text", exist_ok=True)

# Defaults (RU language)
MODEL_NAME = os.environ.get("WHISPER_MODEL", "medium")     # tiny/base/small/medium/large-v3
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")          # cpu or cuda
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")  # int8 (CPU), float16 (GPU), float32
LANGUAGE = os.environ.get("WHISPER_LANG", "ru")           # force Russian by default

app = Flask(__name__)

print(f"Loading Faster-Whisper: model={MODEL_NAME}, device={DEVICE}, compute={COMPUTE_TYPE}, language={LANGUAGE}")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

TEMPLATE = """
<!doctype html>
<title>Whisper Web</title>
<h1>Загрузите аудиофайл</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value="Распознать">
</form>
{% if transcription %}
<h2>Результат:</h2>
<pre>{{ transcription }}</pre>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            return "Файл не выбран", 400
        src_path = os.path.join("audio", uploaded.filename)
        uploaded.save(src_path)

        segments, info = model.transcribe(src_path, beam_size=5, language=LANGUAGE)
        lines = [f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}" for seg in segments]
        text_result = "\n".join(lines)

        out_path = os.path.join("text", uploaded.filename + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_result)

        return render_template_string(TEMPLATE, transcription=text_result)

    return render_template_string(TEMPLATE, transcription=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
