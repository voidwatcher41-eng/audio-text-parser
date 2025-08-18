import os
import json
from flask import Flask, request, render_template_string, Response
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
<form id="upload-form">
  <input type=file name=file>
  <input type=submit value="Распознать">
</form>
<div id="progress-container" style="width:100%;background:#eee;height:20px;display:none;margin-top:15px;">
  <div id="progress-bar" style="width:0;height:100%;background:#76c7c0;"></div>
</div>
<pre id="result"></pre>
<script>
const form = document.getElementById('upload-form');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const result = document.getElementById('result');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(form);
  progressContainer.style.display = 'block';
  progressBar.style.width = '0%';
  result.textContent = '';
  const response = await fetch('/transcribe', {method:'POST', body: formData});
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    const parts = buffer.split('\n\n');
    buffer = parts.pop();
    for (const part of parts) {
      if (part.startsWith('data:')) {
        const data = JSON.parse(part.slice(5));
        progressBar.style.width = data.progress + '%';
        result.textContent = data.text;
      }
    }
  }
});
</script>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(TEMPLATE)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    uploaded = request.files.get("file")
    if not uploaded or uploaded.filename == "":
        return "Файл не выбран", 400
    src_path = os.path.join("audio", uploaded.filename)
    uploaded.save(src_path)

    def generate():
        segments, info = model.transcribe(src_path, beam_size=5, language=LANGUAGE)
        text_result = ""
        for seg in segments:
            line = f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}"
            text_result += line + "\n"
            progress = int(seg.end / info.duration * 100)
            yield f"data: {json.dumps({'progress': progress, 'text': text_result})}\n\n"
        out_path = os.path.join("text", uploaded.filename + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_result)
        yield f"data: {json.dumps({'progress': 100, 'text': text_result})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
