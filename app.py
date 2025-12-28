from flask import Flask, request, jsonify, send_file
import os
import json
import wave
import traceback
import subprocess
import shlex
from datetime import datetime
from werkzeug.utils import secure_filename

from vosk import Model, KaldiRecognizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor
from gtts import gTTS
import torch

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
GENERATED_AUDIO_FOLDER = "generated_audio"
VOSK_MODEL_PATH = "models/vosk/vosk-model-small-en-us-0.15"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_AUDIO_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac"}

# --------------------------------------------------
# Device
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load VOSK model (ONCE)
# --------------------------------------------------
vosk_model = Model(VOSK_MODEL_PATH)

# --------------------------------------------------
# IndicTrans2 setup
# --------------------------------------------------
translation_models = {}

INDIC_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "od": "ory_Orya",
}

GTTS_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "bn": "bn",
    "mr": "mr",
}

# --------------------------------------------------
# Utils
# --------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_wav_16k(src):
    """
    Converts any audio file to mono 16kHz PCM WAV for VOSK
    """
    base = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(UPLOAD_FOLDER, f"{base}_16k.wav")

    cmd = f'ffmpeg -y -i "{src}" -ac 1 -ar 16000 -acodec pcm_s16le "{out}"'
    subprocess.run(
        shlex.split(cmd),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out

# --------------------------------------------------
# VOSK ASR
# --------------------------------------------------
def transcribe_audio(wav_path):
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)

    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text += " " + res.get("text", "")

    final = json.loads(rec.FinalResult())
    text += " " + final.get("text", "")
    return text.strip()

# --------------------------------------------------
# IndicTrans2 Translation
# --------------------------------------------------
def get_translation_model(src, tgt):
    key = f"{src}-{tgt}"
    if key in translation_models:
        return translation_models[key]

    if src == "en":
        model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
    elif tgt == "en":
        model_name = "ai4bharat/indictrans2-indic-en-dist-320M"
    else:
        model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    processor = IndicProcessor(inference=True)
    translation_models[key] = (model, tokenizer, processor)
    return model, tokenizer, processor


def translate_text(text, src, tgt):
    model, tokenizer, processor = get_translation_model(src, tgt)

    batch = processor.preprocess_batch(
        [text],
        src_lang=INDIC_LANG_MAP[src],
        tgt_lang=INDIC_LANG_MAP[tgt],
    )

    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=5)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return processor.postprocess_batch(decoded, lang=INDIC_LANG_MAP[tgt])[0]

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/translate", methods=["POST"])
def translate_api():
    try:
        start = datetime.now()

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        src = request.form.get("input_lang")
        tgt = request.form.get("target_lang")

        if src not in INDIC_LANG_MAP or tgt not in INDIC_LANG_MAP:
            return jsonify({"error": "Unsupported language"}), 400

        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        name = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, name)
        file.save(upload_path)

        # -------- Auto conversion for VOSK --------
        wav_path = convert_to_wav_16k(upload_path)

        # ASR
        recognized_text = transcribe_audio(wav_path)

        # Translation
        translated_text = translate_text(recognized_text, src, tgt)

        # TTS
        out_audio = os.path.join(GENERATED_AUDIO_FOLDER, "output.mp3")
        tts_lang = GTTS_LANG_MAP.get(tgt, "en")
        gTTS(translated_text, lang=tts_lang).save(out_audio)

        return jsonify({
            "recognized_text": recognized_text,
            "translated_text": translated_text,
            "audio_url": "/audio/output.mp3",
            "processing_time": (datetime.now() - start).total_seconds(),
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/audio/<name>")
def serve_audio(name):
    return send_file(
        os.path.join(GENERATED_AUDIO_FOLDER, name),
        mimetype="audio/mpeg"
    )

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=False)
