from flask import Flask, request, jsonify, send_file
import os
import json
import wave
import traceback
import subprocess
import shlex
from datetime import datetime
from werkzeug.utils import secure_filename

import torch
from vosk import Model, KaldiRecognizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Paths
# --------------------------------------------------
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
# mBART-50 Language Map (FINAL)
# --------------------------------------------------
MBART_LANG_MAP = {
    "en": "en_XX",
    "hi": "hi_IN",
    "bn": "bn_IN",
    "mr": "mr_IN",
    "gu": "gu_IN",
    "ta": "ta_IN",
    "te": "te_IN",
    "kn": "kn_IN",
    "ml": "ml_IN",
}

GTTS_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "bn": "bn",
    "mr": "mr",
    "gu": "gu",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "ml": "ml",
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
# mBART-50 (ONLY translation model)
# --------------------------------------------------
mbart_model = None
mbart_tokenizer = None

def load_mbart_model():
    global mbart_model, mbart_tokenizer

    model_name = "facebook/mbart-large-50-many-to-many-mmt"

    mbart_tokenizer = AutoTokenizer.from_pretrained(model_name)
    mbart_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()


def translate_text(text, src, tgt):
    if src not in MBART_LANG_MAP or tgt not in MBART_LANG_MAP:
        raise ValueError("Language not supported by mBART-50")

    src_lang = MBART_LANG_MAP[src]
    tgt_lang = MBART_LANG_MAP[tgt]

    mbart_tokenizer.src_lang = src_lang

    inputs = mbart_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    forced_bos_token_id = mbart_tokenizer.lang_code_to_id[tgt_lang]

    with torch.no_grad():
        outputs = mbart_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=5,
        )

    return mbart_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

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

        if src not in MBART_LANG_MAP or tgt not in MBART_LANG_MAP:
            return jsonify({"error": "Unsupported language"}), 400

        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        name = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, name)
        file.save(upload_path)

        # Convert for VOSK
        wav_path = convert_to_wav_16k(upload_path)

        # ASR
        recognized_text = transcribe_audio(wav_path)

        # Translation (mBART-50 ONLY)
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
    load_mbart_model()
    app.run(host="0.0.0.0", port=8000, debug=False)
