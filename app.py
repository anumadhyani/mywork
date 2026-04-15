import os
import tempfile

from flask import Flask, jsonify, request
from flask_cors import CORS

from model import DEFAULT_MODEL_PATH, predict_image


app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Missing multipart form field 'file'"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    model_path = request.args.get("model_path", DEFAULT_MODEL_PATH)

    suffix = os.path.splitext(f.filename)[1] or ".jpg"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        result = predict_image(tmp_path, model_path=model_path)
        return jsonify(result)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
