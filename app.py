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


@app.post("/api/c2pa")
def api_c2pa():
    if "file" not in request.files:
        return jsonify({"error": "No file part", "status": "failure"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No selected file", "status": "failure"}), 400

    suffix = os.path.splitext(f.filename)[1] or ".jpg"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        try:
            import c2pa  # type: ignore
        except Exception as e:
            return jsonify({"error": f"c2pa-python is not available: {str(e)}", "status": "failure"}), 500

        settings = None
        verify = request.args.get("verify", "false").lower() in ("1", "true", "yes")
        if verify:
            settings = c2pa.Settings.from_dict({"verify": {"verify_cert_anchors": False}})

        with c2pa.Context(settings) as context:
            with c2pa.Reader(tmp_path, context=context) as reader:
                detailed = reader.detailed_json()

        return jsonify({"status": "success", "c2pa": detailed})
    except Exception as e:
        return jsonify({"error": f"C2PA read failed: {str(e)}", "status": "failure"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post("/api/ai-detect")
def api_ai_detect():
    if "file" not in request.files:
        return jsonify({"error": "No file part", "status": "failure"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No selected file", "status": "failure"}), 400

    model_path = request.args.get("model_path", DEFAULT_MODEL_PATH)
    suffix = os.path.splitext(f.filename)[1] or ".jpg"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        result = predict_image(tmp_path, model_path=model_path)

        confidence = None
        if "proba" in result and isinstance(result["proba"], list) and len(result["proba"]) >= 2:
            confidence = {"real": float(result["proba"][0]), "fake": float(result["proba"][1])}

        return jsonify(
            {
                "status": "success",
                "prediction": result.get("label"),
                "confidence": confidence,
            }
        )
    except Exception as e:
        return jsonify({"error": f"AI Detection failed: {str(e)}", "status": "failure"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
