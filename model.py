from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np

from feature_extraction import FEATURE_NAMES, extract_all_features


DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "ai_detector_random_forest_model.pkl")
)

DEFAULT_EMBEDDING_ONNX_PATH = os.getenv(
    "EMBEDDING_ONNX_PATH",
    os.path.join(os.path.dirname(__file__), "models", "mobilenetv3_embedding.onnx"),
)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_DEFAULT_SGD_CLF_V2 = os.path.join(_MODELS_DIR, "embedding_sgd_logreg_v2.joblib")
_DEFAULT_SGD_CLF = os.path.join(_MODELS_DIR, "embedding_sgd_logreg.joblib")
_DEFAULT_LOGREG_CLF = os.path.join(_MODELS_DIR, "embedding_logreg.joblib")
_DEFAULT_CLF = next(
    (path for path in (_DEFAULT_SGD_CLF_V2, _DEFAULT_SGD_CLF, _DEFAULT_LOGREG_CLF) if os.path.exists(path)),
    _DEFAULT_LOGREG_CLF,
)

DEFAULT_EMBEDDING_CLF_PATH = os.getenv("EMBEDDING_CLF_PATH") or _DEFAULT_CLF

class SigmoidCalibratedBinaryClassifier:
    """Simple Platt scaling (sigmoid calibration) wrapper.

    Lives in an importable module so joblib artifacts produced by the
    streaming trainer can be deserialized in production.
    """

    def __init__(self, base_estimator, calibrator):
        self.base_estimator = base_estimator
        self.calibrator = calibrator

    def _scores(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.base_estimator, "decision_function"):
            s = self.base_estimator.decision_function(X)
            return np.asarray(s).reshape(-1, 1)
        if hasattr(self.base_estimator, "predict_proba"):
            p = self.base_estimator.predict_proba(X)
            p_ai = np.asarray(p)[:, 1]
            eps = 1e-6
            p_ai = np.clip(p_ai, eps, 1.0 - eps)
            logit = np.log(p_ai / (1.0 - p_ai))
            return logit.reshape(-1, 1)
        raise RuntimeError("Base estimator must support decision_function or predict_proba")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = self._scores(X)
        p_ai = self.calibrator.predict_proba(s)[:, 1]
        p_ai = np.asarray(p_ai, dtype=np.float64)
        p_real = 1.0 - p_ai
        return np.stack([p_real, p_ai], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int64)


# Back-compat alias: artifacts pickled while the class lived in __main__.
_SigmoidCalibratedBinaryClassifier = SigmoidCalibratedBinaryClassifier


AI_THRESHOLD = float(os.getenv("AI_THRESHOLD", "0.90"))
LIKELY_AI_THRESHOLD = float(os.getenv("LIKELY_AI_THRESHOLD", "0.80"))
LIKELY_REAL_THRESHOLD = float(os.getenv("LIKELY_REAL_THRESHOLD", "0.10"))
REAL_THRESHOLD = float(os.getenv("REAL_THRESHOLD", "0.05"))
UNCERTAIN_EPS = float(os.getenv("UNCERTAIN_EPS", "0.01"))


def _label_from_p_ai(p_ai: float):
    if p_ai >= AI_THRESHOLD:
        return "Likely AI", "ai"

    if abs(p_ai - 0.5) <= UNCERTAIN_EPS:
        return "Uncertain", "uncertain"

    if p_ai >= LIKELY_AI_THRESHOLD:
        return "Likely AI", "likely_ai"

    # Real thresholds should apply when the model believes p_ai is low.
    if p_ai <= REAL_THRESHOLD:
        return "Real", "real"

    if p_ai < LIKELY_REAL_THRESHOLD:
        return "Likely Real", "likely_real"

    return "Uncertain", "uncertain"


@lru_cache(maxsize=1)
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    return joblib.load(model_path)


def _load_embedding_stack():
    if not (os.path.exists(DEFAULT_EMBEDDING_ONNX_PATH) and os.path.exists(DEFAULT_EMBEDDING_CLF_PATH)):
        return None

    try:
        import onnxruntime as ort
        from PIL import Image

        session = ort.InferenceSession(DEFAULT_EMBEDDING_ONNX_PATH, providers=["CPUExecutionProvider"])
        clf = joblib.load(DEFAULT_EMBEDDING_CLF_PATH)
        return session, clf, Image
    except Exception:
        return None


def _preprocess_224_rgb(image_path: str, Image):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize((224, 224))
        arr = np.asarray(im).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)


def _predict_with_embeddings(image_path: str):
    stack = _load_embedding_stack()
    if stack is None:
        return None

    session, clf, Image = stack
    inp = _preprocess_224_rgb(image_path, Image)
    input_name = session.get_inputs()[0].name
    emb = session.run(None, {input_name: inp})[0]
    emb = np.asarray(emb)

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(emb)[0]
        pred = int(np.argmax(proba))
        confidence = float(np.max(proba))
        return {
            "pred": pred,
            "proba": [float(proba[0]), float(proba[1])],
            "confidence": confidence,
            "backend": "embedding_logreg",
        }

    pred = int(clf.predict(emb)[0])
    return {
        "pred": pred,
        "proba": None,
        "confidence": None,
        "backend": "embedding_logreg",
    }


def predict_image(image_path: str, model_path: str = DEFAULT_MODEL_PATH):
    emb_result = _predict_with_embeddings(image_path)

    if emb_result is not None:
        pred = emb_result["pred"]
        proba = emb_result["proba"]
        confidence = emb_result["confidence"]
        backend = emb_result["backend"]

        label = "AI" if pred == 1 else "Real"
        band = "unknown"
        uncertain = False
        if proba and len(proba) >= 2:
            p_ai = float(proba[1])
            label, band = _label_from_p_ai(p_ai)
            uncertain = band == "uncertain"

        return {
            "pred": pred,
            "label": label,
            "confidence": confidence,
            "proba": proba,
            "features": None,
            "backend": backend,
            "uncertain": uncertain,
            "band": band,
        }

    model = load_model(model_path)
    features = extract_all_features(image_path)
    features_2d = [features]

    pred = int(model.predict(features_2d)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_2d)[0].tolist()

    label = "AI" if pred == 1 else "Real"
    confidence = None
    band = "unknown"
    uncertain = False

    if proba and len(proba) >= 2:
        confidence = float(max(proba))
        p_ai = float(proba[1])
        label, band = _label_from_p_ai(p_ai)
        uncertain = band == "uncertain"

    return {
        "pred": pred,
        "label": label,
        "confidence": confidence,
        "proba": proba,
        "features": features.tolist(),
        "backend": "random_forest",
        "uncertain": uncertain,
        "band": band,
    }
