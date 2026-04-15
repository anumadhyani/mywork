from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np

from feature_extraction import FEATURE_NAMES, extract_all_features


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "ai_detector_random_forest_model.pkl",
)


@lru_cache(maxsize=1)
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    return joblib.load(model_path)


def predict_image(image_path: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    model = load_model(model_path)
    features = extract_all_features(image_path)

    x = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
    pred = model.predict(x)[0]

    if isinstance(pred, (np.integer, int)):
        pred_value = int(pred)
    else:
        try:
            pred_value = int(round(float(pred)))
        except (TypeError, ValueError):
            pred_value = None

    result: dict = {
        "prediction": pred_value if pred_value is not None else str(pred),
        "features": {name: float(features[i]) for i, name in enumerate(FEATURE_NAMES)},
    }

    if pred_value in (0, 1):
        result["label"] = "AI" if pred_value == 1 else "Real"

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        proba_list = [float(p) for p in proba]
        result["proba"] = proba_list
        if pred_value in (0, 1) and len(proba_list) > pred_value:
            result["confidence"] = float(proba_list[pred_value])

    return result
