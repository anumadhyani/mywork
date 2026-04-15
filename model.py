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

    result: dict = {
        "prediction": int(pred) if isinstance(pred, (np.integer, int)) else str(pred),
        "features": {name: float(features[i]) for i, name in enumerate(FEATURE_NAMES)},
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        result["proba"] = [float(p) for p in proba]

    return result
