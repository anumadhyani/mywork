import argparse
import json
import os
import tempfile
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix


# --- HuggingFace streaming training (no local image storage) ---
# Trains a lightweight classifier on MobileNetV3-Large embeddings computed on-the-fly.
# This avoids exporting/storing the full image dataset locally.
#
# Dataset: Rajarshi-Roy-research/Defactify_Image_Dataset
# Label_A: 0=Real, 1=AI
# Label_B: 0=Real, 1=SD2.1, 2=SDXL, 3=SD3, 4=DALL·E 3, 5=Midjourney


@dataclass
class _Batch:
    X: np.ndarray
    y: np.ndarray


def _iter_stream(dataset: str, split: str, seed: int):
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset, split=split, streaming=True)

    # Streaming datasets are iterable; shuffle uses a buffer.
    try:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    except Exception:
        pass

    for ex in ds:
        yield ex


def _get_cols(example: dict) -> tuple[object | None, int | None]:
    # Accept both canonical and variant casing.
    img = None
    if "image" in example:
        img = example.get("image")
    elif "Image" in example:
        img = example.get("Image")

    label = None
    for k in ("Label_A", "label_a", "labelA"):
        if k in example:
            label = example.get(k)
            break

    try:
        label_i = int(label) if label is not None else None
    except Exception:
        label_i = None

    return img, label_i


def _get_label_b(example: dict) -> int | None:
    # Optional: generator/source label for AI images (Defactify uses Label_B).
    for k in ("Label_B", "label_b", "labelB"):
        if k in example:
            try:
                return int(example.get(k))
            except Exception:
                return None
    return None


def _load_image_pil(img_obj):
    # HF Image feature returns a PIL-like object with .convert/.save.
    if hasattr(img_obj, "convert"):
        return img_obj.convert("RGB")
    return None


def _make_embedder(device: str):
    import torch
    from torchvision import models

    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier = torch.nn.Identity()

    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    model.eval().to(device)
    preprocess = weights.transforms()

    return model, preprocess, device


def _embed_batch(model, preprocess, device: str, pil_images: list):
    import torch

    tensor = torch.stack([preprocess(im) for im in pil_images]).to(device)
    with torch.no_grad():
        out = model(tensor)
    return out.detach().cpu().numpy()


def _iter_batches(dataset: str, split: str, seed: int, limit: int, batch_size: int, device: str):
    # --- Stream images and embed them in batches ---
    model, preprocess, device = _make_embedder(device=device)

    pil_batch = []
    y_batch = []
    seen = 0
    skipped = 0

    for ex in _iter_stream(dataset, split=split, seed=seed):
        if limit and seen >= limit:
            break

        img_obj, label_a = _get_cols(ex)
        if label_a not in (0, 1):
            skipped += 1
            continue

        pil = _load_image_pil(img_obj)
        if pil is None:
            skipped += 1
            continue

        pil_batch.append(pil)
        y_batch.append(int(label_a))
        seen += 1

        if len(pil_batch) >= batch_size:
            X = _embed_batch(model, preprocess, device=device, pil_images=pil_batch)
            y = np.asarray(y_batch, dtype=np.int64)
            yield _Batch(X=X, y=y), skipped
            pil_batch = []
            y_batch = []

    if pil_batch:
        X = _embed_batch(model, preprocess, device=device, pil_images=pil_batch)
        y = np.asarray(y_batch, dtype=np.int64)
        yield _Batch(X=X, y=y), skipped


def _export_onnx(model, out_path: str):
    import torch

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model.cpu(),
        dummy,
        out_path,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )


def _report_precision_coverage(y_true: np.ndarray, p_ai: np.ndarray, target_precision: float):
    # --- Threshold selection for production-style gating ---
    # We choose:
    # - AI when p_ai >= t_ai
    # - Real when p_ai <= t_real
    # - otherwise Uncertain
    # with both AI and Real precision >= target_precision.

    if y_true.size == 0:
        return

    # AI: precision among predicted AI.
    order = np.argsort(-p_ai)
    y_sorted = y_true[order]
    p_sorted = p_ai[order]

    cum_pred_ai = np.arange(1, y_sorted.size + 1)
    cum_true_ai = np.cumsum(y_sorted == 1)
    prec_ai = cum_true_ai / np.maximum(cum_pred_ai, 1)

    ai_idx = np.where(prec_ai >= target_precision)[0]
    t_ai = None
    if ai_idx.size > 0:
        # Take the lowest threshold that still meets precision (max coverage).
        k = int(ai_idx[-1])
        t_ai = float(p_sorted[k])

    # Real: precision among predicted Real => y_true == 0.
    order_r = np.argsort(p_ai)
    y_sorted_r = y_true[order_r]
    p_sorted_r = p_ai[order_r]

    cum_pred_real = np.arange(1, y_sorted_r.size + 1)
    cum_true_real = np.cumsum(y_sorted_r == 0)
    prec_real = cum_true_real / np.maximum(cum_pred_real, 1)

    real_idx = np.where(prec_real >= target_precision)[0]
    t_real = None
    if real_idx.size > 0:
        k = int(real_idx[-1])
        t_real = float(p_sorted_r[k])

    if t_ai is None or t_real is None:
        print(f"No thresholds found to achieve precision>={target_precision:.3f} for both classes")
        return

    pred_ai = p_ai >= t_ai
    pred_real = p_ai <= t_real
    pred_unc = ~(pred_ai | pred_real)

    def safe_precision(mask: np.ndarray, positive_label: int):
        denom = int(mask.sum())
        if denom == 0:
            return 0.0, 0
        num = int(((y_true == positive_label) & mask).sum())
        return num / denom, denom

    prec_ai_final, n_ai = safe_precision(pred_ai, 1)
    prec_real_final, n_real = safe_precision(pred_real, 0)
    coverage = 1.0 - (pred_unc.sum() / y_true.size)

    print("\nHigh-precision gating report")
    print("Target precision:", target_precision)
    print("AI threshold (p_ai >=):", f"{t_ai:.4f}")
    print("Real threshold (p_ai <=):", f"{t_real:.4f}")
    print("AI precision:", f"{prec_ai_final:.4f}", "n=", n_ai)
    print("Real precision:", f"{prec_real_final:.4f}", "n=", n_real)
    print("Coverage (non-uncertain):", f"{coverage:.4f}")
    print("Uncertain n=", int(pred_unc.sum()))


class _SigmoidCalibratedBinaryClassifier:
    # --- Simple Platt scaling (sigmoid calibration) ---
    # Fits a 1D logistic regression on the base model's score.
    # This avoids sklearn's CalibratedClassifierCV(cv='prefit') compatibility issues.

    def __init__(self, base_estimator, calibrator: LogisticRegression):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="Rajarshi-Roy-research/Defactify_Image_Dataset",
        help="HuggingFace dataset name.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--limit-train", type=int, default=0, help="0 means no limit")
    parser.add_argument("--limit-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of streaming passes over the training split (re-iterates the HF stream).",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="balanced",
        help="Use class weighting to reduce majority-class bias (recommended: balanced).",
    )
    parser.add_argument(
        "--calibrate",
        choices=["none", "sigmoid"],
        default="sigmoid",
        help="Optional probability calibration using a small streamed calibration buffer.",
    )
    parser.add_argument(
        "--calibration-n",
        type=int,
        default=2000,
        help="Number of streamed training examples to reserve for calibration (embeddings only).",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.95,
        help="Target precision for both classes when reporting conservative Real/AI thresholds.",
    )
    parser.add_argument(
        "--oversample-real",
        choices=["none", "match_ai"],
        default="match_ai",
        help="Mitigate class imbalance by oversampling Real embeddings during training (no image storage).",
    )
    parser.add_argument(
        "--real-pool-size",
        type=int,
        default=20000,
        help="Max number of real embeddings to keep in memory for oversampling.",
    )

    parser.add_argument(
        "--clf-out",
        default="models/embedding_sgd_logreg.joblib",
        help="Output classifier path.",
    )
    parser.add_argument(
        "--meta-out",
        default="models/embedding_sgd_meta.json",
        help="Metadata JSON output.",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=False,
        help="If set, export the MobileNet embedding backbone to ONNX (requires onnx dependencies).",
    )
    parser.add_argument("--onnx-out", default="models/mobilenetv3_embedding.onnx")
    args = parser.parse_args()

    # --- Online classifier (logistic regression via SGD) ---
    # We use partial_fit so we never need to store all embeddings in memory.
    # NOTE: Some sklearn versions do not support class_weight='balanced' with partial_fit.
    # We implement balanced weighting by passing sample_weight per batch instead.
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1,
        tol=None,
    )
    classes = np.asarray([0, 1], dtype=np.int64)

    # --- Shared embedder (reuse one MobileNet instance for train + test) ---
    model, preprocess, device = _make_embedder(device=args.device)

    def embed_batch(pil_images: list):
        return _embed_batch(model, preprocess, device=device, pil_images=pil_images)

    # --- Train loop (optionally reserve a calibration buffer) ---
    print("Streaming train split:", args.train_split)
    seen_train = 0
    skipped_train = 0
    seen_fit = 0
    cal_X: list[np.ndarray] = []
    cal_y: list[np.ndarray] = []
    cal_seen = 0

    # --- Real embedding pool for oversampling ---
    # Stores embeddings only (not images) to allow 1:1 balancing against AI during streaming.
    real_pool: list[np.ndarray] = []

    def batch_sample_weight(y: np.ndarray) -> np.ndarray | None:
        if args.class_weight == "none":
            return None
        n = int(y.size)
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        if n0 == 0 or n1 == 0:
            return None
        # Standard balanced weights: n / (n_classes * n_c)
        w0 = n / (2.0 * n0)
        w1 = n / (2.0 * n1)
        return np.where(y == 0, w0, w1).astype(np.float64)

    for epoch in range(max(1, int(args.epochs))):
        pil_batch = []
        y_batch: list[int] = []
        print("Epoch:", epoch + 1, "/", max(1, int(args.epochs)))

        for ex in _iter_stream(args.dataset, split=args.train_split, seed=args.seed + epoch):
            if args.limit_train and seen_train >= args.limit_train:
                break

            img_obj, label_a = _get_cols(ex)
            if label_a not in (0, 1):
                skipped_train += 1
                continue

            pil = _load_image_pil(img_obj)
            if pil is None:
                skipped_train += 1
                continue

            pil_batch.append(pil)
            y_batch.append(int(label_a))
            seen_train += 1

            if len(pil_batch) >= args.batch_size:
                X = embed_batch(pil_batch)
                y = np.asarray(y_batch, dtype=np.int64)

                # --- Update pool with any real embeddings in this batch ---
                if args.oversample_real != "none":
                    real_mask = y == 0
                    if np.any(real_mask):
                        X_real = X[real_mask]
                        for r in X_real:
                            if len(real_pool) < int(args.real_pool_size):
                                real_pool.append(r.astype(np.float32, copy=False))

                # --- Optionally oversample real to match AI count (1:1) ---
                if args.oversample_real == "match_ai" and len(real_pool) > 0:
                    n_ai = int((y == 1).sum())
                    if n_ai > 0:
                        # Sample real embeddings with replacement.
                        idx = np.random.randint(0, len(real_pool), size=n_ai)
                        X_os = np.stack([real_pool[j] for j in idx], axis=0)
                        y_os = np.zeros((n_ai,), dtype=np.int64)
                        X = np.concatenate([X, X_os], axis=0)
                        y = np.concatenate([y, y_os], axis=0)

                # Reserve a calibration buffer from the first epoch only.
                if args.calibrate != "none" and epoch == 0 and (cal_seen < args.calibration_n):
                    cal_X.append(X)
                    cal_y.append(y)
                    cal_seen += int(y.size)
                else:
                    sw = batch_sample_weight(y)
                    if sw is None:
                        clf.partial_fit(X, y, classes=classes)
                    else:
                        clf.partial_fit(X, y, classes=classes, sample_weight=sw)
                    seen_fit += int(y.size)

                pil_batch = []
                y_batch = []

                if (seen_train % (args.batch_size * 50)) == 0:
                    print(f"Trained on {seen_train} streamed examples...")

        if pil_batch:
            X = embed_batch(pil_batch)
            y = np.asarray(y_batch, dtype=np.int64)

            if args.oversample_real != "none":
                real_mask = y == 0
                if np.any(real_mask):
                    X_real = X[real_mask]
                    for r in X_real:
                        if len(real_pool) < int(args.real_pool_size):
                            real_pool.append(r.astype(np.float32, copy=False))

            if args.oversample_real == "match_ai" and len(real_pool) > 0:
                n_ai = int((y == 1).sum())
                if n_ai > 0:
                    idx = np.random.randint(0, len(real_pool), size=n_ai)
                    X_os = np.stack([real_pool[j] for j in idx], axis=0)
                    y_os = np.zeros((n_ai,), dtype=np.int64)
                    X = np.concatenate([X, X_os], axis=0)
                    y = np.concatenate([y, y_os], axis=0)

            if args.calibrate != "none" and epoch == 0 and (cal_seen < args.calibration_n):
                cal_X.append(X)
                cal_y.append(y)
                cal_seen += int(y.size)
            else:
                sw = batch_sample_weight(y)
                if sw is None:
                    clf.partial_fit(X, y, classes=classes)
                else:
                    clf.partial_fit(X, y, classes=classes, sample_weight=sw)
                seen_fit += int(y.size)

    print("Train done")
    print("Train streamed examples:", seen_train, "skipped:", skipped_train)
    print("Train fit examples:", seen_fit)

    # --- Optional probability calibration ---
    final_clf = clf
    cal_n = int(sum(int(x.shape[0]) for x in cal_X))
    if args.calibrate != "none" and cal_n > 0:
        X_cal = np.vstack(cal_X)
        y_cal = np.concatenate(cal_y)
        if args.calibrate != "sigmoid":
            raise RuntimeError("Only 'sigmoid' calibration is supported in streaming mode")

        scores = None
        if hasattr(clf, "decision_function"):
            scores = np.asarray(clf.decision_function(X_cal)).reshape(-1, 1)
        else:
            proba = np.asarray(clf.predict_proba(X_cal))[:, 1]
            eps = 1e-6
            proba = np.clip(proba, eps, 1.0 - eps)
            scores = np.log(proba / (1.0 - proba)).reshape(-1, 1)

        calibrator = LogisticRegression(max_iter=2000)
        calibrator.fit(scores, y_cal)
        final_clf = _SigmoidCalibratedBinaryClassifier(base_estimator=clf, calibrator=calibrator)
        print("Calibrated probabilities with: sigmoid (platt scaling)", "n_cal=", cal_n)

    # --- Evaluate loop ---
    print("Streaming test split:", args.test_split)
    y_true: list[int] = []
    y_pred: list[int] = []
    p_ai_list: list[float] = []
    seen_test = 0
    skipped_test = 0
    pil_batch = []
    y_batch = []

    for ex in _iter_stream(args.dataset, split=args.test_split, seed=args.seed):
        if args.limit_test and seen_test >= args.limit_test:
            break

        img_obj, label_a = _get_cols(ex)
        if label_a not in (0, 1):
            skipped_test += 1
            continue

        pil = _load_image_pil(img_obj)
        if pil is None:
            skipped_test += 1
            continue

        pil_batch.append(pil)
        y_batch.append(int(label_a))
        seen_test += 1

        if len(pil_batch) >= args.batch_size:
            X = embed_batch(pil_batch)
            yp = final_clf.predict(X)
            proba = final_clf.predict_proba(X)
            p_ai_list.extend([float(v) for v in proba[:, 1].tolist()])
            y_true.extend(y_batch)
            y_pred.extend([int(v) for v in yp.tolist()])
            pil_batch = []
            y_batch = []

    if pil_batch:
        X = embed_batch(pil_batch)
        yp = final_clf.predict(X)
        proba = final_clf.predict_proba(X)
        p_ai_list.extend([float(v) for v in proba[:, 1].tolist()])
        y_true.extend(y_batch)
        y_pred.extend([int(v) for v in yp.tolist()])

    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    p_ai = np.asarray(p_ai_list, dtype=np.float64)

    print("Test examples:", len(yt), "skipped:", skipped_test)
    print("Confusion matrix:\n", confusion_matrix(yt, yp))
    print(
        classification_report(
            yt,
            yp,
            target_names=["Real", "AI"],
            digits=4,
            zero_division=0,
        )
    )

    if p_ai.size == yt.size:
        _report_precision_coverage(yt, p_ai, target_precision=float(args.target_precision))

    # --- Save classifier + meta (no images stored) ---
    os.makedirs(os.path.dirname(args.clf_out) or ".", exist_ok=True)
    joblib.dump(final_clf, args.clf_out)

    meta = {
        "dataset": args.dataset,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_examples": int(seen_train),
        "test_examples": int(len(yt)),
        "classifier": "sgd_log_loss",
        "calibration": args.calibrate,
        "calibration_n": int(cal_n),
        "epochs": int(args.epochs),
        "class_weight": args.class_weight,
        "batch_size": int(args.batch_size),
        "device": device,
        "classifier_path": args.clf_out,
        "onnx": args.onnx_out,
    }
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # --- Optional: export ONNX embedding backbone (used by model.py at inference time) ---
    if args.export_onnx and not os.path.exists(args.onnx_out):
        try:
            import torch
            from torchvision import models as tv_models

            weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT
            backbone = tv_models.mobilenet_v3_large(weights=weights)
            backbone.classifier = torch.nn.Identity()
            _export_onnx(backbone, args.onnx_out)
            print("Saved ONNX backbone to:", args.onnx_out)
        except Exception as e:
            print("ONNX export skipped:", str(e))

    print("Saved classifier to:", args.clf_out)
    print("Saved meta to:", args.meta_out)


if __name__ == "__main__":
    main()
