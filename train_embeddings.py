import argparse
import json
import os
import random
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def _iter_images(root: str):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            _, ext = os.path.splitext(lower)
            if ext in exts:
                yield os.path.join(dirpath, name)


def _label_from_path(path: str):
    base = os.path.basename(path).lower()
    if base.startswith("input"):
        return 0
    if base.startswith("output"):
        return 1

    parts = [p.lower() for p in path.split(os.sep) if p]
    for p in reversed(parts):
        if p == "real":
            return 0
        if p == "fake":
            return 1
    return None


def _dataset_from_path(path: str, dataset_a_root: str, dataset_b_root: str, dataset_a_name: str, dataset_b_name: str):
    p = os.path.abspath(path)
    a = os.path.abspath(dataset_a_root)
    b = os.path.abspath(dataset_b_root)
    if p.startswith(a + os.sep) or p == a:
        return dataset_a_name
    if p.startswith(b + os.sep) or p == b:
        return dataset_b_name
    return "unknown"


def _load_paths(dataset_root: str):
    items = []
    for p in _iter_images(dataset_root):
        y = _label_from_path(p)
        if y is None:
            continue
        items.append((p, y))
    return items


def _load_image_pil(path: str):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def _compute_embeddings(paths, batch_size: int, device: str):
    import torch
    from torchvision import models, transforms

    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier = torch.nn.Identity()
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    model.eval().to(device)

    preprocess = weights.transforms()

    embs = []
    total = len(paths)
    for i in range(0, total, batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = [_load_image_pil(p) for p in batch_paths]
        tensor = torch.stack([preprocess(img) for img in imgs]).to(device)
        with torch.no_grad():
            out = model(tensor)
        embs.append(out.detach().cpu().numpy())

        done = min(i + batch_size, total)
        if done == total or done % (batch_size * 20) == 0:
            print(f"Embedded {done}/{total} images")

    return np.vstack(embs), model


def _cache_key(dataset_root: str, name: str, max_images: int | None):
    root = os.path.abspath(dataset_root)
    safe = root.replace(os.sep, "_").replace(":", "")
    suffix = "all" if not max_images else f"max{max_images}"
    return f"{name}__{safe}__{suffix}.npz"


def _load_or_compute_embeddings(
    dataset_root: str,
    items,
    batch_size: int,
    device: str,
    cache_dir: str,
    name: str,
    max_images: int,
):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        _cache_key(dataset_root, name=name, max_images=(max_images if max_images > 0 else None)),
    )

    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        X = z["X"]
        y = z["y"]
        paths = z["paths"].tolist()
        print(f"Loaded cached embeddings: {cache_path} (n={len(paths)})")
        return X, y, paths

    paths = [p for p, _ in items]
    y = np.asarray([yy for _, yy in items], dtype=np.int64)
    print(f"Computing embeddings for {name}: n={len(paths)}")
    X, backbone = _compute_embeddings(paths, batch_size=batch_size, device=device)

    np.savez_compressed(cache_path, X=X, y=y, paths=np.asarray(paths, dtype=object))
    print(f"Saved cached embeddings: {cache_path}")
    return X, y, paths


def _export_onnx(model, out_path: str):
    import torch

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)
    try:
        torch.onnx.export(
            model.cpu(),
            dummy,
            out_path,
            input_names=["input"],
            output_names=["embedding"],
            opset_version=17,
            dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        )
    except ModuleNotFoundError as e:
        missing = str(e)
        raise RuntimeError(
            "ONNX export failed due to missing dependency. Install with: "
            "python -m pip install onnx onnxscript"
        ) from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-a", required=True, help="First dataset root (e.g. CIFAKE)")
    parser.add_argument("--dataset-b", required=True, help="Second dataset root (e.g. Pico-banana)")
    parser.add_argument("--dataset-a-name", default="cifake")
    parser.add_argument("--dataset-b-name", default="pico_banana")
    parser.add_argument("--holdout", choices=["a", "b"], default="b", help="Which dataset to hold out as test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If > 0, randomly sample up to this many images per dataset (for quick experiments)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling when --max-images is used",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/embeddings",
        help="Directory to store precomputed embeddings so reruns are fast",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=False,
        help="If set, export the MobileNet embedding backbone to ONNX (requires onnx + onnxscript)",
    )
    parser.add_argument(
        "--classifier",
        choices=["logreg", "hgb"],
        default="logreg",
        help="Classifier to train on embeddings. 'hgb' often generalizes better than 'logreg'.",
    )
    parser.add_argument(
        "--calibrate",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Calibrate probabilities for more trustworthy confidence values.",
    )
    parser.add_argument(
        "--calibration-size",
        type=float,
        default=0.15,
        help="Fraction of training data reserved for calibration.",
    )
    parser.add_argument("--onnx-out", default="models/mobilenetv3_embedding.onnx")
    parser.add_argument("--clf-out", default="models/embedding_logreg.joblib")
    parser.add_argument("--meta-out", default="models/embedding_meta.json")
    args = parser.parse_args()

    random.seed(args.seed)

    items_a = _load_paths(args.dataset_a)
    items_b = _load_paths(args.dataset_b)

    if args.max_images and args.max_images > 0:
        if len(items_a) > args.max_images:
            items_a = random.sample(items_a, args.max_images)
        if len(items_b) > args.max_images:
            items_b = random.sample(items_b, args.max_images)

    if not items_a:
        raise RuntimeError(f"No labeled images found under dataset-a: {args.dataset_a}")
    if not items_b:
        raise RuntimeError(f"No labeled images found under dataset-b: {args.dataset_b}")

    if args.holdout == "b":
        train_items = items_a
        test_items = items_b
        holdout_name = args.dataset_b_name
    else:
        train_items = items_b
        test_items = items_a
        holdout_name = args.dataset_a_name

    X_a, y_a, paths_a = _load_or_compute_embeddings(
        args.dataset_a,
        items_a,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        name=args.dataset_a_name,
        max_images=args.max_images,
    )
    X_b, y_b, paths_b = _load_or_compute_embeddings(
        args.dataset_b,
        items_b,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        name=args.dataset_b_name,
        max_images=args.max_images,
    )

    if args.holdout == "b":
        X_train, y_train = X_a, y_a
        X_test, y_test = X_b, y_b
        holdout_name = args.dataset_b_name
        backbone_for_export = None
    else:
        X_train, y_train = X_b, y_b
        X_test, y_test = X_a, y_a
        holdout_name = args.dataset_a_name
        backbone_for_export = None

    if args.export_onnx and not (os.path.exists(args.onnx_out)):
        _, backbone_for_export = _compute_embeddings(paths_a[:1], batch_size=1, device=args.device)

    if args.classifier == "logreg":
        base_clf = LogisticRegression(max_iter=3000, n_jobs=-1)
    else:
        base_clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08)

    if args.calibrate != "none":
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train,
            y_train,
            test_size=args.calibration_size,
            random_state=args.seed,
            stratify=y_train,
        )
        base_clf.fit(X_fit, y_fit)
        clf = CalibratedClassifierCV(base_clf, method=args.calibrate, cv="prefit")
        clf.fit(X_cal, y_cal)
    else:
        base_clf.fit(X_train, y_train)
        clf = base_clf

    y_pred = clf.predict(X_test)
    print("Holdout dataset:", holdout_name)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Real", "AI"],
            digits=4,
            zero_division=0,
        )
    )

    if backbone_for_export is not None:
        try:
            _export_onnx(backbone_for_export, args.onnx_out)
        except RuntimeError as e:
            print(str(e))
    joblib.dump(clf, args.clf_out)

    meta = {
        "model": "mobilenetv3_large_embedding",
        "classifier_type": args.classifier,
        "classifier_calibration": args.calibrate,
        "dataset_a": args.dataset_a_name,
        "dataset_b": args.dataset_b_name,
        "holdout": args.holdout,
        "holdout_name": holdout_name,
        "onnx": args.onnx_out,
        "classifier_path": args.clf_out,
        "embedding_dim": int(X_train.shape[1]),
    }
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved ONNX backbone to:", args.onnx_out)
    print("Saved classifier to:", args.clf_out)
    print("Saved meta to:", args.meta_out)


if __name__ == "__main__":
    main()
