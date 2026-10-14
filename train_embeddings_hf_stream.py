import argparse
import json
import os
import tempfile
from dataclasses import dataclass

import joblib
import numpy as np
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
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1, tol=None)
    classes = np.asarray([0, 1], dtype=np.int64)

    # --- Shared embedder (reuse one MobileNet instance for train + test) ---
    model, preprocess, device = _make_embedder(device=args.device)

    def embed_batch(pil_images: list):
        return _embed_batch(model, preprocess, device=device, pil_images=pil_images)

    # --- Train loop ---
    print("Streaming train split:", args.train_split)
    seen_train = 0
    skipped_train = 0
    pil_batch: list = []
    y_batch: list[int] = []

    for ex in _iter_stream(args.dataset, split=args.train_split, seed=args.seed):
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
            clf.partial_fit(X, y, classes=classes)
            pil_batch = []
            y_batch = []

            if (seen_train % (args.batch_size * 50)) == 0:
                print(f"Trained on {seen_train} examples...")

    if pil_batch:
        X = embed_batch(pil_batch)
        y = np.asarray(y_batch, dtype=np.int64)
        clf.partial_fit(X, y, classes=classes)

    print("Train done")
    print("Train examples:", seen_train, "skipped:", skipped_train)

    # --- Evaluate loop ---
    print("Streaming test split:", args.test_split)
    y_true: list[int] = []
    y_pred: list[int] = []
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
            yp = clf.predict(X)
            y_true.extend(y_batch)
            y_pred.extend([int(v) for v in yp.tolist()])
            pil_batch = []
            y_batch = []

    if pil_batch:
        X = embed_batch(pil_batch)
        yp = clf.predict(X)
        y_true.extend(y_batch)
        y_pred.extend([int(v) for v in yp.tolist()])

    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

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

    # --- Save classifier + meta (no images stored) ---
    os.makedirs(os.path.dirname(args.clf_out) or ".", exist_ok=True)
    joblib.dump(clf, args.clf_out)

    meta = {
        "dataset": args.dataset,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_examples": int(seen_train),
        "test_examples": int(len(yt)),
        "classifier": "sgd_log_loss",
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
