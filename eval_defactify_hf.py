import argparse
import os
import random
import tempfile
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from model import predict_image


# --- Defactify HF evaluation ---
# Evaluates our current model on the HuggingFace dataset:
#   Rajarshi-Roy-research/Defactify_Image_Dataset
# Label_A: 0=Real, 1=AI
# Label_B: 0=Real, 1=SD2.1, 2=SDXL, 3=SD3, 4=DALL·E 3, 5=Midjourney


_LABEL_B_NAMES = {
    0: "real",
    1: "stable_diffusion_2_1",
    2: "stable_diffusion_xl",
    3: "stable_diffusion_3",
    4: "dalle_3",
    5: "midjourney",
}


def _pick_col(cols: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _iter_examples(ds, seed: int, limit: int):
    # Shuffle deterministically without requiring a full materialization if possible.
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    if limit and limit > 0:
        idxs = idxs[:limit]

    for i in idxs:
        yield ds[int(i)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (depends on HF dataset; commonly: train/validation/test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max number of examples to evaluate (0 means all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        default="Rajarshi-Roy-research/Defactify_Image_Dataset",
        help="HuggingFace dataset name.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency 'datasets'. Install locally with: python -m pip install datasets"
        ) from e

    ds = load_dataset(args.dataset, split=args.split)

    cols = list(getattr(ds, "column_names", []) or [])
    image_col = _pick_col(cols, ["image", "Image", "img", "Img", "pixel_values"])
    label_a_col = _pick_col(cols, ["Label_A", "label_a", "labelA"])
    label_b_col = _pick_col(cols, ["Label_B", "label_b", "labelB"])

    if not image_col:
        raise SystemExit(f"Could not find image column. Available columns: {cols}")
    if not label_a_col:
        raise SystemExit(f"Could not find Label_A column. Available columns: {cols}")

    y_true: list[int] = []
    y_pred: list[int] = []
    band_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()

    per_source_total: Counter[str] = Counter()
    per_source_correct: Counter[str] = Counter()

    p_ai_values: list[float] = []
    skipped = 0
    failed = 0

    # Errors by band help diagnose whether "uncertain" is swallowing the distribution.
    band_errors: dict[str, Counter[str]] = defaultdict(Counter)

    for n, ex in enumerate(_iter_examples(ds, seed=args.seed, limit=args.limit)):
        try:
            label_a = ex.get(label_a_col)
            try:
                label_a = int(label_a)
            except Exception:
                skipped += 1
                continue
            if label_a not in (0, 1):
                skipped += 1
                continue

            label_b = None
            if label_b_col:
                try:
                    label_b = int(ex.get(label_b_col))
                except Exception:
                    label_b = None

            im = ex.get(image_col)
            if im is None:
                skipped += 1
                continue

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp_path = tmp.name

                # HF datasets Image feature returns a PIL image-like object with .save().
                if hasattr(im, "save"):
                    im.save(tmp_path)
                elif isinstance(im, dict) and "path" in im and os.path.exists(str(im["path"])):
                    # Some datasets provide {"path": ...} or {"bytes": ...}.
                    src_path = str(im["path"])
                    with open(src_path, "rb") as f:
                        with open(tmp_path, "wb") as out:
                            out.write(f.read())
                elif isinstance(im, dict) and im.get("bytes"):
                    with open(tmp_path, "wb") as out:
                        out.write(im["bytes"])
                else:
                    skipped += 1
                    continue

                out = predict_image(tmp_path)
                pred = int(out.get("pred")) if isinstance(out, dict) and out.get("pred") is not None else None
                if pred not in (0, 1):
                    skipped += 1
                    continue

                band = str(out.get("band") or "unknown") if isinstance(out, dict) else "unknown"
                backend = str(out.get("backend") or "unknown") if isinstance(out, dict) else "unknown"
                band_counts[band] += 1
                backend_counts[backend] += 1

                proba = out.get("proba") if isinstance(out, dict) else None
                if isinstance(proba, (list, tuple)) and len(proba) >= 2:
                    try:
                        p_ai = float(proba[1])
                        if np.isfinite(p_ai):
                            p_ai_values.append(p_ai)
                    except Exception:
                        pass

                if int(pred) != int(label_a):
                    if int(label_a) == 1 and int(pred) == 0:
                        band_errors[band]["ai_as_real"] += 1
                    elif int(label_a) == 0 and int(pred) == 1:
                        band_errors[band]["real_as_ai"] += 1
                    else:
                        band_errors[band]["other"] += 1

                y_true.append(int(label_a))
                y_pred.append(int(pred))

                if label_b is not None:
                    src = _LABEL_B_NAMES.get(int(label_b), f"label_b_{label_b}")
                    per_source_total[src] += 1
                    if int(pred) == int(label_a):
                        per_source_correct[src] += 1

                if (len(y_true) % 100) == 0:
                    print(f"Evaluated {len(y_true)} examples...")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        except Exception:
            failed += 1

    if not y_true:
        raise SystemExit("No examples evaluated (all skipped/failed).")

    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    acc = float((yt == yp).mean())
    print("Dataset:", args.dataset)
    print("Split:", args.split)
    print("Examples evaluated:", len(y_true))
    print("Skipped:", skipped, "Failed:", failed)
    print("Accuracy:", f"{acc * 100:.2f}%")
    print("Confusion matrix:\n", confusion_matrix(yt, yp))

    print("Backends:")
    for b, c in backend_counts.most_common():
        print(f"  {b}: {c}")

    print("Band counts:")
    for band, c in band_counts.most_common():
        print(f"  {band}: {c}")

    if band_errors:
        print("Errors by band:")
        for band, ctr in sorted(band_errors.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
            total_err = sum(ctr.values())
            print(
                f"  {band}: {total_err} (ai_as_real={ctr.get('ai_as_real', 0)}, real_as_ai={ctr.get('real_as_ai', 0)}, other={ctr.get('other', 0)})"
            )

    if per_source_total:
        print("Accuracy by Label_B (source model):")
        for src, total in sorted(per_source_total.items(), key=lambda kv: (-kv[1], kv[0])):
            correct = int(per_source_correct.get(src, 0))
            print(f"  {src}: {correct}/{total} ({(correct / total) * 100:.2f}%)")

    if p_ai_values:
        p = np.asarray(p_ai_values, dtype=np.float32)
        print(
            "p_ai stats:",
            f"n={len(p)}",
            f"min={float(np.min(p)):.4f}",
            f"p10={float(np.quantile(p, 0.10)):.4f}",
            f"p50={float(np.quantile(p, 0.50)):.4f}",
            f"p90={float(np.quantile(p, 0.90)):.4f}",
            f"max={float(np.max(p)):.4f}",
        )

    print(
        classification_report(
            yt,
            yp,
            target_names=["Real", "AI"],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
