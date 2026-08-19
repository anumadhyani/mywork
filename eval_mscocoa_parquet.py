import argparse
import glob
import os
import random
import tempfile
from typing import Iterable
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import classification_report, confusion_matrix

from model import predict_image


def _iter_parquet_rows(parquet_paths: list[str], seed: int) -> Iterable[tuple[bytes, int]]:
    rng = random.Random(seed)
    shuffled = parquet_paths[:]
    rng.shuffle(shuffled)

    for path in shuffled:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=["Image", "Label_A"]):
            img_col = batch.column(batch.schema.get_field_index("Image"))
            label_col = batch.column(batch.schema.get_field_index("Label_A"))

            # Convert batch columns to python lists. Image is struct<bytes, path>.
            img_py = img_col.to_pylist()
            label_py = label_col.to_pylist()
            for img_struct, label in zip(img_py, label_py):
                if img_struct is None:
                    continue
                b = img_struct.get("bytes") if isinstance(img_struct, dict) else None
                if b is None:
                    continue
                try:
                    y = int(label)
                except Exception:
                    continue
                yield b, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mscocoa-dir",
        default="MSCocoa_dataset",
        help="Path to folder containing MSCocoa parquet shards.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max number of examples to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mscocoa_dir = os.path.abspath(args.mscocoa_dir)
    parquet_paths = sorted(glob.glob(os.path.join(mscocoa_dir, f"{args.split}-*.parquet")))
    if not parquet_paths:
        raise SystemExit(f"No parquet shards found for split '{args.split}' under: {mscocoa_dir}")

    y_true: list[int] = []
    y_pred: list[int] = []
    bands: list[str] = []
    p_ai_values: list[float] = []
    skipped = 0
    failed = 0

    band_counts: Counter[str] = Counter()
    band_errors: dict[str, Counter[str]] = defaultdict(Counter)

    for i, (img_bytes, label_a) in enumerate(_iter_parquet_rows(parquet_paths, seed=args.seed)):
        if args.limit and len(y_true) >= args.limit:
            break

        # Label_A: 0=Real, 1=Fake (AI)
        if label_a not in (0, 1):
            skipped += 1
            continue

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp_path = tmp.name
                tmp.write(img_bytes)

            out = predict_image(tmp_path)
            pred = int(out.get("pred")) if isinstance(out, dict) and out.get("pred") is not None else None
            if pred not in (0, 1):
                skipped += 1
                continue

            band = "unknown"
            if isinstance(out, dict) and out.get("band"):
                band = str(out.get("band"))
            band_counts[band] += 1

            proba = out.get("proba") if isinstance(out, dict) else None
            if isinstance(proba, (list, tuple)) and len(proba) >= 2:
                try:
                    p_ai = float(proba[1])
                    if np.isfinite(p_ai):
                        p_ai_values.append(p_ai)
                except Exception:
                    pass

            # Track errors by band
            if int(pred) != int(label_a):
                if int(label_a) == 1 and int(pred) == 0:
                    band_errors[band]["ai_as_real"] += 1
                elif int(label_a) == 0 and int(pred) == 1:
                    band_errors[band]["real_as_ai"] += 1
                else:
                    band_errors[band]["other"] += 1

            y_true.append(int(label_a))
            y_pred.append(int(pred))
            bands.append(band)

            if (len(y_true) % 100) == 0:
                print(f"Evaluated {len(y_true)} examples...")
        except Exception:
            failed += 1
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    if not y_true:
        raise SystemExit("No examples evaluated (all skipped/failed).")

    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    acc = float((yt == yp).mean())
    print("Split:", args.split)
    print("Examples evaluated:", len(y_true))
    print("Skipped:", skipped, "Failed:", failed)
    print("Accuracy:", f"{acc * 100:.2f}%")
    print("Confusion matrix:\n", confusion_matrix(yt, yp))

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
