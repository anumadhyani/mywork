import argparse
import glob
import os
from pathlib import Path


def _default_parquet_files(mscocoa_dir: str):
    patterns = [
        os.path.join(mscocoa_dir, "train-*.parquet"),
        os.path.join(mscocoa_dir, "validation-*.parquet"),
        os.path.join(mscocoa_dir, "test-*.parquet"),
    ]
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mscocoa-dir",
        required=True,
        help="Path to the local 'MSCocoa dataset' folder containing *.parquet shards.",
    )
    parser.add_argument(
        "--out-dir",
        default="mscocoa_images",
        help="Output directory to write images into Real/ and Fake/ subfolders.",
    )
    parser.add_argument(
        "--splits",
        default="train,validation,test",
        help="Comma-separated splits to export: train,validation,test",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, export at most this many examples per split (for quick tests).",
    )
    parser.add_argument(
        "--by-model",
        action="store_true",
        default=False,
        help="If set, also nest AI images by Label_B (e.g. SDXL, DALLE3).",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency 'datasets'. Install locally with: python -m pip install datasets pyarrow"
        ) from e

    mscocoa_dir = os.path.abspath(args.mscocoa_dir)
    out_dir = os.path.abspath(args.out_dir)

    parquet_files = _default_parquet_files(mscocoa_dir)
    if not parquet_files:
        raise SystemExit(
            f"No parquet shards found under {mscocoa_dir}. Expected files like train-00000-of-*.parquet"
        )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    os.makedirs(out_dir, exist_ok=True)

    for split in splits:
        split_glob = os.path.join(mscocoa_dir, f"{split}-*.parquet")
        split_files = sorted(glob.glob(split_glob))
        if not split_files:
            print(f"Skipping split '{split}' (no files matching {split_glob})")
            continue

        ds = load_dataset("parquet", data_files=split_files, split="train")

        real_dir = os.path.join(out_dir, split, "Real")
        fake_dir = os.path.join(out_dir, split, "Fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        n = 0
        for i, row in enumerate(ds):
            if args.limit and n >= args.limit:
                break

            label_a = int(row.get("Label_A"))
            label_b = row.get("Label_B")

            image = row.get("Image")
            if image is None:
                continue

            if label_a == 0:
                target_dir = real_dir
            else:
                target_dir = fake_dir
                if args.by_model:
                    model_name = str(label_b)
                    target_dir = os.path.join(fake_dir, model_name)
                    os.makedirs(target_dir, exist_ok=True)

            out_path = os.path.join(target_dir, f"{split}_{i:08d}.jpg")
            try:
                image.save(out_path, format="JPEG", quality=95)
            except Exception:
                continue

            n += 1
            if n % 1000 == 0:
                print(f"Exported {n} images for split '{split}'...")

        print(f"Done split '{split}': exported {n} images")

    print("Export complete. Output:", out_dir)


if __name__ == "__main__":
    main()
