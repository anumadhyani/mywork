import argparse
import os
import random


# --- Defactify HF export ---
# Exports the HuggingFace dataset:
#   Rajarshi-Roy-research/Defactify_Image_Dataset
# into a folder structure compatible with train_embeddings.py:
#   out_dir/Real/*.jpg
#   out_dir/Fake/*.jpg
#
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="Rajarshi-Roy-research/Defactify_Image_Dataset",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to export (depends on HF dataset; commonly: train/validation/test).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory to write images into Real/ and Fake/ subfolders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, export at most this many examples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--by-model",
        action="store_true",
        default=False,
        help="If set, also nest AI images by Label_B (e.g., Fake/midjourney, Fake/dalle_3).",
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
    image_col = _pick_col(cols, ["image", "Image", "img", "Img"])
    label_a_col = _pick_col(cols, ["Label_A", "label_a", "labelA"])
    label_b_col = _pick_col(cols, ["Label_B", "label_b", "labelB"])

    if not image_col:
        raise SystemExit(f"Could not find image column. Available columns: {cols}")
    if not label_a_col:
        raise SystemExit(f"Could not find Label_A column. Available columns: {cols}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    if args.limit and args.limit > 0:
        idxs = idxs[: args.limit]

    exported = 0
    skipped = 0

    for i in idxs:
        ex = ds[int(i)]

        try:
            label_a = int(ex.get(label_a_col))
        except Exception:
            skipped += 1
            continue

        if label_a == 0:
            cls_dir = os.path.join(out_dir, "Real")
        elif label_a == 1:
            if args.by_model and label_b_col:
                try:
                    label_b = int(ex.get(label_b_col))
                except Exception:
                    label_b = -1
                src = _LABEL_B_NAMES.get(label_b, f"label_b_{label_b}")
                cls_dir = os.path.join(out_dir, "Fake", src)
            else:
                cls_dir = os.path.join(out_dir, "Fake")
        else:
            skipped += 1
            continue

        os.makedirs(cls_dir, exist_ok=True)

        im = ex.get(image_col)
        if im is None:
            skipped += 1
            continue

        out_path = os.path.join(cls_dir, f"defactify_{args.split}_{exported:08d}.jpg")

        try:
            if hasattr(im, "save"):
                im.save(out_path)
            elif isinstance(im, dict) and im.get("bytes"):
                with open(out_path, "wb") as f:
                    f.write(im["bytes"])
            else:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        exported += 1
        if (exported % 500) == 0:
            print(f"Exported {exported}/{len(idxs)}")

    print("Export complete")
    print("Split:", args.split)
    print("Exported:", exported)
    print("Skipped:", skipped)
    print("Output dir:", out_dir)


if __name__ == "__main__":
    main()
