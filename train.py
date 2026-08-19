import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from feature_extraction import extract_all_features


def iter_images(dataset_root: str):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for root, _, files in os.walk(dataset_root):
        for name in files:
            lower = name.lower()
            _, ext = os.path.splitext(lower)
            if ext not in exts:
                continue
            yield os.path.join(root, name)


def label_from_filename(path: str):
    base = os.path.basename(path).lower()
    if base.startswith("input"):
        return 0
    if base.startswith("output"):
        return 1
    return None


def label_from_folders(path: str):
    parts = [p.lower() for p in path.split(os.sep) if p]
    for p in reversed(parts):
        if p == "real":
            return 0
        if p in ("fake", "ai", "synthetic"):
            return 1
    return None


def label_from_path(path: str):
    label = label_from_filename(path)
    if label is not None:
        return label
    return label_from_folders(path)


def group_from_path(path: str):
    base = os.path.basename(path).lower()
    if base.startswith("input") or base.startswith("output"):
        return os.path.basename(os.path.dirname(path))
    return os.path.splitext(os.path.basename(path))[0]


def load_dataset(dataset_root: str, dataset_tag: str):
    X = []
    y = []
    groups = []
    skipped = 0

    for p in iter_images(dataset_root):
        label = label_from_path(p)
        if label is None:
            continue

        feats = extract_all_features(p)
        if feats is None or not np.all(np.isfinite(feats)):
            skipped += 1
            continue

        X.append(np.asarray(feats, dtype=np.float32))
        y.append(label)
        groups.append(f"{dataset_tag}:{group_from_path(p)}")

    if not X:
        raise RuntimeError(f"No labeled images found under: {dataset_root}")

    return np.vstack(X), np.asarray(y, dtype=np.int64), np.asarray(groups), skipped


def _normalize_dataset_args(values):
    roots = []
    for v in values:
        if not v:
            continue
        parts = [p.strip() for p in v.split(",")]
        roots.extend([p for p in parts if p])
    deduped = []
    seen = set()
    for r in roots:
        abs_r = os.path.abspath(r)
        if abs_r not in seen:
            seen.add(abs_r)
            deduped.append(abs_r)
    return deduped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        action="append",
        help="Dataset root path. Repeat --dataset for multiple roots, or pass comma-separated paths.",
    )
    parser.add_argument(
        "--out",
        default="models/ai_detector_random_forest_model.pkl",
        help="Output path for trained model .pkl",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    dataset_roots = _normalize_dataset_args(args.dataset)
    Xs = []
    ys = []
    gs = []
    skipped_total = 0

    for root in dataset_roots:
        tag = os.path.basename(root) or "dataset"
        X_i, y_i, g_i, skipped_i = load_dataset(root, dataset_tag=tag)
        Xs.append(X_i)
        ys.append(y_i)
        gs.append(g_i)
        skipped_total += skipped_i

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Datasets:", dataset_roots)
    print("Skipped images:", skipped_total)
    print("Train size:", len(train_idx), "Test size:", len(test_idx))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Real (input)", "AI (output)"],
            digits=4,
            zero_division=0,
        )
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(clf, args.out)
    print("Saved model to:", args.out)


if __name__ == "__main__":
    main()
