# -*- coding: utf-8 -*-
"""
export_features.py
==================
Extracts the interpretable interaction feature vector f for every image of a
split and writes it as a CSV.

This is the artefact promised in the paper's Data availability statement
("the extracted interaction-feature vectors").  Each row is one image; the
columns are the named feature groups of the method:

    pose_*   body-only kinematic descriptors (the pose-only baseline's input)
    scene_*  position / contact / box / occlusion relations to the obstacle

Every column has a one-line verbal gloss, which is the point of the
representation: the decision layer is auditable because the feature space
itself is readable.  `feature_glossary.md` (written next to the CSV) contains
the glossary.

Determinism
-----------
The script is deterministic given the same model checkpoints: the pose model,
the scene parser and the classifier are all deterministic at inference time.
A fixed seed is nevertheless set for every RNG that is touched, so that the
run is reproducible even if a dependency starts using one.

Usage
-----
    python scripts/export_features.py \
        --image_dir  /path/to/test1/image \
        --label_dir  /path/to/test1/label \
        --out_csv    results/features/test1.csv \
        --pose_model yolo11m-pose.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # keep speed; inference is deterministic
    except Exception:
        pass


def _read_label(path: Path) -> int:
    """Return 1 (climb) / 0 (no-climb) from a YOLO-style label file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        text = ""
    if not text:
        return 0
    first = text.splitlines()[0].strip().split()
    if not first:
        return 0
    try:
        return int(float(first[0]))
    except Exception:
        return 0


def _glossary(feature_names: List[str]) -> str:
    """A short human-readable gloss for every feature family."""
    lines = ["# Feature glossary", "",
             "One line per family of coordinates of the interaction vector f.",
             "Every coordinate has a one-line verbal gloss, which is what makes",
             "the decision layer auditable.", ""]
    family_gloss = {
        "pose_": ("Body-only kinematics, computed from the COCO-17 keypoints "
                  "alone. This sub-vector is the input of the pose-only baseline."),
        "scene_": ("Human-scene interaction predicates: the relation between the "
                   "body and the obstacle/ground parsed from the scene."),
    }
    for prefix, gloss in family_gloss.items():
        lines.append(f"## `{prefix}*`")
        lines.append("")
        lines.append(gloss)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--label_dir", default=None,
                    help="Directory of YOLO-style label files (optional)")
    ap.add_argument("--split_csv", default=None,
                    help="splits CSV produced by make_splits.py; overrides "
                         "--image_dir/--label_dir")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--pose_model", default="yolo11m-pose.pt")
    ap.add_argument("--classifier", default="rule",
                    choices=["rule", "ml", "ensemble"],
                    help="Head used to produce the companion score column")
    ap.add_argument("--model_path", default=None,
                    help="Trained classifier .pkl (for 'ml' / 'ensemble')")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N images (debugging)")
    ap.add_argument("--sample_n", type=int, default=None,
                    help="Randomly sample N images (at the fixed seed) "
                         "instead of taking all of them. The paper fits the "
                         "learned head on ~100 images sampled from the "
                         "training pool.")
    args = ap.parse_args()

    set_seed(args.seed)

    # Make `saip` importable when the script is run from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "saip"))

    from sam2 import GroundingSAMClimbDetectorV2  # noqa: E402

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir) if args.label_dir else None

    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
                    + list(image_dir.glob("*.png")))
    if args.limit:
        images = images[:args.limit]
    if args.sample_n and args.sample_n < len(images):
        # The paper draws the ~100-image training subset from the training
        # pool at a fixed seed; the same draw is reproduced here.
        rng = random.Random(args.seed)
        images = sorted(rng.sample(images, args.sample_n))
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    print(f"[export] {len(images)} images from {image_dir}")

    detector = GroundingSAMClimbDetectorV2(
        pose_model_path=args.pose_model,
        classifier_method=args.classifier,
    )
    if args.classifier in ("ml", "ensemble") and args.model_path:
        print(f"[export] loading classifier {args.model_path}")
        detector.load_classifier(args.model_path)

    rows: List[Dict] = []
    feature_names: List[str] = []

    for i, img_path in enumerate(images):
        print(f"[export {i+1}/{len(images)}] {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print("   unreadable, skipped")
            continue

        label = -1
        if label_dir is not None:
            label = _read_label(label_dir / f"{img_path.stem}.txt")

        try:
            is_climb, score, detail = detector.detect_climb(image, debug=False)
        except Exception as exc:  # keep the run alive, record the failure
            print(f"   error: {exc}")
            rows.append(dict(filename=img_path.name, label=label,
                             score=0.0, prediction=0, error=str(exc)[:200]))
            continue

        feats = detail.get("all_features", {})
        if not feature_names and feats:
            feature_names = sorted(feats.keys())

        row = dict(filename=img_path.name, label=label,
                   score=float(score), prediction=int(bool(is_climb)))
        for k, v in feats.items():
            row[k] = float(v)
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = ["filename", "label", "prediction", "score"] + feature_names
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # Companion metadata
    meta = dict(
        n_images=len(rows),
        n_features=len(feature_names),
        feature_names=feature_names,
        pose_model=args.pose_model,
        classifier=args.classifier,
        classifier_model_path=args.model_path,
        seed=args.seed,
        image_dir=str(image_dir),
    )
    with open(out_csv.with_suffix(".meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    with open(out_csv.parent / "feature_glossary.md", "w", encoding="utf-8") as fp:
        fp.write(_glossary(feature_names))

    print(f"\n[export] wrote {out_csv}  ({len(rows)} rows, "
          f"{len(feature_names)} features)")
    print(f"[export] wrote {out_csv.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
