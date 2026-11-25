# -*- coding: utf-8 -*-
"""
task_unified.py  —  Combined Task 1 + Task 2 runner (updated for scene_sam2.py).

What changed vs the original two scripts
========================================
1) Uses the V2 pipeline from `scene_sam2.py`:
     - GroundingSAMClimbDetectorV2     (new detector)
     - ClimbFeatureExtractorV2         (new feature extractor, v2 features)
     - ClimbClassifierV2               (rule / ml / ensemble heads)

2) **Actually uses the trained model.**  When `--classifier ml` or
   `--classifier ensemble` is given and `--model_path` points to a trained
   `.pkl` file (produced by `scene_sam2.py --mode train`), the script
   calls `detector.load_classifier(...)` and all downstream predictions
   go through the ML / ensemble head instead of the plain rule head.

3) Task 1 (sequence visualisation) and Task 2 (test1 evaluation) are
   merged into a single file, selectable via `--mode`:
       --mode sequence   →  Task 1
       --mode eval       →  Task 2
       --mode all        →  run both back-to-back

4) Visualisation uses the V2 `detector.visualize(...)` method, so the
   output frames already include:
       - barrier (cyan) / wall (red) / ground (green) overlays
       - the robust yellow obstacle-top line
       - person skeleton + wrists highlighted
       - RED bbox + "CLIMB DETECTED" / GREEN bbox + "person"
       - bottom-left evidence panel
       - top CLIMB ALERT / NORMAL banner

Usage examples
--------------
    # Task 1 only (sequence), using the trained ensemble classifier
    python task_unified.py --mode sequence \
        --input_dir sequence_input --output_dir sequence_output \
        --classifier ensemble --model_path climb_classifier_v2.pkl

    # Task 2 only (test1 eval), using the trained ML classifier
    python task_unified.py --mode eval \
        --test_dir test1 --out_dir results \
        --classifier ml --model_path climb_classifier_v2.pkl

    # Both
    python task_unified.py --mode all \
        --input_dir sequence_input --output_dir sequence_output \
        --test_dir test1 --out_dir results \
        --classifier ensemble --model_path climb_classifier_v2.pkl
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Make the SAIP modules importable no matter where the script is invoked from.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam2 import (
    GroundingSAMClimbDetectorV2,
    ClimbFeatureExtractorV2,
)


# =============================================================================
# Detector factory — builds a V2 detector and (optionally) loads a trained pkl
# =============================================================================
def build_detector(pose_model: str,
                   classifier: str,
                   model_path: str = None) -> GroundingSAMClimbDetectorV2:
    """Instantiate V2 detector and, if needed, load the trained classifier."""
    detector = GroundingSAMClimbDetectorV2(
        pose_model_path=pose_model,
        classifier_method=classifier,
    )
    if classifier in ('ml', 'ensemble'):
        if model_path and Path(model_path).exists():
            print(f"[init] Loading trained classifier  ->  {model_path}")
            detector.load_classifier(model_path)
        else:
            print(f"[warn] classifier='{classifier}' but no trained model "
                  f"at '{model_path}'.")
            if classifier == 'ml':
                print("       ML predictions will fail — falling back to 'rule'.")
                detector.classifier.method = 'rule'
            else:
                print("       Ensemble will fall back to rule-only scoring.")
    return detector


# =============================================================================
# TASK 1  —  Sequence visualisation
# =============================================================================
def _collect_images(folder: Path) -> List[Path]:
    return sorted(list(folder.glob('*.jpg')) +
                  list(folder.glob('*.jpeg')) +
                  list(folder.glob('*.png')))


def run_sequence(detector: GroundingSAMClimbDetectorV2,
                 input_dir: Path,
                 output_dir: Path) -> List[Dict]:
    """Task 1: per-frame detection + visualisation for a consecutive sequence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = _collect_images(input_dir)
    if not frame_paths:
        print(f"[sequence] no images in {input_dir}")
        return []
    print(f"[sequence] found {len(frame_paths)} frames in {input_dir}")

    summary = []
    for i, p in enumerate(frame_paths):
        print(f"\n[sequence {i+1}/{len(frame_paths)}] {p.name}")
        image = cv2.imread(str(p))
        if image is None:
            print("  failed to read, skipped")
            continue

        try:
            is_climb, score, detail = detector.detect_climb(image, debug=False)
        except Exception as e:
            print(f"  error: {e}")
            continue

        # Use the V2 visualisation (includes evidence panel + banner)
        if 'keypoints' in detail and 'masks' in detail:
            vis = detector.visualize(image,
                                     detail['keypoints'],
                                     detail['masks'],
                                     is_climb, score, detail)
        else:
            # no person detected — still emit the frame with a note
            vis = image.copy()
            cv2.putText(vis, "no person detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Frame index banner (top-right)
        banner = f"Frame {i+1:02d}/{len(frame_paths):02d}"
        cv2.putText(vis, banner, (vis.shape[1] - 260, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        prefix = 'CLIMB_' if is_climb else 'NORMAL_'
        out_path = output_dir / f"{prefix}{i+1:02d}_{p.stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  -> {out_path.name}  (is_climb={is_climb}, score={score:.3f})")

        summary.append({
            'frame':    p.name,
            'is_climb': bool(is_climb),
            'score':    float(score),
            'output':   str(out_path),
        })

    # Summary to stdout
    climb_frames = [s for s in summary if s['is_climb']]
    print("\n" + "=" * 70)
    print(f"Sequence summary: {len(climb_frames)}/{len(summary)} "
          f"frames flagged as CLIMB")
    for s in summary:
        flag = 'CLIMB ' if s['is_climb'] else 'normal'
        print(f"  {flag}  score={s['score']:.3f}   {s['frame']}")
    print("=" * 70)
    print(f"All visualizations saved to: {output_dir}")

    # Persist sequence summary
    json.dump(summary, open(output_dir / 'sequence_summary.json', 'w'),
              indent=2)
    return summary


# =============================================================================
# TASK 2  —  Evaluate on test1/ and compare FULL vs POSE-ONLY
# =============================================================================
def load_label(label_path: Path, stem: str) -> int:
    """Return 1 (climb) or 0 (no-climb); fall back to filename prefix."""
    if label_path.exists():
        try:
            text = label_path.read_text(encoding='utf-8',
                                        errors='ignore').strip()
        except Exception:
            text = ''
        if text:
            first = text.splitlines()[0].strip().lower()
            if first in ('1', 'climb', 'climbing'):
                return 1
            if first in ('0', 'no_climb', 'normal', 'noclimb'):
                return 0
            for line in text.splitlines():
                tok = line.split()
                if len(tok) >= 1 and tok[0] == '1':
                    return 1
            return 0
    if stem.lower().startswith(('climb_', 'climbing_')):
        return 1
    if stem.lower().startswith(('noclimb_', 'no_climb_', 'normal_')):
        return 0
    print(f"  [warn] label missing & no filename hint for {stem} -> assumed 0")
    return 0


# -----------------------------------------------------------------------------
# Pose-only rule head (updated for V2 feature names)
# -----------------------------------------------------------------------------
def pose_only_rule(features: Dict[str, float]) -> Tuple[bool, float]:
    """
    Simple rule based purely on V2 pose features (no segmentation,
    no interaction). Returns (is_climb, score).
    """
    max_arm_lift  = features.get('pose_max_arm_lift', 0.0)
    min_knee_ang  = features.get('pose_min_knee_angle', 180.0)
    max_knee_lift = features.get('pose_max_knee_lift', 0.0)
    trunk_tilt    = features.get('pose_trunk_tilt', 0.0)
    limb_span_x   = features.get('pose_limb_span_x', 0.0)
    limb_span_y   = features.get('pose_limb_span_y', 0.0)
    # V2-only: lower-body occlusion signal
    low_vis       = features.get('pose_lower_body_visible_ratio', 1.0)
    upp_vis       = features.get('pose_upper_body_visible_ratio', 1.0)
    asym          = features.get('pose_upper_minus_lower_visible', 0.0)

    score = 0.0
    if max_arm_lift > 100:   score += 0.28
    elif max_arm_lift > 50:  score += 0.14

    if 60 < min_knee_ang < 120:    score += 0.22
    elif 120 < min_knee_ang < 150: score += 0.10

    if max_knee_lift > 50:   score += 0.12
    if trunk_tilt > 15:      score += 0.08
    if limb_span_y > limb_span_x * 1.3: score += 0.15

    # Occlusion asymmetry is a strong climb signal even with pose-only.
    if upp_vis > 0.6 and low_vis < 0.3 and asym > 0.3:
        score += 0.20
    elif asym > 0.2:
        score += 0.10

    score = float(min(score, 1.0))
    return (score >= 0.5), score


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def precision_recall(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc  = (tp + tn) / len(y_true) if y_true else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, f1=f1, accuracy=acc)


def mean_score_by_class(y_true, scores):
    pos = [s for t, s in zip(y_true, scores) if t == 1]
    neg = [s for t, s in zip(y_true, scores) if t == 0]
    return (float(np.mean(pos)) if pos else 0.0,
            float(np.mean(neg)) if neg else 0.0)


# -----------------------------------------------------------------------------
# Per-detector inference
# -----------------------------------------------------------------------------
def infer_full_pipeline(detector: GroundingSAMClimbDetectorV2, image):
    """
    Full V2 method: segmentation + pose + interaction features + classifier
    (rule / ml / ensemble — whatever the detector was configured with).
    """
    h, w = image.shape[:2]
    masks = detector.segment_scene(image)
    pose_res = detector.pose_model.predict(image, conf=0.1, verbose=False)
    if not pose_res or not hasattr(pose_res[0], 'keypoints'):
        return 0, 0.0
    kp = pose_res[0].keypoints
    if kp is None or len(kp.data) == 0:
        return 0, 0.0
    keypoints = kp.data[0].cpu().numpy()
    features = detector.feature_extractor.extract_all_features(
        keypoints, masks, (h, w))
    is_climb, score, _ = detector.classifier.predict(features)
    return int(is_climb), float(score)


def infer_pose_only(pose_model, feat_ext: ClimbFeatureExtractorV2, image):
    """Pose-only: YOLO-pose + pose-only rule (uses V2 pose feature names)."""
    pose_res = pose_model.predict(image, conf=0.1, verbose=False)
    if not pose_res or not hasattr(pose_res[0], 'keypoints'):
        return 0, 0.0
    kp = pose_res[0].keypoints
    if kp is None or len(kp.data) == 0:
        return 0, 0.0
    keypoints = kp.data[0].cpu().numpy()
    pose_features = feat_ext.extract_pose_features(keypoints)
    features = {f'pose_{k}': v for k, v in pose_features.items()}
    is_climb, score = pose_only_rule(features)
    return int(is_climb), float(score)


def run_eval(detector: GroundingSAMClimbDetectorV2,
             test_dir: Path,
             out_dir: Path) -> Dict:
    """Task 2: evaluate FULL pipeline vs POSE_ONLY on test_dir/."""
    img_dir = test_dir / 'image'
    lab_dir = test_dir / 'label'
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = _collect_images(img_dir)
    if not img_paths:
        print(f"[eval] no images in {img_dir}")
        return {}
    print(f"[eval] found {len(img_paths)} test images in {img_dir}")

    pose_model = detector.pose_model
    feat_ext   = ClimbFeatureExtractorV2()

    y_true, pred_full, score_full = [], [], []
    pred_pose, score_pose = [], []
    per_image = []

    for i, p in enumerate(img_paths):
        stem = p.stem
        print(f"\n[eval {i+1}/{len(img_paths)}] {p.name}")
        image = cv2.imread(str(p))
        if image is None:
            print("  failed to read, skipped")
            continue

        gt = load_label(lab_dir / f"{stem}.txt", stem)

        try:
            pf, sf = infer_full_pipeline(detector, image)
        except Exception as e:
            print(f"  [full] error: {e}")
            pf, sf = 0, 0.0
        try:
            pp, sp = infer_pose_only(pose_model, feat_ext, image)
        except Exception as e:
            print(f"  [pose] error: {e}")
            pp, sp = 0, 0.0

        y_true.append(gt)
        pred_full.append(pf); score_full.append(sf)
        pred_pose.append(pp); score_pose.append(sp)

        print(f"  gt={gt}  full={pf} ({sf:.3f})  "
              f"pose_only={pp} ({sp:.3f})")

        per_image.append(dict(
            filename=p.name, gt=gt,
            full_pred=pf, full_score=sf,
            pose_pred=pp, pose_score=sp,
        ))

    # Aggregate
    m_full = precision_recall(y_true, pred_full)
    m_pose = precision_recall(y_true, pred_pose)
    pos_full, neg_full = mean_score_by_class(y_true, score_full)
    pos_pose, neg_pose = mean_score_by_class(y_true, score_pose)

    summary = {
        'classifier_method': detector.classifier.method,
        'threshold':         float(detector.classifier.threshold),
        'n_total':           len(y_true),
        'n_climb':           int(sum(y_true)),
        'n_no_climb':        int(len(y_true) - sum(y_true)),
        'full_pipeline': {
            **m_full,
            'mean_score_climb':    pos_full,
            'mean_score_no_climb': neg_full,
        },
        'pose_only': {
            **m_pose,
            'mean_score_climb':    pos_pose,
            'mean_score_no_climb': neg_pose,
        },
    }

    # Save
    json.dump(per_image, open(out_dir / 'task2_per_image.json', 'w'),
              indent=2)
    json.dump(summary, open(out_dir / 'task2_summary.json', 'w'),
              indent=2)

    # Pretty print
    print("\n" + "=" * 92)
    print(f"  Task 2 — Comparison on {summary['n_total']} images "
          f"({summary['n_climb']} climb, {summary['n_no_climb']} no-climb)")
    print(f"  FULL head = '{summary['classifier_method']}', "
          f"threshold = {summary['threshold']:.3f}")
    print("=" * 92)
    hdr = (f"{'Method':<18} {'P':>7} {'R':>7} {'F1':>7} {'Acc':>7} "
           f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
           f"{'mean(climb)':>12} {'mean(noclimb)':>14}")
    print(hdr)
    print("-" * 92)
    for name, m in [('FULL (ours)', summary['full_pipeline']),
                    ('POSE_ONLY',   summary['pose_only'])]:
        print(f"{name:<18} "
              f"{m['precision']:>7.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>7.3f} {m['accuracy']:>7.3f} "
              f"{m['tp']:>4d} {m['fp']:>4d} {m['fn']:>4d} {m['tn']:>4d} "
              f"{m['mean_score_climb']:>12.3f} "
              f"{m['mean_score_no_climb']:>14.3f}")
    print("=" * 92)
    print(f"Detailed per-image results : {out_dir / 'task2_per_image.json'}")
    print(f"Summary                    : {out_dir / 'task2_summary.json'}")

    return summary


# =============================================================================
# CLI
# =============================================================================
def _parse_args():
    ap = argparse.ArgumentParser(
        description="Unified Task 1 (sequence viz) + Task 2 (test1 eval) "
                    "for the V2 climb-over-barrier detector.")
    ap.add_argument('--mode', required=True,
                    choices=['sequence', 'eval', 'all'],
                    help="'sequence' = Task 1, 'eval' = Task 2, "
                         "'all' = run both")

    # Common
    ap.add_argument('--pose_model', default='yolo11m-pose.pt')
    ap.add_argument('--classifier', default='ensemble',
                    choices=['rule', 'ml', 'ensemble'],
                    help="Which classifier head to use in the FULL pipeline.")
    ap.add_argument('--model_path', default='climb_classifier_v2.pkl',
                    help="Path to the trained classifier (.pkl). "
                         "Used when --classifier is 'ml' or 'ensemble'.")

    # Task 1
    ap.add_argument('--input_dir',  default='sequence_input',
                    help="Folder with the 10 consecutive frames (Task 1).")
    ap.add_argument('--output_dir', default='sequence_output',
                    help="Where to write per-frame visualisations (Task 1).")

    # Task 2
    ap.add_argument('--test_dir', default='test1',
                    help="Root folder containing image/ and label/ (Task 2).")
    ap.add_argument('--out_dir',  default='results',
                    help="Where to write eval summaries (Task 2).")
    return ap.parse_args()


def main():
    args = _parse_args()
    detector = build_detector(
        pose_model=args.pose_model,
        classifier=args.classifier,
        model_path=args.model_path,
    )

    if args.mode in ('sequence', 'all'):
        run_sequence(detector,
                     Path(args.input_dir),
                     Path(args.output_dir))

    if args.mode in ('eval', 'all'):
        run_eval(detector,
                 Path(args.test_dir),
                 Path(args.out_dir))


if __name__ == '__main__':
    main()
