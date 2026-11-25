# -*- coding: gb18030 -*-
"""
noise_robustness_test.py  (patched)
====================================

Evaluate two methods on the climbing-over-barrier 50-image test split under
two progressive perturbation modes: square occlusion and motion blur.

  Methods compared
  ----------------
    * yolo500    : closed-vocabulary YOLO climbing detector trained on 500
                   labeled images. Loaded from a .pt checkpoint.
    * full       : the proposed scene + pose + interaction + ensemble head
                   from `sam2.GroundingSAMClimbDetectorV2`.

  Perturbations
  -------------
    * occlusion  : a square gray patch of side `s = level * min(H, W)` is
                   pasted on the person (centered on the person bbox if
                   available, otherwise on the image center).
    * motion_blur: a horizontal averaging kernel of length k = 1 + 10*level
                   is convolved with the image.

  Outputs
  -------
    out_dir/
        qual_occlusion/
            full_L<i>_climb_<stem>.jpg       # 1 climb example per (method, level)
            full_L<i>_noclimb_<stem>.jpg     # 1 non-climb example per (method, level)
            yolo500_L<i>_climb_<stem>.jpg
            yolo500_L<i>_noclimb_<stem>.jpg
        qual_blur/
            (same structure)
        accuracy_table.csv
        accuracy_curves.png
        accuracy_table.tex
        per_image_predictions.json

Fixes vs. original
------------------
  1. YOLO_OFFLINE / ULTRALYTICS_OFFLINE / HUB_OFFLINE env-vars are set at
     module import time (before ultralytics is imported anywhere) so the
     network-timeout crash is prevented.
  2. Yolo500Wrapper validates the checkpoint path before loading; raises a
     clear FileNotFoundError instead of timing out on GitHub.
  3. ultralytics telemetry sync is disabled via settings.update({'sync': False}).
  4. evaluate() now saves 2 qualitative images per (method, kind, level):
     one climb sample and one non-climb sample (filenames tagged accordingly).
  5. _pick_qual_indices() automatically selects the first climb image and
     the first non-climb image from the test set.
"""

from __future__ import annotations

# ============================================================
# ①  Set offline flags BEFORE ultralytics is ever imported.
# ============================================================
import os
os.environ['YOLO_OFFLINE']          = '1'
os.environ['ULTRALYTICS_OFFLINE']   = '1'
os.environ['HUB_OFFLINE']           = '1'

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
OCCLUSION_LEVELS = [0.0, 0.03, 0.09, 0.12, 0.15, 0.18]  # patch side / min(H,W)
BLUR_LEVELS      = [0, 1, 2, 3, 4, 5]                    # kernel = 1 + 10*level
PERTURBATIONS    = ['occlusion', 'blur']

IMG_EXTS             = ('.jpg', '.jpeg', '.png', '.bmp')
CLIMB_CLASS_ID_IN_GT = 1   # overrideable via --climb_class_id


# =============================================================================
#  Data loading  (YOLO-format: image/ + label/)
# =============================================================================

def _resolve_yolo_dirs(test_folder: Path) -> Tuple[Path, Path]:
    """Locate the image/ and label/ subdirectories of a YOLO-style test set."""
    cand_img = test_folder / 'image'
    cand_lbl = test_folder / 'label'
    if cand_img.is_dir() and cand_lbl.is_dir():
        return cand_img, cand_lbl
    if (test_folder.parent / 'label').is_dir():
        return test_folder, test_folder.parent / 'label'
    raise FileNotFoundError(
        f"Could not locate YOLO image/ and label/ relative to {test_folder}")


def collect_test_image(test_folder: Path) -> List[Path]:
    """List every image under <test_folder>/image/ in sorted order."""
    img_dir, _ = _resolve_yolo_dirs(test_folder)
    paths = sorted([p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMG_EXTS and p.is_file()])
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return paths


def _read_yolo_label_file(lbl_path: Path) -> int:
    """Image-level label aggregated from one YOLO .txt file.

    Returns 1 iff at least one row has class_id == CLIMB_CLASS_ID_IN_GT.
    """
    if not lbl_path.exists():
        return 0
    try:
        with open(lbl_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cid = int(float(parts[0]))
                except (ValueError, IndexError):
                    continue
                if cid == CLIMB_CLASS_ID_IN_GT:
                    return 1
    except OSError:
        return 0
    return 0


def _read_yolo_label_boxes(
        lbl_path: Path,
) -> List[Tuple[int, float, float, float, float]]:
    """Read every YOLO row of one .txt file as (cid, cx, cy, w, h) in
    normalized [0, 1] form. Returns [] on any error."""
    if not lbl_path.exists():
        return []
    rows: List[Tuple[int, float, float, float, float]] = []
    try:
        with open(lbl_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                                      float(parts[3]), float(parts[4]))
                except ValueError:
                    continue
                rows.append((cid, cx, cy, bw, bh))
    except OSError:
        return []
    return rows


def _yolo_to_xyxy(
        box: Tuple[int, float, float, float, float],
        img_w: int, img_h: int,
) -> Tuple[int, int, int, int]:
    """Convert one YOLO row (cid, cx, cy, w, h) to pixel-space (x1,y1,x2,y2)."""
    _, cx, cy, bw, bh = box
    x1 = int(round((cx - bw / 2.0) * img_w))
    y1 = int(round((cy - bh / 2.0) * img_h))
    x2 = int(round((cx + bw / 2.0) * img_w))
    y2 = int(round((cy + bh / 2.0) * img_h))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2


def load_person_bboxes_normalized(
        test_paths: List[Path],
) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    """Return {filename -> list of YOLO rows} in normalized coordinates."""
    if not test_paths:
        return {}
    img_dir = test_paths[0].parent
    lbl_dir = img_dir.parent / 'label'
    return {p.name: _read_yolo_label_boxes(lbl_dir / (p.stem + '.txt'))
            for p in test_paths}


def pick_person_bbox_xyxy(
        boxes_for_image: List[Tuple[int, float, float, float, float]],
        img_w: int, img_h: int,
        prefer_class_id: int = CLIMB_CLASS_ID_IN_GT,
) -> Optional[Tuple[int, int, int, int]]:
    """Pick the largest bbox (preferred class first) in pixel space."""
    if not boxes_for_image:
        return None
    pref = [b for b in boxes_for_image if b[0] == prefer_class_id]
    pool = pref if pref else boxes_for_image
    pool_sorted = sorted(pool, key=lambda b: b[3] * b[4], reverse=True)
    return _yolo_to_xyxy(pool_sorted[0], img_w, img_h)


def load_label(test_paths: List[Path],
               label_csv: Optional[Path] = None) -> Dict[str, int]:
    """Return {image filename -> 0/1} from YOLO label files.

    label_csv is kept for backward compatibility; it overrides YOLO labels.
    """
    if not test_paths:
        return {}
    img_dir = test_paths[0].parent
    lbl_dir = img_dir.parent / 'label'
    if not lbl_dir.is_dir():
        raise FileNotFoundError(
            f"Expected YOLO label/ next to image/, but {lbl_dir} is missing.")

    label: Dict[str, int] = {
        p.name: _read_yolo_label_file(lbl_dir / (p.stem + '.txt'))
        for p in test_paths
    }

    if label_csv is not None and label_csv.exists():
        df = pd.read_csv(label_csv)
        if len(df.columns) >= 2:
            fcol, ycol = df.columns[0], df.columns[1]
            for _, row in df.iterrows():
                label[str(row[fcol])] = int(row[ycol])

    return label


# =============================================================================
#  Perturbations
# =============================================================================

def apply_occlusion(img: np.ndarray, level: float,
                    person_bbox: Optional[Tuple[int, int, int, int]] = None,
                    ) -> np.ndarray:
    """Paste a gray square of side level*min(H,W) on the person bbox center."""
    if level <= 0.0:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    side = max(1, int(level * min(h, w)))
    if person_bbox is not None:
        x1, y1, x2, y2 = person_bbox
        cx = (x1 + x2) // 2
        cy = int(y1 + (y2 - y1) * 0.7)
    else:
        cx, cy = w // 2, h // 2
    rx1 = max(0, cx - side // 2); ry1 = max(0, cy - side // 2)
    rx2 = min(w, rx1 + side);     ry2 = min(h, ry1 + side)
    out[ry1:ry2, rx1:rx2] = (128, 128, 128)
    return out


def apply_motion_blur(img: np.ndarray, level: int) -> np.ndarray:
    """Horizontal averaging of length k = 1 + 10*level. level=0 = identity."""
    if level <= 0:
        return img
    k = 1 + 10 * level
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def perturb(img: np.ndarray, kind: str, level_value,
            person_bbox=None) -> np.ndarray:
    if kind == 'occlusion':
        return apply_occlusion(img, float(level_value), person_bbox)
    elif kind == 'blur':
        return apply_motion_blur(img, int(level_value))
    raise ValueError(f"unknown perturbation: {kind}")


# =============================================================================
#  Method wrappers
# =============================================================================

class FullDetectorWrapper:
    """Thin adapter around GroundingSAMClimbDetectorV2."""

    def __init__(self, pose_model_path: str,
                 classifier_pkl: Optional[str],
                 method: str = 'ensemble',
                 threshold: float = 0.220):
        sys.path.append(str(Path(__file__).parent.resolve()))
        from sam2 import GroundingSAMClimbDetectorV2  # noqa
        self.detector = GroundingSAMClimbDetectorV2(
            pose_model_path=pose_model_path,
            classifier_method=method)
        if classifier_pkl is not None and Path(classifier_pkl).exists():
            self.detector.load_classifier(classifier_pkl)
        self.detector.classifier.threshold = float(threshold)
        self.threshold = float(threshold)

    def predict(self, img: np.ndarray) -> Tuple[int, float, Dict]:
        try:
            is_climb, score, detail = self.detector.detect_climb(img, debug=False)
        except Exception as e:
            return 0, 0.0, {'error': str(e)}
        return int(bool(is_climb)), float(score), detail

    def visualize(self, img: np.ndarray, score: float,
                  is_climb: int, detail: Dict) -> np.ndarray:
        if 'keypoints' in detail and 'masks' in detail:
            try:
                return self.detector.visualize(
                    img, detail['keypoints'], detail['masks'],
                    bool(is_climb), score, detail)
            except Exception:
                pass
        return img.copy()


class Yolo500Wrapper:
    """Closed-vocabulary YOLO climbing detector."""

    def __init__(self, ckpt_path: str, conf_thr: float = 0.25,
                 climb_class_id: int = 0):
        # ② Validate the checkpoint exists BEFORE ultralytics tries to
        #    resolve it (which would trigger a GitHub network request).
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"[Yolo500Wrapper] file not found: {ckpt.resolve()}\n"
                f" --yolo500_path to .pt "
                f"--skip_yolo500")

        # ③ Disable ultralytics telemetry / sync to prevent any network calls.
        from ultralytics import YOLO
        from ultralytics import settings as ult_settings
        try:
            ult_settings.update({'sync': False})
        except Exception:
            pass

        self.model     = YOLO(str(ckpt), task="detect", verbose=False)
        self.conf_thr  = float(conf_thr)
        self.cid       = int(climb_class_id)

    def predict(self, img: np.ndarray) -> Tuple[int, float, Dict]:
        try:
            res = self.model.predict(img, conf=self.conf_thr,
                                     verbose=False, amp=False)
        except Exception as e:
            return 0, 0.0, {'error': str(e)}
        if not res or len(res) == 0 or res[0].boxes is None:
            return 0, 0.0, {'boxes': []}
        boxes = res[0].boxes
        if len(boxes) == 0:
            return 0, 0.0, {'boxes': []}
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)
        keep = (cls == self.cid) & (conf > self.conf_thr)
        detail = {
            'boxes': boxes.xyxy.cpu().numpy().tolist(),
            'conf':  conf.tolist(),
            'cls':   cls.tolist(),
        }
        if not keep.any():
            return 0, float(conf.max() if len(conf) else 0.0), detail
        return 1, float(conf[keep].max()), detail

    def visualize(self, img: np.ndarray, score: float,
                  is_climb: int, detail: Dict) -> np.ndarray:
        out = img.copy()
        for b, c, k in zip(detail.get('boxes', []),
                            detail.get('conf',  []),
                            detail.get('cls',   [])):
            x1, y1, x2, y2 = [int(v) for v in b]
            color = (0, 0, 255) if (k == self.cid and c > self.conf_thr) \
                else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{c:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        tag = 'CLIMB' if is_climb else 'NORMAL'
        col = (0, 0, 255) if is_climb else (0, 200, 0)
        cv2.putText(out, f"YOLO500 {tag} {score:.2f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        return out


# =============================================================================
#  Metric utilities
# =============================================================================

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp  = int(((y_true == 1) & (y_pred == 1)).sum())
    fp  = int(((y_true == 0) & (y_pred == 1)).sum())
    fn  = int(((y_true == 1) & (y_pred == 0)).sum())
    tn  = int(((y_true == 0) & (y_pred == 0)).sum())
    p   = tp / (tp + fp + 1e-9)
    r   = tp / (tp + fn + 1e-9)
    f1  = 2 * p * r / (p + r + 1e-9)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-9)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'P': p, 'R': r, 'F1': f1, 'Acc': acc}


# =============================================================================
#  ④ Helper: pick 2 qualitative-image indices (1 climb + 1 non-climb)
# =============================================================================

def _pick_qual_indices(test_paths: List[Path],
                       label: Dict[str, int]) -> List[int]:
    """Return [first_climb_idx, first_noclimb_idx].

    Falls back to index 0 if one category is absent.
    """
    climb_idx, noclimb_idx = None, None
    for i, p in enumerate(test_paths):
        y = label.get(p.name, 0)
        if climb_idx is None and y == 1:
            climb_idx = i
        if noclimb_idx is None and y == 0:
            noclimb_idx = i
        if climb_idx is not None and noclimb_idx is not None:
            break
    return [climb_idx   if climb_idx   is not None else 0,
            noclimb_idx if noclimb_idx is not None else 0]


# =============================================================================
#  Main evaluation
# =============================================================================

def evaluate(method_name: str,
             method_obj,
             test_paths: List[Path],
             label: Dict[str, int],
             out_dir: Path,
             bboxes_norm: Optional[Dict[str, List[Tuple[int, float, float, float, float]]]] = None,
             qual_save_indices: Optional[List[int]] = None,
             per_image_log: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Run method_obj on every (image, perturbation_kind, level) triple.

    Qualitative images
    ------------------
    For each (method, kind, level) combination two images are saved:
      * one climb sample  -> filename contains '_climb_'
      * one non-climb sample -> filename contains '_noclimb_'
    The indices of the two images are given by `qual_save_indices`
    (default: first climb + first non-climb from the test set).
    """
    if qual_save_indices is None:
        qual_save_indices = _pick_qual_indices(test_paths, label)

    rows = []

    for kind, levels in (('occlusion', OCCLUSION_LEVELS),
                          ('blur',      BLUR_LEVELS)):
        qual_dir = out_dir / f'qual_{kind}'
        qual_dir.mkdir(parents=True, exist_ok=True)

        for li, lv in enumerate(levels):
            print(f"\n[{method_name}] perturb={kind:9s}  level={lv}")
            preds: List[int] = []
            ys:    List[int] = []

            for j, p in enumerate(test_paths):
                img = cv2.imread(str(p))
                if img is None:
                    continue

                # Person bbox for occlusion patch placement.
                bbox_xyxy = None
                if kind == 'occlusion' and bboxes_norm is not None:
                    h, w = img.shape[:2]
                    bbox_xyxy = pick_person_bbox_xyxy(
                        bboxes_norm.get(p.name, []), w, h,
                        prefer_class_id=CLIMB_CLASS_ID_IN_GT)

                img_p = perturb(img, kind, lv, person_bbox=bbox_xyxy)
                yhat, score, detail = method_obj.predict(img_p)
                preds.append(int(yhat))
                ys.append(int(label.get(p.name, 0)))

                if per_image_log is not None:
                    per_image_log.append({
                        'method':      method_name,
                        'perturbation': kind,
                        'level':       li,
                        'level_value': lv,
                        'filename':    p.name,
                        'y_true':      int(label.get(p.name, 0)),
                        'y_pred':      int(yhat),
                        'score':       float(score),
                    })

                # ④ Save 2 qualitative images per (method, kind, level).
                if j in qual_save_indices:
                    tag = 'climb' if label.get(p.name, 0) == 1 else 'noclimb'
                    vis = method_obj.visualize(img_p, score, yhat, detail)
                    out_path = qual_dir / \
                        f"{method_name}_L{li}_{tag}_{p.stem}.jpg"
                    cv2.imwrite(str(out_path), vis)

            stats = confusion(np.asarray(ys), np.asarray(preds))
            stats.update({
                'method':      method_name,
                'perturbation': kind,
                'level':       li,
                'level_value': lv,
            })
            rows.append(stats)
            print(f"  -> Acc={stats['Acc']:.3f}  F1={stats['F1']:.3f}  "
                  f"TP={stats['TP']} FP={stats['FP']} "
                  f"FN={stats['FN']} TN={stats['TN']}")

    return pd.DataFrame(rows)


# =============================================================================
#  Reporting: LaTeX table + accuracy curves
# =============================================================================

def write_latex_table(df: pd.DataFrame, out_path: Path):
    """Write a LaTeX accuracy table grouped by perturbation type."""
    methods = sorted(df['method'].unique())

    def _row(method: str, kind: str, levels) -> str:
        cells = []
        for lv in levels:
            sel = df[(df['method'] == method) &
                     (df['perturbation'] == kind) &
                     (df['level_value'] == lv)]
            cells.append('--' if len(sel) == 0
                         else f"{float(sel['Acc'].iloc[0]):.3f}")
        return ' & '.join(cells)

    lines: List[str] = []
    lines.append(r"% noise_robustness_test.py  auto-generated table")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Image-level accuracy under progressive occlusion "
        r"and motion blur on the 50-image test split.}")
    lines.append(r"\label{tab:robustness}")

    # ---- Occlusion sub-table ----
    n_occ = len(OCCLUSION_LEVELS)
    lines.append(r"\begin{tabular}{l|" + "c" * n_occ + r"}")
    lines.append(r"\toprule")
    lines.append(
        r"\multicolumn{" + str(n_occ + 1) +
        r"}{c}{\textbf{(a) Occlusion (patch side / min(H,W))}} \\")
    lines.append(r"\midrule")
    lines.append(
        "Method & " +
        " & ".join(f"{lv:.2f}" for lv in OCCLUSION_LEVELS) + r" \\")
    lines.append(r"\midrule")
    for m in methods:
        lines.append(f"{m} & " + _row(m, 'occlusion', OCCLUSION_LEVELS) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1.0em}")

    # ---- Motion blur sub-table ----
    n_blur = len(BLUR_LEVELS)
    lines.append(r"\begin{tabular}{l|" + "c" * n_blur + r"}")
    lines.append(r"\toprule")
    lines.append(
        r"\multicolumn{" + str(n_blur + 1) +
        r"}{c}{\textbf{(b) Motion blur "
        r"(kernel level, size $=1+10\cdot\text{level}$)}} \\")
    lines.append(r"\midrule")
    lines.append(
        "Method & " +
        " & ".join(str(lv) for lv in BLUR_LEVELS) + r" \\")
    lines.append(r"\midrule")
    for m in methods:
        lines.append(f"{m} & " + _row(m, 'blur', BLUR_LEVELS) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"[write_latex_table] -> {out_path}")


def plot_accuracy_curves(df: pd.DataFrame, out_path: Path):
    """Plot accuracy vs perturbation level for occlusion and blur side by side."""
    methods      = sorted(df['method'].unique())
    color_cycle  = ['#1F77B4', '#D62728', '#2CA02C', '#FF7F0E', '#9467BD']
    marker_cycle = ['o', 's', '^', 'D', 'v']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, kind, levels, xlabel in (
            (axes[0], 'occlusion', OCCLUSION_LEVELS,
             'Occlusion patch fraction of $\\min(H,W)$'),
            (axes[1], 'blur',      BLUR_LEVELS,
             'Motion-blur kernel level (size $= 1+10\\cdot$level)')):
        for i, m in enumerate(methods):
            sel = (df[(df['method'] == m) & (df['perturbation'] == kind)]
                   .sort_values('level_value'))
            ax.plot(sel['level_value'].values, sel['Acc'].values,
                    color=color_cycle[i % len(color_cycle)],
                    marker=marker_cycle[i % len(marker_cycle)],
                    linewidth=2.0, markersize=7, label=m)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(levels)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(
            'Occlusion robustness' if kind == 'occlusion'
            else 'Motion-blur robustness', fontsize=12)

    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].legend(loc='lower left', frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot_accuracy_curves] -> {out_path}")


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Climbing-detector robustness test: occlusion & motion blur",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--test_folder', type=str, required=True,
                    help='YOLO-format test set with image/ and label/ subdirs.')
    ap.add_argument('--climb_class_id', type=int, default=1,
                    help='Class id in YOLO .txt files that denotes climbing.')
    ap.add_argument('--label_csv', type=str, default=None,
                    help='Optional CSV (filename, label) override (legacy).')
    ap.add_argument('--our_model_path', type=str,
                    default='climb_classifier_v2.pkl')
    ap.add_argument('--pose_model', type=str, default='yolo11m-pose.pt')
    ap.add_argument('--our_method', type=str, default='ensemble',
                    choices=['rule', 'ml', 'ensemble'])
    ap.add_argument('--our_threshold', type=float, default=0.220)
    ap.add_argument('--yolo500_path', type=str,
                    default=os.environ.get(
                        'YOLO500_CHECKPOINT',
                        'results_yolo_experiments/exp_500/best_weights.pt'),
                    help='Local path to the YOLO500 .pt checkpoint. '
                         'The file MUST already exist on disk; the script '
                         'will NOT attempt to download it.')
    ap.add_argument('--yolo500_conf', type=float, default=0.25)
    ap.add_argument('--yolo500_class', type=int, default=1)
    ap.add_argument('--out_dir', type=str, default='results/robustness')
    ap.add_argument('--skip_full', action='store_true',
                    help='Skip the FULL detector.')
    ap.add_argument('--skip_yolo500', action='store_true',
                    help='Skip the YOLO500 detector.')
    return ap.parse_args()


def main():
    args = parse_args()

    global CLIMB_CLASS_ID_IN_GT
    CLIMB_CLASS_ID_IN_GT = int(args.climb_class_id)

    test_folder = Path(args.test_folder)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data and labels ----
    test_paths  = collect_test_image(test_folder)
    label_csv   = Path(args.label_csv) if args.label_csv else None
    label       = load_label(test_paths, label_csv)
    bboxes_norm = load_person_bboxes_normalized(test_paths)

    n_pos      = sum(1 for p in test_paths if label.get(p.name, 0) == 1)
    n_neg      = len(test_paths) - n_pos
    n_with_box = sum(1 for p in test_paths if bboxes_norm.get(p.name))

    print("=" * 72)
    print(f"[noise_robustness_test] test set : {len(test_paths)} images "
          f"({n_pos} climb, {n_neg} no-climb)")
    print(f"[noise_robustness_test] bboxes   : "
          f"{n_with_box}/{len(test_paths)} images have more than 1 YOLO box")
    print(f"[noise_robustness_test] out dir  : {out_dir}")
    print(f"[noise_robustness_test] occlusion levels : {OCCLUSION_LEVELS}")
    print(f"[noise_robustness_test] blur levels      : {BLUR_LEVELS} "
          f"(kernel sizes {[1 + 10*lv for lv in BLUR_LEVELS]})")
    print("=" * 72)

    # ---- 2. Pick 2 fixed qualitative-image indices ----
    # ⑤  One climb sample + one non-climb sample saved per (method, kind, level)
    qual_indices = _pick_qual_indices(test_paths, label)
    print(f"[noise_robustness_test] qualitative images: "
          f"{[test_paths[i].name for i in qual_indices]}")

    # ---- 3. Run every method ----
    all_dfs: List[pd.DataFrame] = []
    per_image_log: List[Dict]   = []

    if not args.skip_full:
        print("\n" + "=" * 72)
        print("Method 1/2 : FULL (scene + pose + interaction + ensemble)")
        print("=" * 72)
        full = FullDetectorWrapper(
            pose_model_path=args.pose_model,
            classifier_pkl=args.our_model_path,
            method=args.our_method,
            threshold=args.our_threshold)
        all_dfs.append(evaluate(
            method_name='full',
            method_obj=full,
            test_paths=test_paths,
            label=label,
            out_dir=out_dir,
            bboxes_norm=bboxes_norm,
            qual_save_indices=qual_indices,
            per_image_log=per_image_log))

    if not args.skip_yolo500:
        print("\n" + "=" * 72)
        print("Method 2/2 : YOLO500 (closed-vocabulary, 500-image training)")
        print("=" * 72)
        # FileNotFoundError is raised inside Yolo500Wrapper if .pt is missing.
        yolo500 = Yolo500Wrapper(
            ckpt_path=args.yolo500_path,
            conf_thr=args.yolo500_conf,
            climb_class_id=args.yolo500_class)
        all_dfs.append(evaluate(
            method_name='yolo500',
            method_obj=yolo500,
            test_paths=test_paths,
            label=label,
            out_dir=out_dir,
            bboxes_norm=bboxes_norm,
            qual_save_indices=qual_indices,
            per_image_log=per_image_log))

    if not all_dfs:
        print("[noise_robustness_test] No methods evaluated. Exiting.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    # ---- 4. Save CSV ----
    csv_path = out_dir / 'accuracy_table.csv'
    df_out = df[['method', 'perturbation', 'level', 'level_value',
                 'Acc', 'F1', 'P', 'R', 'TP', 'FP', 'FN', 'TN']].copy()
    df_out.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n[noise_robustness_test] CSV   -> {csv_path}")

    # ---- 5. LaTeX table ----
    write_latex_table(df, out_dir / 'accuracy_table.tex')

    # ---- 6. Accuracy curves ----
    plot_accuracy_curves(df, out_dir / 'accuracy_curves.png')

    # ---- 7. Per-image predictions JSON ----
    pi_path = out_dir / 'per_image_predictions.json'
    with open(pi_path, 'w', encoding='utf-8') as fp:
        json.dump(per_image_log, fp, indent=2, ensure_ascii=False)
    print(f"[noise_robustness_test] per-image predictions -> {pi_path}")

    # ---- 8. Summary pivot table ----
    print("\n" + "=" * 72)
    print("Summary  (image-level accuracy)")
    print("=" * 72)
    print(df_out.pivot_table(
        index='method',
        columns=['perturbation', 'level_value'],
        values='Acc',
        aggfunc='first').round(3).to_string())
    print("=" * 72)


if __name__ == '__main__':
    main()