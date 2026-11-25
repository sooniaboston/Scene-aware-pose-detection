# -*- coding: utf-8 -*-
"""
multi_occlusion_test.py
=======================

Evaluate two methods on the climbing-over-barrier test split under a
5 x 5 matrix of occlusion conditions:
  - Rows    : number of occlusion patches (1, 2, 3, 4, 5)
  - Columns : patch size level            (1, 2, 3, 4, 5)
              → patch side = level * 0.05 * min(H, W)
                (i.e. 5 %, 10 %, 15 %, 20 %, 25 % of the shorter image edge)

  Methods compared
  ----------------
    yolo500 : closed-vocabulary YOLO climbing detector trained on 500 labeled
              images, loaded from a local .pt checkpoint.
    full    : scene + pose + interaction + ensemble head from
              `sam2.GroundingSAMClimbDetectorV2`.

  For every (num_patches, size_level) cell one representative *pre-prediction*
  occluded image is saved (the first climbing sample found in the test set).
  All 25 cells are then composited into a single 5 × 5 grid PNG.

  Outputs
  -------
    out_dir/
        occluded_samples/
            occ_np<N>_sz<S>_<stem>.jpg    # pre-prediction occluded frame
        grid_5x5.png                      # 5-row × 5-col composite
        accuracy_matrix_yolo500.csv
        accuracy_matrix_full.csv
        accuracy_heatmap.png
        per_image_predictions.json
"""

from __future__ import annotations

# ── Block ultralytics network calls before the package is imported ────────────
import os
os.environ['YOLO_OFFLINE']        = '1'
os.environ['ULTRALYTICS_OFFLINE'] = '1'
os.environ['HUB_OFFLINE']         = '1'

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Condition grids ───────────────────────────────────────────────────────────
PATCH_SIZE_LEVELS = [0, 0.15, 0.20, 0.25, 0.30]   # fraction of min(H,W)
NUM_PATCHES_LIST  = [10, 15, 20, 25,30]                   # patch counts

IMG_EXTS             = ('.jpg', '.jpeg', '.png', '.bmp')
CLIMB_CLASS_ID_IN_GT = 1   # overrideable via --climb_class_id


# =============================================================================
#  Data loading  (YOLO-format: image/ + label/)
# =============================================================================

def _resolve_yolo_dirs(test_folder: Path) -> Tuple[Path, Path]:
    cand_img = test_folder / 'image'
    cand_lbl = test_folder / 'label'
    if cand_img.is_dir() and cand_lbl.is_dir():
        return cand_img, cand_lbl
    if (test_folder.parent / 'label').is_dir():
        return test_folder, test_folder.parent / 'label'
    raise FileNotFoundError(
        f"Could not locate YOLO image/ and label/ relative to {test_folder}")


def collect_test_images(test_folder: Path) -> List[Path]:
    img_dir, _ = _resolve_yolo_dirs(test_folder)
    paths = sorted([p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMG_EXTS and p.is_file()])
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return paths


def _read_yolo_label_file(lbl_path: Path, climb_cid: int = CLIMB_CLASS_ID_IN_GT) -> int:
    if not lbl_path.exists():
        return 0
    try:
        with open(lbl_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    if int(float(parts[0])) == climb_cid:
                        return 1
                except (ValueError, IndexError):
                    continue
    except OSError:
        return 0
    return 0


def _read_yolo_boxes(lbl_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not lbl_path.exists():
        return []
    rows = []
    try:
        with open(lbl_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    rows.append((int(float(parts[0])),
                                 float(parts[1]), float(parts[2]),
                                 float(parts[3]), float(parts[4])))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def _yolo_to_xyxy(box, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    _, cx, cy, bw, bh = box
    x1 = max(0, min(img_w - 1, int(round((cx - bw / 2) * img_w))))
    y1 = max(0, min(img_h - 1, int(round((cy - bh / 2) * img_h))))
    x2 = max(0, min(img_w - 1, int(round((cx + bw / 2) * img_w))))
    y2 = max(0, min(img_h - 1, int(round((cy + bh / 2) * img_h))))
    return x1, y1, x2, y2


def load_labels(test_paths: List[Path]) -> Dict[str, int]:
    if not test_paths:
        return {}
    lbl_dir = test_paths[0].parent.parent / 'label'
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Expected label/ at {lbl_dir}")
    return {p.name: _read_yolo_label_file(lbl_dir / (p.stem + '.txt'))
            for p in test_paths}


def load_bboxes(test_paths: List[Path]) -> Dict[str, List]:
    if not test_paths:
        return {}
    lbl_dir = test_paths[0].parent.parent / 'label'
    return {p.name: _read_yolo_boxes(lbl_dir / (p.stem + '.txt'))
            for p in test_paths}


def pick_person_bbox(boxes, img_w: int, img_h: int,
                     prefer_cid: int = CLIMB_CLASS_ID_IN_GT
                     ) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    pref = [b for b in boxes if b[0] == prefer_cid]
    pool = sorted(pref if pref else boxes,
                  key=lambda b: b[3] * b[4], reverse=True)
    return _yolo_to_xyxy(pool[0], img_w, img_h)


# =============================================================================
#  Multi-patch occlusion
# =============================================================================

def apply_multi_occlusion(
        img: np.ndarray,
        num_patches: int,
        size_level_idx: int,           # 0-based → PATCH_SIZE_LEVELS[idx]
        person_bbox: Optional[Tuple[int, int, int, int]] = None,
        seed: int = 42,
) -> np.ndarray:
    """
    Overlay `num_patches` gray squares of side PATCH_SIZE_LEVELS[size_level_idx]
    * min(H,W) on `img`.

    Placement strategy
    ------------------
    * If a person bounding box is available the patches are distributed
      horizontally across the torso region (upper 70 % of the bbox) so they
      cover relevant body features.
    * Otherwise the patches are spread evenly across the horizontal centre
      strip of the image.
    """
    out = img.copy()
    h, w = img.shape[:2]
    side = max(1, int(PATCH_SIZE_LEVELS[size_level_idx] * min(h, w)))
    rng  = np.random.RandomState(seed)

    for i in range(num_patches):
      t = (i + 1) / (num_patches + 1)
      cx = rng.randint(0, w)
      cy = rng.randint(0, h)
  
      # 绘制遮挡块
      rx1 = max(0, cx - side // 2)
      ry1 = max(0, cy - side // 2)
      rx2 = min(w, rx1 + side)
      ry2 = min(h, ry1 + side)
      out[ry1:ry2, rx1:rx2] = (128, 128, 128)
  
#      if person_bbox is not None:
#          x1, y1, y2, y2 = person_bbox  # 你原来的变量名
#          x1, y1, x2, y2 = person_bbox  # 正确写法（我帮你修正）
#          bw = max(1, x2 - x1)
#          bh = max(1, y2 - y1)
#  
#          cx = int(x1 + t * bw)
#          rand_val = rng.rand()
#  
#          # ==============================
#          # 严格按你要求的概率
#          # ==============================
#          if rand_val < 0.10:
#              # 10% → 头部、肩膀、上胸 (0% ~ 25% 高度)
#              cy = int(y1 + rng.rand() * 0.25 * bh)
#  
#          elif rand_val < 0.45:
#              # 35% → 身体下方 (65% ~ 100%)
#              cy = int(y1 + (0.65 + rng.rand() * 0.35) * bh)
#  
#          elif rand_val < 0.80:
#              # 35% → 身体周边（在bbox内，靠近四周边缘）
#              # 上下左右边缘区域，不跑出人体框
#              edge_type = rng.choice(['left', 'right', 'top', 'bottom'])
#  
#              if edge_type == 'left':
#                  cx = int(x1 + 0.05 * bw + rng.rand() * 0.15 * bw)
#                  cy = int(y1 + rng.rand() * bh)
#              elif edge_type == 'right':
#                  cx = int(x2 - 0.05 * bw - rng.rand() * 0.15 * bw)
#                  cy = int(y1 + rng.rand() * bh)
#              elif edge_type == 'top':
#                  cy = int(y1 + 0.05 * bh + rng.rand() * 0.15 * bh)
#                  cx = int(x1 + rng.rand() * bw)
#              else:  # bottom
#                  cy = int(y2 - 0.05 * bh - rng.rand() * 0.15 * bh)
#                  cx = int(x1 + rng.rand() * bw)
#  
#          else:
#              # 20% → 全身随机
#              cy = int(y1 + rng.rand() * bh)
#  
#      else:
#          # 没有 bounding box → 全图随机
        
  
    return out


# =============================================================================
#  Method wrappers  (unchanged from noise_robustness_test.py)
# =============================================================================

class FullDetectorWrapper:
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

    def predict(self, img: np.ndarray) -> Tuple[int, float, dict]:
        try:
            is_climb, score, detail = self.detector.detect_climb(img, debug=False)
        except Exception as e:
            return 0, 0.0, {'error': str(e)}
        return int(bool(is_climb)), float(score), detail


class Yolo500Wrapper:
    def __init__(self, ckpt_path: str, conf_thr: float = 0.25,
                 climb_class_id: int = 0):
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"[Yolo500Wrapper] checkpoint not found: {ckpt.resolve()}\n"
                f"Use --skip_yolo500 to skip this method.")
        from ultralytics import YOLO
        from ultralytics import settings as ult_settings
        try:
            ult_settings.update({'sync': False})
        except Exception:
            pass
        self.model    = YOLO(str(ckpt), task='detect', verbose=False)
        self.conf_thr = float(conf_thr)
        self.cid      = int(climb_class_id)

    def predict(self, img: np.ndarray) -> Tuple[int, float, dict]:
        try:
            res = self.model.predict(img, conf=self.conf_thr,
                                     verbose=False, amp=False)
        except Exception as e:
            return 0, 0.0, {'error': str(e)}
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return 0, 0.0, {}
        boxes = res[0].boxes
        conf  = boxes.conf.cpu().numpy()
        cls   = boxes.cls.cpu().numpy().astype(int)
        keep  = (cls == self.cid) & (conf > self.conf_thr)
        if not keep.any():
            return 0, float(conf.max() if len(conf) else 0.0), {}
        return 1, float(conf[keep].max()), {}


# =============================================================================
#  Metrics
# =============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-9)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'P': p, 'R': r, 'F1': f1, 'Acc': acc}


# =============================================================================
#  5 × 5 grid compositor
# =============================================================================

def make_5x5_grid(
        cell_images: List[List[np.ndarray]],   # [row][col] = BGR ndarray
        out_path: Path,
        gap: int = 10,
        cell_w: int = 160,
        cell_h: int = 200,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Composite a 5-row × 5-col grid of BGR images with white gaps.
    Optional text labels are drawn along the top (column) and left (row) edges.
    """
    label_margin = 26 if (row_labels or col_labels) else 0
    full_w = label_margin + 5 * cell_w + 4 * gap
    full_h = label_margin + 5 * cell_h + 4 * gap
    canvas = np.ones((full_h, full_w, 3), dtype=np.uint8) * 255  # white bg

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1

    # Draw column labels
    if col_labels:
        for c, lbl in enumerate(col_labels[:5]):
            x = label_margin + c * (cell_w + gap) + cell_w // 2 - 30
            cv2.putText(canvas, lbl, (max(0, x), 18),
                        font, font_scale, (50, 50, 50), thickness, cv2.LINE_AA)

    # Draw row labels
    if row_labels:
        for r, lbl in enumerate(row_labels[:5]):
            y = label_margin + r * (cell_h + gap) + cell_h // 2 + 5
            cv2.putText(canvas, lbl, (2, y),
                        font, font_scale, (50, 50, 50), thickness, cv2.LINE_AA)

    # Place cell images
    for r in range(5):
        for c in range(5):
            if r >= len(cell_images) or c >= len(cell_images[r]):
                continue
            img = cell_images[r][c]
            if img is None:
                continue
            img_resized = cv2.resize(img, (cell_w, cell_h))
            y0 = label_margin + r * (cell_h + gap)
            x0 = label_margin + c * (cell_w + gap)
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = img_resized

    cv2.imwrite(str(out_path), canvas)
    print(f"[grid] 5×5 grid saved → {out_path}")
    return canvas


# =============================================================================
#  Accuracy heat-map
# =============================================================================

def plot_heatmaps(
        acc_full: np.ndarray,
        acc_yolo: np.ndarray,
        out_path: Path,
) -> None:
    """
    Side-by-side heat-maps (5 × 5) for the two methods.
    Rows = num_patches (1-5), Cols = patch_size_level (1-5).
    """
    col_ticks = [f"{int(lv*100)}%" for lv in PATCH_SIZE_LEVELS]
    row_ticks = [f"{n}" for n in NUM_PATCHES_LIST]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title in (
            (axes[0], acc_full,  'Our Framework (full)'),
            (axes[1], acc_yolo,  'YOLO-500')):
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(5)); ax.set_xticklabels(col_ticks, fontsize=10)
        ax.set_yticks(range(5)); ax.set_yticklabels(row_ticks, fontsize=10)
        ax.set_xlabel('Patch size (fraction of min(H,W))', fontsize=11)
        ax.set_ylabel('Number of patches', fontsize=11)
        ax.set_title(title, fontsize=12)
        for r in range(5):
            for c in range(5):
                val = mat[r, c]
                ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color='black' if 0.3 < val < 0.8 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle('Image-level accuracy under multi-patch occlusion\n'
                 '(rows = #patches, cols = patch size)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"[heatmap] accuracy heat-maps saved → {out_path}")


# =============================================================================
#  Main evaluation loop
# =============================================================================

def run_evaluation(
        method_name: str,
        method_obj,
        test_paths: List[Path],
        label: Dict[str, int],
        bboxes: Dict[str, list],
        sample_path_idx: int,           # index of the image to save as sample
        samples_dir: Path,
        per_image_log: List[dict],
) -> Tuple[np.ndarray, np.ndarray, List[List[np.ndarray]]]:
    """
    Returns
    -------
    acc_matrix  : shape (5, 5) float   [row=num_patches, col=size_level]
    f1_matrix   : shape (5, 5) float
    cell_images : 5×5 list of BGR ndarrays (occluded sample, pre-prediction)
    """
    acc_mat   = np.zeros((5, 5), dtype=float)
    f1_mat    = np.zeros((5, 5), dtype=float)
    cell_imgs = [[None] * 5 for _ in range(5)]

    sample_path = test_paths[sample_path_idx]

    for ri, num_p in enumerate(NUM_PATCHES_LIST):
        for ci, sz_idx in enumerate(range(5)):
            preds, ys = [], []

            for j, p in enumerate(test_paths):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                h, w = img.shape[:2]
                bbox = pick_person_bbox(bboxes.get(p.name, []), w, h)
                img_occ = apply_multi_occlusion(
                    img, num_patches=num_p, size_level_idx=sz_idx,
                    person_bbox=bbox, seed=42)

                # ── Save pre-prediction occluded sample image ─────────────
                if p.name == sample_path.name:
                    stem     = p.stem
                    fname    = f"occ_np{num_p}_sz{sz_idx+1}_{stem}.jpg"
                    cv2.imwrite(str(samples_dir / fname), img_occ)
                    cell_imgs[ri][ci] = img_occ.copy()

                yhat, score, detail = method_obj.predict(img_occ)
                preds.append(int(yhat))
                ys.append(int(label.get(p.name, 0)))

                per_image_log.append({
                    'method':       method_name,
                    'num_patches':  num_p,
                    'size_level':   sz_idx + 1,
                    'patch_frac':   PATCH_SIZE_LEVELS[sz_idx],
                    'filename':     p.name,
                    'y_true':       int(label.get(p.name, 0)),
                    'y_pred':       int(yhat),
                    'score':        float(score),
                })

            stats = confusion(np.asarray(ys), np.asarray(preds))
            acc_mat[ri, ci] = stats['Acc']
            f1_mat[ri, ci]  = stats['F1']
            print(f"  [{method_name}] patches={num_p}  size_level={sz_idx+1} "
                  f"({int(PATCH_SIZE_LEVELS[sz_idx]*100):2d}%)  "
                  f"Acc={stats['Acc']:.3f}  F1={stats['F1']:.3f}  "
                  f"TP={stats['TP']} FP={stats['FP']} "
                  f"FN={stats['FN']} TN={stats['TN']}")

    return acc_mat, f1_mat, cell_imgs


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description='Multi-patch occlusion robustness test (5×5 grid)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--test_folder',    type=str, required=True)
    ap.add_argument('--climb_class_id', type=int, default=1)
    ap.add_argument('--our_model_path', type=str, default='climb_classifier_v2.pkl')
    ap.add_argument('--pose_model',     type=str, default='yolo11m-pose.pt')
    ap.add_argument('--our_method',     type=str, default='ensemble',
                    choices=['rule', 'ml', 'ensemble'])
    ap.add_argument('--our_threshold',  type=float, default=0.220)
    ap.add_argument('--yolo500_path',   type=str,
                    default=os.environ.get(
                        'YOLO500_CHECKPOINT',
                        'results_yolo_experiments/exp_500/best_weights.pt'))
    ap.add_argument('--yolo500_conf',   type=float, default=0.25)
    ap.add_argument('--yolo500_class',  type=int,   default=1)
    ap.add_argument('--out_dir',        type=str,   default='results/multi_occlusion')
    ap.add_argument('--skip_full',      action='store_true')
    ap.add_argument('--skip_yolo500',   action='store_true')
    ap.add_argument('--sample_index',   type=int,   default=-1,
                    help='Index of image to use as the sample for the 5×5 grid. '
                         '-1 = first climbing image found automatically.')
    return ap.parse_args()


def main():
    args = parse_args()
    global CLIMB_CLASS_ID_IN_GT
    CLIMB_CLASS_ID_IN_GT = int(args.climb_class_id)

    test_folder = Path(args.test_folder)
    out_dir     = Path(args.out_dir)
    samples_dir = out_dir / 'occluded_samples'
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    test_paths = collect_test_images(test_folder)
    label      = load_labels(test_paths)
    bboxes     = load_bboxes(test_paths)

    n_pos = sum(v == 1 for v in label.values())
    n_neg = len(test_paths) - n_pos
    print("=" * 72)
    print(f"[multi_occlusion_test]  test set : {len(test_paths)} images "
          f"({n_pos} climb / {n_neg} no-climb)")
    print(f"[multi_occlusion_test]  out dir  : {out_dir}")
    print(f"[multi_occlusion_test]  conditions: "
          f"{len(NUM_PATCHES_LIST)} patch counts × "
          f"{len(PATCH_SIZE_LEVELS)} sizes = "
          f"{len(NUM_PATCHES_LIST)*len(PATCH_SIZE_LEVELS)} cells")
    print("=" * 72)

    # ── Pick sample image for the grid ───────────────────────────────────────
    if args.sample_index >= 0:
        sample_idx = min(args.sample_index, len(test_paths) - 1)
    else:
        # Use first climbing image
        sample_idx = next(
            (i for i, p in enumerate(test_paths) if label.get(p.name, 0) == 1),
            0)
    print(f"[multi_occlusion_test]  grid sample : "
          f"{test_paths[sample_idx].name} (index {sample_idx})")

    per_image_log:  list = []
    results: dict[str, dict] = {}

    row_labels = [f"N={n}" for n in NUM_PATCHES_LIST]
    col_labels = [f"S={int(f*100)}%" for f in PATCH_SIZE_LEVELS]

    # ── FULL method ───────────────────────────────────────────────────────────
    acc_full = np.zeros((5, 5))
    f1_full  = np.zeros((5, 5))
    cell_full: List[List] = [[None]*5 for _ in range(5)]

    if not args.skip_full:
        print("\n" + "─"*72)
        print("Method 1/2 : FULL (scene + pose + interaction + ensemble)")
        print("─"*72)
        full = FullDetectorWrapper(
            pose_model_path=args.pose_model,
            classifier_pkl=args.our_model_path,
            method=args.our_method,
            threshold=args.our_threshold)
        acc_full, f1_full, cell_full = run_evaluation(
            'full', full, test_paths, label, bboxes,
            sample_idx, samples_dir, per_image_log)
        results['full'] = {'acc': acc_full.tolist(), 'f1': f1_full.tolist()}

    # ── YOLO-500 ──────────────────────────────────────────────────────────────
    acc_yolo = np.zeros((5, 5))
    f1_yolo  = np.zeros((5, 5))
    cell_yolo: List[List] = [[None]*5 for _ in range(5)]

    if not args.skip_yolo500:
        print("\n" + "─"*72)
        print("Method 2/2 : YOLO-500")
        print("─"*72)
        yolo500 = Yolo500Wrapper(
            ckpt_path=args.yolo500_path,
            conf_thr=args.yolo500_conf,
            climb_class_id=args.yolo500_class)
        acc_yolo, f1_yolo, cell_yolo = run_evaluation(
            'yolo500', yolo500, test_paths, label, bboxes,
            sample_idx, samples_dir, per_image_log)
        results['yolo500'] = {'acc': acc_yolo.tolist(), 'f1': f1_yolo.tolist()}

    # ── Save accuracy matrices as CSV ─────────────────────────────────────────
    for method_tag, acc_mat in [('full', acc_full), ('yolo500', acc_yolo)]:
        df = pd.DataFrame(
            acc_mat,
            index=[f"np={n}" for n in NUM_PATCHES_LIST],
            columns=[f"sz={int(f*100)}%" for f in PATCH_SIZE_LEVELS])
        path = out_dir / f'accuracy_matrix_{method_tag}.csv'
        df.to_csv(path, float_format='%.4f')
        print(f"[{method_tag}] accuracy matrix → {path}")

    # ── Heat-maps ─────────────────────────────────────────────────────────────
    if not args.skip_full or not args.skip_yolo500:
        plot_heatmaps(acc_full, acc_yolo, out_dir / 'accuracy_heatmap.png')

    # ── 5 × 5 grid of occluded sample images ─────────────────────────────────
    # Use FULL's cell images if available, else YOLO's
    grid_cells = cell_full if not args.skip_full else cell_yolo

    # Fill any None cells with a gray placeholder
    for r in range(5):
        for c in range(5):
            if grid_cells[r][c] is None:
                grid_cells[r][c] = np.full((200, 160, 3), 200, dtype=np.uint8)

    make_5x5_grid(
        grid_cells,
        out_dir / 'grid_5x5.png',
        gap=10,
        cell_w=160, cell_h=200,
        row_labels=row_labels,
        col_labels=col_labels)

    # ── Per-image predictions JSON ────────────────────────────────────────────
    pi_path = out_dir / 'per_image_predictions.json'
    with open(pi_path, 'w', encoding='utf-8') as fp:
        json.dump(per_image_log, fp, indent=2, ensure_ascii=False)
    print(f"[multi_occlusion_test] per-image predictions → {pi_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY  — image-level accuracy  (rows=num_patches, cols=size_level)")
    print("=" * 72)
    if not args.skip_full:
        print("\n  FULL method")
        print(pd.DataFrame(acc_full,
                           index=row_labels,
                           columns=col_labels).round(3).to_string())
    if not args.skip_yolo500:
        print("\n  YOLO-500")
        print(pd.DataFrame(acc_yolo,
                           index=row_labels,
                           columns=col_labels).round(3).to_string())
    print("=" * 72)


if __name__ == '__main__':
    main()
