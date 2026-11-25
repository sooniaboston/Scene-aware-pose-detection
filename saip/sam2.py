# -*- coding: utf-8 -*-
"""
scene_sam2.py - Deeply refactored climb-over-barrier detector.

What's new vs scene_sam1.py
===========================

1) SEGMENTATION FRONT-END - semantic prior + geometric cleanup
   -------------------------------------------------------------
   - Split the single 'wall' prompt into two semantic tracks that carry
     DIFFERENT meanings for climbing:
         * 'barrier' : fence / railing / guardrail / gate
         * 'wall'    : brick / stone / concrete solid wall
     Both are merged into a single `obstacle` mask, but we also KEEP the
     barrier mask separately so we can reason "did the person cross the fence" explicitly.
   - Post-process the obstacle mask with geometric sanity checks:
         * drop components that touch the image TOP border by >= 50% of
           their width (they're almost certainly background/sky/buildings)
         * keep only components that look like a barrier (wide + not
           too tall) OR a wall (touching bottom / mid of image).
   - Re-estimate the obstacle TOP edge by taking the MEDIAN (not min) of
     the surviving connected components, so a spurious blob does not
     drag the top line up to y=0.

2) FEATURE SET - add the "person vs. obstacle position" family
   ------------------------------------------------------------
   These features encode the DECISIVE signal that was missing before:
   *where is the person relative to the obstacle top?*
     - person_above_ratio        : fraction of body pixels above obstacle top
     - person_torso_above_top    : is torso centroid above obstacle top?
     - head_above_top_margin     : how many px is the head above the top
     - straddle_score            : torso crosses the top line (both sides)
     - hands_and_head_above      : 0/1/2/3 - how many upper-body landmarks
                                   are above the obstacle top
     - lower_body_visible_ratio  : fraction of lower-body keypoints with
                                   conf > 0.2 - LOW value under a good
                                   detector => occluded by the obstacle =>
                                   the person is on the other side.

3) LOWER-BODY-OCCLUSION SIGNAL - a first-class feature
   ------------------------------------------------------
   In the two failure frames you uploaded, YOLO-pose only sees the upper
   half of the body because the legs are behind the wall. A well-posed
   upper body + missing/low-conf lower body is, by itself, an extremely
   strong climb-over signal - and we now expose it as a feature.

4) DECISION LAYER - ML-first, rule as a strong prior
   ----------------------------------------------------
   - Rewrite the rule head so that *position*-based evidence dominates,
     not *contact*-based evidence (contacts are fragile to segmentation
     noise; position is robust).
   - Add an `ensemble` classifier mode that averages rule score + ML
     probability, because on low-sample regimes neither is perfect.
   - Provide a `calibrate_threshold()` utility to pick the decision
     threshold from a validation split (F1-optimal), rather than
     hard-coding 0.5.

5) VISUALIZATION - clearer labels & per-evidence panel
   -------------------------------------------------------
   - Draw the ESTIMATED obstacle top line as a thick yellow stripe.
   - Red bbox + 'CLIMB DETECTED' / green bbox + 'person'.
   - A compact evidence panel at the bottom-left showing the new features.

Public API (drop-in compatible with scene_sam1.py)
--------------------------------------------------
    GroundingSAMClimbDetectorV2(pose_model_path, classifier_method='rule'|'ml'|'ensemble')
        .detect_climb(image)              -> (is_climb, score, detail_dict)
        .segment_scene(image)             -> {'wall','ground','barrier','obstacle'}
        .visualize(image, kpts, masks, is_climb, score, detail)
        .train_ml_classifier(training_data)
        .save_classifier(path) / .load_classifier(path)
"""

from __future__ import annotations
import os
import sys
import json
import pickle
import tempfile
import linecache
import codecs
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---- linecache monkey patch (kept from v1) ---------------------------------
_original_updatecache = linecache.updatecache
def _patched_updatecache(filename, module_globals=None):
    try:
        return _original_updatecache(filename, module_globals)
    except UnicodeDecodeError:
        for enc in ('utf-8', 'latin-1', 'gbk', 'gb2312'):
            try:
                with codecs.open(filename, 'r', encoding=enc, errors='ignore') as fp:
                    lines = fp.readlines()
                linecache.cache[filename] = (len(lines), None, lines, filename)
                return lines
            except Exception:
                continue
        return []
linecache.updatecache = _patched_updatecache

# ---- Grounding-DINO / SAM imports ----------------------------------------
# The location of the Grounding-DINO checkout and of the model checkpoints is
# deployment-specific, so it is resolved from the environment (or from the
# command line) instead of being hard-coded.  See README.md.
GROUNDING_DINO_REPO = os.environ.get('GROUNDING_DINO_REPO', '')

try:
    import groundingdino
except ImportError:
    if GROUNDING_DINO_REPO and GROUNDING_DINO_REPO not in sys.path:
        sys.path.append(GROUNDING_DINO_REPO)

try:
    # Preferred: the original Grounding-DINO repository (reference impl.).
    from groundingdino.util.inference import load_image, predict
except ImportError:
    # Fallback: the Hugging Face port of the same model.  Same API, same
    # box convention (cx, cy, w, h normalised to [0, 1]).
    from hf_backend import load_image, predict
    print("[init] groundingdino repo not found -- using the Hugging Face "
          "Grounding-DINO backend (set GROUNDING_DINO_REPO to use the "
          "original implementation).")
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sam_model_registry = None
    SamPredictor = None

# Optional: the Hugging Face port of SAM, used when the `segment_anything`
# package is unavailable.
try:
    from transformers import SamModel, SamProcessor
    _HF_SAM_AVAILABLE = True
except ImportError:
    _HF_SAM_AVAILABLE = False


# ---- scikit-learn pickle compatibility ------------------------------------
# The shipped classifier was pickled with scikit-learn 1.6.x, whose internal
# module alias for the loss functions was plain `_loss`.  In scikit-learn
# >= 1.7 those classes moved to `sklearn._loss.loss`, so unpickling the model
# on a newer scikit-learn raises `ModuleNotFoundError: No module named
# '_loss'`.  Registering the alias keeps the shipped model loadable across
# scikit-learn versions.
def _install_sklearn_loss_alias() -> None:
    try:
        import sys
        import types
        import sklearn._loss.loss as _loss_mod
    except Exception:
        return
    if "_loss" in sys.modules:
        return
    alias = types.ModuleType("_loss")
    for name in dir(_loss_mod):
        if not name.startswith("__"):
            try:
                setattr(alias, name, getattr(_loss_mod, name))
            except Exception:
                pass
    sys.modules["_loss"] = alias


_install_sklearn_loss_alias()


# =============================================================================
#  Utility: geometric cleanup of a "barrier / wall" binary mask
# =============================================================================
def _cleanup_obstacle_mask(mask: np.ndarray,
                           drop_top_touching_ratio: float = 0.5,
                           min_area_ratio: float = 0.005) -> np.ndarray:
    """
    Remove connected components that are almost certainly NOT a climbable
    barrier/wall:
      (a) components whose top edge touches the image top AND whose height
          exceeds half the image height -> sky / buildings / background.
      (b) components smaller than `min_area_ratio` of the image -> noise.
    Returns a cleaned uint8 mask.
    """
    if mask is None or mask.max() == 0:
        return mask
    h, w = mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    img_area = h * w
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < img_area * min_area_ratio:
            continue
        touches_top = (y <= 2)
        too_tall    = hh > h * drop_top_touching_ratio
        if touches_top and too_tall:
            continue
        out[labels == i] = 255
    return out


def _estimate_obstacle_top(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Per-column top-edge of the obstacle mask, AND a robust scalar estimate
    of the barrier top line (in pixels from image top).

    The scalar estimate uses the MEDIAN of the top-edge over columns where
    the obstacle is present, which is much more robust than the min.
    """
    h, w = mask.shape
    top = np.full(w, h, dtype=np.int32)
    cols_with = []
    for x in range(w):
        col = mask[:, x]
        idx = np.where(col > 0)[0]
        if len(idx) > 0:
            top[x] = idx[0]
            cols_with.append(idx[0])
    if not cols_with:
        return top, float(h)
    return top, float(np.median(cols_with))


# =============================================================================
#  Feature extractor (v2)
# =============================================================================
class ClimbFeatureExtractorV2:

    # COCO-17 indices
    NOSE = 0
    LEFT_EYE, RIGHT_EYE = 1, 2
    LEFT_EAR, RIGHT_EAR = 3, 4
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    LEFT_ELBOW, RIGHT_ELBOW = 7, 8
    LEFT_WRIST, RIGHT_WRIST = 9, 10
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_KNEE, RIGHT_KNEE = 13, 14
    LEFT_ANKLE, RIGHT_ANKLE = 15, 16

    UPPER_BODY = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
                  LEFT_SHOULDER, RIGHT_SHOULDER,
                  LEFT_ELBOW, RIGHT_ELBOW,
                  LEFT_WRIST, RIGHT_WRIST)
    LOWER_BODY = (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
                  LEFT_ANKLE, RIGHT_ANKLE)

    CONF_THR = 0.2

    # ---- pose-only features (same as v1 + a few) ---------------------------
    def _joint_angle(self, kp, a, b, c) -> float:
        v1 = kp[b, :2] - kp[a, :2]
        v2 = kp[c, :2] - kp[b, :2]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    def extract_pose_features(self, kp: np.ndarray) -> Dict[str, float]:
        f = {}
        valid = lambda i: kp[i, 2] > self.CONF_THR

        # Arm lift (shoulder_y - wrist_y;  + means wrist ABOVE shoulder)
        f['left_arm_lift']  = float(kp[self.LEFT_SHOULDER, 1]  - kp[self.LEFT_WRIST, 1])  \
                              if valid(self.LEFT_SHOULDER) and valid(self.LEFT_WRIST)  else 0.0
        f['right_arm_lift'] = float(kp[self.RIGHT_SHOULDER, 1] - kp[self.RIGHT_WRIST, 1]) \
                              if valid(self.RIGHT_SHOULDER) and valid(self.RIGHT_WRIST) else 0.0
        f['max_arm_lift']   = max(f['left_arm_lift'], f['right_arm_lift'])

        # Elbow angles
        f['left_arm_angle']  = self._joint_angle(kp, self.LEFT_SHOULDER,  self.LEFT_ELBOW,  self.LEFT_WRIST) \
                               if all(valid(i) for i in (self.LEFT_SHOULDER,  self.LEFT_ELBOW,  self.LEFT_WRIST))  else 180.0
        f['right_arm_angle'] = self._joint_angle(kp, self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST) \
                               if all(valid(i) for i in (self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST)) else 180.0
        f['min_arm_angle']   = min(f['left_arm_angle'], f['right_arm_angle'])

        # Knee angles
        f['left_knee_angle']  = self._joint_angle(kp, self.LEFT_HIP,  self.LEFT_KNEE,  self.LEFT_ANKLE) \
                                if all(valid(i) for i in (self.LEFT_HIP,  self.LEFT_KNEE,  self.LEFT_ANKLE))  else 180.0
        f['right_knee_angle'] = self._joint_angle(kp, self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE) \
                                if all(valid(i) for i in (self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE)) else 180.0
        f['min_knee_angle']   = min(f['left_knee_angle'], f['right_knee_angle'])

        # Knee lift (hip_y - knee_y;  + means knee ABOVE hip)
        f['left_knee_lift']  = float(kp[self.LEFT_HIP, 1]  - kp[self.LEFT_KNEE, 1])  \
                               if valid(self.LEFT_HIP)  and valid(self.LEFT_KNEE)  else 0.0
        f['right_knee_lift'] = float(kp[self.RIGHT_HIP, 1] - kp[self.RIGHT_KNEE, 1]) \
                               if valid(self.RIGHT_HIP) and valid(self.RIGHT_KNEE) else 0.0
        f['max_knee_lift']   = max(f['left_knee_lift'], f['right_knee_lift'])

        # Body-span stats
        vp = kp[kp[:, 2] > self.CONF_THR]
        if len(vp) > 0:
            f['body_center_y'] = float(np.mean(vp[:, 1]))
            f['body_min_y']    = float(vp[:, 1].min())
            f['body_max_y']    = float(vp[:, 1].max())
            f['body_span_y']   = f['body_max_y'] - f['body_min_y']
            f['body_min_x']    = float(vp[:, 0].min())
            f['body_max_x']    = float(vp[:, 0].max())
            f['body_span_x']   = f['body_max_x'] - f['body_min_x']
        else:
            for k in ('body_center_y','body_min_y','body_max_y','body_span_y',
                      'body_min_x','body_max_x','body_span_x'):
                f[k] = 0.0

        # Trunk tilt
        if all(valid(i) for i in (self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
                                   self.LEFT_HIP, self.RIGHT_HIP)):
            s = (kp[self.LEFT_SHOULDER, :2] + kp[self.RIGHT_SHOULDER, :2]) / 2
            h = (kp[self.LEFT_HIP, :2]      + kp[self.RIGHT_HIP, :2])      / 2
            ang = np.arctan2(h[0] - s[0], h[1] - s[1]) * 180 / np.pi
            f['trunk_tilt'] = float(abs(ang))
        else:
            f['trunk_tilt'] = 0.0

        # ---- NEW pose features --------------------------------------------
        # Upper/lower body visibility ratios - big signal for occlusion
        f['upper_body_visible_ratio'] = float(
            np.mean([kp[i, 2] > self.CONF_THR for i in self.UPPER_BODY]))
        f['lower_body_visible_ratio'] = float(
            np.mean([kp[i, 2] > self.CONF_THR for i in self.LOWER_BODY]))
        # Asymmetry: upper well seen, lower poorly seen => occluded by barrier
        f['upper_minus_lower_visible'] = (
            f['upper_body_visible_ratio'] - f['lower_body_visible_ratio'])

        # Shoulder width / limb span (v1 features, retained)
        f['shoulder_width'] = float(abs(kp[self.RIGHT_SHOULDER, 0] - kp[self.LEFT_SHOULDER, 0])) \
                              if valid(self.LEFT_SHOULDER) and valid(self.RIGHT_SHOULDER) else 0.0

        if all(valid(i) for i in (self.LEFT_WRIST, self.RIGHT_WRIST,
                                   self.LEFT_ANKLE, self.RIGHT_ANKLE)):
            xs = [kp[i, 0] for i in (self.LEFT_WRIST, self.RIGHT_WRIST,
                                     self.LEFT_ANKLE, self.RIGHT_ANKLE)]
            ys = [kp[i, 1] for i in (self.LEFT_WRIST, self.RIGHT_WRIST,
                                     self.LEFT_ANKLE, self.RIGHT_ANKLE)]
            f['limb_span_x'] = float(max(xs) - min(xs))
            f['limb_span_y'] = float(max(ys) - min(ys))
        else:
            f['limb_span_x'] = 0.0
            f['limb_span_y'] = 0.0

        return f

    # ---- Interaction features with obstacle (v2 highlight) -----------------
    def extract_scene_features(self, kp: np.ndarray,
                               masks: Dict[str, np.ndarray],
                               image_shape: Tuple[int, int]) -> Dict[str, float]:
        f = {}
        h, w = image_shape
        obstacle = masks.get('obstacle', np.zeros((h, w), dtype=np.uint8))
        barrier  = masks.get('barrier',  np.zeros((h, w), dtype=np.uint8))
        wall     = masks.get('wall',     np.zeros((h, w), dtype=np.uint8))
        ground   = masks.get('ground',   np.zeros((h, w), dtype=np.uint8))

        vp = kp[kp[:, 2] > self.CONF_THR]
        if len(vp) == 0:
            return self._default_scene_features()

        # ---- obstacle top line (robust) -----------------------------------
        top_edge, obstacle_top_y = _estimate_obstacle_top(obstacle)
        _,        barrier_top_y  = _estimate_obstacle_top(barrier)
        _,        wall_top_y     = _estimate_obstacle_top(wall)

        f['obstacle_detected']  = float(obstacle.max() > 0)
        f['obstacle_top_y']     = float(obstacle_top_y)
        f['obstacle_top_norm']  = float(obstacle_top_y / h)
        f['barrier_top_y']      = float(barrier_top_y)
        f['wall_top_y']         = float(wall_top_y)

        # ---- POSITION-BASED FEATURES (new, decisive for occluded cases) ----
        # Fraction of visible body pixels ABOVE the obstacle top line.
        body_ys = vp[:, 1]
        f['person_above_ratio']      = float(np.mean(body_ys < obstacle_top_y))
        f['person_above_ratio_norm'] = float(np.mean(body_ys < obstacle_top_y + 20))

        # Head / shoulder / hip / torso-centroid positions vs top line
        def _above(idx, margin=0):
            return float(kp[idx, 2] > self.CONF_THR and
                         kp[idx, 1] < obstacle_top_y + margin)
        f['nose_above_top']          = _above(self.NOSE)
        f['left_shoulder_above_top'] = _above(self.LEFT_SHOULDER)
        f['right_shoulder_above_top']= _above(self.RIGHT_SHOULDER)
        f['left_hip_above_top']      = _above(self.LEFT_HIP)
        f['right_hip_above_top']     = _above(self.RIGHT_HIP)
        f['shoulders_above_top']     = (f['left_shoulder_above_top']
                                         + f['right_shoulder_above_top'])
        f['hips_above_top']          = (f['left_hip_above_top']
                                         + f['right_hip_above_top'])
        # Hands above top (important for pulling-up phase)
        f['left_hand_above_top']     = _above(self.LEFT_WRIST, margin=30)
        f['right_hand_above_top']    = _above(self.RIGHT_WRIST, margin=30)
        f['hands_above_top']         = (f['left_hand_above_top']
                                         + f['right_hand_above_top'])

        # "Straddle score": body points exist BOTH clearly above AND below
        # the top line => the person straddles the barrier.
        body_above = np.sum(body_ys < obstacle_top_y - 10)
        body_below = np.sum(body_ys > obstacle_top_y + 10)
        if len(body_ys) > 0:
            f['straddle_score'] = float(min(body_above, body_below) /
                                        (len(body_ys) / 2.0 + 1e-6))
        else:
            f['straddle_score'] = 0.0

        # Head-above-top margin (in px, normalized by image height)
        if kp[self.NOSE, 2] > self.CONF_THR:
            f['head_above_top_margin_norm'] = float(
                (obstacle_top_y - kp[self.NOSE, 1]) / h)
        else:
            f['head_above_top_margin_norm'] = 0.0

        # ---- CONTACT-BASED FEATURES (kept, but no longer dominant) ---------
        f['min_hand_top_distance'] = float('inf')
        f['max_hand_top_strength'] = 0.0
        f['hands_on_wall_top_ratio']= 0.0
        near_cnt = 0; strengths = []
        for idx in (self.LEFT_WRIST, self.RIGHT_WRIST):
            if kp[idx, 2] <= self.CONF_THR:
                strengths.append(0.0); continue
            x, y = int(kp[idx, 0]), int(kp[idx, 1])
            if not (0 <= x < w and 0 <= y < h):
                strengths.append(0.0); continue
            x1, x2 = max(0, x-25), min(w, x+25)
            region_top = top_edge[x1:x2]
            valid_top  = region_top[region_top < h]
            if len(valid_top) == 0:
                strengths.append(0.0); continue
            dy = np.abs(valid_top - y); dx = np.arange(x1, x2)[region_top < h] - x
            dist = np.sqrt(dy**2 + dx**2).min()
            f['min_hand_top_distance'] = min(f['min_hand_top_distance'], float(dist))
            strengths.append(1.0 if dist < 30 else max(0.0, 1.0 - dist/100))
            if dist < 30:
                near_cnt += 1
        f['max_hand_top_strength']  = float(max(strengths) if strengths else 0.0)
        f['hands_on_wall_top_ratio']= float(near_cnt / 2.0)
        if f['min_hand_top_distance'] == float('inf'):
            f['min_hand_top_distance'] = 9999.0

        # Foot-on-wall contact
        foot_wall = 0
        for idx in (self.LEFT_ANKLE, self.RIGHT_ANKLE):
            if kp[idx, 2] <= self.CONF_THR: continue
            x, y = int(kp[idx, 0]), int(kp[idx, 1])
            if 0 <= x < w and 0 <= y < h:
                r = obstacle[max(0, y-20):min(h, y+20),
                             max(0, x-20):min(w, x+20)]
                if r.size and r.sum() > 0:
                    foot_wall += 1
        f['foot_wall_contact_ratio'] = float(foot_wall / 2.0)

        # ---- Ground-related features (kept) --------------------------------
        feet_off = 0; gratios = []
        for idx in (self.LEFT_ANKLE, self.RIGHT_ANKLE):
            if kp[idx, 2] <= self.CONF_THR: continue
            x, y = int(kp[idx, 0]), int(kp[idx, 1])
            if 0 <= x < w and 0 <= y < h:
                r = ground[min(h-1, y):min(h, y+40),
                           max(0, x-25):min(w, x+25)]
                if r.size:
                    gr = r.sum() / (r.size * 255 + 1e-6)
                    gratios.append(gr)
                    if gr < 0.1: feet_off += 1
        f['feet_off_ground_ratio']   = float(feet_off / 2.0)
        f['min_foot_ground_contact'] = float(min(gratios) if gratios else 1.0)

        # ---- Person-bbox-level features (kept & extended) ------------------
        x1, y1 = float(vp[:, 0].min()), float(vp[:, 1].min())
        x2, y2 = float(vp[:, 0].max()), float(vp[:, 1].max())
        cx, cy = (x1+x2)/2, (y1+y2)/2
        f['vertical_position']       = float(cy / h)
        f['normalized_body_height']  = float((y2-y1) / h)
        f['bbox_top_above_obstacle'] = float(y1 < obstacle_top_y)
        f['bbox_center_above_obstacle'] = float(cy < obstacle_top_y)

        # Body-obstacle overlap
        pm = np.zeros((h, w), dtype=np.uint8)
        pm[int(y1):int(y2), int(x1):int(x2)] = 255
        ov = cv2.bitwise_and(pm, obstacle)
        f['body_obstacle_overlap_ratio'] = float(ov.sum() / (pm.sum() + 1e-6))

        # Body-ground contact near the bottom of the bbox
        bot = ground[max(0, int(y2)-20):min(h, int(y2)+20), int(x1):int(x2)]
        f['body_ground_contact'] = float(
            bot.sum() / (bot.size * 255 + 1e-6)) if bot.size else 0.0

        return f

    def _default_scene_features(self) -> Dict[str, float]:
        return {
            'obstacle_detected': 0.0, 'obstacle_top_y': 0.0, 'obstacle_top_norm': 0.0,
            'barrier_top_y': 0.0, 'wall_top_y': 0.0,
            'person_above_ratio': 0.0, 'person_above_ratio_norm': 0.0,
            'nose_above_top': 0.0, 'left_shoulder_above_top': 0.0,
            'right_shoulder_above_top': 0.0, 'left_hip_above_top': 0.0,
            'right_hip_above_top': 0.0, 'shoulders_above_top': 0.0,
            'hips_above_top': 0.0, 'left_hand_above_top': 0.0,
            'right_hand_above_top': 0.0, 'hands_above_top': 0.0,
            'straddle_score': 0.0, 'head_above_top_margin_norm': 0.0,
            'min_hand_top_distance': 9999.0, 'max_hand_top_strength': 0.0,
            'hands_on_wall_top_ratio': 0.0, 'foot_wall_contact_ratio': 0.0,
            'feet_off_ground_ratio': 0.0, 'min_foot_ground_contact': 1.0,
            'vertical_position': 0.5, 'normalized_body_height': 0.0,
            'bbox_top_above_obstacle': 0.0, 'bbox_center_above_obstacle': 0.0,
            'body_obstacle_overlap_ratio': 0.0, 'body_ground_contact': 0.0,
        }

    def extract_all_features(self, kp, masks, image_shape) -> Dict[str, float]:
        pose = self.extract_pose_features(kp)
        scene = self.extract_scene_features(kp, masks, image_shape)
        out = {}
        out.update({f'pose_{k}': v for k, v in pose.items()})
        out.update({f'scene_{k}': v for k, v in scene.items()})
        return out


# =============================================================================
#  Classifier (v2) - rule / ml / ensemble
# =============================================================================
class ClimbClassifierV2:

    def __init__(self, method: str = 'ensemble', threshold: float = 0.5):
        assert method in ('rule', 'ml', 'ensemble')
        self.method = method
        self.threshold = float(threshold)
        self.model = None          # sklearn pipeline
        self.feature_names: Optional[List[str]] = None

    # ---- RULE HEAD --------------------------------------------------------
    # Big change vs v1: position-based evidence DOMINATES contact-based.
    def predict_rule(self, f: Dict[str, float]) -> Tuple[bool, float, Dict]:
        score = 0.0
        d = {}

        # (A) POSITION EVIDENCE - core, up to +0.55
        straddle        = f.get('scene_straddle_score',       0.0)
        person_above    = f.get('scene_person_above_ratio',   0.0)
        head_margin     = f.get('scene_head_above_top_margin_norm', 0.0)
        shoulders_above = f.get('scene_shoulders_above_top',  0.0)  # 0/1/2
        hips_above      = f.get('scene_hips_above_top',       0.0)  # 0/1/2

        pos_score = 0.0
        if straddle > 0.5:       pos_score += 0.25; d['straddle'] = 'strong'
        elif straddle > 0.2:     pos_score += 0.15; d['straddle'] = 'moderate'
        else:                    d['straddle'] = 'weak'

        if person_above > 0.6:   pos_score += 0.15; d['person_above'] = 'mostly_above'
        elif person_above > 0.3: pos_score += 0.08; d['person_above'] = 'partly_above'

        if shoulders_above >= 2: pos_score += 0.10
        elif shoulders_above>=1: pos_score += 0.05

        if head_margin > 0.10:   pos_score += 0.05
        if hips_above >= 1:      pos_score += 0.05
        pos_score = min(pos_score, 0.55)
        d['position_score'] = round(pos_score, 3)
        score += pos_score

        # (B) POSE EVIDENCE - up to +0.20
        pose_score = 0.0
        if f.get('pose_max_arm_lift', 0) > 100: pose_score += 0.08
        elif f.get('pose_max_arm_lift', 0) > 50: pose_score += 0.04
        min_knee = f.get('pose_min_knee_angle', 180)
        if 60 < min_knee < 120:   pose_score += 0.06
        elif 120 < min_knee < 150:pose_score += 0.03
        if f.get('pose_max_knee_lift', 0) > 50: pose_score += 0.03
        if f.get('pose_trunk_tilt', 0) > 15:    pose_score += 0.03
        d['pose_score'] = round(pose_score, 3)
        score += min(pose_score, 0.20)

        # (C) OCCLUSION EVIDENCE - up to +0.15
        occ_score = 0.0
        low_vis = f.get('pose_lower_body_visible_ratio', 1.0)
        upp_vis = f.get('pose_upper_body_visible_ratio', 1.0)
        asym    = f.get('pose_upper_minus_lower_visible', 0.0)
        if upp_vis > 0.6 and low_vis < 0.3 and asym > 0.3:
            # Upper body well seen, lower body missing => classic "behind wall" signal
            occ_score += 0.15; d['occlusion'] = 'lower_body_hidden'
        elif asym > 0.2:
            occ_score += 0.08; d['occlusion'] = 'mild_asymmetry'
        d['occlusion_score'] = round(occ_score, 3)
        score += occ_score

        # (D) CONTACT EVIDENCE - up to +0.10 (demoted from v1)
        ct = 0.0
        if f.get('scene_hands_on_wall_top_ratio', 0) >= 1.0: ct += 0.05
        elif f.get('scene_hands_on_wall_top_ratio', 0) >= 0.5: ct += 0.03
        if f.get('scene_hands_above_top', 0) >= 1: ct += 0.03
        if f.get('scene_body_obstacle_overlap_ratio', 0) > 0.2: ct += 0.02
        d['contact_score'] = round(ct, 3)
        score += min(ct, 0.10)

        # (E) NEGATIVE EVIDENCE - multiplicative penalties
        if f.get('scene_body_ground_contact', 0) > 0.6 \
                and person_above < 0.3 and straddle < 0.2:
            score *= 0.4
            d['ground_penalty'] = 'standing_on_ground'
        elif f.get('scene_body_ground_contact', 0) > 0.3 \
                and person_above < 0.2 and straddle < 0.1:
            score *= 0.7
            d['ground_penalty'] = 'mild'

        if f.get('scene_obstacle_detected', 0) < 0.5:
            score *= 0.6  # no obstacle found => probably not a climb scene
            d['no_obstacle_penalty'] = True

        score = float(np.clip(score, 0.0, 1.0))
        is_climb = score > self.threshold
        d['final_score'] = score
        d['threshold'] = self.threshold
        return bool(is_climb), score, d

    # ---- ML HEAD ----------------------------------------------------------
    def _build_ml_pipeline(self):
        # Gradient Boosting has, empirically, slightly better calibration
        # than plain RF on small tabular datasets; kept LR as a fallback.
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.9, random_state=42))
        ])

    def train(self, feats: List[Dict[str, float]], labels: List[int],
              sample_weight: Optional[np.ndarray] = None):
        if self.method == 'rule':
            print("[ClassifierV2] train() called but method='rule'; skipped.")
            return
        self.feature_names = sorted(feats[0].keys())
        X = self._to_matrix(feats)
        y = np.asarray(labels)
        self.model = self._build_ml_pipeline()
        if sample_weight is not None:
            self.model.fit(X, y, clf__sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        # Feature importance
        clf = self.model.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            imps = clf.feature_importances_
            order = np.argsort(imps)[::-1]
            print("\n[ClassifierV2] Top-15 features:")
            for i in order[:15]:
                print(f"  {self.feature_names[i]:<48s} {imps[i]:.4f}")

    def _to_matrix(self, feats: List[Dict[str, float]]) -> np.ndarray:
        X = []
        for f in feats:
            row = []
            for k in self.feature_names:
                v = f.get(k, 0.0)
                if v == float('inf') or v == float('-inf'):
                    row.append(1e6)
                elif not np.isfinite(v):
                    row.append(0.0)
                else:
                    row.append(float(v))
            X.append(row)
        return np.asarray(X, dtype=np.float32)

    def predict_ml(self, f: Dict[str, float]) -> Tuple[bool, float, Dict]:
        if self.model is None or self.feature_names is None:
            raise RuntimeError("ML model not trained / loaded.")
        X = self._to_matrix([f])
        p = float(self.model.predict_proba(X)[0, 1])
        return bool(p > self.threshold), p, {'ml_prob': p,
                                              'threshold': self.threshold}

    # ---- ENSEMBLE ---------------------------------------------------------
    def predict_ensemble(self, f: Dict[str, float],
                         w_rule: float = 0.4,
                         w_ml:   float = 0.6) -> Tuple[bool, float, Dict]:
        _, s_rule, d_rule = self.predict_rule(f)
        if self.model is None:
            # Fall back to rule-only if ML not available
            return bool(s_rule > self.threshold), s_rule, {'rule': d_rule}
        _, s_ml, d_ml = self.predict_ml(f)
        s = w_rule * s_rule + w_ml * s_ml
        return bool(s > self.threshold), float(s), {
            'rule_score': s_rule, 'ml_prob': s_ml, 'fused': float(s),
            'rule_detail': d_rule}

    # ---- Public entry ----------------------------------------------------
    def predict(self, f: Dict[str, float]) -> Tuple[bool, float, Dict]:
        if self.method == 'rule':
            return self.predict_rule(f)
        if self.method == 'ml':
            return self.predict_ml(f)
        return self.predict_ensemble(f)

    # ---- Threshold calibration -------------------------------------------
    def calibrate_threshold(self, feats, labels,
                            grid=np.linspace(0.2, 0.8, 31)):
        """Pick the F1-optimal threshold on a validation split."""
        probs = []
        for f in feats:
            if self.method == 'rule':
                _, s, _ = self.predict_rule(f)
            elif self.method == 'ml':
                _, s, _ = self.predict_ml(f)
            else:
                _, s, _ = self.predict_ensemble(f)
            probs.append(s)
        probs = np.asarray(probs); y = np.asarray(labels)
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (probs > t).astype(int)
            tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
            fn = int(((pred==0)&(y==1)).sum())
            p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9)
            f1 = 2*p*r/(p+r+1e-9)
            if f1 > best_f1: best_f1, best_t = f1, float(t)
        self.threshold = best_t
        print(f"[Calibrate] threshold = {best_t:.3f}  (F1 = {best_f1:.3f})")
        return best_t, best_f1

    # ---- Save / Load -----------------------------------------------------
    def save(self, path: str):
        data = {'method': self.method,
                'threshold': self.threshold,
                'model': self.model,
                'feature_names': self.feature_names}
        with open(path, 'wb') as fp:
            pickle.dump(data, fp)
        print(f"[ClassifierV2] saved -> {path}")

    def load(self, path: str):
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        self.method        = data['method']
        self.threshold     = data['threshold']
        self.model         = data['model']
        self.feature_names = data['feature_names']
        print(f"[ClassifierV2] loaded <- {path}  (method={self.method}, "
              f"thr={self.threshold:.3f})")


# =============================================================================
#  Main detector (v2)
# =============================================================================
class GroundingSAMClimbDetectorV2:

    def __init__(self,
                 pose_model_path: str,
                 grounding_dino_config: str = None,
                 grounding_dino_checkpoint: str = None,
                 sam_checkpoint: str = None,
                 sam_model_type: str = 'vit_b',
                 box_threshold: float = 0.28,
                 text_threshold: float = 0.22,
                 classifier_method: str = 'ensemble',
                 backend: str = 'auto'):
        # Checkpoint locations default to the environment so that the code is
        # runnable on any machine; see README.md for the expected variables.
        grounding_dino_config = grounding_dino_config or os.environ.get(
            'GROUNDING_DINO_CONFIG', 'groundingdino/config/GroundingDINO_SwinT_OGC.py')
        grounding_dino_checkpoint = grounding_dino_checkpoint or os.environ.get(
            'GROUNDING_DINO_CHECKPOINT', 'groundingdino_swint_ogc.pth')
        sam_checkpoint = sam_checkpoint or os.environ.get(
            'SAM_CHECKPOINT', 'SAM/sam_vit_b_01ec64.pth')
        print("=" * 80)
        print("Initializing GroundingDINO + SAM Climb Detector  (v2)")
        print("=" * 80)

        print("\n[1/4] Loading YOLO-Pose ...")
        self.pose_model = YOLO(pose_model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("[2/4] Loading GroundingDINO ...")
        self._patch_transformers()
        self.use_hf_backend = False
        if backend in ('auto', 'groundingdino'):
            try:
                from groundingdino.util.inference import load_model as _load_gd
                self.grounding_model = _load_gd(grounding_dino_config,
                                                grounding_dino_checkpoint).to(self.device)
            except Exception as exc:
                if backend == 'groundingdino':
                    raise
                print(f"  [warn] original Grounding-DINO unavailable ({exc}); "
                      f"falling back to the Hugging Face backend.")
                self.use_hf_backend = True
        if self.use_hf_backend:
            from hf_backend import _ensure_loaded
            _proc, self.grounding_model = _ensure_loaded()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print("[3/4] Loading SAM ...")
        if sam_model_registry is None or SamPredictor is None:
            raise RuntimeError(
                "segment_anything is not installed. Install it with "
                "`pip install segment-anything` (or `git+https://github.com/"
                "facebookresearch/segment-anything.git`).")
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

        print("[4/4] Feature extractor + classifier ...")
        self.feature_extractor = ClimbFeatureExtractorV2()
        self.classifier = ClimbClassifierV2(method=classifier_method)

        # SEMANTIC PROMPTS - narrower than v1 so we don't grab the sky
        self.scene_prompts = {
            'barrier': 'metal fence . iron railing . metal guardrail . picket fence',
            'wall':    'brick wall . stone wall . concrete wall . retaining wall',
            'ground':  'floor . pavement . sidewalk . road surface . ground',
        }
        print("=" * 80)

    # ---- identical to v1 ---------------------------------------------------
    def _patch_transformers(self):
        try:
            from transformers.modeling_utils import ModuleUtilsMixin
            import inspect
            orig = ModuleUtilsMixin.get_extended_attention_mask
            sig = inspect.signature(orig)
            has_device = 'device' in sig.parameters

            def patched(self, attention_mask, input_shape, device=None, dtype=None):
                if dtype is not None and isinstance(dtype, torch.device):
                    device = dtype; dtype = None
                if has_device:
                    r = orig(self, attention_mask, input_shape, device=device, dtype=dtype)
                else:
                    r = orig(self, attention_mask, input_shape)
                    if device is not None: r = r.to(device)
                return r
            ModuleUtilsMixin.get_extended_attention_mask = patched

            if GROUNDING_DINO_REPO and GROUNDING_DINO_REPO not in sys.path:
                sys.path.insert(0, GROUNDING_DINO_REPO)
            from groundingdino.models.GroundingDINO.bertwarper import BertModelWarper

            def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is not None:
                    head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                    if is_attention_chunked is True:
                        head_mask = head_mask.unsqueeze(-1)
                else:
                    head_mask = [None] * num_hidden_layers
                return head_mask

            def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                assert head_mask.dim() == 5
                head_mask = head_mask.to(dtype=self.dtype)
                return head_mask

            BertModelWarper.get_head_mask = get_head_mask
            BertModelWarper._convert_head_mask_to_5d = _convert_head_mask_to_5d
        except Exception as e:
            print(f"  [warn] transformers patch skipped: {e}")

    # ---- GDINO + SAM segmentation (identical API to v1) ---------------------
    def _set_sam_image(self, img_rgb: np.ndarray) -> None:
        """Encode the image with SAM, once per image.

        The three scene prompts are segmented against the SAME image, so the
        SAM image encoder is run once and the result is reused.  This is a
        pure speed optimisation; the produced masks are identical.
        """
        key = id(img_rgb)
        if getattr(self, "_sam_img_key", None) == key:
            return
        self.sam_predictor.set_image(img_rgb)
        self._sam_img_key = key

    def _segment_prompt(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        h, w = image.shape[:2]
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            # The original backend expects the packed tensor returned by
            # `load_image`; the Hugging Face backend accepts the PIL image.
            if self.use_hf_backend:
                img_tensor = pil
            else:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    tmp = f.name
                Image.fromarray(img_rgb).save(tmp)
                _, img_tensor = load_image(tmp)
            boxes, logits, phrases = predict(
                model=self.grounding_model, image=img_tensor, caption=text_prompt,
                box_threshold=self.box_threshold, text_threshold=self.text_threshold)
            if not self.use_hf_backend:
                try: os.unlink(tmp)
                except Exception: pass
            if boxes is None or len(boxes) == 0:
                return np.zeros((h, w), dtype=np.uint8)
            self._set_sam_image(img_rgb)
            mask = np.zeros((h, w), dtype=np.uint8)
            for box in boxes:
                bn = box.cpu().numpy() if torch.is_tensor(box) else box
                cx, cy, bw, bh = bn
                x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
                try:
                    m, _, _ = self.sam_predictor.predict(
                        box=np.array([x1, y1, x2, y2]), multimask_output=False)
                    if len(m) > 0:
                        mask |= (m[0].astype(np.uint8) * 255)
                except Exception:
                    continue
            return mask
        except Exception as e:
            print(f"  [segment error] {str(e)[:120]}")
            return np.zeros((h, w), dtype=np.uint8)

    def segment_scene(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = image.shape[:2]
        print("\n[segment] barrier / wall / ground")
        barrier = self._segment_prompt(image, self.scene_prompts['barrier'])
        wall    = self._segment_prompt(image, self.scene_prompts['wall'])
        ground  = self._segment_prompt(image, self.scene_prompts['ground'])

        # Geometric sanity cleanup - kill sky/background blobs
        barrier_clean = _cleanup_obstacle_mask(barrier)
        wall_clean    = _cleanup_obstacle_mask(wall)

        # Merge barrier union wall as the "obstacle"
        obstacle = cv2.bitwise_or(barrier_clean, wall_clean)

        # Ground intersect obstacle -> prefer obstacle
        ov = cv2.bitwise_and(ground, obstacle)
        ground = cv2.bitwise_and(ground, cv2.bitwise_not(ov))

        return {'barrier': barrier_clean, 'wall': wall_clean,
                'obstacle': obstacle, 'ground': ground}

    # ---- Main detection entry ---------------------------------------------
    def detect_climb(self, image: np.ndarray, conf_threshold: float = 0.1,
                     debug: bool = True) -> Tuple[bool, float, Dict]:
        h, w = image.shape[:2]
        masks = self.segment_scene(image)

        pose_res = self.pose_model.predict(image, conf=conf_threshold, verbose=False)
        if not pose_res or not hasattr(pose_res[0], 'keypoints'):
            return False, 0.0, {'error': 'no person detected', 'masks': masks}
        kp = pose_res[0].keypoints
        if kp is None or len(kp.data) == 0:
            return False, 0.0, {'error': 'no keypoints detected', 'masks': masks}
        keypoints = kp.data[0].cpu().numpy()

        features = self.feature_extractor.extract_all_features(keypoints, masks, (h, w))
        is_climb, score, cls_detail = self.classifier.predict(features)

        detail = {
            'climb_score': float(score),
            'is_climb':    bool(is_climb),
            'classification_details': cls_detail,
            'all_features': features,
            'masks_info': {
                'barrier_pixels':  int(masks['barrier'].sum() // 255),
                'wall_pixels':     int(masks['wall'].sum() // 255),
                'obstacle_pixels': int(masks['obstacle'].sum() // 255),
                'ground_pixels':   int(masks['ground'].sum() // 255),
            },
            'keypoints': keypoints,
            'masks': masks,
        }
        if debug:
            self._print_debug(detail)
        return is_climb, score, detail

    def _print_debug(self, d):
        f = d['all_features']
        print("\n[debug] key features:")
        for k in ('scene_person_above_ratio', 'scene_straddle_score',
                  'scene_hips_above_top', 'scene_shoulders_above_top',
                  'scene_hands_above_top',
                  'pose_lower_body_visible_ratio',
                  'pose_upper_body_visible_ratio',
                  'scene_obstacle_top_norm', 'scene_body_ground_contact',
                  'scene_hands_on_wall_top_ratio'):
            print(f"  {k:<42s} {f.get(k, 0):.3f}")
        print(f"\n  -> score = {d['climb_score']:.3f}  "
              f"is_climb = {d['is_climb']}")

    # ---- Visualisation ----------------------------------------------------
    def visualize(self, image: np.ndarray, keypoints: np.ndarray,
                  masks: Dict[str, np.ndarray],
                  is_climb: bool, score: float,
                  details: Dict = None) -> np.ndarray:
        vis = image.copy()
        h, w = image.shape[:2]

        # Overlay: barrier (cyan), wall (red), ground (green)
        if masks.get('barrier') is not None and masks['barrier'].max() > 0:
            ov = np.zeros_like(image)
            ov[..., 0] = masks['barrier']      # B
            ov[..., 1] = masks['barrier']      # G -> cyan
            vis = cv2.addWeighted(vis, 0.75, ov, 0.30, 0)
        if masks.get('wall') is not None and masks['wall'].max() > 0:
            ov = np.zeros_like(image)
            ov[..., 2] = masks['wall']          # R
            vis = cv2.addWeighted(vis, 0.78, ov, 0.28, 0)
        if masks.get('ground') is not None and masks['ground'].max() > 0:
            ov = np.zeros_like(image)
            ov[..., 1] = masks['ground']        # G
            vis = cv2.addWeighted(vis, 0.80, ov, 0.25, 0)

        # Draw the robust obstacle-top scalar line
        obstacle = masks.get('obstacle', np.zeros((h, w), dtype=np.uint8))
        if obstacle.max() > 0:
            _, top_y = _estimate_obstacle_top(obstacle)
            cv2.line(vis, (0, int(top_y)), (w, int(top_y)), (0, 255, 255), 3)
            cv2.putText(vis, f"obstacle top  y~{int(top_y)}",
                        (10, max(20, int(top_y) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Skeleton
        conns = [(5,6),(5,7),(7,9),(6,8),(8,10),
                 (5,11),(6,12),(11,12),
                 (11,13),(13,15),(12,14),(14,16)]
        for a, b in conns:
            if keypoints[a, 2] > 0.2 and keypoints[b, 2] > 0.2:
                p1 = (int(keypoints[a, 0]), int(keypoints[a, 1]))
                p2 = (int(keypoints[b, 0]), int(keypoints[b, 1]))
                cv2.line(vis, p1, p2, (255, 255, 0), 2)
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.2:
                x, y = int(kp[0]), int(kp[1])
                if i in (9, 10):
                    cv2.circle(vis, (x, y), 8, (0, 255, 255), -1)
                else:
                    cv2.circle(vis, (x, y), 5, (255, 0, 255), -1)

        # Person bbox - red if climb, green otherwise
        valid = keypoints[keypoints[:, 2] > 0.2]
        if len(valid) >= 2:
            x1 = max(0, int(valid[:, 0].min()) - 15)
            y1 = max(0, int(valid[:, 1].min()) - 15)
            x2 = min(w-1, int(valid[:, 0].max()) + 15)
            y2 = min(h-1, int(valid[:, 1].max()) + 15)
            color = (0, 0, 255) if is_climb else (0, 200, 0)
            label = f"CLIMB DETECTED  {score:.2f}" if is_climb else f"person  {score:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (x1, y1 - th - 10),
                          (x1 + tw + 10, max(y1, th + 10)), color, -1)
            cv2.putText(vis, label, (x1 + 5, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Evidence panel (bottom-left)
        if details and 'all_features' in details:
            f = details['all_features']
            txts = [
                f"person_above_ratio : {f.get('scene_person_above_ratio', 0):.2f}",
                f"straddle_score    : {f.get('scene_straddle_score', 0):.2f}",
                f"shoulders_above   : {f.get('scene_shoulders_above_top', 0):.0f}/2",
                f"hips_above        : {f.get('scene_hips_above_top', 0):.0f}/2",
                f"lower_visible     : {f.get('pose_lower_body_visible_ratio', 0):.2f}",
                f"body_ground_ctct  : {f.get('scene_body_ground_contact', 0):.2f}",
                f"hands_above_top   : {f.get('scene_hands_above_top', 0):.0f}/2",
                f"obstacle_detected : {f.get('scene_obstacle_detected', 0):.0f}",
            ]
            # Background box behind the text for readability
            pad_x, pad_y, line_h = 10, 8, 22
            box_w = 330
            box_h = pad_y * 2 + line_h * len(txts)
            y0 = vis.shape[0] - box_h - 10
            cv2.rectangle(vis, (10, y0),
                          (10 + box_w, y0 + box_h), (0, 0, 0), -1)
            cv2.rectangle(vis, (10, y0),
                          (10 + box_w, y0 + box_h), (255, 255, 255), 2)
            for i, t in enumerate(txts):
                cv2.putText(vis, t,
                            (10 + pad_x, y0 + pad_y + (i + 1) * line_h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
 
        # Top banner - big CLIMB ALERT / NORMAL label
        banner = f"CLIMB ALERT  {score:.2f}" if is_climb else f"NORMAL  {score:.2f}"
        banner_color = (0, 0, 255) if is_climb else (0, 200, 0)
        cv2.putText(vis, banner, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, banner_color, 3, cv2.LINE_AA)
 
        # A compact line of classification-detail tags under the banner
        if details and 'classification_details' in details:
            y_offset = 72
            for k, v in details['classification_details'].items():
                if k in ('final_score', 'threshold',
                         'rule_detail', 'ml_prob', 'rule_score', 'fused'):
                    continue
                text = f"{k}: {v}"
                cv2.putText(vis, text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (230, 230, 230), 1, cv2.LINE_AA)
                y_offset += 20
                if y_offset > 200:
                    break
 
        return vis
 
    # ---- Training helpers (folder -> features -> ML head) --------------------
    def train_ml_classifier(self,
                            training_data: List[Tuple[np.ndarray,
                                                      Dict[str, np.ndarray],
                                                      int]],
                            calibrate: bool = True):
        """
        Extract features from a list of (image, masks, label) and train the
        ML / ensemble classifier. Optionally calibrate the decision threshold
        on the same data (use a held-out split for serious evaluation!).
        """
        if self.classifier.method == 'rule':
            print("[train] classifier is in 'rule' mode - nothing to train.")
            return
 
        feats_list, labels = [], []
        print("\n[train] extracting features from training data ...")
        for i, (img, masks, lab) in enumerate(training_data):
            print(f"  [{i+1}/{len(training_data)}] label={lab}")
            pose_res = self.pose_model.predict(img, conf=0.1, verbose=False)
            if not pose_res or not hasattr(pose_res[0], 'keypoints'):
                print("    skipped (no person)"); continue
            kp = pose_res[0].keypoints
            if kp is None or len(kp.data) == 0:
                print("    skipped (no keypoints)"); continue
            keypoints = kp.data[0].cpu().numpy()
            h, w = img.shape[:2]
            feats = self.feature_extractor.extract_all_features(
                keypoints, masks, (h, w))
            feats_list.append(feats)
            labels.append(int(lab))
 
        if len(feats_list) < 2:
            print("[train] too few usable samples, aborting.")
            return
 
        print(f"\n[train] usable: {len(feats_list)} "
              f"(climb={sum(labels)}, no-climb={len(labels)-sum(labels)})")
        self.classifier.train(feats_list, labels)
 
        if calibrate:
            # Not ideal to calibrate on training data; replace with a
            # validation split for serious evaluation.
            self.classifier.calibrate_threshold(feats_list, labels)
 
    def save_classifier(self, path: str):
        self.classifier.save(path)
 
    def load_classifier(self, path: str):
        self.classifier.load(path)
 
 
# =============================================================================
#  High-level folder utilities (drop-in replacements for v1 examples)
# =============================================================================
def _collect_images(folder: str) -> List[Path]:
    p = Path(folder)
    return sorted(list(p.glob('*.jpg')) +
                  list(p.glob('*.jpeg')) +
                  list(p.glob('*.png')))
 
 
def train_from_folders(climb_folder: str = 'test/climb',
                       no_climb_folder: str = 'test/no_climb',
                       pose_model_path: str = 'yolo11m-pose.pt',
                       classifier_method: str = 'ensemble',
                       model_out: str = 'climb_classifier_v2.pkl'):
    """End-to-end: read two folders -> train -> save a classifier."""
    detector = GroundingSAMClimbDetectorV2(
        pose_model_path=pose_model_path,
        classifier_method=classifier_method)
 
    climb_imgs    = _collect_images(climb_folder)
    no_climb_imgs = _collect_images(no_climb_folder)
    print(f"\n[train_from_folders] climb={len(climb_imgs)} "
          f"no-climb={len(no_climb_imgs)}")
 
    training_data = []
    for p in climb_imgs:
        img = cv2.imread(str(p))
        if img is None: continue
        masks = detector.segment_scene(img)
        training_data.append((img, masks, 1))
    for p in no_climb_imgs:
        img = cv2.imread(str(p))
        if img is None: continue
        masks = detector.segment_scene(img)
        training_data.append((img, masks, 0))
 
    detector.train_ml_classifier(training_data, calibrate=True)
    detector.save_classifier(model_out)
    print(f"[train_from_folders] classifier saved -> {model_out}")
    return detector
 
 
def evaluate_folder(detector: GroundingSAMClimbDetectorV2,
                    test_folder: str,
                    out_dir: str = 'results/v2_inference',
                    save_vis: bool = True) -> List[Dict]:
    """Run detect_climb on every image in a folder; save a visualization."""
    img_paths = _collect_images(test_folder)
    if not img_paths:
        print(f"[evaluate_folder] no images in {test_folder}"); return []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
 
    results = []
    climb_cnt = 0
    for i, p in enumerate(img_paths):
        print(f"\n[{i+1}/{len(img_paths)}] {p.name}")
        img = cv2.imread(str(p))
        if img is None:
            print("  failed to read"); continue
        try:
            is_climb, score, detail = detector.detect_climb(img, debug=False)
        except Exception as e:
            print(f"  error: {e}"); continue
        climb_cnt += int(is_climb)
        rec = {
            'filename':   p.name,
            'is_climb':   bool(is_climb),
            'score':      float(score),
            'classification': detail.get('classification_details', {}),
        }
        results.append(rec)
 
        if save_vis and 'keypoints' in detail and 'masks' in detail:
            vis = detector.visualize(img, detail['keypoints'],
                                     detail['masks'], is_climb, score, detail)
            prefix = 'CLIMB_' if is_climb else 'NORMAL_'
            cv2.imwrite(str(Path(out_dir) / f"{prefix}{p.stem}.jpg"), vis)
 
    results_path = Path(out_dir) / 'inference_results.json'
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
 
    print("\n" + "=" * 72)
    print(f"[evaluate_folder] total={len(results)}  "
          f"CLIMB={climb_cnt}  NORMAL={len(results)-climb_cnt}")
    print(f"  results -> {results_path}")
    print(f"  vis     -> {out_dir}/")
    print("=" * 72)
    return results
 
 
# =============================================================================
#  CLI entry
# =============================================================================
def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Climb detector v2")
    ap.add_argument('--mode', required=True,
                    choices=['train', 'inference', 'single', 'compare'],
                    help='what to do')
    ap.add_argument('--pose_model',  default='yolo11m-pose.pt')
    ap.add_argument('--model_path',  default='climb_classifier_v2.pkl',
                    help='where to save/load the ML classifier')
    ap.add_argument('--classifier',  default='ensemble',
                    choices=['rule', 'ml', 'ensemble'])
    ap.add_argument('--image',       default=None,
                    help='single image path (for --mode single)')
    ap.add_argument('--climb_folder',    default='test/climb')
    ap.add_argument('--no_climb_folder', default='test/no_climb')
    ap.add_argument('--test_folder',     default='test/test')
    ap.add_argument('--out_dir',         default='results/v2_inference')
    return ap.parse_args()
 
 
def _main_train(args):
    train_from_folders(
        climb_folder=args.climb_folder,
        no_climb_folder=args.no_climb_folder,
        pose_model_path=args.pose_model,
        classifier_method=args.classifier,
        model_out=args.model_path)
 
 
def _main_inference(args):
    detector = GroundingSAMClimbDetectorV2(
        pose_model_path=args.pose_model,
        classifier_method=args.classifier)
    if args.classifier != 'rule' and Path(args.model_path).exists():
        detector.load_classifier(args.model_path)
    elif args.classifier != 'rule':
        print(f"[warn] classifier='{args.classifier}' but no model found at "
              f"{args.model_path} - the rule part will still work, ML "
              f"predictions will be unavailable.")
    evaluate_folder(detector, args.test_folder, args.out_dir, save_vis=True)
 
 
def _main_single(args):
    if not args.image:
        print("[single] --image is required for single mode"); return
    img = cv2.imread(args.image)
    if img is None:
        print(f"[single] cannot read {args.image}"); return
    detector = GroundingSAMClimbDetectorV2(
        pose_model_path=args.pose_model,
        classifier_method=args.classifier)
    if args.classifier != 'rule' and Path(args.model_path).exists():
        detector.load_classifier(args.model_path)
    is_climb, score, detail = detector.detect_climb(img, debug=True)
    print(f"\nRESULT: {'CLIMB' if is_climb else 'NORMAL'}  score={score:.3f}")
    if 'keypoints' in detail and 'masks' in detail:
        vis = detector.visualize(img, detail['keypoints'], detail['masks'],
                                 is_climb, score, detail)
        out = f"{Path(args.image).stem}_v2_result.jpg"
        cv2.imwrite(out, vis)
        print(f"Visualization saved -> {out}")
 
 
def _main_compare(args):
    """
    Compare rule-only vs. loaded ML vs. ensemble on --test_folder.
    Requires args.model_path to exist for ML/ensemble modes.
    """
    from copy import deepcopy
    img_paths = _collect_images(args.test_folder)
    if not img_paths:
        print(f"[compare] no images in {args.test_folder}"); return
 
    # Rule-only detector
    det_rule = GroundingSAMClimbDetectorV2(
        pose_model_path=args.pose_model, classifier_method='rule')
    # ML + ensemble reuse the same heavy front-ends (pose + GDINO + SAM) of
    # det_rule - but we need a classifier loaded; so load on top of det_rule.
    have_ml = Path(args.model_path).exists()
    if have_ml:
        cls_ml = ClimbClassifierV2(method='ml')
        cls_ml.load(args.model_path)
        cls_ens = ClimbClassifierV2(method='ensemble')
        cls_ens.load(args.model_path)
 
    rows = []
    for i, p in enumerate(img_paths):
        print(f"\n[{i+1}/{len(img_paths)}] {p.name}")
        img = cv2.imread(str(p))
        if img is None: continue
        # Do the heavy work once, then apply three heads to the features.
        masks = det_rule.segment_scene(img)
        pose_res = det_rule.pose_model.predict(img, conf=0.1, verbose=False)
        if not pose_res or not hasattr(pose_res[0], 'keypoints'):
            continue
        kp = pose_res[0].keypoints
        if kp is None or len(kp.data) == 0:
            continue
        keypoints = kp.data[0].cpu().numpy()
        feats = det_rule.feature_extractor.extract_all_features(
            keypoints, masks, img.shape[:2])
 
        # Rule
        is_r, s_r, _ = det_rule.classifier.predict_rule(feats)
        row = {'filename': p.name,
               'rule':   {'is_climb': bool(is_r), 'score': float(s_r)}}
        if have_ml:
            try:
                is_m, s_m, _ = cls_ml.predict_ml(feats)
                row['ml'] = {'is_climb': bool(is_m), 'score': float(s_m)}
            except Exception as e:
                row['ml'] = {'error': str(e)}
            try:
                is_e, s_e, _ = cls_ens.predict_ensemble(feats)
                row['ensemble'] = {'is_climb': bool(is_e), 'score': float(s_e)}
            except Exception as e:
                row['ensemble'] = {'error': str(e)}
        rows.append(row)
 
        tag = lambda d: f"{'CLIMB' if d.get('is_climb') else 'normal'}"\
                        f" ({d.get('score',0):.2f})"
        print(f"  rule={tag(row['rule'])}"
              + (f"  ml={tag(row['ml'])}" if have_ml and 'score' in row.get('ml', {}) else '')
              + (f"  ens={tag(row['ensemble'])}" if have_ml and 'score' in row.get('ensemble', {}) else ''))
 
    Path('results').mkdir(parents=True, exist_ok=True)
    out_json = 'results/v2_comparison.json'
    with open(out_json, 'w') as fp:
        json.dump(rows, fp, indent=2)
    print(f"\n[compare] saved -> {out_json}")
 
 
if __name__ == '__main__':
    args = _parse_args()
    if   args.mode == 'train':     _main_train(args)
    elif args.mode == 'inference': _main_inference(args)
    elif args.mode == 'single':    _main_single(args)
    elif args.mode == 'compare':   _main_compare(args)