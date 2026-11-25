# -*- coding: utf-8 -*-
"""
pose_only_baseline.py  (v2)
===========================
Same pose-only classifier as before, but now supports the test directory
structure:

    test1/
    ├── image/   *.jpg / *.jpeg / *.png
    └── label/   *.txt  (one per image, same stem)

Supported label-file formats (auto-detected):
  • Single integer  : "0" or "1"
  • YOLO detection  : "class cx cy w h"  → class is the label (0/1)
  • YOLO multi-line : any line's first token taken as class (majority vote)

Usage examples
--------------
  # Full table (train from two folders, test from image+label dir)
  python pose_only_baseline.py --mode table \
      --pose_model      yolo11m-pose.pt \
      --climb_folder    test/climb \
      --no_climb_folder test/no_climb \
      --test_dir        test1 \
      --out_dir         results_pose_only

  # Evaluate rule prior only (no training, unlabelled folder)
  python pose_only_baseline.py --mode eval_rule \
      --pose_model  yolo11m-pose.pt \
      --test_folder test1/image \
      --out_dir     results_pose_only

  # Train + test with image/label dir
  python pose_only_baseline.py --mode train_eval \
      --pose_model      yolo11m-pose.pt \
      --climb_folder    test/climb \
      --no_climb_folder test/no_climb \
      --test_dir        test1 \
      --out_dir         results_pose_only
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
#  COCO-17 index constants
# ─────────────────────────────────────────────────────────────────────────────
NOSE         = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SH,  R_SH  = 5, 6
L_EL,  R_EL  = 7, 8
L_WR,  R_WR  = 9, 10
L_HI,  R_HI  = 11, 12
L_KN,  R_KN  = 13, 14
L_AN,  R_AN  = 15, 16

UPPER_IDX = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR,
             L_SH, R_SH, L_EL, R_EL, L_WR, R_WR)
LOWER_IDX = (L_HI, R_HI, L_KN, R_KN, L_AN, R_AN)

CONF_THR  = 0.20
SKELETON  = [(5,6),(5,7),(7,9),(6,8),(8,10),
             (5,11),(6,12),(11,12),
             (11,13),(13,15),(12,14),(14,16)]


# ─────────────────────────────────────────────────────────────────────────────
#  Label-file parser  (auto-detects format)
# ─────────────────────────────────────────────────────────────────────────────
def _read_label(label_path: Path) -> Optional[int]:
    """
    Read a .txt label file and return 0 or 1.
    Handles:
      single integer "0"/"1", YOLO detection "class cx cy w h",
      and multi-line YOLO (majority class across all lines).
    Returns None if file is missing or unreadable.
    """
    if not label_path.exists():
        return None
    try:
        text = label_path.read_text(encoding='utf-8').strip()
    except Exception:
        return None
    if not text:
        return None
    classes = []
    for line in text.splitlines():
        token = line.strip().split()
        if token:
            try:
                classes.append(int(token[0]))
            except ValueError:
                pass
    if not classes:
        return None
    label = int(round(sum(classes) / len(classes)))
    return int(bool(label))   # normalise to 0 / 1


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def _angle(kp: np.ndarray, a: int, b: int, c: int) -> float:
    v1 = kp[b, :2] - kp[a, :2]
    v2 = kp[c, :2] - kp[b, :2]
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/denom, -1, 1))))


def _ok(*idxs, kp: np.ndarray) -> bool:
    return all(kp[i, 2] > CONF_THR for i in idxs)


# ─────────────────────────────────────────────────────────────────────────────
#  Pose feature extractor  (35-dim, pose-ONLY)
# ─────────────────────────────────────────────────────────────────────────────
class PoseFeatureExtractor:
    def extract(self, kp: np.ndarray,
                img_h: int, img_w: int) -> Dict[str, float]:
        f: Dict[str, float] = {}

        # F1 – Limb lift (positive ⟺ raised above anchor joint)
        f['lift_arm_L']        = float(kp[L_SH,1]-kp[L_WR,1]) if _ok(L_SH,L_WR,kp=kp) else 0.0
        f['lift_arm_R']        = float(kp[R_SH,1]-kp[R_WR,1]) if _ok(R_SH,R_WR,kp=kp) else 0.0
        f['lift_arm_max']      = max(f['lift_arm_L'], f['lift_arm_R'])
        f['lift_arm_max_norm'] = f['lift_arm_max'] / (img_h + 1e-6)
        f['lift_knee_L']       = float(kp[L_HI,1]-kp[L_KN,1]) if _ok(L_HI,L_KN,kp=kp) else 0.0
        f['lift_knee_R']       = float(kp[R_HI,1]-kp[R_KN,1]) if _ok(R_HI,R_KN,kp=kp) else 0.0
        f['lift_knee_max']      = max(f['lift_knee_L'], f['lift_knee_R'])
        f['lift_knee_max_norm'] = f['lift_knee_max'] / (img_h + 1e-6)

        # F2 – Joint angles (neutral = 180° when joint invisible)
        f['angle_elbow_L'] = _angle(kp,L_SH,L_EL,L_WR) if _ok(L_SH,L_EL,L_WR,kp=kp) else 180.0
        f['angle_elbow_R'] = _angle(kp,R_SH,R_EL,R_WR) if _ok(R_SH,R_EL,R_WR,kp=kp) else 180.0
        f['angle_elbow_min'] = min(f['angle_elbow_L'], f['angle_elbow_R'])
        f['angle_knee_L']  = _angle(kp,L_HI,L_KN,L_AN) if _ok(L_HI,L_KN,L_AN,kp=kp) else 180.0
        f['angle_knee_R']  = _angle(kp,R_HI,R_KN,R_AN) if _ok(R_HI,R_KN,R_AN,kp=kp) else 180.0
        f['angle_knee_min'] = min(f['angle_knee_L'], f['angle_knee_R'])
        f['knee_bend_max']  = 180.0 - f['angle_knee_min']

        # F3 – Trunk geometry
        if _ok(L_SH, R_SH, L_HI, R_HI, kp=kp):
            sh = (kp[L_SH,:2] + kp[R_SH,:2]) / 2
            hi = (kp[L_HI,:2] + kp[R_HI,:2]) / 2
            f['trunk_tilt_deg'] = float(abs(np.degrees(np.arctan2(hi[0]-sh[0], hi[1]-sh[1]))))
            f['trunk_len_norm'] = float(np.linalg.norm(hi - sh)) / img_h
            f['forward_lean']   = float(sh[0] - hi[0]) / (img_w + 1e-6)
        else:
            f['trunk_tilt_deg'] = 0.0; f['trunk_len_norm'] = 0.0; f['forward_lean'] = 0.0

        # F4 – Body-span statistics
        vp = kp[kp[:,2] > CONF_THR]
        if len(vp) >= 2:
            f['body_h_norm']   = (vp[:,1].max()-vp[:,1].min()) / img_h
            f['body_w_norm']   = (vp[:,0].max()-vp[:,0].min()) / img_w
            f['body_cy_norm']  = float(np.mean(vp[:,1])) / img_h
            f['body_cx_norm']  = float(np.mean(vp[:,0])) / img_w
            f['body_aspect']   = (vp[:,1].max()-vp[:,1].min()) / (vp[:,0].max()-vp[:,0].min()+1e-6)
            f['body_top_norm'] = float(vp[:,1].min()) / img_h
        else:
            f['body_h_norm'] = 0.0; f['body_w_norm'] = 0.0
            f['body_cy_norm']= 0.5; f['body_cx_norm']= 0.5
            f['body_aspect'] = 1.0; f['body_top_norm']= 0.5

        # F5 – Visibility / occlusion asymmetry  Δv = v_upper − v_lower
        f['vis_upper']       = float(np.mean([kp[i,2]>CONF_THR for i in UPPER_IDX]))
        f['vis_lower']       = float(np.mean([kp[i,2]>CONF_THR for i in LOWER_IDX]))
        f['vis_asymmetry']   = f['vis_upper'] - f['vis_lower']
        f['n_upper_valid']   = float(sum(kp[i,2]>CONF_THR for i in UPPER_IDX))
        f['n_lower_valid']   = float(sum(kp[i,2]>CONF_THR for i in LOWER_IDX))
        f['conf_upper_mean'] = float(np.mean([kp[i,2] for i in UPPER_IDX]))
        f['conf_lower_mean'] = float(np.mean([kp[i,2] for i in LOWER_IDX]))

        # F6 – Head vertical position in frame
        f['head_y_norm'] = float(kp[NOSE,1])/img_h if kp[NOSE,2]>CONF_THR else f['body_cy_norm']

        # F7 – Wrist / ankle / diagonal span
        if _ok(L_WR, R_WR, kp=kp):
            f['wrist_span_norm']  = abs(kp[L_WR,0]-kp[R_WR,0]) / img_w
            f['wrist_y_min_norm'] = min(kp[L_WR,1],kp[R_WR,1]) / img_h
        else:
            f['wrist_span_norm'] = 0.0; f['wrist_y_min_norm'] = f['body_cy_norm']
        if _ok(L_AN, R_AN, kp=kp):
            f['ankle_span_norm'] = abs(kp[L_AN,0]-kp[R_AN,0]) / img_w
            f['ankle_y_norm']    = (kp[L_AN,1]+kp[R_AN,1])/2 / img_h
        else:
            f['ankle_span_norm'] = 0.0; f['ankle_y_norm'] = 1.0
        if _ok(L_WR, R_AN, kp=kp):
            f['diag_span_LW_RA'] = float(np.linalg.norm(kp[L_WR,:2]-kp[R_AN,:2])) / (img_h+1e-6)
        elif _ok(R_WR, L_AN, kp=kp):
            f['diag_span_LW_RA'] = float(np.linalg.norm(kp[R_WR,:2]-kp[L_AN,:2])) / (img_h+1e-6)
        else:
            f['diag_span_LW_RA'] = 0.0
        return f


# ─────────────────────────────────────────────────────────────────────────────
#  Rule-prior head (training-free)
# ─────────────────────────────────────────────────────────────────────────────
class PoseRuleClassifier:
    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    def score(self, f: Dict) -> Tuple[bool, float, Dict]:
        s = 0.0; d = {}
        aln = f.get('lift_arm_max_norm', 0.0)
        a = 0.30 if aln>0.15 else (0.20 if aln>0.08 else (0.10 if aln>0.03 else 0.0))
        d['arm_lift'] = round(a,3); s += a
        b = 0.0
        if f.get('knee_bend_max',0)>80:  b+=0.12
        elif f.get('knee_bend_max',0)>50: b+=0.07
        if f.get('lift_knee_max_norm',0)>0.08: b+=0.08
        elif f.get('lift_knee_max_norm',0)>0.04: b+=0.04
        b=min(b,0.20); d['knee_signal']=round(b,3); s+=b
        tilt=f.get('trunk_tilt_deg',0.0)
        c=0.10 if tilt>45 else (0.06 if tilt>20 else (0.03 if tilt>10 else 0.0))
        d['trunk_tilt']=round(c,3); s+=c
        asym=f.get('vis_asymmetry',0.0); vu=f.get('vis_upper',0.0); vl=f.get('vis_lower',1.0)
        e = (0.25 if vu>0.6 and vl<0.25 and asym>0.35 else
             0.15 if vu>0.5 and vl<0.50 and asym>0.20 else
             0.08 if asym>0.15 else 0.0)
        d['occlusion_asym']=round(e,3); s+=e
        wr_y=f.get('wrist_y_min_norm',1.0); cy=f.get('body_cy_norm',0.5)
        w=0.15 if wr_y<cy-0.10 else (0.08 if wr_y<cy-0.05 else 0.0)
        d['wrist_height']=round(w,3); s+=w
        if f.get('body_cy_norm',0.5)>0.70 and f.get('trunk_tilt_deg',0)<10 and asym<0.10:
            s*=0.5; d['standing_penalty']=True
        s=float(np.clip(s,0.0,1.0))
        return bool(s>self.threshold), s, d

    def calibrate(self, feats, labels, grid=np.linspace(0.10, 0.85, 50)):
        probs=np.array([self.score(f)[1] for f in feats]); y=np.asarray(labels)
        best_f1, best_t = -1.0, self.threshold
        for t in grid:
            pred=(probs>t).astype(int)
            tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
            fn=int(((pred==0)&(y==1)).sum())
            p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); f1=2*p*r/(p+r+1e-9)
            if f1>best_f1: best_f1,best_t=f1,float(t)
        self.threshold=best_t
        print(f"  [rule calibrate] τ* = {best_t:.3f}  (val F1 = {best_f1:.3f})")
        return best_t, best_f1


# ─────────────────────────────────────────────────────────────────────────────
#  ML heads
# ─────────────────────────────────────────────────────────────────────────────
def _make_pipelines():
    return {
        'GradBoost':  Pipeline([('sc',StandardScaler()),
                                ('cl',GradientBoostingClassifier(n_estimators=200,max_depth=3,
                                     learning_rate=0.05,subsample=0.9,random_state=42))]),
        'RandForest': Pipeline([('sc',StandardScaler()),
                                ('cl',RandomForestClassifier(n_estimators=300,
                                     min_samples_leaf=2,random_state=42))]),
        'LogReg':     Pipeline([('sc',StandardScaler()),
                                ('cl',LogisticRegression(C=1.0,max_iter=1000,random_state=42))]),
    }


def _to_matrix(feats, names):
    return np.array([[float(f.get(k,0.0)) for k in names] for f in feats],dtype=np.float32)


def _calibrate_thr(pipe, X, y, grid=np.linspace(0.10, 0.85, 40)):
    probs=pipe.predict_proba(X)[:,1]; best_f1,best_t=-1.0,0.50
    for t in grid:
        pred=(probs>t).astype(int)
        tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
        fn=int(((pred==0)&(y==1)).sum())
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); f1=2*p*r/(p+r+1e-9)
        if f1>best_f1: best_f1,best_t=f1,float(t)
    return best_t, best_f1


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics / table
# ─────────────────────────────────────────────────────────────────────────────
def _metrics(y_true, y_pred):
    tp=int(((y_pred==1)&(y_true==1)).sum()); fp=int(((y_pred==1)&(y_true==0)).sum())
    fn=int(((y_pred==0)&(y_true==1)).sum()); tn=int(((y_pred==0)&(y_true==0)).sum())
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); f1=2*p*r/(p+r+1e-9)
    return dict(P=p,R=r,F1=f1,Acc=(tp+tn)/(tp+fp+fn+tn+1e-9),TP=tp,FP=fp,FN=fn,TN=tn)

SEP    = "─" * 94
HEADER = (f"\n{'Method':<32s} {'Train':>7s}  "
          f"{'P':>6s} {'R':>6s} {'F1':>6s} {'Acc':>6s}  "
          f"{'TP':>4s} {'FP/FN':>7s} {'TN':>4s}")

def _row(method, train_n, m):
    print(f"{method:<32s} {train_n:>7s}  "
          f"{m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} {m['Acc']:>6.3f}  "
          f"{m['TP']:>4d} {m['FP']:>3d}/{m['FN']:<3d} {m['TN']:>4d}")


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def _vis(image, kp, is_climb, score, f):
    vis=image.copy(); h,w=image.shape[:2]
    for a,b in SKELETON:
        if kp[a,2]>CONF_THR and kp[b,2]>CONF_THR:
            cv2.line(vis,(int(kp[a,0]),int(kp[a,1])),(int(kp[b,0]),int(kp[b,1])),(255,200,0),2)
    for i in range(17):
        if kp[i,2]>CONF_THR:
            cv2.circle(vis,(int(kp[i,0]),int(kp[i,1])),5,
                       (0,255,255) if i in (L_WR,R_WR) else (200,0,255),-1)
    vp=kp[kp[:,2]>CONF_THR]
    if len(vp)>=2:
        x1=max(0,int(vp[:,0].min())-15); y1=max(0,int(vp[:,1].min())-15)
        x2=min(w-1,int(vp[:,0].max())+15); y2=min(h-1,int(vp[:,1].max())+15)
        col=(0,0,220) if is_climb else (0,180,0)
        lbl=f"CLIMB {score:.2f}" if is_climb else f"NORMAL {score:.2f}"
        cv2.rectangle(vis,(x1,y1),(x2,y2),col,3)
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
        cv2.rectangle(vis,(x1,y1-th-10),(x1+tw+10,max(y1,th+10)),col,-1)
        cv2.putText(vis,lbl,(x1+5,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    lines=[f"vis_asymmetry  : {f.get('vis_asymmetry',0):.2f}",
           f"vis_upper      : {f.get('vis_upper',0):.2f}",
           f"vis_lower      : {f.get('vis_lower',0):.2f}",
           f"lift_arm_max_n : {f.get('lift_arm_max_norm',0):.3f}",
           f"knee_bend_max  : {f.get('knee_bend_max',0):.1f}deg",
           f"trunk_tilt_deg : {f.get('trunk_tilt_deg',0):.1f}deg",
           f"wrist_y_min_n  : {f.get('wrist_y_min_norm',0):.3f}",
           f"body_cy_norm   : {f.get('body_cy_norm',0):.3f}"]
    pad,lh,bw=8,21,280; bh=pad*2+lh*len(lines); y0=h-bh-10
    cv2.rectangle(vis,(10,y0),(10+bw,y0+bh),(0,0,0),-1)
    cv2.rectangle(vis,(10,y0),(10+bw,y0+bh),(200,200,200),1)
    for i,t in enumerate(lines):
        cv2.putText(vis,t,(18,y0+pad+(i+1)*lh-4),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(vis,"[POSE-ONLY] "+(f"CLIMB ALERT {score:.2f}" if is_climb else f"NORMAL {score:.2f}"),
                (10,38),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,220) if is_climb else (0,180,0),3,cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────
IMG_EXTS = ('*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG')

def _collect(folder):
    p=Path(folder); out=[]
    for ext in IMG_EXTS: out.extend(p.glob(ext))
    return sorted(set(out))

def _load_pose_model(path):
    from ultralytics import YOLO
    return YOLO(path)

def _extract_kp(model, image, conf=0.10):
    res=model.predict(image,conf=conf,verbose=False)
    if not res: return None
    kp=res[0].keypoints
    if kp is None or len(kp.data)==0: return None
    return kp.data[0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────
class PoseOnlyBaseline:

    def __init__(self, pose_model_path: str):
        print("[PoseOnlyBaseline] loading YOLO-pose …")
        self.pose_model = _load_pose_model(pose_model_path)
        self.extractor  = PoseFeatureExtractor()
        self.rule       = PoseRuleClassifier(threshold=0.40)
        self.pipelines  = _make_pipelines()
        self.thresholds = {k: 0.50 for k in self.pipelines}
        self.feat_names: Optional[List[str]] = None
        self._trained   = False

    def _process(self, path: Path):
        img=cv2.imread(str(path))
        if img is None: return None
        kp=_extract_kp(self.pose_model,img)
        if kp is None: return None
        return kp, self.extractor.extract(kp,img.shape[0],img.shape[1]), img

    # ── NEW: load from test1/image + test1/label ──────────────────────────
    def load_from_dir(self, base_dir: str) -> List[Tuple]:
        """
        Load labelled data from base_dir/image/ and base_dir/label/.
        Label files are .txt with the same stem as the image.
        """
        base    = Path(base_dir)
        img_dir = base / 'image'
        lbl_dir = base / 'label'

        if not img_dir.exists():
            raise FileNotFoundError(
                f"\n  Image folder not found: {img_dir}\n"
                f"  Expected layout:\n"
                f"    {base}/\n"
                f"    ├── image/   <- put images here\n"
                f"    └── label/   <- put .txt labels here")

        paths = _collect(str(img_dir))
        print(f"\n  Found {len(paths)} images in '{img_dir}'")
        if not lbl_dir.exists():
            print(f"  [warn] label dir not found: {lbl_dir}")
            print(f"         Treating all images as unlabelled (label=-1).")

        data = []; skip_no_label=0; skip_no_person=0
        for p in paths:
            lbl = _read_label(lbl_dir / (p.stem+'.txt')) if lbl_dir.exists() else None
            if lbl is None:
                skip_no_label += 1; continue
            res = self._process(p)
            if res is None:
                skip_no_person += 1; continue
            kp, f, img = res
            data.append((kp, f, img, lbl, p))

        n_pos = sum(d[3] for d in data)
        print(f"  Loaded: {len(data)} usable  "
              f"(climb={n_pos}, no-climb={len(data)-n_pos})")
        if skip_no_label:  print(f"  Skipped {skip_no_label}: label file missing/unreadable")
        if skip_no_person: print(f"  Skipped {skip_no_person}: no person detected by YOLO-pose")
        return data

    # ── original: load from plain class folder ────────────────────────────
    def load_data(self, folder: str, label: int) -> List[Tuple]:
        paths = _collect(folder)
        print(f"  Loading {len(paths)} images from '{folder}' (label={label})")
        out = []
        for p in paths:
            res = self._process(p)
            if res is None: print(f"    skipped (no person): {p.name}"); continue
            kp, f, img = res
            out.append((kp, f, img, label, p))
        print(f"  → {len(out)} usable")
        return out

    # ── training ──────────────────────────────────────────────────────────
    def train(self, climb_data, no_climb_data, val_ratio: float = 0.20):
        """
        80% of training data is used to fit the model.
        The remaining 20% (val split) is used ONLY to calibrate the
        decision threshold τ*, ensuring test metrics are not inflated.
        """
        from sklearn.model_selection import train_test_split

        all_data = climb_data + no_climb_data
        feats  = [d[1] for d in all_data]
        labels = [d[3] for d in all_data]
        self.feat_names = sorted(feats[0].keys())
        X = _to_matrix(feats, self.feat_names)
        y = np.asarray(labels)

        # Stratified 80/20 split: fit on 80%, calibrate threshold on 20%
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=val_ratio, stratify=y, random_state=42)
        feats_val = [dict(zip(self.feat_names, row)) for row in X_val]

        print(f"\n  [split] fit={len(y_fit)}  val={len(y_val)}  "
              f"(stratified 80/20, seed=42)")

        # Rule prior: calibrate on val split
        self.rule.calibrate(feats_val, y_val.tolist())

        print(f"\n  [ML] fitting on {len(y_fit)} samples  "
              f"(climb={y_fit.sum()}, no-climb={(y_fit==0).sum()})")
        for name, pipe in self.pipelines.items():
            pipe.fit(X_fit, y_fit)
            # Threshold calibrated on val split (held-out from fitting)
            t, f1_val = _calibrate_thr(pipe, X_val, y_val)
            self.thresholds[name] = t
            clf = pipe.named_steps['cl']
            if hasattr(clf,'feature_importances_') and name=='GradBoost':
                imps=clf.feature_importances_; order=np.argsort(imps)[::-1]
                print(f"\n  [GradBoost] top-10 features:")
                for i in order[:10]: print(f"    {self.feat_names[i]:<36s} {imps[i]:.4f}")
            # fit-set F1 shown for reference only (not used for threshold)
            t_tr, f1_tr = _calibrate_thr(pipe, X_fit, y_fit)
            print(f"  [{name}] τ* = {t:.3f}  "
                  f"(val F1 = {f1_val:.3f}  |  fit F1 ref = {f1_tr:.3f})")
        self._trained = True

    def predict(self, f: Dict, method='GradBoost') -> Tuple[bool, float]:
        if method=='rule':
            is_c,s,_=self.rule.score(f); return is_c,s
        if not self._trained: raise RuntimeError("Not trained.")
        X=_to_matrix([f],self.feat_names)
        p=float(self.pipelines[method].predict_proba(X)[0,1])
        return bool(p>self.thresholds[method]),p

    def evaluate(self, data, method='GradBoost') -> Dict:
        yt,yp=[],[]
        for kp,f,img,lbl,_ in data:
            is_c,_=self.predict(f,method=method)
            yt.append(lbl); yp.append(int(is_c))
        return _metrics(np.array(yt),np.array(yp))

    def visualize_dataset(self, data, out_dir, method='GradBoost'):
        Path(out_dir).mkdir(parents=True,exist_ok=True)
        for kp,f,img,lbl,p in data:
            is_c,s=self.predict(f,method=method)
            gt_tag='gt1' if lbl else 'gt0'
            vis=_vis(img,kp,is_c,s,f)
            cv2.imwrite(str(Path(out_dir)/f"{'CLIMB' if is_c else 'NORMAL'}_{gt_tag}_{p.stem}.jpg"),vis)
        print(f"  [vis] {len(data)} images → {out_dir}/")

    def save(self, path):
        with open(path,'wb') as fp:
            pickle.dump({'feat_names':self.feat_names,'rule_thr':self.rule.threshold,
                         'pipelines':self.pipelines,'thresholds':self.thresholds},fp)
        print(f"  [save] → {path}")

    def load(self, path):
        with open(path,'rb') as fp: d=pickle.load(fp)
        self.feat_names=d['feat_names']; self.rule.threshold=d['rule_thr']
        self.pipelines=d['pipelines']; self.thresholds=d['thresholds']
        self._trained=True; print(f"  [load] ← {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry points
# ─────────────────────────────────────────────────────────────────────────────
def run_table(args):
    bl = PoseOnlyBaseline(args.pose_model)
    print("\n── Training data ─────────────────────────────────────────────")
    c_tr  = bl.load_data(args.climb_folder,    label=1)
    nc_tr = bl.load_data(args.no_climb_folder, label=0)
    n_train = len(c_tr) + len(nc_tr)
    print("\n── Training ──────────────────────────────────────────────────")
    bl.train(c_tr, nc_tr)
    print("\n── Test data ─────────────────────────────────────────────────")
    test_data = bl.load_from_dir(args.test_dir)
    if not test_data:
        print("\n  ✗  No test data loaded — check test_dir/image/ and test_dir/label/"); return
    n_pos=sum(d[3] for d in test_data); n_neg=len(test_data)-n_pos
    print(f"\n  Test: {len(test_data)} total  (climb={n_pos}, no-climb={n_neg})")
    print(f"\n{SEP}{HEADER}\n{SEP}")
    rows={}
    for method,ts in [('rule','---'),('GradBoost',f'~{n_train}'),
                       ('RandForest',f'~{n_train}'),('LogReg',f'~{n_train}')]:
        m=bl.evaluate(test_data,method=method)
        rows[f"Pose-{method}"]=m; _row(f"Pose-{method}",ts,m)
    print(SEP)
    best=max([k for k in rows if 'rule' not in k],key=lambda k:rows[k]['F1'])
    best_method = best.replace('Pose-', '')   # "Pose-LogReg" -> "LogReg"
    print(f"\n  Best ML head: {best}  (F1={rows[best]['F1']:.3f})\n")
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    with open(out/'metrics.json','w') as fp:
        json.dump({k:{kk:float(vv) for kk,vv in v.items()} for k,v in rows.items()},fp,indent=2)
    bl.save(str(out/'pose_only_model.pkl'))
    print("── Visualisations ────────────────────────────────────────────")
    bl.visualize_dataset(test_data,str(out/'vis'),method=best_method)
    print(f"\n  metrics → {out/'metrics.json'}")


def run_train_eval(args):
    bl=PoseOnlyBaseline(args.pose_model)
    c_tr=bl.load_data(args.climb_folder,label=1)
    nc_tr=bl.load_data(args.no_climb_folder,label=0)
    n_train=len(c_tr)+len(nc_tr)
    bl.train(c_tr,nc_tr)
    test_data=bl.load_from_dir(args.test_dir)
    if not test_data: print("✗  No test data."); return
    n_pos=sum(d[3] for d in test_data)
    print(f"\n  Test: {len(test_data)} (climb={n_pos}, no-climb={len(test_data)-n_pos})\n")
    print(SEP); print(HEADER); print(SEP)
    for method,ts in [('rule','---'),('GradBoost',f'~{n_train}'),
                       ('RandForest',f'~{n_train}'),('LogReg',f'~{n_train}')]:
        m=bl.evaluate(test_data,method=method); _row(f"Pose-{method}",ts,m)
    print(SEP)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    bl.save(str(out/'pose_only_model.pkl'))


def run_eval_rule(args):
    bl=PoseOnlyBaseline(args.pose_model)
    paths=_collect(args.test_folder)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    print(f"\n  Rule-only on {len(paths)} images …")
    results=[]; cnt=0
    for p in paths:
        res=bl._process(p)
        if res is None: continue
        kp,f,img=res; is_c,s,d=bl.rule.score(f); cnt+=int(is_c)
        results.append({'file':p.name,'is_climb':bool(is_c),'score':float(s)})
        vis=_vis(img,kp,is_c,s,f)
        cv2.imwrite(str(out/f"{'CLIMB' if is_c else 'NORMAL'}_{p.stem}.jpg"),vis)
    with open(out/'results.json','w') as fp: json.dump(results,fp,indent=2)
    print(f"  CLIMB={cnt}  NORMAL={len(results)-cnt}  → {out}/")


def _parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=['eval_rule','train_eval','table'])
    ap.add_argument('--pose_model',      default='yolo11m-pose.pt')
    ap.add_argument('--climb_folder',    default='test/climb')
    ap.add_argument('--no_climb_folder', default='test/no_climb')
    ap.add_argument('--test_dir',        default='test1',
                    help='folder containing image/ and label/ subfolders')
    ap.add_argument('--test_folder',     default='test1/image',
                    help='for eval_rule: plain image folder (no labels needed)')
    ap.add_argument('--out_dir',         default='results_pose_only')
    return ap.parse_args()


if __name__ == '__main__':
    args=_parse()
    if   args.mode=='eval_rule':  run_eval_rule(args)
    elif args.mode=='train_eval': run_train_eval(args)
    elif args.mode=='table':      run_table(args)