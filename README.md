# Scene-Aware Interaction Perception (SAIP)

Reference implementation of

> **Scene-Aware Interaction Perception: A Few-Shot Framework for Human
> Behavior Recognition in Context**
> Shukuo Zhao, Department of Precision Instrument, Tsinghua University
> *International Journal of Computer Vision* (2026)

SAIP detects **safety-critical barrier-climbing behaviour** from a single
RGB frame.  It promotes the *human-scene geometric relation* from an implicit
side-channel to the primary representational unit: the decision is made on an
interpretable vector of interaction predicates rather than on a pose skeleton
or on an opaque feature activation.

The method reaches **F1 = 0.971** (precision = recall = 0.971, accuracy =
0.960) on the 50-image held-out test split using roughly **100 labelled
images**, matching a YOLO11m detector trained on 500 images, and degrades
more gracefully under multi-patch occlusion.

---

## 1. Method in one page

SAIP factorises into four stages.

| stage | module | output |
|---|---|---|
| Scene perception `Φ_scene` | Grounding-DINO + SAM | pixel-level obstacle / ground masks and the robust obstacle **top line** `y_top` |
| Pose perception `Φ_pose` | YOLO11m-pose | 17 COCO keypoints `K` + limb-group **visibility statistics** `v` |
| Interaction `Φ_int` | `saip/sam2.py` | the interpretable interaction vector `f` |
| Decision `Ψ` | rule prior + shallow learned head | calibrated score `s`, binary decision `y_hat`, five-stage semantic assessment |

The interaction vector `f` is grouped into four families, every coordinate of
which admits a one-line verbal gloss:

* **position** (`scene_person_above_ratio`, `scene_straddle_score`,
  `scene_shoulders_above_top`, `scene_hands_above_top`, ...) — where the body
  is relative to the obstacle top line;
* **contact** (`scene_min_hand_top_distance`, `scene_hands_on_wall_top_ratio`,
  `scene_feet_off_ground_ratio`, ...) — the body–surface interface;
* **bounding box / body mask** (`scene_body_obstacle_overlap_ratio`,
  `scene_bbox_top_above_obstacle`, ...);
* **scene-conditioned occlusion** (`pose_delta_v`,
  `pose_upper_minus_lower_visible`) — the visibility deficit, reinterpreted
  *only* when an obstacle is actually present.

The visibility asymmetry `Δv = v_upper − v_lower` is the decisive signal
that a skeleton-only classifier cannot disambiguate from estimator failure:
a large positive `Δv` means the legs are hidden **behind a foreground
obstacle**, which in the presence of a detected barrier is evidence of
climbing, and in an empty scene is estimator noise.

### Decision layer

Three interchangeable heads:

* **rule prior** — training-free, capped-additive sub-scores with
  multiplicative negative-evidence penalties;
* **learned classifier** — a shallow gradient-boosted head (200 trees, depth 3,
  learning rate 0.05) over standardised features; ~100 examples per class
  suffice because `f` already encodes the task semantics;
* **ensemble** — the default head blends the two.

The operating threshold `τ*` is selected by maximising F1 on a held-out
validation split; the shipped model uses `τ* = 0.22`.

A **five-stage semantic interpreter** resolves the features into an
operational alert level:

```
S0 Normal  →  S1 Preparation  →  S2 Climbing Up  →  S3 Straddling  →  S4 Breach
```

---

## 2. Repository layout

```
saip/
  sam2.py                    the method: feature extractor, classifier heads,
                             Grounding-DINO+SAM detector, visualiser
  hf_backend.py              drop-in Grounding-DINO backend on top of
                             `transformers` (used when the original
                             repository is not installed)
  task_unified.py            Task 1 (sequence visualisation) and
                             Task 2 (test-split evaluation), selectable
                             with --mode
  multi_occlusion_test.py    the 5x5 multi-patch occlusion grid
  noise_robustness_test.py   square-occlusion and motion-blur robustness
  pose_only_baseline.py      the pose-only baseline (YOLO11m-pose)
  task3_yolo_baseline.py     the closed-vocabulary YOLO11m baseline
  task3_500.py               the 500-image YOLO11m baseline
  train_yolo_baseline.py     YOLO11m training with 5-fold CV splits

scripts/
  make_splits.py             materialise the file lists + labels of every split
  export_features.py         extract the interaction feature vectors to CSV

splits/                      the exact file lists and labels of every subset
results/                     the numbers reported in the paper
configs/                     model / prompt configuration
```

---

## 3. Setup

### 3.1 Python environment

```bash
pip install -r requirements.txt
```

### 3.2 The scene parser

The reference implementation uses the original Grounding-DINO repository with
the SwinT-OGC checkpoint:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e GroundingDINO
```

**Alternative (no CUDA extension required).**  `saip/hf_backend.py` provides
an equivalent implementation on top of the Hugging Face port of the same
model.  Point the environment variable at it and the code will use it
automatically whenever the original repository is not importable:

```bash
export GROUNDING_DINO_HF_ID=IDEA-Research/Grounding-DINO-base
```

`Grounding-DINO-base` is the Swin-T backbone variant, i.e. the same
architecture family as the SwinT-OGC checkpoint used in the paper.
`Grounding-DINO-tiny` is also supported and is considerably faster, at some
loss of mask quality.

### 3.3 Checkpoints

| checkpoint | how to obtain |
|---|---|
| `yolo11m-pose.pt` | `yolo detect` / `yolo pose` auto-download, or `wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt` |
| `SAM/sam_vit_b_01ec64.pth` | `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` |
| `climb_classifier_v2.pkl` | shipped in `models/` (see below) |

All checkpoint locations are read from the environment, so nothing is
hard-coded:

| variable | default |
|---|---|
| `GROUNDING_DINO_REPO` | — (path to the GroundingDINO checkout) |
| `GROUNDING_DINO_CONFIG` | `groundingdino/config/GroundingDINO_SwinT_OGC.py` |
| `GROUNDING_DINO_CHECKPOINT` | `groundingdino_swint_ogc.pth` |
| `GROUNDING_DINO_HF_ID` | `IDEA-Research/Grounding-DINO-tiny` |
| `SAM_CHECKPOINT` | `SAM/sam_vit_b_01ec64.pth` |
| `YOLO500_CHECKPOINT` | `results_yolo_experiments/exp_500/best_weights.pt` |

---

## 4. Reproducing the paper

### 4.1 The splits

The images are **not** redistributed with this repository (see §6).  The exact
file lists and labels of every subset are, so that anybody who has obtained
the images can reproduce every number:

```bash
python scripts/make_splits.py --data_root /path/to/dataset --out_dir splits
```

| subset | images | climb | no-climb |
|---|---|---|---|
| `train1` | 500 | 371 | 129 |
| `test1` | 50 | 35 | 15 |

### 4.2 The interaction feature vectors

```bash
python scripts/export_features.py \
    --image_dir  /path/to/dataset/test1/image \
    --label_dir  /path/to/dataset/test1/label \
    --out_csv    results/features/test1.csv \
    --pose_model yolo11m-pose.pt \
    --classifier ensemble \
    --model_path models/climb_classifier_v2.pkl
```

The script is deterministic and writes the feature names and the run
configuration next to the CSV.  A fixed seed (42) is set for every RNG that is
touched.

### 4.3 Evaluation on the test split (Task 2)

```bash
python saip/task_unified.py --mode eval \
    --test_dir /path/to/dataset/test1 \
    --out_dir  results \
    --classifier ensemble \
    --model_path models/climb_classifier_v2.pkl
```

Reported result on the 50-image test split (35 climb / 15 no-climb):

| method | train. imgs | P | R | F1 | Acc | TP | FP/FN | TN |
|---|---|---|---|---|---|---|---|---|
| YOLO11m-pose (pose-only) | 500 | 0.968 | 0.857 | 0.909 | 0.880 | 30 | 1/5 | 14 |
| **SAIP** | **~100** | **0.971** | **0.971** | **0.971** | **0.960** | **34** | **1/1** | **14** |

The score separability is the reason for the difference: the pose-only
baseline separates climbing from non-climbing by **0.018**, SAIP by **0.481**
(26x larger).

### 4.4 Sequence visualisation (Task 1)

```bash
python saip/task_unified.py --mode sequence \
    --input_dir  /path/to/sequence_frames \
    --output_dir results/sequence \
    --classifier ensemble \
    --model_path models/climb_classifier_v2.pkl
```

### 4.5 Robustness to multi-patch occlusion

```bash
python saip/multi_occlusion_test.py \
    --test_folder /path/to/dataset/test1 \
    --our_model_path models/climb_classifier_v2.pkl \
    --yolo500_path /path/to/exp_500/best_weights.pt \
    --out_dir results/occlusion
```

The occlusion grid is a 5x5 factorial: rows are the number of patches
(1, 2, 3, 4, 5), columns the patch size level (5 %, 10 %, 15 %, 20 %, 25 % of
the shorter image edge).  Patch positions are pseudo-randomised at a **fixed
seed (42)**, so the grid is exactly reproducible.

### 4.6 The YOLO11m baselines

```bash
python saip/task3_yolo_baseline.py --n_train 500 ...
python saip/task3_500.py ...
```

---

## 5. The shipped model

`models/climb_classifier_v2.pkl` holds the trained shallow head
(standardiser + gradient-boosted classifier, 200 trees, depth 3, learning rate
0.05, `random_state=42`) together with the ordered list of the 57 feature
names and the operating threshold `τ* = 0.22`.

It was pickled with scikit-learn 1.6.x.  On scikit-learn >= 1.7 the internal
`_loss` module moved, so `sam2.py` registers a compatibility alias and the
model loads unchanged.

---

## 6. Data availability

The climbing-over-barrier images analysed in this study come from a
**third-party dataset that was not collected by the authors**.  It is
distributed by its compiler through a Baidu Netdisk link given on the page
cited in the paper; a small payment to the compiler is requested for access,
and no open licence is stated.  As we do not hold redistribution rights, **we
do not redistribute the images**.

What *is* redistributed here is:

* the file lists of the training and test subsets **with their labels**
  (`splits/`),
* the extracted interaction-feature vectors (`results/features/`),
* the occlusion-generation script **with its fixed random seed**
  (`saip/multi_occlusion_test.py`),
* the source code.

Other raw and processed data generated during the study are available from the
corresponding author upon reasonable request.

---

## 7. Citation

```bibtex
@article{zhao2026saip,
  title   = {Scene-Aware Interaction Perception: A Few-Shot Framework for
             Human Behavior Recognition in Context},
  author  = {Zhao, Shukuo},
  journal = {International Journal of Computer Vision},
  year    = {2026}
}
```

## 8. Licence

The source code in this repository is released under the MIT licence.  The
**images are not included** and remain the property of their original
compiler, who states no open licence.
