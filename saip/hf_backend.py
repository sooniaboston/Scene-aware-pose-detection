# -*- coding: utf-8 -*-
"""
hf_backend.py
=============
A drop-in replacement for the two functions of the original Grounding-DINO
repository that SAIP uses:

    from groundingdino.util.inference import load_image, predict

The original repository is the reference implementation and is used
unchanged whenever it is importable.  This module provides an equivalent
implementation on top of the Hugging Face `transformers` port of
Grounding-DINO (`IDEA-Research/Grounding-DINO-tiny` or `-small`), so that the
method can be run without building the original CUDA extension.

The two entry points keep the original signatures and semantics:

    load_image(path)                  -> (PIL-like) image tensor
    predict(model, image, caption,
            box_threshold, text_threshold)
                                     -> (boxes, logits, phrases)

`boxes` are returned in the ORIGINAL Grounding-DINO convention:
    (cx, cy, w, h), each NORMALISED to [0, 1] w.r.t. the image size.

The prompt handling is identical: the caption is a '.'-separated list of
phrases, and each returned box is attributed to the phrase it matched.
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# The HF model id / local path of the Grounding-DINO port to use.
DEFAULT_MODEL_ID = os.environ.get(
    "GROUNDING_DINO_HF_ID", "IDEA-Research/Grounding-DINO-tiny")

_PROC = None
_MODEL = None
_MODEL_ID = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _ensure_loaded(model_id: str = None):
    """Load (and cache) the HF processor and model."""
    global _PROC, _MODEL, _MODEL_ID
    model_id = model_id or DEFAULT_MODEL_ID
    if _MODEL is not None and _MODEL_ID == model_id:
        return _PROC, _MODEL

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    _PROC = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
    _MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, local_files_only=local_only).eval()
    _MODEL_ID = model_id
    return _PROC, _MODEL


# ---------------------------------------------------------------------------
# The original API
# ---------------------------------------------------------------------------
def load_image(image_path: str):
    """Load an image the way `groundingdino.util.inference.load_image` does.

    The original returns a CUDA tensor of shape (3, H, W) with the image
    scaled so that the longest side is 800 px.  Here the image is returned as
    a PIL image; the HF processor performs the same normalisation.
    """
    return Image.open(image_path).convert("RGB")


def predict(model=None,
            image=None,
            caption: str = "",
            box_threshold: float = 0.28,
            text_threshold: float = 0.22,
            **kwargs):
    """Ground `caption` in `image`.

    Returns
    -------
    boxes   : np.ndarray (N, 4) -- (cx, cy, w, h), normalised to [0, 1]
    logits  : np.ndarray (N,)   -- the matching scores
    phrases : list[str]          -- the phrase each box was attributed to
    """
    proc, mdl = _ensure_loaded()
    if model is not None and model is not mdl:
        # The caller passed its own model; use it with our processor.
        mdl = model

    if isinstance(image, Image.Image):
        pil = image
    else:  # a numpy array (BGR) or a path
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image[:, :, ::-1])
        else:
            pil = Image.open(str(image)).convert("RGB")

    # The original API takes the '.'-separated caption; HF expects a list of
    # candidate strings, one per phrase.
    phrases_in = [p.strip() for p in caption.split(".") if p.strip()]
    if not phrases_in:
        phrases_in = [caption]

    inputs = proc(images=pil, text=[phrases_in], return_tensors="pt")
    with torch.no_grad():
        outputs = mdl(**inputs)

    target_sizes = [(pil.height, pil.width)]
    # The keyword for the score threshold was renamed in transformers 4.51
    # (`box_threshold` -> `threshold`); support both.
    try:
        post = proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
    except TypeError:
        post = proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
    r = post[0]

    scores = r["scores"].detach().cpu().numpy()
    # `text_labels` since transformers >= 4.51; `labels` before that.
    if "text_labels" in r:
        labels = list(r["text_labels"])
    else:
        labels = [str(x) for x in r["labels"]]
    if labels and isinstance(labels[0], (int, np.integer)):
        labels = [phrases_in[i] if i < len(phrases_in) else caption
                  for i in labels]

    boxes_xyxy = r["boxes"].detach().cpu().numpy()
    boxes = np.zeros_like(boxes_xyxy)
    if len(boxes_xyxy):
        x1, y1, x2, y2 = (boxes_xyxy[:, 0], boxes_xyxy[:, 1],
                          boxes_xyxy[:, 2], boxes_xyxy[:, 3])
        boxes[:, 0] = (x1 + x2) / 2.0 / pil.width    # cx
        boxes[:, 1] = (y1 + y2) / 2.0 / pil.height   # cy
        boxes[:, 2] = (x2 - x1) / pil.width          # w
        boxes[:, 3] = (y2 - y1) / pil.height         # h

    return boxes, scores.astype(np.float32), labels
