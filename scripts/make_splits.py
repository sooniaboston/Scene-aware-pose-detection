# -*- coding: utf-8 -*-
"""
make_splits.py
==============
Materialises the FILE LISTS of the training and test subsets used in the
paper, together with their labels.

The climbing-over-barrier images are NOT redistributed with this repository
(the third-party dataset that provides them states no open licence and is
distributed through a Baidu Netdisk link by its compiler; see the Data
availability statement in the paper).  What *is* redistributed here is the
exact index of every subset -- file name, class label and bounding box --
so that anybody who has obtained the images can reproduce every split and
every number in the paper bit-for-bit.

Subsets
-------
    train1/   500 images  (371 climb / 129 no-climb)
              the training pool from which the ~100-image subset used to
              fit the shallow head and the operating threshold was sampled
    test1/     50 images  ( 35 climb /  15 no-climb)
              the held-out test split of the paper

Output
------
    splits/train1.csv   filename,label,cx,cy,w,h
    splits/test1.csv    filename,label,cx,cy,w,h
    splits/folders_train.csv  filename,label,source   (the 200-image
              climb/no-climb folder pool used by the baseline scripts)
    splits/SPLITS.md     human-readable description of every split

Label semantics
---------------
    label 1 = climb,  label 0 = no-climb
    (cx, cy, w, h) is the YOLO-normalised person bounding box of the
    labelled instance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple


def _read_yolo_label(path: Path) -> Optional[Tuple[int, float, float, float, float]]:
    """Read a YOLO detection label -> (class, cx, cy, w, h), or None."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None
    if not text:
        return None
    first = text.splitlines()[0].strip().split()
    if len(first) < 5:
        # a bare class index (e.g. "1" / "0")
        try:
            return (int(float(first[0])), 0.0, 0.0, 0.0, 0.0)
        except Exception:
            return None
    try:
        return (int(float(first[0])), float(first[1]), float(first[2]),
                float(first[3]), float(first[4]))
    except Exception:
        return None


def collect_image_label_dir(split_dir: Path) -> List[dict]:
    """Collect (image, label) pairs from a YOLO-style <split>/{image,label}."""
    img_dir = split_dir / "image"
    lab_dir = split_dir / "label"
    rows = []
    for img in sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
                      + list(img_dir.glob("*.png"))):
        lbl = _read_yolo_label(lab_dir / f"{img.stem}.txt")
        if lbl is None:
            continue
        cls, cx, cy, w, h = lbl
        rows.append(dict(filename=img.name, label=cls, cx=cx, cy=cy, w=w, h=h))
    return rows


def collect_class_folders(climb_dir: Path, no_climb_dir: Path) -> List[dict]:
    """Collect a folder pool that is already split by class directory."""
    rows = []
    for folder, label, tag in ((climb_dir, 1, "climb"), (no_climb_dir, 0, "no_climb")):
        if not folder.exists():
            continue
        for img in sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
                          + list(folder.glob("*.png"))):
            rows.append(dict(filename=img.name, label=label, source=tag))
    return rows


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True,
                    help="Root that contains train1/, test1/ and test/")
    ap.add_argument("--out_dir", default="splits")
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ("train1", "test1"):
        rows = collect_image_label_dir(root / split)
        write_csv(out / f"{split}.csv", rows,
                  ["filename", "label", "cx", "cy", "w", "h"])
        pos = sum(1 for r in rows if r["label"] == 1)
        summary[split] = dict(n=len(rows), climb=pos, no_climb=len(rows) - pos)
        print(f"{split}: {len(rows)} images ({pos} climb / {len(rows)-pos} no-climb)")

    folder_rows = collect_class_folders(root / "test" / "climb",
                                       root / "test" / "no_climb")
    write_csv(out / "folders_pool.csv", folder_rows,
              ["filename", "label", "source"])
    summary["folders_pool"] = dict(
        n=len(folder_rows),
        climb=sum(1 for r in folder_rows if r["label"] == 1),
        no_climb=sum(1 for r in folder_rows if r["label"] == 0))
    print(f"folders_pool: {len(folder_rows)} images")

    with open(out / "splits_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    # Human-readable summary
    md = ["# Dataset splits used in the paper", "",
          "The images themselves are **not** redistributed (see the Data",
          "availability statement in the paper). These are the exact file lists",
          "and labels of the subsets that were used.", ""]
    md.append("| subset | images | climb | no-climb |")
    md.append("|---|---|---|---|")
    for k, v in summary.items():
        md.append(f"| `{k}` | {v['n']} | {v['climb']} | {v['no_climb']} |")
    md += ["", "## Label semantics", "",
           "`label = 1` means *climb*, `label = 0` means *no-climb*.",
           "The columns `cx, cy, w, h` are the YOLO-normalised person bounding",
           "box of the labelled instance.", ""]
    (out / "SPLITS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
