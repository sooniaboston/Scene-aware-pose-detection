# -*- coding: utf-8 -*-
import argparse
import json
import shutil
import random
from pathlib import Path
import yaml
from ultralytics import YOLO
import pandas as pd
import numpy as np

# 固定随机种子
random.seed(42)
np.random.seed(42)

# =============================================================================
# 工具函数（已修复编码问题）
# =============================================================================
def is_yolo_bbox_label(text: str) -> bool:
    return False  # 强制重新生成标签，彻底避免class 1错误

def prepare_split(src_dir: Path, dst_dir: Path, split_name: str, sample_size: int = None) -> int:
    img_dir = src_dir / 'image'
    lab_dir = src_dir / 'label'
    out_img = dst_dir / 'images' / split_name
    out_lab = dst_dir / 'labels' / split_name
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(list(img_dir.glob('*.jpg')) +
                       list(img_dir.glob('*.jpeg')) +
                       list(img_dir.glob('*.png')))
    
    if sample_size and sample_size < len(img_paths):
        img_paths = random.sample(img_paths, sample_size)

    n_climb = n_noclimb = 0
    for p in img_paths:
        dst_img = out_img / p.name
        if not dst_img.exists():
            shutil.copy(p, dst_img)

        src_lab = lab_dir / f"{p.stem}.txt"
        dst_lab = out_lab / f"{p.stem}.txt"
        is_climb = False

        if src_lab.exists():
            # ? 修复编码报错：使用 errors='ignore'
            txt = src_lab.read_text(encoding='gbk', errors='ignore').strip()
            first = txt.splitlines()[0].strip().lower() if txt else ''
            is_climb = first in ('1', 'climb', 'climbing')
        else:
            stem = p.stem.lower()
            is_climb = stem.startswith(('climb_', 'climbing_'))

        # 强制写入正确标签，类别一定是0
        if is_climb:
            dst_lab.write_text("0 0.5 0.5 0.9 0.9\n")
            n_climb += 1
        else:
            dst_lab.write_text("")
            n_noclimb += 1

    print(f"  {split_name}: {len(img_paths)} images ({n_climb} climb / {n_noclimb} no-climb)")
    return len(img_paths)

def build_yolo_dataset(train_src: Path, test_src: Path, yolo_root: Path):
    print(f"\n[prepare] 构建数据集 → {yolo_root}")
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    
    prepare_split(train_src, yolo_root, 'train', sample_size=500)
    prepare_split(test_src, yolo_root, 'val')

    data_yaml = yolo_root / 'data.yaml'
    data_yaml.write_text(yaml.safe_dump({
        'path': str(yolo_root.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'climb'},
    }), encoding='utf-8')
    return data_yaml

# =============================================================================
# 主训练：500样本 + 保存最优模型
# =============================================================================
def train_500_and_save_best():
    # 配置
    train_dir = "train1"
    test_dir = "test1"
    yolo_root = "yolo_data_500"
    out_dir = "results_yolo_500"
    model_name = "yolo11m.pt"
    epochs = 50
    imgsz = 640
    batch = 16
    device = 0
    save_best_to = "best_model_500.pt"

    # 构建数据集
    data_yaml = build_yolo_dataset(Path(train_dir), Path(test_dir), Path(yolo_root))

    # 训练
    print("\n[train] 开始训练 500 样本")
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=out_dir,
        name="yolo11m_500",
        exist_ok=True,
        patience=10,
        seed=42,
        amp=False,
        verbose=True
    )

    # 保存最优模型
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy(best_pt, save_best_to)
    print(f"\n? 最优模型已保存到：{save_best_to}")

    # 评估
    print("\n[eval] 测试集评估")
    val = model.val(data=str(data_yaml), split='val', device=device)
    print(f"mAP50: {val.box.map50:.4f}")
    print(f"mAP50-95: {val.box.map:.4f}")
    print(f"Precision: {val.box.mp:.4f}")
    print(f"Recall: {val.box.mr:.4f}")

if __name__ == "__main__":
    train_500_and_save_best()