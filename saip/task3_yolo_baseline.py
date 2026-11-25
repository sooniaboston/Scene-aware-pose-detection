# -*- coding: utf-8 -*-
"""
Task 3: Train a plain YOLOv11m detector on train1/ with different dataset sizes
and evaluate on test1/. Output metrics as a table.
只使用你原生已有label，不自动生成任何伪标签
约定：0=no-climb，1=climb
"""
import argparse
import sys
import json
import shutil
import random
from pathlib import Path
import yaml
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np

# 设置随机种子保证可复现性
random.seed(42)
np.random.seed(42)


def prepare_split(src_dir: Path, dst_dir: Path, split_name: str, sample_size: int = None) -> int:
    """
    纯拷贝：图片、label 原样复制，不做任何标签生成与转换
    """
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

    for p in img_paths:
        # 复制图片
        dst_img = out_img / p.name
        if not dst_img.exists():
            shutil.copy(p, dst_img)

        # 【关键】原样复制标签，不修改、不生成
        src_lab = lab_dir / f"{p.stem}.txt"
        dst_lab = out_lab / f"{p.stem}.txt"
        if src_lab.exists():
            shutil.copy(src_lab, dst_lab)

    print(f"  {split_name}: 拷贝 {len(img_paths)} 张图片及对应原生标签")
    return len(img_paths)


def build_yolo_dataset(train_src: Path, test_src: Path, yolo_root: Path, train_sample_size: int) -> Path:
    """
    从train划分90% train /10% val，全部原样拷贝原生标签
    0=no-climb，1=climb
    """
    print(f"\n[prepare] Building YOLO dataset (train size: {train_sample_size}) at {yolo_root}")
    
    if yolo_root.exists():
        shutil.rmtree(yolo_root)

    img_dir = train_src / 'image'
    lab_dir = train_src / 'label'
    all_img_paths = sorted(list(img_dir.glob('*.jpg')) +
                           list(img_dir.glob('*.jpeg')) +
                           list(img_dir.glob('*.png')))

    if train_sample_size and train_sample_size < len(all_img_paths):
        all_img_paths = random.sample(all_img_paths, train_sample_size)

    random.shuffle(all_img_paths)
    split_idx = int(len(all_img_paths) * 0.9)
    train_imgs = all_img_paths[:split_idx]
    val_imgs = all_img_paths[split_idx:]

    print(f"[split] 总训练样本: {len(all_img_paths)}")
    print(f"[split] 训练集(train): {len(train_imgs)} 张")
    print(f"[split] 验证集(val): {len(val_imgs)} 张")

    # 拷贝 train
    out_img_train = yolo_root / 'images' / 'train'
    out_lab_train = yolo_root / 'labels' / 'train'
    out_img_train.mkdir(parents=True, exist_ok=True)
    out_lab_train.mkdir(parents=True, exist_ok=True)
    for p in train_imgs:
        shutil.copy(p, out_img_train / p.name)
        src_lab = lab_dir / f"{p.stem}.txt"
        dst_lab = out_lab_train / f"{p.stem}.txt"
        if src_lab.exists():
            shutil.copy(src_lab, dst_lab)

    # 拷贝 val
    out_img_val = yolo_root / 'images' / 'val'
    out_lab_val = yolo_root / 'labels' / 'val'
    out_img_val.mkdir(parents=True, exist_ok=True)
    out_lab_val.mkdir(parents=True, exist_ok=True)
    for p in val_imgs:
        shutil.copy(p, out_img_val / p.name)
        src_lab = lab_dir / f"{p.stem}.txt"
        dst_lab = out_lab_val / f"{p.stem}.txt"
        if src_lab.exists():
            shutil.copy(src_lab, dst_lab)

    # yaml 保持你要的 0=no-climb 1=climb
    data_yaml = yolo_root / 'data.yaml'
    data_yaml.write_text(yaml.safe_dump({
        'path': str(yolo_root.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'no-climb', 1: 'climb'},
    }))

    print(f"[prepare] 数据集构建完成，全部使用原生标签: {data_yaml}")
    return data_yaml


def image_level_eval(model: YOLO, test_dir: Path, conf: float = 0.25) -> dict:
    img_dir = test_dir / 'image'
    lab_dir = test_dir / 'label'

    img_paths = sorted(list(img_dir.glob('*.jpg')) +
                       list(img_dir.glob('*.jpeg')) +
                       list(img_dir.glob('*.png')))

    def _gt(stem):
        p = lab_dir / f"{stem}.txt"
        if not p.exists():
            return 0
        txt = p.read_text(encoding='utf-8', errors='ignore').strip()
        if not txt:
            return 0
        # 只要有类别1就是climb
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1 and parts[0] == '1':
                return 1
        return 0

    tp = fp = fn = tn = 0
    scores_pos, scores_neg = [], []
    
    for p in img_paths:
        gt = _gt(p.stem)
        res = model.predict(str(p), conf=conf, verbose=False)
        boxes = res[0].boxes
        
        pred = 0
        max_conf = 0.0
        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.cpu().numpy()
            if 1 in cls_ids:
                pred = 1
                max_conf = float(boxes.conf[cls_ids == 1].max().item())

        if   gt == 1 and pred == 1: tp += 1
        elif gt == 0 and pred == 1: fp += 1
        elif gt == 1 and pred == 0: fn += 1
        else:                       tn += 1

        (scores_pos if gt == 1 else scores_neg).append(max_conf)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc  = (tp + tn) / len(img_paths) if img_paths else 0.0
    mean_pos = sum(scores_pos) / len(scores_pos) if scores_pos else 0.0
    mean_neg = sum(scores_neg) / len(scores_neg) if scores_neg else 0.0

    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=prec, recall=rec, f1=f1, accuracy=acc,
        mean_score_climb=mean_pos,
        mean_score_no_climb=mean_neg
    )


# 下面 run_experiment、main 函数完全不变
def run_experiment(train_dir, test_dir, yolo_root, out_dir, model_name, 
                   epochs, imgsz, batch, device, conf, train_sample_size):
    print(f"\n{'='*80}")
    print(f"Experiment: Training with {train_sample_size} images")
    print(f"{'='*80}")
    
    data_yaml = build_yolo_dataset(
        Path(train_dir), 
        Path(test_dir), 
        Path(yolo_root) / f"exp_{train_sample_size}",
        train_sample_size
    )
    
    print(f"\n[train] Starting YOLOv11m training ({epochs} epochs)")
    model = YOLO(model_name)
    exp_out_dir = Path(out_dir) / f"exp_{train_sample_size}"
    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(exp_out_dir),
        name='yolo11m_climb',
        exist_ok=True,
        patience=30,
        seed=42,
        verbose=False,
        amp=False,
        save=True
    )
    
    best_weights = Path(train_results.save_dir) / 'weights' / 'best.pt'
    if not best_weights.exists():
        best_weights = Path(train_results.save_dir) / 'weights' / 'last.pt'
    final_best_path = exp_out_dir / "best_weights.pt"
    final_best_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, final_best_path)
    print(f"[save] 最优模型已永久保存到: {final_best_path}")
    
    model = YOLO(str(final_best_path))
    
    print("\n[eval] Running model.val() on val split ...")
    val_metrics = model.val(
        data=str(data_yaml), split='val',
        imgsz=imgsz, batch=batch, device=device,
        verbose=False
    )
    box = val_metrics.box
    yolo_metrics = {
        'mAP50': float(box.map50),
        'mAP50_95': float(box.map),
        'detection_P': float(box.mp),
        'detection_R': float(box.mr),
        'detection_F1': float(2 * box.mp * box.mr / (box.mp + box.mr + 1e-9)),
    }
    
    print(f"\n[eval] Computing image-level metrics @ conf={conf} ...")
    img_metrics = image_level_eval(model, Path(test_dir), conf=conf)
    
    results = {
        'train_samples': train_sample_size,
        'yolo_metrics': yolo_metrics,
        'image_metrics': img_metrics
    }
    
    exp_summary = {
        'train_samples': train_sample_size,
        'yolo_box_metrics': yolo_metrics,
        'image_level_metrics': img_metrics
    }
    exp_out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(exp_summary, open(exp_out_dir / 'experiment_summary.json', 'w'), indent=2)
    
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', default='train2')
    ap.add_argument('--test_dir',  default='test1')
    ap.add_argument('--out_dir',   default='results_yolo_experiments')
    ap.add_argument('--yolo_root', default='yolo_data_experiments')
    ap.add_argument('--model',     default='yolo11m.pt')
    ap.add_argument('--epochs',    type=int, default=100)
    ap.add_argument('--imgsz',     type=int, default=640)
    ap.add_argument('--batch',     type=int, default=16)
    ap.add_argument('--device',    default=0)
    ap.add_argument('--conf',      type=float, default=0.25)
    ap.add_argument('--sample_sizes', nargs='+', type=int, 
                   default=[100, 200, 400, 500, 1000])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    for sample_size in args.sample_sizes:
        try:
            exp_result = run_experiment(
                train_dir=args.train_dir,
                test_dir=args.test_dir,
                yolo_root=args.yolo_root,
                out_dir=args.out_dir,
                model_name=args.model,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                conf=args.conf,
                train_sample_size=sample_size
            )
            all_results.append(exp_result)
        except Exception as e:
            print(f"Error running experiment with {sample_size} images: {e}")
            continue

    if all_results:
        table_data = []
        for res in all_results:
            row = {
                '训练样本数': res['train_samples'],
                'mAP50': res['yolo_metrics']['mAP50'],
                'mAP50-95': res['yolo_metrics']['mAP50_95'],
                '检测精度(P)': res['yolo_metrics']['detection_P'],
                '检测召回(R)': res['yolo_metrics']['detection_R'],
                '检测F1': res['yolo_metrics']['detection_F1'],
                '图像级精度': res['image_metrics']['precision'],
                '图像级召回': res['image_metrics']['recall'],
                '图像级F1': res['image_metrics']['f1'],
                '图像级准确率': res['image_metrics']['accuracy'],
                'TP': res['image_metrics']['tp'],
                'FP': res['image_metrics']['fp'],
                'FN': res['image_metrics']['fn'],
                'TN': res['image_metrics']['tn']
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('训练样本数').reset_index(drop=True)
        
        csv_path = out_dir / 'experiment_results_table.csv'
        excel_path = out_dir / 'experiment_results_table.xlsx'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        df.to_excel(excel_path, index=False)
        
        print(f"\n{'='*80}")
        print("所有实验结果汇总表")
        print(f"{'='*80}")
        print("\n1. 检测级指标:")
        detection_cols = ['训练样本数', 'mAP50', 'mAP50-95', '检测精度(P)', '检测召回(R)', '检测F1']
        print(df[detection_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        print("\n2. 图像级指标:")
        image_cols = ['训练样本数', '图像级精度', '图像级召回', '图像级F1', '图像级准确率', 'TP', 'FP', 'FN', 'TN']
        print(df[image_cols].to_string(index=False, 
                                      float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
        
        full_results = {'experiments': all_results, 'table_data': table_data}
        json.dump(full_results, open(out_dir / 'all_experiments_results.json', 'w'), indent=2)
        
        print(f"\n结果已保存：\n - CSV: {csv_path}\n - Excel: {excel_path}\n - JSON: {out_dir}/all_experiments_results.json")
    else:
        print("没有成功完成任何实验")


if __name__ == '__main__':
    main()