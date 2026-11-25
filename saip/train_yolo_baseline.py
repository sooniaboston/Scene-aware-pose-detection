# -*- coding: utf-8 -*-
import os
import yaml
import torch
from ultralytics import YOLO
import mlflow
from pathlib import Path
import glob
import random
from dotenv import load_dotenv

# 强制关闭 AMP 自动检查，避免下载损坏的 yolo26n.pt
os.environ['ULTRALYTICS_AMP_CHECK'] = '0'

try:
    from sklearn.model_selection import KFold
except Exception:
    KFold = None

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://10.0.15.62:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'COCO_Pose_Detection')
DATA_YAML_PATH = os.getenv('DATA_YAML_PATH', 'coco-pose.yaml')
PRE_MODEL_PATH = os.getenv('PRE_MODEL_PATH', 'yolo11m-pose.pt')
RUN_NAME = os.getenv('MLFLOW_RUN_NAME', 'YOLO11_Pose_Run')

EPOCHS = int(os.getenv('EPOCHS', '100'))
IMG_SIZE = int(os.getenv('IMG_SIZE', 640))
BATCH = int(os.getenv('BATCH', 16))
DEVICE = os.getenv('DEVICE', '0')
WORKERS = int(os.getenv('WORKERS', 8))
PATIENCE = int(os.getenv('PATIENCE', 30))
K_FOLDS = int(os.getenv('K_FOLDS', 5))

LR0 = float(os.getenv('LR0', 0.01))
AUGMENT = os.getenv('AUGMENT', 'True').lower() == 'true'
MOSAIC = float(os.getenv('MOSAIC', 1.0))
DEGREES = float(os.getenv('DEGREES', 10.0))
FLIPUD = float(os.getenv('FLIPUD', 0.0))
FLIPLR = float(os.getenv('FLIPLR', 0.5))
SCALE = float(os.getenv('SCALE', 0.5))

def make_cv_splits(train_dir, k, out_dir, shuffle=True, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    images = []
    for e in exts:
        images.extend(glob.glob(str(Path(train_dir) / '**' / e), recursive=True))
    images = sorted(images)

    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(images)

    n = len(images)
    if n == 0:
        raise RuntimeError('No images found in train dir')

    folds = []
    if KFold is not None:
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
        for train_idx, val_idx in kf.split(images):
            train_files = [images[i] for i in train_idx]
            val_files = [images[i] for i in val_idx]
            folds.append((train_files, val_files))
    else:
        fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
        current = 0
        for size in fold_sizes:
            val_idx = list(range(current, current + size))
            train_idx = [i for i in range(n) if i not in val_idx]
            folds.append(([images[i] for i in train_idx], [images[i] for i in val_idx]))
            current += size

    yaml_paths = []
    for i, (train_files, val_files) in enumerate(folds):
        train_txt = out_dir / f'train_fold_{i}.txt'
        val_txt = out_dir / f'val_fold_{i}.txt'
        train_txt.write_text('\n'.join(train_files), encoding='utf-8')
        val_txt.write_text('\n'.join(val_files), encoding='utf-8')
        yaml_paths.append((str(train_txt), str(val_txt)))
    return yaml_paths

def _extract_scalar_from_metrics(metrics_dict, keys):
    if not metrics_dict:
        return None
    for k, v in metrics_dict.items():
        if any(key in k.lower() for key in keys):
            try:
                return float(v)
            except:
                try:
                    return float(v.item())
                except:
                    return None
    return None

def evaluate_and_log(model, data_yaml, prefix, imgsz=IMG_SIZE, batch=BATCH):
    try:
        results = model.val(data=str(data_yaml), imgsz=imgsz, batch=batch, verbose=False)
    except:
        results = model.val(data=str(data_yaml), imgsz=imgsz, batch=batch)

    metrics_obj = None
    if hasattr(results, 'metrics'):
        metrics_obj = results.metrics
    elif isinstance(results, dict):
        metrics_obj = results

    box_map50 = _extract_scalar_from_metrics(metrics_obj, ['box_map50', 'box/map50'])
    box_map = _extract_scalar_from_metrics(metrics_obj, ['box_map', 'box/map'])
    pose_map50 = _extract_scalar_from_metrics(metrics_obj, ['pose_map50', 'pose/map50'])
    pose_map = _extract_scalar_from_metrics(metrics_obj, ['pose_map', 'pose/map'])

    print("\n" + "=" * 60)
    print(f"Test Results - {prefix}")
    print(f"Box mAP50: {box_map50:.4f}")
    print(f"Box mAP:   {box_map:.4f}")
    print(f"Pose mAP50:{pose_map50:.4f}")
    print(f"Pose mAP:  {pose_map:.4f}")
    print("=" * 60 + "\n")

    try:
        mlflow.log_metric(f"{prefix}_box_mAP50", box_map50)
        mlflow.log_metric(f"{prefix}_box_mAP", box_map)
        mlflow.log_metric(f"{prefix}_pose_mAP50", pose_map50)
        mlflow.log_metric(f"{prefix}_pose_mAP", pose_map)
    except:
        pass

    return box_map50, box_map, pose_map50, pose_map

if __name__ == '__main__':
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 修复编码读取
    with open(DATA_YAML_PATH, 'rb') as f:
        raw_data = f.read()
    try:
        base_data = yaml.safe_load(raw_data.decode('utf-8', 'ignore'))
    except:
        base_data = yaml.safe_load(raw_data.decode('gbk', 'ignore'))

    train_dir = base_data['train']
    test_dir = base_data['val']
    cv_out = Path('cv_splits')

    if not cv_out.exists() or len(list(cv_out.glob("*.txt"))) == 0:
        split_paths = make_cv_splits(train_dir, K_FOLDS, cv_out)
    else:
        print("Loading existing CV splits...")
        split_paths = []
        for i in range(K_FOLDS):
            train_txt = cv_out / f'train_fold_{i}.txt'
            val_txt = cv_out / f'val_fold_{i}.txt'
            split_paths.append((str(train_txt), str(val_txt)))

    for fold, (train_txt, val_txt) in enumerate(split_paths):
        fold_name = f"{RUN_NAME}_fold_{fold}"
        data_yaml = cv_out / f'data_fold_{fold}.yaml'

        if not data_yaml.exists():
            fold_data = {
                'train': str(Path(train_txt).resolve()),
                'val': str(Path(val_txt).resolve()),  # 这里修复！原来写的是 train_txt，BUG！
                'test': str(Path(test_dir).resolve()),
                'nc': base_data['nc'],
                'names': base_data['names'],
                'kpt_shape': base_data.get('kpt_shape'),
                'flip_idx': base_data.get('flip_idx')
            }
            with open(data_yaml, 'w', encoding='utf-8') as f:
                yaml.safe_dump(fold_data, f, allow_unicode=True)

        last_ckpt = Path(f"runs/pose/{fold_name}/weights/last.pt")
        resume_train = False

        with mlflow.start_run(run_name=fold_name):
            try:
                if last_ckpt.exists() and os.path.getsize(last_ckpt) > 1024*1024:
                    print(f"Resuming from: {last_ckpt}")
                    model = YOLO(str(last_ckpt))
                    resume_train = True
                else:
                    print("Starting from pretrained model...")
                    model = YOLO(PRE_MODEL_PATH)
            except:
                print("Checkpoint broken, starting over...")
                model = YOLO(PRE_MODEL_PATH)

            # 训练
            model.train(
                data=str(data_yaml),
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=BATCH,
                device=DEVICE,
                workers=WORKERS,
                patience=PATIENCE,
                lr0=LR0,
                augment=AUGMENT,
                mosaic=MOSAIC,
                degrees=DEGREES,
                flipud=FLIPUD,
                fliplr=FLIPLR,
                scale=SCALE,
                name=fold_name,
                project='runs/pose',
                exist_ok=True,
                resume=resume_train,
                amp=False  # 关键：关闭AMP，彻底避免报错
            )

            print(f"\nFold {fold} training finished. Starting test...")
            evaluate_and_log(model, data_yaml, prefix="test")

    print("\nAll folds training and testing finished!")