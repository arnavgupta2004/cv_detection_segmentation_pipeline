import torch
import ultralytics.nn.modules as um
import ultralytics.nn.tasks as ut
import torch.nn as nn
import os
from ultralytics import YOLO

# Allow all Ultralytics YOLO classes for PyTorch 2.6 safe loading
safe_classes = []

# Collect all classes from ultralytics.nn.modules
for name in dir(um):
    obj = getattr(um, name)
    if isinstance(obj, type):
        safe_classes.append(obj)

# Collect all classes from ultralytics.nn.tasks
for name in dir(ut):
    obj = getattr(ut, name)
    if isinstance(obj, type):
        safe_classes.append(obj)

# Also allow common torch containers used in YOLO checkpoints
safe_classes += [nn.Sequential]

torch.serialization.add_safe_globals(safe_classes)


def train_yolo():
    os.makedirs("runs/yolo", exist_ok=True)

    model = YOLO("yolov8n.pt")  # nano — fast enough for CPU/edge inference

    results = model.train(
        data="coco128.yaml",
        epochs=10,
        imgsz=640,
        project="runs/yolo",
        name="yolov8n_coco128",
        exist_ok=True,
        device="cpu",  # swap to 'cuda' if available
    )

    print(f"mAP@50: {results.box.map50:.4f}  mAP@50-95: {results.box.map:.4f}")
    print("Weights saved to: runs/yolo/yolov8n_coco128/weights/best.pt")


if __name__ == "__main__":
    train_yolo()
