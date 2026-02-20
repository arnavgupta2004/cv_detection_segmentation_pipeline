import torch
import numpy as np
from ultralytics import YOLO
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from unet_model import UNet
from train_unet import SyntheticSegmentationDataset


def calculate_iou(pred_mask, true_mask):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def calculate_dice(pred_mask, true_mask):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    denom = pred_mask.sum() + true_mask.sum()
    return 1.0 if denom == 0 else (2.0 * intersection) / denom


def evaluate_unet():
    print("Evaluating U-Net...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "runs/unet/best_unet.pth"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run train_unet.py first.")
        return 0.0, 0.0

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    val_loader = DataLoader(
        SyntheticSegmentationDataset(num_samples=100), batch_size=1, shuffle=False
    )
    ious, dices = [], []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="U-Net Eval"):
            images = images.to(device)
            true_mask = masks.numpy()[0, 0]
            pred_mask = (torch.sigmoid(model(images)) > 0.5).cpu().numpy()[0, 0]
            ious.append(calculate_iou(pred_mask, true_mask))
            dices.append(calculate_dice(pred_mask, true_mask))

    return np.mean(ious), np.mean(dices)


# Yolo
def evaluate_yolo():
    print("Evaluating YOLOv8...")
    model_path = "runs/yolo/yolov8n_coco128/weights/best.pt"

    if not os.path.exists(model_path):
        print("Fine-tuned weights not found, falling back to yolov8n.pt")
        model_path = "yolov8n.pt"

    metrics = YOLO(model_path).val(data="coco128.yaml")
    return metrics.box.map50, metrics.box.map


def main():
    print("=" * 40)
    yolo_map50, yolo_map50_95 = evaluate_yolo()
    unet_iou, unet_dice = evaluate_unet()

    print("\n" + "=" * 40)
    print("        EVALUATION SUMMARY")
    print("=" * 40)
    print(f"YOLOv8  — mAP@50: {yolo_map50:.4f}  mAP@50-95: {yolo_map50_95:.4f}")
    print(f"U-Net   — IoU:    {unet_iou:.4f}  Dice:      {unet_dice:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
