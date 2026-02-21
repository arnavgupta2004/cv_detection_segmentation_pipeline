import os
from ultralytics import YOLO


def train_yolo():
    os.makedirs("runs/yolo", exist_ok=True)

    # Load YOLOv8 nano pretrained weights
    model = YOLO("yolov8n.pt")

    # Fine-tune on COCO128
    results = model.train(
        data="coco128.yaml",
        epochs=10,
        imgsz=640,
        project="runs/yolo",
        name="yolov8n_coco128",
        exist_ok=True,
        device="cpu",  # change to 'cuda' if GPU available
    )

    print(f"mAP@50: {results.box.map50:.4f}")
    print(f"mAP@50-95: {results.box.map:.4f}")
    print("Weights saved to: runs/yolo/yolov8n_coco128/weights/best.pt")


if __name__ == "__main__":
    train_yolo()
