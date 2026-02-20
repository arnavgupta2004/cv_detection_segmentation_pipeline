import cv2
import torch
import numpy as np
from ultralytics import YOLO
import argparse
import time
import os

from unet_model import UNet


class RealTimeInference:
    def __init__(
        self,
        yolo_weights="runs/yolo/yolov8n_coco128/weights/best.pt",
        unet_weights="runs/unet/best_unet.pth",
        mode="both",
        device="auto",
    ):
        self.mode = mode
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        print(f"Device: {self.device}")

        if self.mode in ["detect", "both"]:
            if not os.path.exists(yolo_weights):
                print(
                    f"YOLO weights not found at {yolo_weights}, falling back to yolov8n.pt"
                )
                yolo_weights = "yolov8n.pt"
            self.yolo = YOLO(yolo_weights)

        if self.mode in ["segment", "both"]:
            self.unet = UNet(in_channels=3, out_channels=1).to(self.device)
            if os.path.exists(unet_weights):
                self.unet.load_state_dict(
                    torch.load(
                        unet_weights, map_location=self.device, weights_only=True
                    )
                )
            else:
                print(
                    f"U-Net weights not found at {unet_weights}, model will output noise."
                )
            self.unet.eval()

    def process_frame(self, frame):
        annotated = frame.copy()
        h, w = frame.shape[:2]

        if self.mode in ["detect", "both"]:
            results = self.yolo(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = f"{results.names[int(box.cls[0])]} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        if self.mode in ["segment", "both"]:
            inp = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB)
            inp = (
                torch.tensor(np.transpose(inp.astype(np.float32) / 255.0, (2, 0, 1)))
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                mask_256 = (torch.sigmoid(self.unet(inp)) > 0.5).squeeze().cpu().numpy()

            if mask_256.any():
                mask = cv2.resize(
                    mask_256.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
                overlay = annotated.copy()
                overlay[mask == 1] = (0, 255, 0)
                annotated = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)

        return annotated

    def run(self, source="0"):
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        is_image = any(source.lower().endswith(e) for e in image_exts)
        is_video = source == "0" or any(source.lower().endswith(e) for e in video_exts)

        if is_image:
            frame = cv2.imread(source)
            if frame is None:
                print(f"Could not load image: {source}")
                return
            t0 = time.time()
            result = self.process_frame(frame)
            cv2.putText(
                result,
                f"Inference: {(time.time() - t0) * 1000:.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif is_video:
            cap = cv2.VideoCapture(0 if source == "0" else source)
            if not cap.isOpened():
                print(f"Could not open video source: {source}")
                return
            prev_time = time.time()
            print("Press 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.process_frame(frame)
                fps = 1 / (time.time() - prev_time)
                prev_time = time.time()
                cv2.putText(
                    result,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Real-Time Inference", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()

        else:
            print(f"Unsupported source: {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument(
        "--mode", type=str, choices=["detect", "segment", "both"], default="both"
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    RealTimeInference(mode=args.mode, device=args.device).run(source=args.source)
