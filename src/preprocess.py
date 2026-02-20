import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob


def load_image(image_path: str):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def preprocess_for_inference(image: np.ndarray, target_size: tuple):
    resized = cv2.resize(image, target_size)
    normalized = resized.astype(np.float32) / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    return torch.tensor(transposed).unsqueeze(0)


class COCOSegmentationDataset(Dataset):
    def __init__(
        self, image_dir: str, mask_dir: str, transform=None, target_size=(256, 256)
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = load_image(img_path)

        basename = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, basename)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = cv2.resize(image, self.target_size)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0  # float in [0,1] for BCEWithLogitsLoss

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, 0)

        return torch.tensor(image), torch.tensor(mask)


if __name__ == "__main__":
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)

    yolo_tensor = preprocess_for_inference(dummy, target_size=(640, 640))
    unet_tensor = preprocess_for_inference(dummy, target_size=(256, 256))

    print(f"YOLO input shape:  {yolo_tensor.shape}")
    print(f"U-Net input shape: {unet_tensor.shape}")
