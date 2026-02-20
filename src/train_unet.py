import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from tqdm import tqdm

from unet_model import UNet


class SyntheticSegmentationDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(256, 256)):
        self.num_samples = num_samples
        self.images = []
        self.masks = []

        for _ in range(num_samples):
            img = np.random.randint(
                0, 50, (img_size[0], img_size[1], 3), dtype=np.uint8
            )
            mask = np.zeros(img_size, dtype=np.uint8)

            shape_type = np.random.choice(["circle", "rectangle"])
            x = np.random.randint(40, img_size[1] - 40)
            y = np.random.randint(40, img_size[0] - 40)
            color = tuple(int(c) for c in np.random.randint(150, 255, 3))

            if shape_type == "circle":
                r = np.random.randint(20, 60)
                cv2.circle(img, (x, y), r, color, -1)
                cv2.circle(mask, (x, y), r, 255, -1)
            else:
                w, h = np.random.randint(30, 80), np.random.randint(30, 80)
                cv2.rectangle(
                    img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, -1
                )
                cv2.rectangle(
                    mask, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), 255, -1
                )

            img = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
            mask = np.expand_dims(mask.astype(np.float32) / 255.0, 0)

            self.images.append(torch.tensor(img))
            self.masks.append(torch.tensor(mask))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


class DiceLoss(nn.Module):
    # BCE handles pixel-wise stability; Dice handles class imbalance (small objects vs large background)
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


def train_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)

    train_loader = DataLoader(
        SyntheticSegmentationDataset(200), batch_size=8, shuffle=True
    )
    val_loader = DataLoader(
        SyntheticSegmentationDataset(50), batch_size=8, shuffle=False
    )

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    combined_loss = lambda logits, targets: (
        0.5 * bce(logits, targets) + 0.5 * dice(logits, targets)
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    os.makedirs("runs/unet", exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(15):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/15 [Train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/15 [Val]"):
                images, masks = images.to(device), masks.to(device)
                val_loss += combined_loss(model(images), masks).item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/15 — Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "runs/unet/best_unet.pth")
            print("  Saved best model.")

    print("\nDone. Best model: runs/unet/best_unet.pth")


if __name__ == "__main__":
    train_unet()
