import albumentations as A
import cv2


def get_unet_augmentation(target_size=(256, 256)):
    return A.Compose(
        [
            A.Resize(target_size[0], target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=30,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            # Randomly drops patches — forces model to not rely on a single region
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        ]
    )


def get_yolo_augmentation(target_size=(640, 640)):
    return A.Compose(
        [
            A.Resize(target_size[0], target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


if __name__ == "__main__":
    import numpy as np

    dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
    dummy_mask = np.zeros((500, 500), dtype=np.uint8)

    unet_transform = get_unet_augmentation(target_size=(256, 256))
    augmented = unet_transform(image=dummy_img, mask=dummy_mask)
    print(f"U-Net — image: {augmented['image'].shape}, mask: {augmented['mask'].shape}")

    dummy_bboxes = [[0.5, 0.5, 0.2, 0.2]]
    dummy_labels = [0]

    yolo_transform = get_yolo_augmentation(target_size=(640, 640))
    aug_yolo = yolo_transform(
        image=dummy_img, bboxes=dummy_bboxes, class_labels=dummy_labels
    )
    print(f"YOLO — image: {aug_yolo['image'].shape}, bboxes: {aug_yolo['bboxes']}")
