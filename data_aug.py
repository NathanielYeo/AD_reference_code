import cv2
import os
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast
)

def apply_and_save_augmentation(augmentation, image, save_path):
    augmented_image = augmentation(image=image)['image']
    cv2.imwrite(save_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

def get_individual_augmentations(cls):
    if cls == "metal_nut":
        # In total 5
        return [
            # Translate
            ("translate", ShiftScaleRotate(shift_limit=0.02, rotate_limit=0, p=1)),  # Translates by up to 10% of image's height/width
            # Rotate 10,10 
            ("rotate", ShiftScaleRotate(shift_limit=0, rotate_limit=(-10, 10), p=1)),
            # Rotate 90
            # Not implemented 
            # ("rotate_90", ShiftScaleRotate(shift_limit=0, rotate_limit=90, p=1)),
            # Add brightness 10,10
            ("brightness", RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), p=1)),
            # Change Contrast
            ("contrast", RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0.9, 1.1), p=1))
        ]
    elif cls == "screw":
        # Compose HorizontalFlip and VerticalFlip together
        composed_aug = Compose([HorizontalFlip(p=1), VerticalFlip(p=1)])
        return [
            # Compose flip
            ("hflip_vflip", composed_aug),
            # Translate
            ("translate", ShiftScaleRotate(shift_limit=0.02, rotate_limit=0, p=1)),  # Translates by up to 10% of image's height/width
            # Rotate 10,10 
            ("rotate", ShiftScaleRotate(shift_limit=0, rotate_limit=(-10, 10), p=1)),
            # Rotate 90
            # ("rotate_90", ShiftScaleRotate(shift_limit=0, rotate_limit=90, p=1)),
            # Zoom
            ("zoom", ShiftScaleRotate(shift_limit=0, rotate_limit=0, scale_limit=(0.02, 0.02), p=1)),  # Zooms by a factor of 0.98 to 1.02
            # Add brightness 10,10
            ("brightness", RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), p=1)),
            # Change Contrast
            ("contrast", RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0.9, 1.1), p=1))
        ]
    else:
        return []


image_dir = "D:\\Documents\\FYP\\patchcore\\anomaly_detection\\datasets\\\pseudo_s\\screw_5\\train\\good"
cls = "screw"  # change this to "metal_nut" or "screw" for the respective class

# Loop through all the files in the folder
for filename in os.listdir(image_dir):
    # Skip files that were saved as original
    if filename.endswith('_original.png') or filename.endswith('_original.jpg') or filename.endswith('_original.jpeg'):
        continue
    
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        individual_augmentations = get_individual_augmentations(cls)

        for aug_name, aug in individual_augmentations:
            base, ext = os.path.splitext(filename)
            save_path = os.path.join(image_dir, f"{base}_{aug_name}{ext}")
            apply_and_save_augmentation(aug, image, save_path)
