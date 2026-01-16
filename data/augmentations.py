"""
Augmentation pipelines using Albumentations library.
Provides robust data augmentation for MRI images.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append('..')
from config import get_config

config = get_config()


def get_train_transforms() -> A.Compose:
    """
    Get training augmentation pipeline.
    Includes geometric and photometric augmentations for robustness.
    """
    return A.Compose([
        # Resize to target size
        A.Resize(
            height=config.data.image_size[0],
            width=config.data.image_size[1]
        ),
        
        # Geometric augmentations
        A.Rotate(
            limit=config.data.rotation_limit,
            p=0.5,
            border_mode=0  # Constant border
        ),
        A.HorizontalFlip(p=config.data.flip_prob),
        A.VerticalFlip(p=config.data.flip_prob * 0.5),
        
        # Affine transforms
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            p=0.5,
            border_mode=0
        ),
        
        # Photometric augmentations
        A.RandomBrightnessContrast(
            brightness_limit=config.data.brightness_limit,
            contrast_limit=config.data.contrast_limit,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        
        # Noise and blur for robustness
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # MRI-specific augmentations
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        
        # Elastic deformation (simulates brain tissue variations)
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            p=0.2
        ),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=config.data.normalize_mean,
            std=config.data.normalize_std
        ),
        ToTensorV2()
    ])


def get_val_transforms() -> A.Compose:
    """
    Get validation/test augmentation pipeline.
    Only includes resize and normalization.
    """
    return A.Compose([
        A.Resize(
            height=config.data.image_size[0],
            width=config.data.image_size[1]
        ),
        A.Normalize(
            mean=config.data.normalize_mean,
            std=config.data.normalize_std
        ),
        ToTensorV2()
    ])


def get_inference_transforms() -> A.Compose:
    """
    Get inference-time transforms.
    Same as validation transforms but can be extended for TTA.
    """
    return get_val_transforms()


def get_tta_transforms() -> list:
    """
    Get Test-Time Augmentation (TTA) transforms.
    Returns a list of different augmentation configs for ensemble.
    """
    base_transforms = [
        A.Resize(
            height=config.data.image_size[0],
            width=config.data.image_size[1]
        ),
        A.Normalize(
            mean=config.data.normalize_mean,
            std=config.data.normalize_std
        ),
        ToTensorV2()
    ]
    
    return [
        # Original
        A.Compose(base_transforms),
        
        # Horizontal flip
        A.Compose([
            A.Resize(config.data.image_size[0], config.data.image_size[1]),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=config.data.normalize_mean, std=config.data.normalize_std),
            ToTensorV2()
        ]),
        
        # Vertical flip
        A.Compose([
            A.Resize(config.data.image_size[0], config.data.image_size[1]),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=config.data.normalize_mean, std=config.data.normalize_std),
            ToTensorV2()
        ]),
        
        # Rotation 90
        A.Compose([
            A.Resize(config.data.image_size[0], config.data.image_size[1]),
            A.Rotate(limit=(90, 90), p=1.0),
            A.Normalize(mean=config.data.normalize_mean, std=config.data.normalize_std),
            ToTensorV2()
        ]),
    ]


if __name__ == "__main__":
    import numpy as np
    
    # Test transforms
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_result = train_transform(image=dummy_image)
    val_result = val_transform(image=dummy_image)
    
    print(f"Training transform output shape: {train_result['image'].shape}")
    print(f"Validation transform output shape: {val_result['image'].shape}")
