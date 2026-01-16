"""
Image preprocessing utilities for MRI data.
Handles loading, normalization, and format conversion.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Union, Optional
from pathlib import Path
import sys
sys.path.append('..')
from config import get_config

config = get_config()


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        path: Path to the image file.
        
    Returns:
        RGB image as numpy array (H, W, C).
    """
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int] = None,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image (H, W, C).
        size: Target size (height, width). Defaults to config.
        interpolation: OpenCV interpolation method.
        
    Returns:
        Resized image.
    """
    if size is None:
        size = config.data.image_size
    
    return cv2.resize(image, (size[1], size[0]), interpolation=interpolation)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = None,
    std: Tuple[float, float, float] = None
) -> np.ndarray:
    """
    Apply z-score normalization to image.
    
    Args:
        image: Input image (H, W, C) with values in [0, 255] or [0, 1].
        mean: Channel means for normalization.
        std: Channel standard deviations.
        
    Returns:
        Normalized image as float32.
    """
    if mean is None:
        mean = config.data.normalize_mean
    if std is None:
        std = config.data.normalize_std
    
    # Convert to float if needed
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    normalized = (image - mean) / std
    
    return normalized.astype(np.float32)


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = None,
    std: Tuple[float, float, float] = None
) -> np.ndarray:
    """
    Reverse normalization for visualization.
    
    Args:
        image: Normalized image.
        mean: Channel means used for normalization.
        std: Channel standard deviations.
        
    Returns:
        Denormalized image with values in [0, 255] as uint8.
    """
    if mean is None:
        mean = config.data.normalize_mean
    if std is None:
        std = config.data.normalize_std
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    denorm = image * std + mean
    denorm = np.clip(denorm * 255, 0, 255).astype(np.uint8)
    
    return denorm


def preprocess_image(
    image: Union[str, Path, np.ndarray],
    size: Tuple[int, int] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.
    
    Args:
        image: Path to image or numpy array.
        size: Target size (height, width).
        normalize: Whether to apply normalization.
        
    Returns:
        Preprocessed image as numpy array.
    """
    # Load if path
    if isinstance(image, (str, Path)):
        image = load_image(image)
    
    # Resize
    image = resize_image(image, size)
    
    # Normalize
    if normalize:
        image = normalize_image(image)
    
    return image


def preprocess_for_model(
    image: Union[str, Path, np.ndarray],
    num_slices: int = None
) -> torch.Tensor:
    """
    Preprocess image for model input with slice stacking.
    
    Args:
        image: Input image path or array.
        num_slices: Number of slices to create for BiLSTM.
        
    Returns:
        Tensor of shape (num_slices, C, H, W).
    """
    if num_slices is None:
        num_slices = config.model.num_slices
    
    # Preprocess
    img = preprocess_image(image)
    
    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    
    # Create slice stack by repeating (simulating MRI slices)
    # In production, actual MRI slices would be loaded
    stack = tensor.unsqueeze(0).repeat(num_slices, 1, 1, 1)
    
    return stack


def create_mri_stack(
    images: list,
    target_slices: int = None
) -> torch.Tensor:
    """
    Create MRI stack from multiple images.
    
    Args:
        images: List of image paths or arrays.
        target_slices: Number of slices to output.
        
    Returns:
        Tensor of shape (target_slices, C, H, W).
    """
    if target_slices is None:
        target_slices = config.model.num_slices
    
    processed = []
    for img in images:
        processed.append(preprocess_image(img))
    
    # Stack and convert
    stack = np.stack(processed)
    tensor = torch.from_numpy(stack).permute(0, 3, 1, 2).float()
    
    # Pad or truncate to target slices
    current_slices = tensor.shape[0]
    if current_slices < target_slices:
        # Pad by repeating last slice
        padding = tensor[-1:].repeat(target_slices - current_slices, 1, 1, 1)
        tensor = torch.cat([tensor, padding], dim=0)
    elif current_slices > target_slices:
        # Uniformly sample
        indices = torch.linspace(0, current_slices - 1, target_slices).long()
        tensor = tensor[indices]
    
    return tensor


def enhance_mri_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE for MRI contrast enhancement.
    
    Args:
        image: Input MRI image (H, W, C).
        
    Returns:
        Contrast-enhanced image.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


if __name__ == "__main__":
    # Test preprocessing
    import matplotlib.pyplot as plt
    
    # Create dummy image
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test preprocessing pipeline
    processed = preprocess_image(dummy)
    print(f"Processed shape: {processed.shape}")
    print(f"Processed dtype: {processed.dtype}")
    print(f"Processed range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Test model preparation
    model_input = preprocess_for_model(dummy)
    print(f"\nModel input shape: {model_input.shape}")
    print(f"Expected: ({config.model.num_slices}, 3, {config.data.image_size[0]}, {config.data.image_size[1]})")
