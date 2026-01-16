"""
PyTorch Dataset class for MRI brain tumor images.
Supports binary classification (glioma vs non-glioma) and multi-class.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Callable
import cv2
import random
import sys
sys.path.append('..')
from config import get_config
from .augmentations import get_train_transforms, get_val_transforms

config = get_config()


class MRIDataset(Dataset):
    """
    Dataset for loading MRI brain tumor images.
    
    Supports the Kaggle Brain Tumor MRI Dataset structure:
        data/raw/Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
    """
    
    def __init__(
        self,
        root_dir: str = None,
        transform: Optional[Callable] = None,
        mode: str = "binary",  # 'binary' or 'multiclass'
        num_slices: int = None,
        split: str = "train"  # 'train', 'val', 'test'
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing class folders.
            transform: Albumentations transform pipeline.
            mode: 'binary' for glioma vs non-glioma, 'multiclass' for all classes.
            num_slices: Number of slices per MRI stack.
            split: Dataset split type.
        """
        self.root_dir = Path(root_dir) if root_dir else config.data.train_dir
        self.transform = transform
        self.mode = mode
        self.num_slices = num_slices or config.model.num_slices
        self.split = split
        
        # Define class mappings
        if mode == "binary":
            self.class_to_idx = {"non_glioma": 0, "glioma": 1}
            self.classes = ["non_glioma", "glioma"]
        else:
            self.class_to_idx = {
                "glioma": 0,
                "meningioma": 1,
                "notumor": 2,
                "pituitary": 3
            }
            self.classes = list(self.class_to_idx.keys())
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_class_distribution()
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths and their labels."""
        samples = []
        
        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # Iterate through class folders
        for class_folder in self.root_dir.iterdir():
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name.lower()
            
            # Determine label based on mode
            if self.mode == "binary":
                if "glioma" in class_name and class_name != "notumor":
                    label = 1  # glioma
                else:
                    label = 0  # non_glioma
            else:
                if class_name in self.class_to_idx:
                    label = self.class_to_idx[class_name]
                else:
                    continue  # Skip unknown classes
            
            # Collect images
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in extensions:
                    samples.append((img_path, label))
        
        # Sort samples for reproducibility (shuffle happens in DataLoader)
        samples.sort(key=lambda x: str(x[0]))

        return samples
    
    def _print_class_distribution(self):
        """Print class distribution for debugging."""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.classes[label] if self.mode == "binary" else list(self.class_to_idx.keys())[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"Class distribution ({self.split}):")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Tuple of (image_stack, label) where:
                - image_stack: Tensor of shape (num_slices, C, H, W)
                - label: Tensor scalar
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: resize and normalize
            image = cv2.resize(image, config.data.image_size[::-1])
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Create slice stack (simulate MRI volume)
        # In real scenario, would load actual 3D MRI slices
        if isinstance(image, torch.Tensor):
            image_stack = image.unsqueeze(0).repeat(self.num_slices, 1, 1, 1)
        else:
            image_stack = torch.from_numpy(image).unsqueeze(0).repeat(self.num_slices, 1, 1, 1)
        
        # Add slight variations to simulate different slices
        for i in range(self.num_slices):
            noise_factor = 0.02 * (i - self.num_slices // 2) / self.num_slices
            image_stack[i] = image_stack[i] + noise_factor * torch.randn_like(image_stack[i])
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_stack, label_tensor
    
    def get_sample_weights(self) -> torch.Tensor:
        """Calculate sample weights for handling class imbalance."""
        class_counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Inverse frequency weighting
        weights = 1.0 / class_counts
        sample_weights = torch.tensor([weights[label] for _, label in self.samples])
        
        return sample_weights


def create_data_loaders(
    train_dir: str = None,
    test_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
    val_split: float = 0.1,
    mode: str = "binary"
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        train_dir: Training data directory.
        test_dir: Test data directory.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        val_split: Fraction of training data for validation.
        mode: 'binary' or 'multiclass'.
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders.
    """
    train_dir = train_dir or str(config.data.train_dir)
    test_dir = test_dir or str(config.data.test_dir)
    batch_size = batch_size or config.training.batch_size
    num_workers = num_workers or config.training.num_workers
    
    # Create datasets
    train_dataset = MRIDataset(
        root_dir=train_dir,
        transform=get_train_transforms(),
        mode=mode,
        split="train"
    )
    
    # Create validation dataset with val transforms (separate instance)
    val_dataset = MRIDataset(
        root_dir=train_dir,
        transform=get_val_transforms(),
        mode=mode,
        split="val"
    )

    # Split training into train/val using same seed for reproducibility
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size

    # Split both datasets with the same seed to get same indices
    train_subset, _ = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    _, val_subset = random_split(
        val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Test dataset
    test_dataset = MRIDataset(
        root_dir=test_dir,
        transform=get_val_transforms(),
        mode=mode,
        split="test"
    ) if Path(test_dir).exists() else None
    
    # Create data loaders
    loaders = {
        'train': DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=config.training.pin_memory,
            drop_last=True
        ),
        'val': DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.training.pin_memory
        )
    }
    
    if test_dataset:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.training.pin_memory
        )
    
    return loaders


if __name__ == "__main__":
    # Test dataset loading
    print("Testing MRIDataset...")
    
    # Create dummy data structure for testing
    test_dir = Path("./test_data")
    for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)
        # Create dummy images
        for i in range(5):
            dummy_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(test_dir / class_name / f"img_{i}.jpg"), dummy_img)
    
    # Test dataset
    dataset = MRIDataset(
        root_dir=str(test_dir),
        transform=get_train_transforms(),
        mode="binary"
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    img_stack, label = dataset[0]
    print(f"Image stack shape: {img_stack.shape}")
    print(f"Label: {label}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print("\nTest completed successfully!")
