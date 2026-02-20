"""
Helper utilities for training and inference.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available.
        
    Returns:
        torch.device object.
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model.
        trainable_only: Whether to count only trainable parameters.
        
    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str = "checkpoints",
    filename: str = None,
    is_best: bool = False,
    scheduler: Optional[Any] = None,
    extra_info: Optional[Dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch.
        metrics: Metric values.
        checkpoint_dir: Directory for checkpoints.
        filename: Checkpoint filename.
        is_best: Whether this is the best model.
        scheduler: Learning rate scheduler.
        extra_info: Additional information to save.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if extra_info is not None:
        checkpoint['extra_info'] = extra_info
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model separately
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = None
) -> Tuple[nn.Module, int, Dict[str, float]]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint file.
        optimizer: Optional optimizer to load state.
        scheduler: Optional scheduler to load state.
        device: Device to load model to.
        
    Returns:
        Tuple of (model, epoch, metrics).
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    
    return model, epoch, metrics


def export_to_torchscript(
    model: nn.Module,
    export_path: str,
    example_input: torch.Tensor = None
):
    """
    Export model to TorchScript for production.
    
    Args:
        model: Model to export.
        export_path: Path to save exported model.
        example_input: Example input for tracing.
    """
    model.eval()
    
    if example_input is None:
        # Default input shape: (1, 16, 3, 128, 128)
        example_input = torch.randn(1, 16, 3, 128, 128)
    
    if torch.cuda.is_available():
        model = model.cuda()
        example_input = example_input.cuda()
    
    # Use scripting instead of tracing for models with control flow
    try:
        scripted_model = torch.jit.script(model)
    except:
        scripted_model = torch.jit.trace(model, example_input)
    
    scripted_model.save(export_path)
    logger.info(f"Exported TorchScript model to {export_path}")


def export_to_onnx(
    model: nn.Module,
    export_path: str,
    example_input: torch.Tensor = None,
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: dict = None
):
    """
    Export model to ONNX format.
    
    Args:
        model: Model to export.
        export_path: Path to save ONNX model.
        example_input: Example input for tracing.
        input_names: Names for input tensors.
        output_names: Names for output tensors.
        dynamic_axes: Dynamic axes for variable batch size.
    """
    model.eval()
    
    if example_input is None:
        example_input = torch.randn(1, 16, 3, 128, 128)
    
    if input_names is None:
        input_names = ['mri_input']
    
    if output_names is None:
        output_names = ['prediction']
    
    if dynamic_axes is None:
        dynamic_axes = {
            'mri_input': {0: 'batch_size'},
            'prediction': {0: 'batch_size'}
        }
    
    torch.onnx.export(
        model,
        example_input,
        export_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14
    )
    
    logger.info(f"Exported ONNX model to {export_path}")


class EarlyStopping:
    """
    Early stopping handler for training.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class AverageMeter:
    """
    Computes and stores average and current value.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def save_training_config(
    config: Dict,
    save_path: str
):
    """
    Save training configuration.
    
    Args:
        config: Configuration dictionary.
        save_path: Path to save config.
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Saved config to {save_path}")


def load_training_config(load_path: str) -> Dict:
    """
    Load training configuration.
    
    Args:
        load_path: Path to config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(load_path, 'r') as f:
        config = json.load(f)
    
    return config


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test seed setting
    set_seed(42)
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='min')
    losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88]
    
    for epoch, loss in enumerate(losses):
        should_stop = early_stop(loss)
        print(f"Epoch {epoch}: loss={loss}, counter={early_stop.counter}, stop={should_stop}")
    
    # Test average meter
    meter = AverageMeter("loss")
    for val in [1.0, 0.5, 0.3]:
        meter.update(val)
    print(f"Average: {meter.avg:.4f}")
    
    print("\nUtility tests passed!")
