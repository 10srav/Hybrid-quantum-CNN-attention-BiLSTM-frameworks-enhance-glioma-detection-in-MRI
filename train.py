"""
Training script for Hybrid Quantum CNN Attention BiLSTM model.
Supports WandB logging, checkpointing, and learning rate scheduling.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from data.dataset import MRIDataset, create_data_loaders
from data.augmentations import get_train_transforms, get_val_transforms
from models.hybrid_qcnn import HybridQCNN, create_model
from utils.metrics import calculate_metrics, MetricsTracker, plot_confusion_matrix, plot_roc_curve
from utils.helpers import (
    set_seed, get_device, save_checkpoint, load_checkpoint,
    EarlyStopping, AverageMeter, count_parameters
)

# Optional WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available. Install with: pip install wandb")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Hybrid QCNN for Glioma Detection")
    
    # Data
    parser.add_argument('--train_dir', type=str, default='data/raw/Training',
                        help='Training data directory')
    parser.add_argument('--test_dir', type=str, default='data/raw/Testing',
                        help='Test data directory')
    
    # Model
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 'light'],
                        help='Model type: full or light')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--use_real_quantum', action='store_true',
                        help='Use real PennyLane quantum circuits')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use WandB for logging')
    parser.add_argument('--wandb_project', type=str, default='hybrid-qcnn-glioma',
                        help='WandB project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """
    Train model for one epoch.
    
    Returns:
        Dictionary of training metrics.
    """
    model.train()
    
    loss_meter = AverageMeter('Loss')
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), images.size(0))
        
        # Collect predictions
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = loss_meter.avg
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    """
    Validate model.
    
    Returns:
        Dictionary of validation metrics.
    """
    model.eval()
    
    loss_meter = AverageMeter('Loss')
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        
        # Update metrics
        loss_meter.update(loss.item(), images.size(0))
        
        # Collect predictions
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = loss_meter.avg
    
    return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = f"hybrid_qcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize WandB
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Hybrid Quantum CNN Training")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    
    # Create data loaders
    print("\nLoading data...")
    
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    
    if not train_dir.exists():
        print(f"Warning: Training directory not found: {train_dir}")
        print("Creating dummy data for demonstration...")
        # Create dummy data
        create_dummy_data(train_dir)
    
    try:
        loaders = create_data_loaders(
            train_dir=str(train_dir),
            test_dir=str(test_dir) if test_dir.exists() else str(train_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        train_loader = loaders['train']
        val_loader = loaders['val']
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using dummy data for demonstration...")
        train_loader, val_loader = create_dummy_loaders(args.batch_size)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    config = get_config()
    
    model = create_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        use_real_quantum=args.use_real_quantum
    )
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer — weighted to handle class imbalance (non-glioma:glioma ~= 3:1)
    class_weights = torch.tensor([1.0, 3.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        model, start_epoch, _ = load_checkpoint(
            model, args.resume, optimizer, scheduler, device
        )
        print(f"Resumed from epoch {start_epoch}")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='min')
    
    # Metrics tracker
    tracker = MetricsTracker(['loss', 'accuracy', 'f1', 'auc'])
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        
        # Validate
        val_metrics, val_labels, val_preds, val_probs = validate(
            model, val_loader, criterion, device, epoch + 1
        )
        
        # Update scheduler
        scheduler.step()
        
        # Update tracker
        tracker.update(train_metrics, 'train')
        tracker.update(val_metrics, 'val')
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        
        # Log to WandB
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/f1': train_metrics['f1'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/f1': val_metrics['f1'],
                'val/auc': val_metrics.get('auc', 0),
                'lr': scheduler.get_last_lr()[0]
            })
        
        # Check for best model
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            print(f"[BEST] New best model! Val Loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=val_metrics,
            checkpoint_dir=str(checkpoint_dir),
            filename=f"checkpoint_epoch_{epoch+1}.pth",
            is_best=is_best,
            scheduler=scheduler
        )
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Load best model
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        model, _, best_metrics = load_checkpoint(model, str(best_model_path))
        print(f"\nBest model metrics:")
        for name, value in best_metrics.items():
            print(f"  {name}: {value:.4f}")
    
    # Save final plots
    print("\nSaving plots...")
    try:
        # Plot confusion matrix
        plot_confusion_matrix(
            val_labels, val_preds,
            class_names=['Non-Glioma', 'Glioma'],
            save_path=str(checkpoint_dir / "confusion_matrix.png")
        )
        
        # Plot ROC curve
        if len(np.unique(val_labels)) == 2:
            plot_roc_curve(
                val_labels, val_probs,
                save_path=str(checkpoint_dir / "roc_curve.png")
            )
        
        # Plot training history
        tracker.plot_history(
            metrics=['loss', 'accuracy', 'f1'],
            save_path=str(checkpoint_dir / "training_history.png")
        )
        
        print(f"Plots saved to {checkpoint_dir}")
    except Exception as e:
        print(f"Error saving plots: {e}")
    
    # Finish WandB
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print("\nDone!")


def create_dummy_data(data_dir: Path):
    """Create dummy data for testing."""
    import cv2
    
    for class_name in ['glioma', 'notumor']:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(10):
            img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"img_{i}.jpg"), img)
    
    print(f"Created dummy data in {data_dir}")


def create_dummy_loaders(batch_size: int):
    """Create dummy data loaders for testing."""
    from torch.utils.data import TensorDataset
    
    # Create dummy tensors
    X_train = torch.randn(32, 16, 3, 128, 128)
    y_train = torch.randint(0, 2, (32,))
    
    X_val = torch.randn(8, 16, 3, 128, 128)
    y_val = torch.randint(0, 2, (8,))
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    main()
