"""
Evaluation metrics for glioma classification.
Includes accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "binary"
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for positive class.
        average: Averaging method ('binary', 'macro', 'weighted').
        
    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Add AUC if probabilities provided
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            else:
                metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except ValueError:
            metrics['auc'] = 0.0
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Names of classes.
        normalize: Whether to normalize by row.
        
    Returns:
        Confusion matrix array.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    return cm


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Names of classes.
        normalize: Whether to normalize.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    cm = compute_confusion_matrix(y_true, y_pred, class_names, normalize)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compute_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC.
    
    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities.
        
    Returns:
        Tuple of (fpr, tpr, auc_value).
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_value = roc_auc_score(y_true, y_proba)
    
    return fpr, tpr, auc_value


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fpr, tpr, auc_value = compute_roc_auc(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_value:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.fill_between(recall, precision, alpha=0.2)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


class MetricsTracker:
    """
    Track metrics over training epochs.
    """
    
    def __init__(self, metrics_names: List[str] = None):
        """
        Initialize tracker.
        
        Args:
            metrics_names: List of metric names to track.
        """
        self.metrics_names = metrics_names or ['loss', 'accuracy', 'f1', 'auc']
        self.history = {name: {'train': [], 'val': []} for name in self.metrics_names}
        self.best_metrics = {name: {'train': 0.0, 'val': 0.0} for name in self.metrics_names}
        self.best_metrics['loss'] = {'train': float('inf'), 'val': float('inf')}
    
    def update(self, metrics: Dict[str, float], phase: str = 'train'):
        """
        Update metrics history.
        
        Args:
            metrics: Dictionary of metric values.
            phase: 'train' or 'val'.
        """
        for name, value in metrics.items():
            if name in self.history:
                self.history[name][phase].append(value)
                
                # Update best
                if name == 'loss':
                    if value < self.best_metrics[name][phase]:
                        self.best_metrics[name][phase] = value
                else:
                    if value > self.best_metrics[name][phase]:
                        self.best_metrics[name][phase] = value
    
    def get_last(self, phase: str = 'val') -> Dict[str, float]:
        """Get last recorded metrics."""
        return {name: values[phase][-1] if values[phase] else 0.0 
                for name, values in self.history.items()}
    
    def is_best(self, metric_name: str = 'val_loss') -> bool:
        """Check if current epoch is best for given metric."""
        phase = 'val' if 'val' in metric_name else 'train'
        name = metric_name.replace('val_', '').replace('train_', '')
        
        if not self.history[name][phase]:
            return False
        
        current = self.history[name][phase][-1]
        
        if name == 'loss':
            return current <= self.best_metrics[name][phase]
        return current >= self.best_metrics[name][phase]
    
    def plot_history(
        self,
        metrics: List[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            metrics: Metrics to plot.
            save_path: Path to save figure.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure.
        """
        metrics = metrics or ['loss', 'accuracy', 'f1']
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        for ax, name in zip(axes, metrics):
            train_values = self.history.get(name, {}).get('train', [])
            val_values = self.history.get(name, {}).get('val', [])
            
            epochs = range(1, len(train_values) + 1)
            
            if train_values:
                ax.plot(epochs, train_values, 'b-', label='Train')
            if val_values:
                ax.plot(epochs, val_values, 'r-', label='Val')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(name.capitalize())
            ax.set_title(f'{name.capitalize()} over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Generate sample predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
    y_proba = np.array([0.1, 0.6, 0.9, 0.8, 0.2, 0.75, 0.15, 0.4, 0.85, 0.3])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    for epoch in range(5):
        tracker.update({'loss': 1.0 - epoch * 0.1, 'accuracy': 0.5 + epoch * 0.1}, 'train')
        tracker.update({'loss': 1.1 - epoch * 0.08, 'accuracy': 0.45 + epoch * 0.08}, 'val')
    
    print(f"\nBest metrics: {tracker.best_metrics}")
