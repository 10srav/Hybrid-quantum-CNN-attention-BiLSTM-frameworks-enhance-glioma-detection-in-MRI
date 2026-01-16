"""
Configuration module for Hybrid Quantum CNN Attention BiLSTM Framework.
Contains all hyperparameters, paths, and settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

# Base paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class QuantumConfig:
    """Quantum circuit configuration."""
    num_qubits: int = 4
    num_layers: int = 2
    device_name: str = "default.qubit"
    
    # Gate angles
    rx_scale: float = 3.14159  # π for RX gates
    crz_angle: float = 1.5708  # π/2 for CRZ gates
    crx_angle: float = 1.5708  # π/2 for CRX gates
    
    # Patch settings for quantum convolution
    patch_size: int = 2
    stride: int = 2


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Input dimensions
    input_channels: int = 3
    input_height: int = 128
    input_width: int = 128
    num_slices: int = 16  # Number of MRI slices per stack
    
    # CNN layers
    conv1_out_channels: int = 32
    conv2_out_channels: int = 64
    kernel_size: int = 3
    pool_size: int = 2
    
    # Attention
    embed_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    
    # BiLSTM
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_bidirectional: bool = True
    
    # Classification
    num_classes: int = 2  # Glioma vs Non-Glioma
    
    @property
    def lstm_output_size(self) -> int:
        """Calculate BiLSTM output size."""
        return self.lstm_hidden_size * (2 if self.lstm_bidirectional else 1)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 50
    
    # Learning rate scheduling
    scheduler_type: str = "cosine"  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_every_n_epochs: int = 5
    checkpoint_dir: Path = MODEL_DIR
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "hybrid-qcnn-glioma"
    wandb_entity: str = None  # Set your WandB username
    log_every_n_steps: int = 10


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    # Paths
    raw_data_dir: Path = DATA_DIR / "raw"
    train_dir: Path = DATA_DIR / "raw" / "Training"
    test_dir: Path = DATA_DIR / "raw" / "Testing"
    
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Preprocessing
    image_size: Tuple[int, int] = (128, 128)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Augmentation
    use_augmentation: bool = True
    rotation_limit: int = 30
    flip_prob: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    
    # Class labels
    class_names: List[str] = field(default_factory=lambda: ["non_glioma", "glioma"])
    glioma_folder_name: str = "glioma"


@dataclass
class APIConfig:
    """FastAPI configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    allowed_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000"
    ])
    
    # Model path
    model_path: Path = MODEL_DIR / "best_model.pth"


@dataclass
class Config:
    """Main configuration container."""
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


if __name__ == "__main__":
    # Print configuration summary
    cfg = get_config()
    print("=" * 60)
    print("Hybrid Quantum CNN Configuration")
    print("=" * 60)
    print(f"\nQuantum Config:")
    print(f"  - Qubits: {cfg.quantum.num_qubits}")
    print(f"  - Device: {cfg.quantum.device_name}")
    print(f"\nModel Config:")
    print(f"  - Input: {cfg.model.num_slices}x{cfg.model.input_channels}x{cfg.model.input_height}x{cfg.model.input_width}")
    print(f"  - Classes: {cfg.model.num_classes}")
    print(f"\nTraining Config:")
    print(f"  - LR: {cfg.training.learning_rate}")
    print(f"  - Batch Size: {cfg.training.batch_size}")
    print(f"  - Epochs: {cfg.training.num_epochs}")
