"""
Complete Hybrid Quantum CNN Attention BiLSTM model.
Combines quantum feature extraction, CNNs, attention, and BiLSTM
for robust glioma detection in MRI scans.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import sys
sys.path.append('..')
from config import get_config
from .quantum_layer import QuantumConv2dFast, get_quantum_layer
from .attention import MultiHeadSelfAttention, CBAM, PositionalEncoding
from .bilstm import BiLSTMEncoder, BiLSTMWithAttention

config = get_config()


class CNNEncoder(nn.Module):
    """
    Classical CNN encoder for spatial feature extraction.
    Applied after quantum layer to extract hierarchical features.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        num_blocks: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize CNN encoder.
        
        Args:
            in_channels: Input channels (from quantum layer).
            base_channels: Base channel count (doubled each block).
            num_blocks: Number of conv blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        # Build conv blocks
        ch_in = in_channels
        ch_out = base_channels
        
        for i in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, 3, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_out, ch_out, 3, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(dropout)
                )
            )
            ch_in = ch_out
            ch_out = min(ch_out * 2, 512)
        
        self.out_channels = ch_in
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Feature maps (B, out_channels, H', W').
        """
        for block in self.blocks:
            x = block(x)
        return x


class HybridQCNN(nn.Module):
    """
    Hybrid Quantum CNN Attention BiLSTM model for glioma detection.
    
    Architecture:
        Input (B, num_slices, C, H, W)
        → QuantumConv (per slice)
        → CNN Encoder
        → CBAM Attention
        → Flatten → Sequence
        → Positional Encoding
        → Multi-Head Self-Attention
        → BiLSTM
        → Classification Head
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        num_slices: int = 16,
        image_size: int = 128,
        quantum_channels: int = 8,
        base_cnn_channels: int = 32,
        num_cnn_blocks: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        lstm_hidden: int = 128,
        dropout: float = 0.1,
        use_real_quantum: bool = False
    ):
        """
        Initialize the hybrid model.
        
        Args:
            num_classes: Number of output classes (2 for binary).
            in_channels: Input image channels.
            num_slices: Number of MRI slices per sample.
            image_size: Input image size.
            quantum_channels: Output channels from quantum layer.
            base_cnn_channels: Base channels for CNN encoder.
            num_cnn_blocks: Number of CNN blocks.
            embed_dim: Embedding dimension for attention.
            num_heads: Number of attention heads.
            lstm_hidden: BiLSTM hidden dimension.
            dropout: Dropout probability.
            use_real_quantum: Whether to use real PennyLane quantum circuits.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_slices = num_slices
        self.embed_dim = embed_dim
        
        # Quantum convolutional layer
        self.quantum_conv = get_quantum_layer(
            use_real_quantum=use_real_quantum,
            in_channels=in_channels,
            out_channels=quantum_channels,
            kernel_size=2,
            stride=2,
            n_qubits=4,
            n_layers=2
        )
        
        # Classical CNN encoder
        self.cnn_encoder = CNNEncoder(
            in_channels=quantum_channels,
            base_channels=base_cnn_channels,
            num_blocks=num_cnn_blocks,
            dropout=dropout
        )
        
        # CBAM attention for spatial focus
        self.cbam = CBAM(self.cnn_encoder.out_channels)
        
        # Calculate feature map size after CNN
        # Input: image_size, after quantum (stride=2): image_size//2
        # After CNN blocks: (image_size//2) / (2^num_cnn_blocks)
        self._feature_size = (image_size // 2) // (2 ** num_cnn_blocks)
        self._feature_dim = self.cnn_encoder.out_channels * self._feature_size * self._feature_size
        
        # Project to embedding dimension
        self.feature_proj = nn.Sequential(
            nn.Linear(self._feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for slice sequence
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=num_slices)
        
        # Multi-head self-attention
        self.self_attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # BiLSTM for sequence modeling
        self.bilstm = BiLSTMWithAttention(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )
        
        # Store attention maps for visualization
        self.attention_maps = {}
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape (B, num_slices, C, H, W).
            return_attention: Whether to return attention maps.
            
        Returns:
            Tuple of:
                - Logits of shape (B, num_classes)
                - Attention maps dict if return_attention=True
        """
        B, S, C, H, W = x.shape
        
        # Process each slice through quantum + CNN
        slice_features = []
        spatial_attns = []
        
        for s in range(S):
            # Quantum convolution
            q_out = self.quantum_conv(x[:, s])
            
            # CNN encoding
            cnn_out = self.cnn_encoder(q_out)
            
            # CBAM attention
            cbam_out, spatial_attn = self.cbam(cnn_out)
            
            # Flatten
            flat = cbam_out.view(B, -1)
            
            slice_features.append(flat)
            spatial_attns.append(spatial_attn)
        
        # Stack slice features: (B, S, feature_dim)
        features = torch.stack(slice_features, dim=1)
        
        # Project to embedding dimension
        features = self.feature_proj(features)
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Multi-head self-attention
        attn_out, self_attn_weights = self.self_attention(features, return_attention=True)
        
        # BiLSTM with attention
        lstm_out, lstm_attn = self.bilstm(attn_out, return_attention=True)
        
        # Classification
        logits = self.classifier(lstm_out)
        
        if return_attention:
            self.attention_maps = {
                'spatial': torch.stack(spatial_attns, dim=1),  # (B, S, 1, H', W')
                'self_attention': self_attn_weights,  # (B, num_heads, S, S)
                'lstm_attention': lstm_attn  # (B, S, 1)
            }
            return logits, self.attention_maps
        
        return logits, None
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.
        
        Args:
            x: Input tensor (B, S, C, H, W).
            
        Returns:
            Predicted class indices (B,).
        """
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input tensor (B, S, C, H, W).
            
        Returns:
            Class probabilities (B, num_classes).
        """
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embedding before classification.
        
        Args:
            x: Input tensor (B, S, C, H, W).
            
        Returns:
            Feature embedding (B, lstm_hidden * 2).
        """
        B, S, C, H, W = x.shape
        
        slice_features = []
        for s in range(S):
            q_out = self.quantum_conv(x[:, s])
            cnn_out = self.cnn_encoder(q_out)
            cbam_out, _ = self.cbam(cnn_out)
            slice_features.append(cbam_out.view(B, -1))
        
        features = torch.stack(slice_features, dim=1)
        features = self.feature_proj(features)
        features = self.pos_encoding(features)
        attn_out, _ = self.self_attention(features)
        lstm_out, _ = self.bilstm(attn_out)
        
        return lstm_out


class HybridQCNNLight(nn.Module):
    """
    Lightweight version of HybridQCNN for faster inference.
    Uses fewer layers and simpler attention.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        num_slices: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_slices = num_slices
        
        # Simple quantum-inspired encoding
        self.encoder = nn.Sequential(
            QuantumConv2dFast(in_channels, 16, 2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # Feature dimension: 64 * 4 * 4 = 1024
        self.fc_proj = nn.Linear(1024, hidden_dim)
        
        # Simple BiLSTM
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            batch_first=True, bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        B, S, C, H, W = x.shape
        
        # Encode each slice
        features = []
        for s in range(S):
            feat = self.encoder(x[:, s])
            features.append(feat.view(B, -1))
        
        # Stack and project
        features = torch.stack(features, dim=1)
        features = self.fc_proj(features)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(features)
        
        # Use last timestep
        logits = self.classifier(lstm_out[:, -1])
        
        return logits, None


def create_model(
    model_type: str = "full",
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create model.
    
    Args:
        model_type: 'full' or 'light'.
        num_classes: Number of output classes.
        **kwargs: Additional model arguments.
        
    Returns:
        Model instance.
    """
    if model_type == "full":
        return HybridQCNN(num_classes=num_classes, **kwargs)
    elif model_type == "light":
        return HybridQCNNLight(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the complete model
    print("Testing HybridQCNN...")
    
    # Create model
    model = HybridQCNN(
        num_classes=2,
        in_channels=3,
        num_slices=16,
        image_size=128,
        use_real_quantum=False  # Use fast approximation
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 16, 3, 128, 128)  # (B, S, C, H, W)
    
    print("\nTesting forward pass...")
    logits, attention = model(x, return_attention=True)
    print(f"Output logits shape: {logits.shape}")
    print(f"Spatial attention shape: {attention['spatial'].shape}")
    print(f"Self-attention shape: {attention['self_attention'].shape}")
    print(f"LSTM attention shape: {attention['lstm_attention'].shape}")
    
    # Test predictions
    probs = model.predict_proba(x)
    print(f"\nPrediction probabilities: {probs}")
    
    # Test light model
    print("\nTesting HybridQCNNLight...")
    light_model = HybridQCNNLight()
    light_params = sum(p.numel() for p in light_model.parameters())
    print(f"Light model parameters: {light_params:,}")
    
    light_logits, _ = light_model(x)
    print(f"Light model output shape: {light_logits.shape}")
    
    print("\nAll tests passed!")
