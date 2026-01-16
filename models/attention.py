"""
Multi-Head Self-Attention module.
Focuses on tumor-relevant regions in feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import sys
sys.path.append('..')
from config import get_config

config = get_config()


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Computes attention over sequence elements to focus on
    relevant regions (e.g., tumor areas in MRI slices).
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            bias: Whether to use bias in projections.
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (B, L, D) where:
               B = batch size, L = sequence length, D = embed_dim.
            mask: Optional attention mask of shape (B, L, L).
            return_attention: Whether to return attention weights.
            
        Returns:
            Tuple of:
                - Output tensor of shape (B, L, D)
                - Attention weights of shape (B, H, L, L) if return_attention=True
        """
        B, L, D = x.shape
        
        # Pre-norm
        x_norm = self.norm(x)
        
        # Project to queries, keys, values
        Q = self.q_proj(x_norm)  # (B, L, D)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        
        # Reshape for multi-head attention: (B, L, H, head_dim) -> (B, H, L, head_dim)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, H, L, head_dim)
        
        # Reshape: (B, H, L, head_dim) -> (B, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)
        
        # Residual connection
        output = x + output
        
        if return_attention:
            return output, attn_weights
        return output, None


class SpatialAttention(nn.Module):
    """
    Spatial attention module for focusing on relevant image regions.
    Produces attention maps highlighting tumor locations.
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        """
        Initialize spatial attention.
        
        Args:
            in_channels: Number of input channels.
            reduction: Channel reduction factor.
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial attention.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Tuple of (attended_features, attention_map).
        """
        attn_map = self.attention(x)  # (B, 1, H, W)
        return x * attn_map, attn_map


class ChannelAttention(nn.Module):
    """
    Channel attention (Squeeze-and-Excitation style).
    Recalibrates channel-wise feature responses.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initialize channel attention.
        
        Args:
            in_channels: Number of input channels.
            reduction: Reduction ratio for squeeze operation.
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Attention-weighted tensor (B, C, H, W).
        """
        B, C, _, _ = x.shape
        
        # Global pooling
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines channel and spatial attention.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initialize CBAM.
        
        Args:
            in_channels: Number of input channels.
            reduction: Reduction ratio.
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels, reduction=8)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CBAM attention.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Tuple of (attended_features, spatial_attention_map).
        """
        # Channel attention first
        x = self.channel_attention(x)
        
        # Then spatial attention
        x, attn_map = self.spatial_attention(x)
        
        return x, attn_map


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence elements.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (B, L, D).
            
        Returns:
            Positionally-encoded tensor.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    # Test attention modules
    print("Testing Attention Modules...")
    
    # Test Multi-Head Self-Attention
    mhsa = MultiHeadSelfAttention(embed_dim=256, num_heads=8)
    x_seq = torch.randn(2, 16, 256)  # (B, L, D)
    out_seq, attn = mhsa(x_seq, return_attention=True)
    print(f"MHSA output shape: {out_seq.shape}")
    print(f"MHSA attention shape: {attn.shape}")
    
    # Test Spatial Attention
    spatial = SpatialAttention(64)
    x_spatial = torch.randn(2, 64, 32, 32)
    out_spatial, attn_map = spatial(x_spatial)
    print(f"Spatial attention output: {out_spatial.shape}, map: {attn_map.shape}")
    
    # Test CBAM
    cbam = CBAM(64)
    out_cbam, cbam_map = cbam(x_spatial)
    print(f"CBAM output: {out_cbam.shape}, map: {cbam_map.shape}")
    
    print("\nAttention tests passed!")
