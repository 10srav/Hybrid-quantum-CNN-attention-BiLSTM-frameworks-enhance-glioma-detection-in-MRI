"""
Bidirectional LSTM module for sequential modeling.
Captures temporal dependencies across MRI slices.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import sys
sys.path.append('..')
from config import get_config

config = get_config()


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for sequence modeling.
    
    Processes sequences of MRI slice features to capture
    temporal/spatial dependencies in volumetric data.
    """
    
    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        """
        Initialize BiLSTM encoder.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: LSTM hidden state dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability (applied between layers).
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection (optional, for dimension matching)
        self.input_proj = nn.Linear(input_size, input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        
        # Output dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights with orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for stable training
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
    
    @property
    def output_size(self) -> int:
        """Return output feature dimension."""
        return self.hidden_size * self.num_directions
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process sequence through BiLSTM.
        
        Args:
            x: Input sequence of shape (B, L, D) where:
               B = batch size, L = sequence length, D = input_size.
            hidden: Optional initial hidden state (h_0, c_0).
            return_hidden: Whether to return final hidden states.
            
        Returns:
            Tuple of:
                - Output tensor of shape (B, L, hidden_size * num_directions)
                - Final hidden state if return_hidden=True
        """
        B, L, D = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                B,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype
            )
            c_0 = torch.zeros_like(h_0)
            hidden = (h_0, c_0)
        
        # Forward pass through LSTM
        output, (h_n, c_n) = self.lstm(x, hidden)
        
        # Apply layer norm and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        if return_hidden:
            return output, (h_n, c_n)
        return output, None
    
    def get_sequence_embedding(
        self,
        x: torch.Tensor,
        method: str = "last"
    ) -> torch.Tensor:
        """
        Get fixed-size embedding for entire sequence.
        
        Args:
            x: Input sequence (B, L, D).
            method: Pooling method - 'last', 'mean', 'max', or 'attention'.
            
        Returns:
            Sequence embedding of shape (B, hidden_size * num_directions).
        """
        output, _ = self.forward(x)
        
        if method == "last":
            # Use last timestep
            embedding = output[:, -1, :]
        elif method == "mean":
            # Mean pooling
            embedding = output.mean(dim=1)
        elif method == "max":
            # Max pooling
            embedding = output.max(dim=1)[0]
        elif method == "attention":
            # Simple self-attention pooling
            attention_weights = torch.softmax(output.sum(dim=-1), dim=-1)
            embedding = (output * attention_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {method}")
        
        return embedding


class BiLSTMWithAttention(nn.Module):
    """
    BiLSTM with attention mechanism for weighted sequence aggregation.
    """
    
    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize BiLSTM with attention.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.bilstm = BiLSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        lstm_output_size = hidden_size * 2
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(lstm_output_size, lstm_output_size)
    
    @property
    def output_size(self) -> int:
        return self.bilstm.output_size
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process sequence with attention-weighted output.
        
        Args:
            x: Input sequence (B, L, D).
            return_attention: Whether to return attention weights.
            
        Returns:
            Tuple of:
                - Context vector (B, hidden_size * 2)
                - Attention weights (B, L, 1) if return_attention=True
        """
        # BiLSTM forward
        lstm_out, _ = self.bilstm(x)  # (B, L, hidden*2)
        
        # Compute attention weights
        attn_weights = self.attention(lstm_out)  # (B, L, 1)
        
        # Weighted sum
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, hidden*2)
        
        # Project output
        output = self.output_proj(context)
        
        if return_attention:
            return output, attn_weights
        return output, None


class StackedBiLSTM(nn.Module):
    """
    Stacked BiLSTM with residual connections for deeper sequence modeling.
    """
    
    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 128,
        num_stacks: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize stacked BiLSTM.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: LSTM hidden dimension.
            num_stacks: Number of stacked BiLSTM blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.num_stacks = num_stacks
        self.hidden_size = hidden_size
        
        # First layer
        self.first_lstm = BiLSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Stacked layers
        self.stacked_lstms = nn.ModuleList([
            BiLSTMEncoder(
                input_size=hidden_size * 2,
                hidden_size=hidden_size,
                dropout=dropout
            )
            for _ in range(num_stacks - 1)
        ])
    
    @property
    def output_size(self) -> int:
        return self.hidden_size * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through stacked BiLSTM layers.
        
        Args:
            x: Input sequence (B, L, D).
            
        Returns:
            Output sequence (B, L, hidden_size * 2).
        """
        # First layer
        output, _ = self.first_lstm(x)
        
        # Stacked layers with residual connections
        for lstm in self.stacked_lstms:
            residual = output
            output, _ = lstm(output)
            output = output + residual
        
        return output


if __name__ == "__main__":
    # Test BiLSTM modules
    print("Testing BiLSTM Modules...")
    
    # Test basic BiLSTM
    bilstm = BiLSTMEncoder(input_size=256, hidden_size=128)
    x = torch.randn(2, 16, 256)  # (B, L, D)
    out, hidden = bilstm(x, return_hidden=True)
    print(f"BiLSTM output shape: {out.shape}")
    print(f"BiLSTM hidden h_n shape: {hidden[0].shape}")
    
    # Test sequence embedding
    emb = bilstm.get_sequence_embedding(x, method="attention")
    print(f"Sequence embedding shape: {emb.shape}")
    
    # Test BiLSTM with attention
    bilstm_attn = BiLSTMWithAttention(input_size=256, hidden_size=128)
    ctx, attn = bilstm_attn(x, return_attention=True)
    print(f"BiLSTM+Attention context shape: {ctx.shape}")
    print(f"BiLSTM+Attention weights shape: {attn.shape}")
    
    # Test stacked BiLSTM
    stacked = StackedBiLSTM(input_size=256, hidden_size=128, num_stacks=3)
    stacked_out = stacked(x)
    print(f"Stacked BiLSTM output shape: {stacked_out.shape}")
    
    print("\nBiLSTM tests passed!")
