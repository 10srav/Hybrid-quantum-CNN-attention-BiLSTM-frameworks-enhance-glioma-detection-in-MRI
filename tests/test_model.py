"""
Unit tests for the HybridQCNN model.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.hybrid_qcnn import HybridQCNN, HybridQCNNLight, create_model
from models.quantum_layer import QuantumConv2dFast, get_quantum_layer
from models.attention import MultiHeadSelfAttention, CBAM
from models.bilstm import BiLSTMEncoder, BiLSTMWithAttention


class TestQuantumLayer:
    """Test quantum layer components."""
    
    def test_quantum_conv_fast_forward(self):
        """Test fast quantum convolution forward pass."""
        layer = QuantumConv2dFast(in_channels=3, out_channels=8)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        
        assert out.shape[0] == 2  # Batch size preserved
        assert out.shape[1] == 8  # Output channels
        assert out.shape[2] <= 32  # Spatial dim reduced
    
    def test_get_quantum_layer_factory(self):
        """Test quantum layer factory function."""
        fast_layer = get_quantum_layer(use_real_quantum=False, in_channels=3)
        assert isinstance(fast_layer, QuantumConv2dFast)


class TestAttention:
    """Test attention modules."""
    
    def test_multihead_attention_shape(self):
        """Test multi-head self-attention output shape."""
        attn = MultiHeadSelfAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 16, 256)
        out, weights = attn(x, return_attention=True)
        
        assert out.shape == (2, 16, 256)
        assert weights.shape == (2, 8, 16, 16)
    
    def test_cbam_attention(self):
        """Test CBAM attention module."""
        cbam = CBAM(in_channels=64)
        x = torch.randn(2, 64, 32, 32)
        out, attn_map = cbam(x)
        
        assert out.shape == x.shape
        assert attn_map.shape == (2, 1, 32, 32)


class TestBiLSTM:
    """Test BiLSTM modules."""
    
    def test_bilstm_encoder(self):
        """Test BiLSTM encoder."""
        lstm = BiLSTMEncoder(input_size=256, hidden_size=128)
        x = torch.randn(2, 16, 256)
        out, hidden = lstm(x, return_hidden=True)
        
        assert out.shape == (2, 16, 256)  # bidirectional: 128*2
        assert hidden[0].shape[0] == 2  # 2 directions
    
    def test_bilstm_with_attention(self):
        """Test BiLSTM with attention pooling."""
        lstm_attn = BiLSTMWithAttention(input_size=256, hidden_size=128)
        x = torch.randn(2, 16, 256)
        ctx, weights = lstm_attn(x, return_attention=True)
        
        assert ctx.shape == (2, 256)
        assert weights.shape == (2, 16, 1)


class TestHybridQCNN:
    """Test the complete hybrid model."""
    
    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return HybridQCNN(
            num_classes=2,
            in_channels=3,
            num_slices=16,
            image_size=64,  # Smaller for faster tests
            use_real_quantum=False
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 16, 3, 64, 64)
    
    def test_forward_pass(self, model, sample_input):
        """Test forward pass produces correct output shape."""
        output, _ = model(sample_input)
        assert output.shape == (2, 2)
    
    def test_forward_with_attention(self, model, sample_input):
        """Test forward pass returns attention maps."""
        output, attention = model(sample_input, return_attention=True)
        
        assert output.shape == (2, 2)
        assert 'spatial' in attention
        assert 'self_attention' in attention
        assert 'lstm_attention' in attention
    
    def test_predict(self, model, sample_input):
        """Test prediction method."""
        preds = model.predict(sample_input)
        
        assert preds.shape == (2,)
        assert all(p in [0, 1] for p in preds.tolist())
    
    def test_predict_proba(self, model, sample_input):
        """Test probability prediction."""
        probs = model.predict_proba(sample_input)
        
        assert probs.shape == (2, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    
    def test_get_embedding(self, model, sample_input):
        """Test embedding extraction."""
        embedding = model.get_embedding(sample_input)
        
        assert embedding.shape == (2, 256)  # hidden_size * 2


class TestHybridQCNNLight:
    """Test the lightweight model variant."""
    
    def test_light_model_forward(self):
        """Test light model forward pass."""
        model = HybridQCNNLight(num_classes=2, num_slices=16)
        x = torch.randn(2, 16, 3, 128, 128)
        output, _ = model(x)
        
        assert output.shape == (2, 2)
    
    def test_light_model_fewer_params(self):
        """Test light model has fewer parameters."""
        full_model = HybridQCNN(num_classes=2, image_size=64)
        light_model = HybridQCNNLight(num_classes=2)
        
        full_params = sum(p.numel() for p in full_model.parameters())
        light_params = sum(p.numel() for p in light_model.parameters())
        
        assert light_params < full_params


class TestModelFactory:
    """Test model factory function."""
    
    def test_create_full_model(self):
        """Test creating full model."""
        model = create_model(model_type="full", num_classes=2)
        assert isinstance(model, HybridQCNN)
    
    def test_create_light_model(self):
        """Test creating light model."""
        model = create_model(model_type="light", num_classes=2)
        assert isinstance(model, HybridQCNNLight)
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError):
            create_model(model_type="invalid")


class TestGradientFlow:
    """Test gradient computation."""
    
    def test_gradients_flow(self):
        """Test that gradients flow through model."""
        model = HybridQCNN(num_classes=2, image_size=32)
        x = torch.randn(1, 16, 3, 32, 32, requires_grad=True)
        
        output, _ = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
