"""
Grad-CAM implementation for model interpretability.
Generates attention heatmaps highlighting tumor-relevant regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
import base64
from io import BytesIO
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Visualizes which regions of the input image contributed
    most to the model's prediction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_cuda: bool = True
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The neural network model.
            target_layer: Layer to extract activations from.
            use_cuda: Whether to use GPU.
        """
        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        slice_idx: int = 0
    ) -> Tuple[np.ndarray, int]:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (B, S, C, H, W) or (B, C, H, W).
            target_class: Target class for CAM. If None, uses predicted class.
            slice_idx: Which slice to visualize (for MRI stacks).
            
        Returns:
            Tuple of (heatmap, predicted_class).
        """
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
            self.model = self.model.cuda()
        
        self.model.eval()
        
        # Forward pass
        output, _ = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute CAM
        gradients = self.gradients[0]  # (C, H, W) or (S, C, H, W)
        activations = self.activations[0]
        
        # Handle slice dimension if present
        if len(gradients.shape) == 4:
            gradients = gradients[slice_idx]
            activations = activations[slice_idx]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Convert to numpy
        heatmap = cam.cpu().numpy()
        
        return heatmap, target_class
    
    def generate_visualization(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Generate visualization with heatmap overlay.
        
        Args:
            input_tensor: Model input tensor.
            original_image: Original image (H, W, C) in RGB.
            target_class: Target class for CAM.
            alpha: Overlay transparency.
            colormap: OpenCV colormap.
            
        Returns:
            Visualization image (H, W, C) in RGB.
        """
        heatmap, pred_class = self(input_tensor, target_class)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        if original_image.max() > 1:
            original_image = original_image.astype(np.float32) / 255.0
        
        overlay = (1 - alpha) * original_image + alpha * heatmap_colored.astype(np.float32) / 255.0
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        
        return overlay


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ for improved localization.
    Uses second-order gradients for more accurate heatmaps.
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        slice_idx: int = 0
    ) -> Tuple[np.ndarray, int]:
        """
        Generate Grad-CAM++ heatmap.
        """
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
            self.model = self.model.cuda()
        
        self.model.eval()
        
        # Forward pass
        output, _ = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        if len(gradients.shape) == 4:
            gradients = gradients[slice_idx]
            activations = activations[slice_idx]
        
        # Grad-CAM++ weights
        grad_sq = gradients ** 2
        grad_cube = gradients ** 3
        
        alpha_num = grad_sq
        alpha_denom = 2 * grad_sq + activations.sum(dim=(1, 2), keepdim=True) * grad_cube + 1e-8
        alpha = alpha_num / alpha_denom
        
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        heatmap = cam.cpu().numpy()
        
        return heatmap, target_class


def generate_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_layer_name: str = "cnn_encoder",
    target_class: Optional[int] = None,
    method: str = "gradcam"
) -> Dict[str, any]:
    """
    High-level function to generate attention heatmap.
    
    Args:
        model: The model.
        input_tensor: Input tensor.
        original_image: Original image for overlay.
        target_layer_name: Name of target layer.
        target_class: Target class.
        method: 'gradcam' or 'gradcam++'.
        
    Returns:
        Dictionary with heatmap, overlay, and base64 encoded image.
    """
    # Find target layer
    target_layer = None
    for name, module in model.named_modules():
        if target_layer_name in name:
            target_layer = module
            break
    
    if target_layer is None:
        # Default to last conv layer
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break
    
    if target_layer is None:
        raise ValueError(f"Could not find target layer: {target_layer_name}")
    
    # Create CAM object
    if method == "gradcam++":
        cam = GradCAMPlusPlus(model, target_layer)
    else:
        cam = GradCAM(model, target_layer)
    
    # Generate visualization
    heatmap, pred_class = cam(input_tensor, target_class)
    overlay = cam.generate_visualization(input_tensor, original_image, target_class)
    
    # Convert to base64 for API responses
    pil_image = Image.fromarray(overlay)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return {
        'heatmap': heatmap,
        'overlay': overlay,
        'base64': base64_image,
        'predicted_class': pred_class
    }


def create_attention_visualization(
    attention_maps: Dict[str, torch.Tensor],
    original_images: np.ndarray,
    slice_idx: int = 0
) -> Dict[str, np.ndarray]:
    """
    Create visualizations from model attention maps.
    
    Args:
        attention_maps: Dictionary from model.forward() with return_attention=True.
        original_images: Original images (B, S, H, W, C) or (B, H, W, C).
        slice_idx: Slice index to visualize.
        
    Returns:
        Dictionary of visualization images.
    """
    results = {}
    
    # Spatial attention maps
    if 'spatial' in attention_maps:
        spatial_attn = attention_maps['spatial'][0, slice_idx, 0].cpu().numpy()
        spatial_attn = cv2.resize(spatial_attn, (original_images.shape[-2], original_images.shape[-3]))
        spatial_attn = (spatial_attn * 255).astype(np.uint8)
        results['spatial_attention'] = cv2.applyColorMap(spatial_attn, cv2.COLORMAP_HOT)
    
    # Self-attention (average over heads)
    if 'self_attention' in attention_maps:
        self_attn = attention_maps['self_attention'][0].mean(dim=0).cpu().numpy()
        self_attn = (self_attn * 255).astype(np.uint8)
        results['self_attention'] = cv2.applyColorMap(self_attn, cv2.COLORMAP_VIRIDIS)
    
    # LSTM attention weights
    if 'lstm_attention' in attention_maps:
        lstm_attn = attention_maps['lstm_attention'][0, :, 0].cpu().numpy()
        # Create bar visualization
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(len(lstm_attn)), lstm_attn)
        ax.set_xlabel('Slice')
        ax.set_ylabel('Attention Weight')
        ax.set_title('LSTM Attention over Slices')
        
        # Convert to image
        fig.canvas.draw()
        lstm_viz = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        lstm_viz = lstm_viz.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        results['lstm_attention'] = lstm_viz
    
    return results


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully.")
    print("Note: Requires model instance for testing.")
