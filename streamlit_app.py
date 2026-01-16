"""
Streamlit Dashboard for Hybrid Quantum CNN Glioma Detection.
Interactive UI for uploading MRI scans and viewing predictions.
"""

import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
import sys
import time
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from models.hybrid_qcnn import HybridQCNN, create_model
from data.preprocessing import preprocess_for_model
from utils.helpers import load_checkpoint, get_device

# Page config
st.set_page_config(
    page_title="Hybrid QCNN Glioma Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    .prediction-glioma {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    .prediction-normal {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load and cache the model."""
    config = get_config()
    device = get_device()
    
    model = create_model(model_type="full", num_classes=2)
    model = model.to(device)
    model.eval()
    
    # Try to load checkpoint
    checkpoint_path = config.api.model_path
    if checkpoint_path.exists():
        try:
            model, _, _ = load_checkpoint(model, str(checkpoint_path), device=device)
            st.sidebar.success("✓ Model loaded from checkpoint")
        except Exception as e:
            st.sidebar.warning(f"Using random weights: {e}")
    else:
        st.sidebar.info("No checkpoint found - using demo mode")
    
    return model, device


def predict_image(model, device, image: np.ndarray):
    """Run prediction on image."""
    # Preprocess
    input_tensor = preprocess_for_model(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        outputs, attention_maps = model(input_tensor, return_attention=True)
        probs = torch.softmax(outputs, dim=1)
    inference_time = (time.time() - start_time) * 1000
    
    return {
        'glioma_prob': probs[0, 1].item(),
        'non_glioma_prob': probs[0, 0].item(),
        'prediction': 'Glioma' if probs[0, 1] > 0.5 else 'Non-Glioma',
        'confidence': max(probs[0, 1].item(), probs[0, 0].item()),
        'inference_time_ms': inference_time,
        'attention_maps': attention_maps
    }


def create_probability_chart(glioma_prob: float, non_glioma_prob: float):
    """Create probability bar chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[non_glioma_prob],
        y=['Non-Glioma'],
        orientation='h',
        marker=dict(color='#26de81'),
        name='Non-Glioma',
        text=[f'{non_glioma_prob:.1%}'],
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        x=[glioma_prob],
        y=['Glioma'],
        orientation='h',
        marker=dict(color='#ff6b6b'),
        name='Glioma',
        text=[f'{glioma_prob:.1%}'],
        textposition='inside'
    ))
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Probability',
        yaxis_title='Class',
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        height=200,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_attention_visualization(attention_maps, slice_idx=0):
    """Create attention heatmap visualization."""
    if attention_maps is None or 'spatial' not in attention_maps:
        return None
    
    # Get spatial attention for the specified slice
    spatial_attn = attention_maps['spatial'][0, slice_idx, 0].cpu().numpy()
    
    # Resize for display
    spatial_attn = cv2.resize(spatial_attn, (256, 256))
    
    fig = px.imshow(
        spatial_attn,
        color_continuous_scale='Hot',
        title=f'Attention Map (Slice {slice_idx})',
        labels={'color': 'Attention'}
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Hybrid Quantum CNN Glioma Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced brain tumor detection using quantum-inspired neural networks</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model()
    
    # Model info
    st.sidebar.subheader("Model Information")
    num_params = sum(p.numel() for p in model.parameters())
    st.sidebar.metric("Parameters", f"{num_params:,}")
    st.sidebar.metric("Device", str(device).upper())
    
    # Options
    st.sidebar.subheader("Display Options")
    show_attention = st.sidebar.checkbox("Show Attention Maps", value=True)
    show_slices = st.sidebar.checkbox("Show Slice Analysis", value=False)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.99, 0.7)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload MRI Scan")
        
        uploaded_file = st.file_uploader(
            "Choose an MRI image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a brain MRI scan for glioma detection"
        )
        
        # Demo images
        st.markdown("---")
        st.markdown("**Or try a demo image:**")
        demo_col1, demo_col2 = st.columns(2)
        
        use_demo = False
        with demo_col1:
            if st.button("🧠 Sample Glioma"):
                use_demo = "glioma"
        with demo_col2:
            if st.button("✅ Sample Normal"):
                use_demo = "normal"
        
        # Display uploaded or demo image
        image = None
        
        if uploaded_file is not None:
            # Read uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        
        elif use_demo:
            # Generate demo image
            np.random.seed(42 if use_demo == "glioma" else 123)
            image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            
            if use_demo == "glioma":
                # Add simulated tumor region
                cv2.circle(image, (150, 120), 40, (255, 200, 200), -1)
                cv2.circle(image, (150, 120), 30, (255, 150, 150), -1)
            
            st.image(image, caption=f"Demo: {use_demo.title()} MRI", use_container_width=True)
    
    with col2:
        st.subheader("🔬 Analysis Results")
        
        if image is not None:
            # Run prediction
            with st.spinner("Analyzing MRI scan..."):
                results = predict_image(model, device, image)
            
            # Display prediction
            prediction = results['prediction']
            confidence = results['confidence']
            
            # Color based on prediction
            if prediction == 'Glioma':
                color = "#ff6b6b"
                icon = "⚠️"
            else:
                color = "#26de81"
                icon = "✅"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 0%, {color}99 100%); 
                        border-radius: 15px; padding: 2rem; text-align: center; color: white;">
                <h2 style="margin: 0; font-size: 2rem;">{icon} {prediction}</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Warning for low confidence
            if confidence < confidence_threshold:
                st.warning(f"⚠️ Confidence below threshold ({confidence_threshold:.0%}). Consider manual review.")
            
            # Metrics
            st.markdown("---")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Glioma Probability", f"{results['glioma_prob']:.1%}")
            with metric_col2:
                st.metric("Non-Glioma Probability", f"{results['non_glioma_prob']:.1%}")
            with metric_col3:
                st.metric("Inference Time", f"{results['inference_time_ms']:.0f} ms")
            
            # Probability chart
            st.plotly_chart(
                create_probability_chart(results['glioma_prob'], results['non_glioma_prob']),
                use_container_width=True
            )
            
            # Attention visualization
            if show_attention and results['attention_maps']:
                st.markdown("---")
                st.subheader("🔍 Attention Analysis")
                
                attn_fig = create_attention_visualization(results['attention_maps'])
                if attn_fig:
                    st.plotly_chart(attn_fig, use_container_width=True)
                
                # LSTM attention over slices
                if 'lstm_attention' in results['attention_maps']:
                    lstm_attn = results['attention_maps']['lstm_attention'][0, :, 0].cpu().numpy()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(range(len(lstm_attn))),
                        y=lstm_attn,
                        marker=dict(color=lstm_attn, colorscale='Viridis')
                    ))
                    fig.update_layout(
                        title='Attention Weights per MRI Slice',
                        xaxis_title='Slice Index',
                        yaxis_title='Attention Weight',
                        height=200
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Upload an MRI scan or try a demo image to see analysis results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for research and educational purposes only. 
        It should not be used as a substitute for professional medical diagnosis.</p>
        <p>Built with ❤️ using Hybrid Quantum CNN + Attention + BiLSTM</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
