"""
Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
import base64
from io import BytesIO
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    return img


@pytest.fixture
def sample_image_bytes(sample_image):
    """Convert sample image to bytes."""
    _, buffer = cv2.imencode('.jpg', sample_image)
    return buffer.tobytes()


@pytest.fixture
def sample_image_base64(sample_image):
    """Convert sample image to base64."""
    _, buffer = cv2.imencode('.jpg', sample_image)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestModelInfo:
    """Test model information endpoint."""
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "num_parameters" in data


class TestAuthentication:
    """Test authentication endpoints."""
    
    def test_login_success(self, client):
        """Test successful login."""
        response = client.post(
            "/token",
            data={"username": "demo", "password": "demo123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_failure(self, client):
        """Test failed login with wrong credentials."""
        response = client.post(
            "/token",
            data={"username": "wrong", "password": "wrong"}
        )
        assert response.status_code == 401
    
    def test_get_user_authenticated(self, client):
        """Test getting user info with token."""
        # First login
        login_response = client.post(
            "/token",
            data={"username": "demo", "password": "demo123"}
        )
        token = login_response.json()["access_token"]
        
        # Get user info
        response = client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["username"] == "demo"
    
    def test_get_user_unauthenticated(self, client):
        """Test getting user info without token."""
        response = client.get("/users/me")
        assert response.status_code == 401


class TestPrediction:
    """Test prediction endpoints."""
    
    def test_predict_file_upload(self, client, sample_image_bytes):
        """Test prediction with file upload."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"return_heatmap": "false"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "glioma_probability" in data
        assert "non_glioma_probability" in data
        assert data["prediction"] in ["glioma", "non_glioma"]
    
    def test_predict_with_heatmap(self, client, sample_image_bytes):
        """Test prediction with heatmap returned."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"return_heatmap": "true"}
        )
        assert response.status_code == 200
        # Note: heatmap may be None if generation fails
    
    def test_predict_invalid_file(self, client):
        """Test prediction with invalid file."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
    
    def test_predict_base64(self, client, sample_image_base64):
        """Test prediction with base64 image."""
        response = client.post(
            "/predict/base64",
            params={"image_base64": sample_image_base64, "return_heatmap": "false"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data


class TestBatchPrediction:
    """Test batch prediction endpoint."""
    
    def test_batch_predict_requires_auth(self, client, sample_image_bytes):
        """Test that batch prediction requires authentication."""
        response = client.post(
            "/predict/batch",
            files=[("files", ("test1.jpg", sample_image_bytes, "image/jpeg"))]
        )
        assert response.status_code == 401
    
    def test_batch_predict_authenticated(self, client, sample_image_bytes):
        """Test batch prediction with authentication."""
        # Login
        login_response = client.post(
            "/token",
            data={"username": "demo", "password": "demo123"}
        )
        token = login_response.json()["access_token"]
        
        # Batch predict
        response = client.post(
            "/predict/batch",
            files=[
                ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
                ("files", ("test2.jpg", sample_image_bytes, "image/jpeg"))
            ],
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
