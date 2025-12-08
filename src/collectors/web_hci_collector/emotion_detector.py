"""
Emotion/Cognitive State Detector

Detects learner cognitive states using DenseAttNet or similar models.
States detected:
- Confusion
- Engagement
- Boredom
- Frustration

This module provides:
1. A demo mode with simulated predictions (for testing without model)
2. Integration with pre-trained DAiSEE models when available
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
import time

# Optional imports for actual model inference
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class CognitiveState:
    """Represents detected cognitive states."""
    confusion: float = 0.0
    engagement: float = 0.0
    boredom: float = 0.0
    frustration: float = 0.0
    timestamp: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "confusion": self.confusion,
            "engagement": self.engagement,
            "boredom": self.boredom,
            "frustration": self.frustration,
            "timestamp": self.timestamp,
            "confidence": self.confidence
        }


class EmotionDetector:
    """
    Detects learner cognitive states from face images.

    Supports multiple backends:
    - 'demo': Simulated predictions for testing
    - 'onnx': ONNX Runtime inference (recommended for production)
    - 'torch': PyTorch inference

    The model should be trained on DAiSEE or similar dataset for
    detecting learning-specific cognitive states.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "demo",
        device: str = "cpu"
    ):
        """
        Initialize the emotion detector.

        Args:
            model_path: Path to model file (.onnx or .pt)
            backend: Inference backend ('demo', 'onnx', 'torch')
            device: Device for inference ('cpu' or 'cuda')
        """
        self.backend = backend
        self.device = device
        self.model = None
        self.session = None
        self.transform = None

        self._demo_state = {
            "confusion": 0.3,
            "engagement": 0.7,
            "boredom": 0.2,
            "frustration": 0.1
        }
        self._demo_trend = {
            "confusion": 0.01,
            "engagement": -0.005,
            "boredom": 0.008,
            "frustration": 0.005
        }

        if backend != "demo":
            self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]):
        """Load the model from file."""
        if model_path is None:
            print("Warning: No model path provided, using demo mode")
            self.backend = "demo"
            return

        model_path = Path(model_path)

        if not model_path.exists():
            print(f"Warning: Model file not found at {model_path}, using demo mode")
            self.backend = "demo"
            return

        if self.backend == "onnx" and ONNX_AVAILABLE:
            self._load_onnx_model(model_path)
        elif self.backend == "torch" and TORCH_AVAILABLE:
            self._load_torch_model(model_path)
        else:
            print(f"Warning: Backend '{self.backend}' not available, using demo mode")
            self.backend = "demo"

        # Setup image transforms
        if TORCH_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _load_onnx_model(self, model_path: Path):
        """Load ONNX model."""
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            print(f"Loaded ONNX model from {model_path}")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            self.backend = "demo"

    def _load_torch_model(self, model_path: Path):
        """Load PyTorch model."""
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            print(f"Loaded PyTorch model from {model_path}")
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")
            self.backend = "demo"

    def predict(self, face_image: np.ndarray) -> CognitiveState:
        """
        Predict cognitive states from a face image.

        Args:
            face_image: Face image as numpy array (H, W, C) in RGB format

        Returns:
            CognitiveState with predictions
        """
        if self.backend == "demo":
            return self._demo_predict()
        elif self.backend == "onnx":
            return self._onnx_predict(face_image)
        elif self.backend == "torch":
            return self._torch_predict(face_image)
        else:
            return self._demo_predict()

    def _demo_predict(self) -> CognitiveState:
        """Generate simulated predictions for demo/testing."""
        # Random walk with bounds
        for key in self._demo_state:
            # Add random noise
            self._demo_state[key] += self._demo_trend[key] + np.random.uniform(-0.05, 0.05)

            # Reverse trend at bounds
            if self._demo_state[key] > 0.9:
                self._demo_trend[key] = -abs(self._demo_trend[key])
            elif self._demo_state[key] < 0.1:
                self._demo_trend[key] = abs(self._demo_trend[key])

            # Clamp to [0, 1]
            self._demo_state[key] = max(0.0, min(1.0, self._demo_state[key]))

        return CognitiveState(
            confusion=self._demo_state["confusion"],
            engagement=self._demo_state["engagement"],
            boredom=self._demo_state["boredom"],
            frustration=self._demo_state["frustration"],
            timestamp=time.time(),
            confidence=0.85
        )

    def _onnx_predict(self, face_image: np.ndarray) -> CognitiveState:
        """Run ONNX model inference."""
        if self.session is None:
            return self._demo_predict()

        try:
            # Preprocess image
            if TORCH_AVAILABLE and self.transform:
                img = Image.fromarray(face_image)
                img_tensor = self.transform(img).unsqueeze(0).numpy()
            else:
                # Basic preprocessing without torchvision
                img = np.resize(face_image, (224, 224, 3)).astype(np.float32)
                img = img / 255.0
                img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                img_tensor = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: img_tensor})

            # Parse outputs (assuming 4 class outputs)
            probs = self._softmax(outputs[0][0])

            return CognitiveState(
                engagement=float(probs[0]),
                boredom=float(probs[1]),
                confusion=float(probs[2]),
                frustration=float(probs[3]),
                timestamp=time.time(),
                confidence=float(np.max(probs))
            )

        except Exception as e:
            print(f"ONNX inference error: {e}")
            return self._demo_predict()

    def _torch_predict(self, face_image: np.ndarray) -> CognitiveState:
        """Run PyTorch model inference."""
        if self.model is None or not TORCH_AVAILABLE:
            return self._demo_predict()

        try:
            # Preprocess image
            img = Image.fromarray(face_image)
            img_tensor = self.transform(img).unsqueeze(0)

            if self.device != "cpu":
                img_tensor = img_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

            return CognitiveState(
                engagement=float(probs[0]),
                boredom=float(probs[1]),
                confusion=float(probs[2]),
                frustration=float(probs[3]),
                timestamp=time.time(),
                confidence=float(np.max(probs))
            )

        except Exception as e:
            print(f"PyTorch inference error: {e}")
            return self._demo_predict()

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class AsyncEmotionDetector:
    """
    Asynchronous wrapper for EmotionDetector.

    Runs inference in a background thread to avoid blocking.
    """

    def __init__(self, detector: EmotionDetector):
        self.detector = detector
        self.latest_state: Optional[CognitiveState] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._image_queue: Optional[np.ndarray] = None

    def start(self):
        """Start the async detector."""
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the async detector."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def submit_image(self, face_image: np.ndarray):
        """Submit a face image for async processing."""
        with self._lock:
            self._image_queue = face_image

    def get_latest_state(self) -> Optional[CognitiveState]:
        """Get the latest cognitive state prediction."""
        with self._lock:
            return self.latest_state

    def _inference_loop(self):
        """Background inference loop."""
        while self._running:
            # Get latest image
            with self._lock:
                image = self._image_queue
                self._image_queue = None

            if image is not None:
                # Run inference
                state = self.detector.predict(image)

                with self._lock:
                    self.latest_state = state

            time.sleep(0.1)  # 10 Hz max


# Download instructions for pre-trained models
MODEL_DOWNLOAD_INSTRUCTIONS = """
To use a real cognitive state detection model, you need to:

1. Download a pre-trained DAiSEE model:

   Option A: DenseAttNet (Recommended)
   - Paper: "3D DenseAttNet for Student Engagement Recognition"
   - GitHub: Search for "DenseAttNet DAiSEE" on GitHub
   - Convert to ONNX for better performance

   Option B: ResNet+TCN
   - Paper: "ResNet+TCN for Students Engagement Level Detection"
   - GitHub: Search for "ResNet TCN engagement detection"

   Option C: Train your own
   - Download DAiSEE dataset from: https://iith.ac.in/~daisee-dataset/
   - Train using the provided architectures

2. Convert to ONNX (if using PyTorch model):

   import torch
   model = torch.load('model.pt')
   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, dummy_input, 'model.onnx')

3. Place the model file in:
   src/collectors/web_hci_collector/models/denseattnet.onnx

4. Update the EmotionDetector initialization:
   detector = EmotionDetector(
       model_path='src/collectors/web_hci_collector/models/denseattnet.onnx',
       backend='onnx'
   )
"""


if __name__ == "__main__":
    # Test demo mode
    print("Testing EmotionDetector in demo mode...")
    detector = EmotionDetector(backend="demo")

    for i in range(10):
        state = detector.predict(np.zeros((224, 224, 3), dtype=np.uint8))
        print(f"Sample {i+1}: {state.to_dict()}")
        time.sleep(0.5)

    print("\n" + MODEL_DOWNLOAD_INSTRUCTIONS)
