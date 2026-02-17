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
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque
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


@dataclass
class FacialFeatures:
    """Extracted facial features from landmarks for a single frame."""
    eye_openness: float = 0.0
    left_eye_openness: float = 0.0
    right_eye_openness: float = 0.0
    brow_height: float = 0.0
    brow_furrow: float = 0.0
    mouth_openness: float = 0.0
    mouth_width: float = 0.0
    is_blinking: bool = False


# MediaPipe FaceMesh landmark indices
_LEFT_EYE_UPPER = 159
_LEFT_EYE_LOWER = 145
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_UPPER = 386
_RIGHT_EYE_LOWER = 374
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263
_LEFT_BROW_INNER = 70
_LEFT_BROW_OUTER = 63
_RIGHT_BROW_INNER = 300
_RIGHT_BROW_OUTER = 293
_UPPER_LIP = 13
_LOWER_LIP = 14
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291


class LandmarkCognitiveEstimator:
    """
    Estimates cognitive states from MediaPipe FaceMesh landmarks using
    FACS-based heuristic indicators.

    Uses temporal buffers to compute baseline statistics and detect
    changes in facial expression over time.
    """

    HISTORY_WINDOW = 90   # ~3 seconds at 30fps
    BLINK_THRESHOLD = 0.015
    BLINK_COOLDOWN_SEC = 0.2
    SMOOTHING_FACTOR = 0.15

    def __init__(self):
        self._eye_openness_history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self._head_pose_history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self._mouth_openness_history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self._brow_history: deque = deque(maxlen=self.HISTORY_WINDOW)

        self._blink_count: int = 0
        self._last_blink_time: float = 0.0

        # Smoothed output states
        self._smoothed = {
            "confusion": 0.0,
            "engagement": 0.5,
            "boredom": 0.0,
            "frustration": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        landmarks: List[Dict],
        head_pose: Optional[Dict] = None,
    ) -> CognitiveState:
        """Estimate cognitive states from a single frame of landmarks."""
        features = self._extract_features(landmarks)
        self._update_history(features, head_pose)
        raw_states = self._calculate_states(features, head_pose)

        # Exponential moving average smoothing
        for key in self._smoothed:
            self._smoothed[key] = (
                self._smoothed[key] * (1 - self.SMOOTHING_FACTOR)
                + raw_states[key] * self.SMOOTHING_FACTOR
            )

        history_ratio = min(1.0, len(self._eye_openness_history) / self.HISTORY_WINDOW)
        confidence = 0.45 + 0.25 * history_ratio

        return CognitiveState(
            confusion=float(self._smoothed["confusion"]),
            engagement=float(self._smoothed["engagement"]),
            boredom=float(self._smoothed["boredom"]),
            frustration=float(self._smoothed["frustration"]),
            timestamp=time.time(),
            confidence=float(confidence),
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, landmarks: List[Dict]) -> FacialFeatures:
        left_eye_openness = abs(
            landmarks[_LEFT_EYE_UPPER]["y"] - landmarks[_LEFT_EYE_LOWER]["y"]
        )
        right_eye_openness = abs(
            landmarks[_RIGHT_EYE_UPPER]["y"] - landmarks[_RIGHT_EYE_LOWER]["y"]
        )
        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2.0

        # Brow height (distance from brow to upper eyelid; positive = raised)
        left_brow_avg_y = (
            landmarks[_LEFT_BROW_INNER]["y"] + landmarks[_LEFT_BROW_OUTER]["y"]
        ) / 2.0
        right_brow_avg_y = (
            landmarks[_RIGHT_BROW_INNER]["y"] + landmarks[_RIGHT_BROW_OUTER]["y"]
        ) / 2.0
        left_brow_height = landmarks[_LEFT_EYE_UPPER]["y"] - left_brow_avg_y
        right_brow_height = landmarks[_RIGHT_EYE_UPPER]["y"] - right_brow_avg_y
        avg_brow_height = (left_brow_height + right_brow_height) / 2.0

        # Inner brow distance (smaller = more furrowed)
        brow_furrow = abs(
            landmarks[_LEFT_BROW_INNER]["x"] - landmarks[_RIGHT_BROW_INNER]["x"]
        )

        # Mouth
        mouth_openness = abs(
            landmarks[_UPPER_LIP]["y"] - landmarks[_LOWER_LIP]["y"]
        )
        mouth_width = abs(
            landmarks[_MOUTH_LEFT]["x"] - landmarks[_MOUTH_RIGHT]["x"]
        )

        # Blink detection
        is_blinking = avg_eye_openness < self.BLINK_THRESHOLD
        now = time.time()
        if is_blinking and (now - self._last_blink_time) > self.BLINK_COOLDOWN_SEC:
            self._last_blink_time = now
            self._blink_count += 1

        return FacialFeatures(
            eye_openness=avg_eye_openness,
            left_eye_openness=left_eye_openness,
            right_eye_openness=right_eye_openness,
            brow_height=avg_brow_height,
            brow_furrow=brow_furrow,
            mouth_openness=mouth_openness,
            mouth_width=mouth_width,
            is_blinking=is_blinking,
        )

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _update_history(self, features: FacialFeatures, head_pose: Optional[Dict]):
        self._eye_openness_history.append(features.eye_openness)
        self._mouth_openness_history.append(features.mouth_openness)
        self._brow_history.append(features.brow_height)
        if head_pose:
            self._head_pose_history.append(head_pose)

    # ------------------------------------------------------------------
    # Cognitive state calculation
    # ------------------------------------------------------------------

    def _calculate_states(
        self, features: FacialFeatures, head_pose: Optional[Dict]
    ) -> Dict[str, float]:
        # Baseline statistics
        if len(self._eye_openness_history) > 5:
            avg_eo = float(np.mean(self._eye_openness_history))
            std_eo = float(np.std(self._eye_openness_history)) + 0.001
        else:
            avg_eo = features.eye_openness
            std_eo = 0.01

        eo_zscore = (features.eye_openness - avg_eo) / std_eo

        head_movement = self._head_movement_variance()
        pitch = head_pose.get("pitch", 0) if head_pose else 0
        yaw = head_pose.get("yaw", 0) if head_pose else 0
        roll = head_pose.get("roll", 0) if head_pose else 0

        # --- ENGAGEMENT ---
        eye_wide = max(0.0, eo_zscore)
        head_stable = max(0.0, 1.0 - head_movement * 10.0)
        looking_forward = max(0.0, 1.0 - (abs(yaw) + abs(pitch)) / 20.0)
        engagement = _clamp(eye_wide * 0.3 + head_stable * 0.3 + looking_forward * 0.4)

        # --- BOREDOM ---
        eye_droopy = max(0.0, -eo_zscore * 0.5)
        yawning = min(1.0, features.mouth_openness * 10.0) if features.mouth_openness > 0.05 else 0.0
        looking_away = min(1.0, (abs(yaw) + abs(pitch)) / 30.0)
        boredom = _clamp(eye_droopy * 0.4 + yawning * 0.3 + looking_away * 0.3)

        # --- CONFUSION ---
        brow_furrowed = max(0.0, 0.1 - features.brow_furrow) * 10.0
        squinting = max(0.0, -eo_zscore * 0.3)
        head_tilt = min(1.0, abs(roll) / 15.0)
        confusion = _clamp(brow_furrowed * 0.4 + squinting * 0.3 + head_tilt * 0.3)

        # --- FRUSTRATION ---
        rapid_blinking = min(1.0, self._blink_count / 10.0)
        head_restless = min(1.0, head_movement * 15.0)
        tense_brows = brow_furrowed * 0.5
        frustration = _clamp(rapid_blinking * 0.4 + head_restless * 0.3 + tense_brows * 0.3)

        # Decay blink count periodically
        if len(self._eye_openness_history) >= self.HISTORY_WINDOW:
            self._blink_count = max(0, self._blink_count - 1)

        return {
            "confusion": confusion,
            "engagement": engagement,
            "boredom": boredom,
            "frustration": frustration,
        }

    def _head_movement_variance(self) -> float:
        if len(self._head_pose_history) < 2:
            return 0.0
        pitches = [h.get("pitch", 0) for h in self._head_pose_history]
        yaws = [h.get("yaw", 0) for h in self._head_pose_history]
        return float(np.var(pitches) + np.var(yaws))


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class EmotionDetector:
    """
    Detects learner cognitive states from face images.

    Supports multiple backends:
    - 'landmarks': Heuristic estimation from face mesh landmarks (default)
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
        self._landmark_estimator: Optional[LandmarkCognitiveEstimator] = None

        if backend == "landmarks":
            self._landmark_estimator = LandmarkCognitiveEstimator()

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

        if backend not in ("demo", "landmarks"):
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
        if self.backend == "demo" or self.backend == "landmarks":
            return self._demo_predict()
        elif self.backend == "onnx":
            return self._onnx_predict(face_image)
        elif self.backend == "torch":
            return self._torch_predict(face_image)
        else:
            return self._demo_predict()

    def predict_from_landmarks(
        self,
        landmarks: List[Dict],
        head_pose: Optional[Dict] = None,
    ) -> CognitiveState:
        """
        Predict cognitive states from MediaPipe face mesh landmarks.

        Args:
            landmarks: List of 468 landmark dicts with 'x', 'y', 'z' keys
            head_pose: Optional dict with 'pitch', 'yaw', 'roll' keys

        Returns:
            CognitiveState with predictions
        """
        if self.backend == "landmarks" and self._landmark_estimator:
            return self._landmark_estimator.predict(landmarks, head_pose)
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
        self._landmarks_queue: Optional[Tuple[List[Dict], Optional[Dict]]] = None

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

    def submit_landmarks(
        self,
        landmarks: List[Dict],
        head_pose: Optional[Dict] = None,
    ):
        """Submit face mesh landmarks for async processing."""
        with self._lock:
            self._landmarks_queue = (landmarks, head_pose)

    def get_latest_state(self) -> Optional[CognitiveState]:
        """Get the latest cognitive state prediction."""
        with self._lock:
            return self.latest_state

    def _inference_loop(self):
        """Background inference loop."""
        while self._running:
            state = None

            with self._lock:
                landmarks_data = self._landmarks_queue
                self._landmarks_queue = None
                image = self._image_queue
                self._image_queue = None

            if landmarks_data is not None:
                landmarks, head_pose = landmarks_data
                state = self.detector.predict_from_landmarks(landmarks, head_pose)
            elif image is not None:
                state = self.detector.predict(image)

            if state is not None:
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
