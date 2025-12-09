"""
L2CS-Net Gaze Estimator for Web HCI Collector

Provides server-side gaze estimation using deep learning models.
This complements WebGazer.js client-side tracking with more accurate predictions.

Features:
- L2CS-Net model for accurate gaze estimation (~3.5° accuracy)
- Roboflow Inference API support (easiest setup)
- Personal calibration support for improved accuracy
- Async processing for non-blocking inference
- Fallback to simple estimation if model unavailable

Installation Options:
1. Roboflow Inference (easiest):
   pip install inference inference-sdk

2. L2CS from GitHub:
   pip install git+https://github.com/Ahmednull/L2CS-Net.git

3. Manual setup with PyTorch:
   pip install torch torchvision opencv-python
"""

import asyncio
import base64
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np

# Optional imports - graceful fallback if not installed
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try Roboflow Inference first (easiest to install)
ROBOFLOW_AVAILABLE = False
try:
    from inference.models.gaze import Gaze
    ROBOFLOW_AVAILABLE = True
except ImportError:
    pass

# Try L2CS from GitHub
L2CS_AVAILABLE = False
try:
    from l2cs import Pipeline as L2CSPipeline
    L2CS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class GazeEstimate:
    """Result of gaze estimation."""
    x: float
    y: float
    pitch: float
    yaw: float
    confidence: float
    timestamp: float
    server_timestamp: float
    source: str = "l2cs"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'pitch': self.pitch,
            'yaw': self.yaw,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'server_timestamp': self.server_timestamp,
            'source': self.source
        }


@dataclass
class CalibrationPoint:
    """Single calibration point data."""
    screen_x: float
    screen_y: float
    pitch: float
    yaw: float
    timestamp: float


@dataclass
class CalibrationData:
    """Calibration data for personal model."""
    points: List[CalibrationPoint] = field(default_factory=list)
    model_fitted: bool = False
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0

    def add_point(self, screen_x: float, screen_y: float, pitch: float, yaw: float, timestamp: float):
        self.points.append(CalibrationPoint(screen_x, screen_y, pitch, yaw, timestamp))
        self.model_fitted = False


class L2CSGazeEstimator:
    """
    Gaze estimator supporting multiple backends:
    1. Roboflow Inference (easiest to install)
    2. L2CS-Net from GitHub
    3. Simple face-based estimation (OpenCV only)
    4. Demo mode (no dependencies)
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        device: str = 'auto',
        model_weights: Optional[str] = None
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration = CalibrationData()

        if device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.pipeline = None
        self.roboflow_model = None
        self.mode = 'demo'

        # Try Roboflow Inference first (easiest setup)
        if ROBOFLOW_AVAILABLE and CV2_AVAILABLE:
            try:
                self.roboflow_model = Gaze()
                self.mode = 'roboflow'
                print("Roboflow gaze estimator initialized (L2CS-Net backend)")
            except Exception as e:
                print(f"Could not initialize Roboflow gaze: {e}")

        # Try L2CS from GitHub
        if self.mode == 'demo' and L2CS_AVAILABLE and TORCH_AVAILABLE and CV2_AVAILABLE:
            try:
                self.pipeline = L2CSPipeline(
                    weights=model_weights,
                    arch='ResNet50',
                    device=torch.device(self.device)
                )
                self.mode = 'l2cs'
                print(f"L2CS gaze estimator initialized on {self.device}")
            except Exception as e:
                print(f"Could not initialize L2CS: {e}")

        # Fallback to simple face-based estimation
        if self.mode == 'demo' and CV2_AVAILABLE:
            self.mode = 'simple'
            print("Using simple geometric gaze estimation (OpenCV face detection)")

        if self.mode == 'demo':
            print("Using demo mode for gaze estimation (install: pip install inference inference-sdk)")

        self._demo_gaze = {'x': screen_width / 2, 'y': screen_height / 2}

    def estimate_from_frame(self, frame: np.ndarray, timestamp: float) -> Optional[GazeEstimate]:
        """Estimate gaze from a video frame."""
        server_timestamp = datetime.now().timestamp() * 1000

        if self.mode == 'roboflow':
            return self._estimate_roboflow(frame, timestamp, server_timestamp)
        elif self.mode == 'l2cs':
            return self._estimate_l2cs(frame, timestamp, server_timestamp)
        elif self.mode == 'simple':
            return self._estimate_simple(frame, timestamp, server_timestamp)
        else:
            return self._estimate_demo(timestamp, server_timestamp)

    def estimate_from_base64(self, base64_data: str, timestamp: float) -> Optional[GazeEstimate]:
        """Estimate gaze from base64 encoded image."""
        if not CV2_AVAILABLE:
            return self._estimate_demo(timestamp, datetime.now().timestamp() * 1000)

        try:
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]

            img_bytes = base64.b64decode(base64_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                return None

            return self.estimate_from_frame(frame, timestamp)

        except Exception as e:
            print(f"Error decoding frame: {e}")
            return None

    def _estimate_roboflow(self, frame: np.ndarray, timestamp: float, server_timestamp: float) -> Optional[GazeEstimate]:
        """Estimate gaze using Roboflow Inference (L2CS-Net backend)."""
        try:
            # Roboflow expects RGB, OpenCV gives BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.roboflow_model.infer(rgb_frame)

            if not results or len(results) == 0:
                return None

            # Roboflow returns gaze predictions with yaw and pitch
            prediction = results[0] if isinstance(results, list) else results

            # Extract yaw and pitch (in degrees, convert to radians)
            yaw_deg = prediction.get('yaw', 0)
            pitch_deg = prediction.get('pitch', 0)

            yaw = np.radians(yaw_deg)
            pitch = np.radians(pitch_deg)

            screen_x, screen_y = self._angles_to_screen(pitch, yaw)

            # Confidence from face detection
            confidence = prediction.get('confidence', 0.8)

            return GazeEstimate(
                x=screen_x,
                y=screen_y,
                pitch=pitch,
                yaw=yaw,
                confidence=confidence,
                timestamp=timestamp,
                server_timestamp=server_timestamp,
                source='roboflow_l2cs'
            )

        except Exception as e:
            print(f"Roboflow estimation error: {e}")
            return None

    def _estimate_l2cs(self, frame: np.ndarray, timestamp: float, server_timestamp: float) -> Optional[GazeEstimate]:
        """Estimate gaze using L2CS-Net model from GitHub."""
        try:
            results = self.pipeline.step(frame)

            if results.pitch is None or len(results.pitch) == 0:
                return None

            pitch = float(results.pitch[0])
            yaw = float(results.yaw[0])
            screen_x, screen_y = self._angles_to_screen(pitch, yaw)
            confidence = self._estimate_confidence(results)

            return GazeEstimate(
                x=screen_x,
                y=screen_y,
                pitch=pitch,
                yaw=yaw,
                confidence=confidence,
                timestamp=timestamp,
                server_timestamp=server_timestamp,
                source='l2cs'
            )

        except Exception as e:
            print(f"L2CS estimation error: {e}")
            return None

    def _estimate_simple(self, frame: np.ndarray, timestamp: float, server_timestamp: float) -> Optional[GazeEstimate]:
        """Simple geometric estimation using face detection."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                return None

            x, y, w, h = faces[0]
            face_center_x = x + w / 2
            face_center_y = y + h / 2

            frame_h, frame_w = frame.shape[:2]
            norm_x = (face_center_x / frame_w - 0.5) * 2
            norm_y = (face_center_y / frame_h - 0.5) * 2

            yaw = -norm_x * 0.5
            pitch = norm_y * 0.35

            screen_x, screen_y = self._angles_to_screen(pitch, yaw)

            return GazeEstimate(
                x=screen_x,
                y=screen_y,
                pitch=pitch,
                yaw=yaw,
                confidence=0.5,
                timestamp=timestamp,
                server_timestamp=server_timestamp,
                source='simple'
            )

        except Exception as e:
            print(f"Simple estimation error: {e}")
            return None

    def _estimate_demo(self, timestamp: float, server_timestamp: float) -> GazeEstimate:
        """Demo mode with random walk gaze simulation."""
        self._demo_gaze['x'] += np.random.normal(0, 30)
        self._demo_gaze['y'] += np.random.normal(0, 20)
        self._demo_gaze['x'] = np.clip(self._demo_gaze['x'], 0, self.screen_width)
        self._demo_gaze['y'] = np.clip(self._demo_gaze['y'], 0, self.screen_height)

        norm_x = (self._demo_gaze['x'] / self.screen_width - 0.5) * 2
        norm_y = (self._demo_gaze['y'] / self.screen_height - 0.5) * 2

        return GazeEstimate(
            x=self._demo_gaze['x'],
            y=self._demo_gaze['y'],
            pitch=norm_y * 0.35,
            yaw=norm_x * 0.5,
            confidence=0.3,
            timestamp=timestamp,
            server_timestamp=server_timestamp,
            source='demo'
        )

    def _angles_to_screen(self, pitch: float, yaw: float) -> Tuple[float, float]:
        """Convert gaze angles to screen coordinates."""
        base_scale_x = self.screen_width / 1.0
        base_scale_y = self.screen_height / 0.7

        if self.calibration.model_fitted:
            x = self.screen_width / 2 + yaw * base_scale_x * self.calibration.scale_x + self.calibration.offset_x
            y = self.screen_height / 2 + pitch * base_scale_y * self.calibration.scale_y + self.calibration.offset_y
        else:
            x = self.screen_width / 2 + yaw * base_scale_x
            y = self.screen_height / 2 + pitch * base_scale_y

        x = np.clip(x, 0, self.screen_width)
        y = np.clip(y, 0, self.screen_height)

        return float(x), float(y)

    def _estimate_confidence(self, results) -> float:
        """Estimate confidence based on detection quality."""
        if results.bboxes is None or len(results.bboxes) == 0:
            return 0.0

        bbox = results.bboxes[0]
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        confidence = min(1.0, face_area / 40000)

        return float(confidence)

    def add_calibration_point(self, screen_x: float, screen_y: float, pitch: float, yaw: float, timestamp: float):
        """Add a calibration point."""
        self.calibration.add_point(screen_x, screen_y, pitch, yaw, timestamp)

    def fit_calibration(self) -> bool:
        """Fit calibration model from collected points."""
        if len(self.calibration.points) < 5:
            print("Need at least 5 calibration points")
            return False

        try:
            screen_x = np.array([p.screen_x for p in self.calibration.points])
            screen_y = np.array([p.screen_y for p in self.calibration.points])
            pitch = np.array([p.pitch for p in self.calibration.points])
            yaw = np.array([p.yaw for p in self.calibration.points])

            if np.std(yaw) > 0.01:
                coeffs_x = np.polyfit(yaw, screen_x, 1)
                self.calibration.scale_x = coeffs_x[0] / (self.screen_width / 1.0)
                self.calibration.offset_x = coeffs_x[1] - self.screen_width / 2

            if np.std(pitch) > 0.01:
                coeffs_y = np.polyfit(pitch, screen_y, 1)
                self.calibration.scale_y = coeffs_y[0] / (self.screen_height / 0.7)
                self.calibration.offset_y = coeffs_y[1] - self.screen_height / 2

            self.calibration.model_fitted = True
            print(f"Calibration fitted with {len(self.calibration.points)} points")
            return True

        except Exception as e:
            print(f"Calibration fitting error: {e}")
            return False

    def clear_calibration(self):
        """Clear calibration data."""
        self.calibration = CalibrationData()

    def save_calibration(self, filepath: str):
        """Save calibration to file."""
        data = {
            'points': [
                {'screen_x': p.screen_x, 'screen_y': p.screen_y, 'pitch': p.pitch, 'yaw': p.yaw, 'timestamp': p.timestamp}
                for p in self.calibration.points
            ],
            'model_fitted': self.calibration.model_fitted,
            'offset_x': self.calibration.offset_x,
            'offset_y': self.calibration.offset_y,
            'scale_x': self.calibration.scale_x,
            'scale_y': self.calibration.scale_y,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_calibration(self, filepath: str) -> bool:
        """Load calibration from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.calibration = CalibrationData(
                points=[CalibrationPoint(**p) for p in data['points']],
                model_fitted=data['model_fitted'],
                offset_x=data['offset_x'],
                offset_y=data['offset_y'],
                scale_x=data['scale_x'],
                scale_y=data['scale_y']
            )
            return True

        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False


class AsyncGazeEstimator:
    """Async wrapper for L2CS gaze estimation."""

    def __init__(self, estimator: Optional[L2CSGazeEstimator] = None, max_queue_size: int = 10):
        self.estimator = estimator or L2CSGazeEstimator()
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._latest_result: Optional[GazeEstimate] = None
        self._result_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background processing thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("Async gaze estimator started")

    def stop(self):
        """Stop the background processing thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def submit_frame(self, frame_data: str, timestamp: float):
        """Submit a frame for processing."""
        try:
            self._queue.put_nowait((frame_data, timestamp))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((frame_data, timestamp))
            except queue.Empty:
                pass

    def get_latest_result(self) -> Optional[GazeEstimate]:
        """Get the most recent gaze estimate."""
        with self._result_lock:
            return self._latest_result

    def _process_loop(self):
        """Background processing loop."""
        while self._running:
            try:
                frame_data, timestamp = self._queue.get(timeout=0.1)
                result = self.estimator.estimate_from_base64(frame_data, timestamp)

                if result:
                    with self._result_lock:
                        self._latest_result = result

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Gaze processing error: {e}")


def create_gaze_estimator(
    screen_width: int = 1920,
    screen_height: int = 1080,
    async_mode: bool = True
) -> AsyncGazeEstimator:
    """Create a gaze estimator with default settings."""
    estimator = L2CSGazeEstimator(screen_width=screen_width, screen_height=screen_height)
    async_estimator = AsyncGazeEstimator(estimator)

    if async_mode:
        async_estimator.start()

    return async_estimator
