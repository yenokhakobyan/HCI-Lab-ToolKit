"""
Web HCI Collector Server

FastAPI server with WebSocket support for real-time HCI data collection.
Collects: eye gaze, face mesh, cognitive states, mouse, keyboard events.
"""

import asyncio
import json
import logging
import os
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import threading
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

from .session_manager import SessionManager, Session, SessionStatus
from .data_processor import DataProcessor
from .emotion_detector import EmotionDetector, AsyncEmotionDetector
from .gaze_estimator import L2CSGazeEstimator, AsyncGazeEstimator, create_gaze_estimator
from .analytics_engine import AnalyticsEngine


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean from an environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


def _default_data_dir() -> str:
    """Portable default data directory (relative to project root)."""
    env = os.environ.get("HCI_DATA_DIR")
    if env:
        return env
    return str(Path(__file__).parent.parent.parent.parent / "data" / "raw" / "web_hci")


@dataclass
class ServerConfig:
    """Configuration for the Web HCI Collector Server."""
    host: str = field(default_factory=lambda: os.environ.get("HCI_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("HCI_PORT", "8000")))
    output_dir: str = field(default_factory=_default_data_dir)
    enable_emotion_detection: bool = field(default_factory=lambda: _env_bool("HCI_ENABLE_EMOTION", True))
    enable_l2cs_gaze: bool = field(default_factory=lambda: _env_bool("HCI_ENABLE_L2CS", False))
    save_interval_seconds: int = 30
    debug: bool = field(default_factory=lambda: _env_bool("HCI_DEBUG", False))
    ssl_enabled: bool = field(default_factory=lambda: _env_bool("HCI_SSL_ENABLED", False))
    ssl_certfile: Optional[str] = field(default_factory=lambda: os.environ.get("HCI_SSL_CERTFILE") or None)
    ssl_keyfile: Optional[str] = field(default_factory=lambda: os.environ.get("HCI_SSL_KEYFILE") or None)
    screen_width: int = field(default_factory=lambda: int(os.environ.get("HCI_SCREEN_WIDTH", "1920")))
    screen_height: int = field(default_factory=lambda: int(os.environ.get("HCI_SCREEN_HEIGHT", "1080")))


class WebHCICollectorServer:
    """
    Web-based HCI data collection server with real-time visualization.

    Collects:
    - Eye gaze (WebGazer.js)
    - Face mesh landmarks (MediaPipe)
    - Cognitive states (DenseAttNet - confusion, engagement, boredom, frustration)
    - Mouse movement and clicks
    - Keyboard events

    Example:
        server = WebHCICollectorServer()
        server.run()  # Opens browser to http://127.0.0.1:8000
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.app = FastAPI(title="Web HCI Collector")

        # CORS middleware
        cors_origins = os.environ.get("HCI_CORS_ORIGINS", "")
        if cors_origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=[o.strip() for o in cors_origins.split(",")],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Initialize components
        self.session_manager = SessionManager()
        self.data_processor = DataProcessor(output_dir=self.config.output_dir)

        # Start periodic data flush
        if self.config.save_interval_seconds > 0:
            self.data_processor.start_periodic_save(self.config.save_interval_seconds)

        # Initialize emotion detector
        if self.config.enable_emotion_detection:
            model_path = Path(__file__).parent / "models" / "denseattnet.onnx"
            if model_path.exists():
                backend = "onnx"
                model_path_str = str(model_path)
            else:
                backend = "landmarks"
                model_path_str = None
            self.emotion_detector = EmotionDetector(
                model_path=model_path_str,
                backend=backend,
            )
            self.async_emotion_detector = AsyncEmotionDetector(self.emotion_detector)
            self.async_emotion_detector.start()
        else:
            self.emotion_detector = None
            self.async_emotion_detector = None

        # Initialize L2CS gaze estimator
        if self.config.enable_l2cs_gaze:
            self.gaze_estimator = create_gaze_estimator(
                screen_width=self.config.screen_width,
                screen_height=self.config.screen_height,
                async_mode=True
            )
            print(f"L2CS gaze estimator enabled (screen: {self.config.screen_width}x{self.config.screen_height})")
        else:
            self.gaze_estimator = None

        # Track connected clients for broadcasting
        self.connected_clients: Dict[str, WebSocket] = {}
        self.dashboard_clients: List[WebSocket] = []

        # Cache session start data so late-joining dashboards get participant dimensions
        self.session_start_cache: Dict[str, Dict[str, Any]] = {}

        # Track actual session start times for duration calculation
        self.session_start_times: Dict[str, datetime] = {}

        # Background task for emotion detection broadcasting
        self._emotion_broadcast_task = None

        # Guard against duplicate finalization from beacon + WS disconnect race.
        # Bounded to prevent unbounded growth; old entries are safe to evict
        # because the race window is only seconds, not hours.
        self._finalized_sessions: set = set()
        self._max_finalized_cache = 200

        # Setup routes and lifecycle
        self._setup_routes()
        self._setup_static_files()
        self._setup_shutdown()

        # Output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _setup_shutdown(self):
        """Register shutdown handler to release camera and background resources."""
        @self.app.on_event("shutdown")
        async def shutdown():
            if self.async_emotion_detector:
                self.async_emotion_detector.stop()
            if self.gaze_estimator:
                self.gaze_estimator.stop()
            # Stop periodic save and flush all remaining in-memory data
            self.data_processor.stop_periodic_save()
            for sid in list(self.data_processor.buffers.keys()):
                try:
                    self.data_processor.flush_session_to_disk(sid)
                except Exception as e:
                    print(f"Shutdown flush error for {sid}: {e}")
            print("Background processors stopped and data flushed.")

    def _setup_static_files(self):
        """Mount static files directory."""
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            """Serve the main experiment page."""
            html_path = Path(__file__).parent / "static" / "index.html"
            if html_path.exists():
                return FileResponse(html_path)
            return HTMLResponse("<h1>Web HCI Collector</h1><p>Static files not found.</p>")

        @self.app.get("/calibration", response_class=HTMLResponse)
        async def calibration():
            """Serve the calibration page."""
            html_path = Path(__file__).parent / "static" / "calibration.html"
            if html_path.exists():
                return FileResponse(html_path)
            return HTMLResponse("<h1>Calibration</h1><p>Calibration page not found.</p>")

        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard():
            """Serve the real-time visualization dashboard."""
            html_path = Path(__file__).parent / "static" / "dashboard.html"
            if html_path.exists():
                return FileResponse(html_path)
            return HTMLResponse("<h1>Dashboard</h1><p>Dashboard not found.</p>")

        # --- Analytics ---

        @self.app.get("/analytics", response_class=HTMLResponse)
        async def analytics_page():
            """Serve the analytics toolbox page."""
            html_path = Path(__file__).parent / "static" / "analytics.html"
            if html_path.exists():
                return FileResponse(html_path)
            return HTMLResponse("<h1>Analytics</h1><p>Analytics page not found.</p>")

        @self.app.get("/analytics/{session_id}", response_class=HTMLResponse)
        async def analytics_session_page(session_id: str):
            """Serve the analytics page for a specific session."""
            html_path = Path(__file__).parent / "static" / "analytics.html"
            if html_path.exists():
                return FileResponse(html_path)
            return HTMLResponse("<h1>Analytics</h1><p>Analytics page not found.</p>")

        @self.app.get("/api/session/{session_id}/analytics")
        async def compute_analytics(session_id: str):
            """Compute and return comprehensive HCI analytics for a session."""
            engine = AnalyticsEngine(self.data_processor)
            try:
                result = engine.analyze_session(session_id)
                return result
            except Exception as e:
                logging.error(f"Analytics computation failed: {e}", exc_info=True)
                return {"error": str(e)}

        @self.app.get("/api/sessions/compare")
        async def compare_sessions(ids: str = ""):
            """Compare metrics across multiple sessions."""
            session_ids = [s.strip() for s in ids.split(",") if s.strip()]
            if len(session_ids) < 2:
                return {"error": "Provide at least 2 session IDs (comma-separated)"}
            engine = AnalyticsEngine(self.data_processor)
            try:
                return engine.compare_sessions(session_ids)
            except Exception as e:
                logging.error(f"Session comparison failed: {e}", exc_info=True)
                return {"error": str(e)}

        @self.app.get("/participate/{session_id}", response_class=HTMLResponse)
        async def participate(session_id: str):
            """Serve the participant page for a specific session."""
            session = self.session_manager.get_session(session_id)
            if not session:
                return HTMLResponse(
                    "<h1>Session not found</h1><p>This session ID is invalid or has expired.</p>",
                    status_code=404,
                )
            if session.status == SessionStatus.COMPLETED:
                return HTMLResponse(
                    "<h1>Session completed</h1><p>This session has already been completed. Thank you!</p>"
                )
            html_path = Path(__file__).parent / "static" / "participate.html"
            if html_path.exists():
                return FileResponse(html_path)
            return HTMLResponse("<h1>Participant page not found</h1>")

        # --- Session API ---

        @self.app.get("/api/sessions")
        async def get_sessions(active_only: bool = False):
            """Get all sessions, optionally filtering to active ones."""
            if active_only:
                sessions = self.session_manager.get_active_sessions()
            else:
                sessions = self.session_manager.get_all_sessions()
            return {"sessions": [s.to_dict() for s in sessions]}

        @self.app.post("/api/sessions")
        async def create_session(request: Request):
            """Create a new participant session from the dashboard."""
            body = await request.json()
            participant_id = body.get("participant_id")
            experiment_config = body.get("experiment_config", {})

            session = self.session_manager.create_session_with_config(
                participant_id=participant_id,
                experiment_config=experiment_config,
            )

            # Derive URL from request Host header so it works across machines
            protocol = "https" if self.config.ssl_enabled else "http"
            host_header = request.headers.get("host", f"{self.config.host}:{self.config.port}")
            participant_url = f"{protocol}://{host_header}/participate/{session.session_id}"

            return {
                "session_id": session.session_id,
                "participant_url": participant_url,
                "status": session.status.value,
                "created_at": session.created_at.isoformat(),
            }

        @self.app.get("/api/session/{session_id}")
        async def get_session(session_id: str):
            """Get session details."""
            session = self.session_manager.get_session(session_id)
            if session:
                return session.to_dict()
            return {"error": "Session not found"}

        @self.app.get("/api/session/{session_id}/config")
        async def get_session_config(session_id: str):
            """Get experiment configuration for a session (used by participant page)."""
            session = self.session_manager.get_session(session_id)
            if not session:
                return {"error": "Session not found"}
            return {
                "session_id": session.session_id,
                "experiment_config": session.experiment_config,
                "status": session.status.value,
            }

        @self.app.patch("/api/session/{session_id}/status")
        async def update_session_status(session_id: str, request: Request):
            """Update session status (state transitions)."""
            body = await request.json()
            new_status_str = body.get("status")
            try:
                new_status = SessionStatus(new_status_str)
            except ValueError:
                return {"success": False, "error": f"Invalid status: {new_status_str}"}

            session = self.session_manager.transition_session(session_id, new_status)
            if session:
                # Broadcast status change to dashboard
                await self._broadcast_to_dashboard(session_id, {
                    "type": "session_status",
                    "data": {"status": session.status.value, "session_id": session_id},
                })
                return {"success": True, "status": session.status.value}
            return {"success": False, "error": "Invalid transition or session not found"}

        @self.app.post("/api/session/{session_id}/export")
        async def export_session(session_id: str, format: str = "csv"):
            """Export session data."""
            filepath = self.data_processor.export_session(session_id, format=format)
            if filepath:
                return {"success": True, "filepath": str(filepath)}
            return {"success": False, "error": "Export failed"}

        @self.app.post("/api/session/{session_id}/save-timeline")
        async def save_timeline(session_id: str, request: Request):
            """Save timeline data from dashboard."""
            try:
                data = await request.json()

                session_dir = Path(self.config.output_dir) / session_id
                session_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                timeline_path = session_dir / f"timeline_{timestamp}.json"

                with open(timeline_path, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Timeline saved for session {session_id}")
                return {"success": True, "filepath": str(timeline_path)}
            except Exception as e:
                print(f"Error saving timeline: {e}")
                return {"success": False, "error": str(e)}

        @self.app.post("/api/session/{session_id}/disconnect")
        async def session_disconnect(session_id: str):
            """Beacon endpoint for reliable participant disconnect notification.

            Called via navigator.sendBeacon() from the participant page during
            beforeunload/pagehide, ensuring the server knows the session ended
            even if the WebSocket message was lost.
            """
            # Guard: only finalize once (beacon and WS disconnect can race)
            if session_id in self._finalized_sessions:
                return {"success": True}

            # Prevent unbounded growth of the finalized set
            if len(self._finalized_sessions) > self._max_finalized_cache:
                self._finalized_sessions.clear()

            session = self.session_manager.get_session(session_id)
            if session and session.status not in (
                SessionStatus.COMPLETED, SessionStatus.ABANDONED
            ):
                self._finalized_sessions.add(session_id)
                self.session_manager.end_session(session_id)
                self.data_processor.flush_session_to_disk(session_id)
                await self._finalize_video(session_id)
                await self._save_session_metadata(session_id)
                await self._broadcast_to_dashboard(session_id, {
                    "type": "session_status",
                    "data": {"status": "abandoned", "session_id": session_id},
                })
            return {"success": True}

        @self.app.post("/api/session/{session_id}/save-video")
        async def save_video(session_id: str):
            """Save video from request body."""
            session_dir = Path(self.config.output_dir) / session_id
            video_path = session_dir / "recording.webm"
            return {"success": True, "filepath": str(video_path)}

        @self.app.get("/api/session/{session_id}/data")
        async def get_session_data(session_id: str):
            """Get all saved session data for replay in dashboard."""
            session_dir = Path(self.config.output_dir) / session_id

            if not session_dir.exists():
                return {"success": False, "error": "Session data not found"}

            result = {
                "success": True,
                "session_id": session_id,
                "files": {}
            }

            # Find all data files
            for file_path in session_dir.glob("*"):
                if file_path.is_file():
                    file_type = file_path.stem.split("_")[0]
                    if file_path.suffix == ".json":
                        try:
                            with open(file_path, 'r') as f:
                                result["files"][file_type] = json.load(f)
                        except:
                            pass
                    elif file_path.suffix == ".csv":
                        result["files"][f"{file_type}_csv"] = str(file_path)
                    elif file_path.suffix == ".webm":
                        result["files"]["video"] = f"/api/session/{session_id}/video"

            return result

        @self.app.get("/api/session/{session_id}/video")
        async def get_session_video(session_id: str):
            """Stream the session video file."""
            session_dir = Path(self.config.output_dir) / session_id
            video_path = session_dir / "recording.webm"

            if video_path.exists():
                return FileResponse(
                    video_path,
                    media_type="video/webm",
                    filename=f"session_{session_id}_recording.webm"
                )
            return {"error": "Video not found"}

        @self.app.get("/api/session/{session_id}/face-mesh")
        async def get_session_face_mesh(session_id: str):
            """Load face mesh data from CSV for session playback.

            Parses the face_mesh CSV file and returns landmarks, key_points,
            head_pose and bounding_box aligned to timeline-relative timestamps.
            Subsamples every 2nd row to reduce payload size.
            """
            session_dir = Path(self.config.output_dir) / session_id
            if not session_dir.exists():
                return {"success": False, "error": "Session not found"}

            # Find face mesh CSV (prefer live version, fall back to timestamped export)
            csv_path = session_dir / "face_mesh_live.csv"
            if not csv_path.exists():
                candidates = sorted(session_dir.glob("face_mesh_*.csv"))
                if candidates:
                    csv_path = candidates[-1]  # Use latest export
                else:
                    return {"success": False, "error": "No face mesh data found"}

            try:
                import csv as csv_mod

                # Find the session start reference timestamp from gaze CSV
                # (gaze is the earliest data stream, and timeline uses the same reference)
                # The gaze CSV has performance.now() timestamps; the timeline uses
                # Date.now() - startTime. We align by subtracting gaze's first timestamp
                # from all CSV timestamps, matching the timeline reference.
                ref_ts = None
                gaze_csv = session_dir / "gaze_live.csv"
                if not gaze_csv.exists():
                    gaze_candidates = sorted(session_dir.glob("gaze_*.csv"))
                    if gaze_candidates:
                        gaze_csv = gaze_candidates[-1]

                if gaze_csv.exists():
                    with open(gaze_csv, 'r') as gf:
                        greader = csv_mod.DictReader(gf)
                        first_gaze = next(greader, None)
                        if first_gaze and first_gaze.get('timestamp'):
                            ref_ts = float(first_gaze['timestamp'])

                data = []

                with open(csv_path, 'r') as f:
                    reader = csv_mod.DictReader(f)
                    for i, row in enumerate(reader):
                        # Subsample: take every 2nd row to reduce payload
                        if i % 2 != 0:
                            continue

                        ts = float(row['timestamp'])
                        # Use gaze first timestamp as reference (aligns with timeline)
                        # Fall back to first face_mesh timestamp if no gaze data
                        if ref_ts is None:
                            ref_ts = ts

                        # Convert to timeline-relative time (ms from session start)
                        timeline_time = round(ts - ref_ts)

                        entry = {'time': timeline_time}

                        # Parse JSON fields
                        if row.get('landmarks'):
                            try:
                                entry['landmarks'] = json.loads(row['landmarks'])
                            except (json.JSONDecodeError, TypeError):
                                pass

                        if row.get('key_points'):
                            try:
                                entry['key_points'] = json.loads(row['key_points'])
                            except (json.JSONDecodeError, TypeError):
                                pass

                        if row.get('head_pose'):
                            try:
                                entry['head_pose'] = json.loads(row['head_pose'])
                            except (json.JSONDecodeError, TypeError):
                                pass

                        if row.get('bounding_box'):
                            try:
                                entry['bounding_box'] = json.loads(row['bounding_box'])
                            except (json.JSONDecodeError, TypeError):
                                pass

                        if row.get('landmark_count'):
                            entry['landmark_count'] = int(row['landmark_count'])

                        data.append(entry)

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Face mesh loaded for {session_id}: {len(data)} frames")
                return {"success": True, "data": data}

            except Exception as e:
                print(f"Error loading face mesh data: {e}")
                return {"success": False, "error": str(e)}

        # --- WebSocket endpoints ---

        @self.app.websocket("/ws/collect/{session_id}")
        async def websocket_collect(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for data collection."""
            await websocket.accept()

            # Create or get session
            session = self.session_manager.get_or_create_session(session_id)
            self.connected_clients[session_id] = websocket

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Client connected: {session_id}")

            try:
                while True:
                    # Receive data from client
                    data = await websocket.receive_json()

                    # Process incoming data
                    await self._process_client_data(session_id, data)

                    # Broadcast to dashboard clients
                    await self._broadcast_to_dashboard(session_id, data)

            except WebSocketDisconnect:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Client disconnected: {session_id}")
                self.connected_clients.pop(session_id, None)
                self.session_start_cache.pop(session_id, None)

                # Guard: only finalize once (beacon and WS disconnect can race)
                if session_id not in self._finalized_sessions:
                    self._finalized_sessions.add(session_id)
                    self.session_manager.end_session(session_id)
                    # Flush data to disk on disconnect
                    self.data_processor.flush_session_to_disk(session_id)
                    # Finalize video and save session metadata with duration
                    await self._finalize_video(session_id)
                    await self._save_session_metadata(session_id)
                    # Notify dashboard
                    await self._broadcast_to_dashboard(session_id, {
                        "type": "session_status",
                        "data": {"status": "abandoned", "session_id": session_id},
                    })
            except Exception as e:
                print(f"WebSocket error: {e}")
                self.connected_clients.pop(session_id, None)

        @self.app.websocket("/ws/dashboard")
        async def websocket_dashboard(websocket: WebSocket):
            """WebSocket endpoint for real-time dashboard."""
            await websocket.accept()
            self.dashboard_clients.append(websocket)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Dashboard client connected")

            # Send cached session start data so late-joining dashboards
            # get participant window dimensions and content URL
            for sid, start_data in self.session_start_cache.items():
                try:
                    await websocket.send_json({
                        "session_id": sid,
                        "type": "session",
                        "timestamp": datetime.now().timestamp() * 1000,
                        "data": {"event": "start", **start_data}
                    })
                except Exception:
                    pass

            try:
                while True:
                    # Keep connection alive, receive any dashboard commands
                    data = await websocket.receive_json()
                    # Handle dashboard commands if needed

            except WebSocketDisconnect:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Dashboard client disconnected")
                if websocket in self.dashboard_clients:
                    self.dashboard_clients.remove(websocket)
            except Exception as e:
                print(f"Dashboard WebSocket error: {e}")
                if websocket in self.dashboard_clients:
                    self.dashboard_clients.remove(websocket)

    async def _process_client_data(self, session_id: str, data: Dict[str, Any]):
        """Process incoming data from a client."""
        data_type = data.get("type")
        timestamp = data.get("timestamp", datetime.now().timestamp() * 1000)
        payload = data.get("data", {})

        # Update session counts
        self.session_manager.update_session_counts(session_id, data_type)

        # Only store data types that have a matching DataBuffer field.
        # Types like session, step_transition, video_chunk, etc. are handled
        # for side effects below but don't need persistent storage.
        _NON_STORABLE_TYPES = {
            "webcam_frame", "session", "step_transition",
            "calibration_complete", "gaze_frame", "l2cs_calibration",
            "video_chunk", "video_complete",
        }

        if data_type not in _NON_STORABLE_TYPES:
            self.data_processor.add_data(
                session_id=session_id,
                data_type=data_type,
                timestamp=timestamp,
                data=payload
            )

        # Handle video chunks - save to file
        if data_type == "video_chunk" and payload.get("data"):
            await self._save_video_chunk(session_id, payload)

        # Cache session start data for late-joining dashboards
        if data_type == "session" and payload.get("event") == "start":
            self.session_start_cache[session_id] = payload
            self.session_start_times[session_id] = datetime.now()

        # Handle session end - finalize video file, save metadata, and clear cache
        if data_type == "session" and payload.get("event") == "end":
            self.session_start_cache.pop(session_id, None)
            await self._finalize_video(session_id)
            await self._save_session_metadata(session_id)

        # Handle step transitions
        if data_type == "step_transition":
            step = payload.get("step")
            status_map = {
                "calibration": SessionStatus.CALIBRATING,
                "content": SessionStatus.COLLECTING,
                "complete": SessionStatus.COMPLETED,
            }
            new_status = status_map.get(step)
            if new_status:
                self.session_manager.transition_session(session_id, new_status)
                await self._broadcast_to_dashboard(session_id, {
                    "type": "session_status",
                    "data": {"status": new_status.value, "session_id": session_id},
                })

        # Handle calibration complete
        if data_type == "calibration_complete":
            self.session_manager.set_calibrated(session_id, True)
            self.session_manager.transition_session(session_id, SessionStatus.COLLECTING)

        # Process face mesh for emotion detection
        if data_type == "face_mesh" and self.async_emotion_detector:
            # Submit landmarks for cognitive state estimation
            landmarks = payload.get("landmarks")
            head_pose_data = payload.get("head_pose")
            if landmarks and len(landmarks) >= 468:
                self.async_emotion_detector.submit_landmarks(landmarks, head_pose_data)

            # Get latest emotion prediction and broadcast
            emotion_state = self.async_emotion_detector.get_latest_state()
            if emotion_state:
                emotion_data = {
                    "type": "emotion",
                    "data": emotion_state.to_dict()
                }
                # Store emotion data
                self.data_processor.add_data(
                    session_id=session_id,
                    data_type="emotion",
                    timestamp=timestamp,
                    data=emotion_state.to_dict()
                )
                # Broadcast to dashboard
                await self._broadcast_to_dashboard(session_id, emotion_data)

        # Process gaze frame for L2CS estimation
        if data_type == "gaze_frame" and self.gaze_estimator:
            frame_data = payload.get("frame")
            if frame_data:
                # Submit frame for async processing
                self.gaze_estimator.submit_frame(frame_data, timestamp)

                # Get latest L2CS gaze result and broadcast
                gaze_result = self.gaze_estimator.get_latest_result()
                if gaze_result:
                    l2cs_data = {
                        "type": "l2cs_gaze",
                        "data": gaze_result.to_dict()
                    }
                    # Store L2CS gaze data
                    self.data_processor.add_data(
                        session_id=session_id,
                        data_type="l2cs_gaze",
                        timestamp=timestamp,
                        data=gaze_result.to_dict()
                    )
                    # Broadcast to dashboard
                    await self._broadcast_to_dashboard(session_id, l2cs_data)

        # Process calibration data for L2CS
        if data_type == "l2cs_calibration" and self.gaze_estimator:
            screen_x = payload.get("screen_x")
            screen_y = payload.get("screen_y")
            pitch = payload.get("pitch")
            yaw = payload.get("yaw")

            if all(v is not None for v in [screen_x, screen_y, pitch, yaw]):
                self.gaze_estimator.estimator.add_calibration_point(
                    screen_x, screen_y, pitch, yaw, timestamp
                )

            # If calibration command is to fit the model
            if payload.get("action") == "fit":
                success = self.gaze_estimator.estimator.fit_calibration()
                await self._broadcast_to_dashboard(session_id, {
                    "type": "l2cs_calibration_result",
                    "data": {"success": success}
                })

    async def _save_video_chunk(self, session_id: str, payload: Dict[str, Any]):
        """Save video chunk to file."""
        try:
            session_dir = Path(self.config.output_dir) / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            video_path = session_dir / "recording.webm"

            # Decode base64 video data
            video_data = base64.b64decode(payload["data"])

            # Append to video file
            mode = 'ab' if video_path.exists() else 'wb'
            with open(video_path, mode) as f:
                f.write(video_data)

            chunk_index = payload.get("chunkIndex", 0)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Video chunk #{chunk_index + 1} saved for session {session_id}")
        except Exception as e:
            print(f"Error saving video chunk: {e}")

    async def _finalize_video(self, session_id: str):
        """Finalize video file after session ends."""
        session_dir = Path(self.config.output_dir) / session_id
        video_path = session_dir / "recording.webm"

        if video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Video finalized for session {session_id}: {size_mb:.2f} MB")

    async def _save_session_metadata(self, session_id: str):
        """Save session metadata (duration, status) to disk for later retrieval."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return

        session_dir = Path(self.config.output_dir) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Use tracked start time if available, otherwise fall back to session created_at
        start_time = self.session_start_times.get(session_id, session.created_at)
        end_time = session.ended_at or datetime.now()

        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        metadata = {
            "session_id": session_id,
            "status": session.status.value,
            "started_at": start_time.isoformat(),
            "ended_at": end_time.isoformat(),
            "duration_ms": max(duration_ms, 0),
        }

        meta_path = session_dir / "session_metadata.json"
        try:
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Session metadata saved for {session_id}: {duration_ms}ms")
        except Exception as e:
            print(f"Error saving session metadata: {e}")

        # Clean up start time cache
        self.session_start_times.pop(session_id, None)

    async def _broadcast_to_dashboard(self, session_id: str, data: Dict[str, Any]):
        """Broadcast data to all connected dashboard clients."""
        if not self.dashboard_clients:
            return

        message = {
            "session_id": session_id,
            "timestamp": datetime.now().timestamp() * 1000,
            **data
        }

        disconnected = []
        for client in self.dashboard_clients:
            try:
                await client.send_json(message)
            except:
                disconnected.append(client)

        # Remove disconnected clients
        for client in disconnected:
            self.dashboard_clients.remove(client)

    def _generate_ssl_certs(self) -> tuple:
        """Generate self-signed SSL certificates for local development."""
        from pathlib import Path
        import subprocess

        certs_dir = Path(__file__).parent / "certs"
        certs_dir.mkdir(exist_ok=True)

        cert_file = certs_dir / "cert.pem"
        key_file = certs_dir / "key.pem"

        if cert_file.exists() and key_file.exists():
            print("  Using existing SSL certificates")
            return str(cert_file), str(key_file)

        print("  Generating self-signed SSL certificates...")

        # Generate self-signed certificate using openssl
        try:
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:4096",
                "-keyout", str(key_file),
                "-out", str(cert_file),
                "-days", "365",
                "-nodes",
                "-subj", "/CN=localhost"
            ], check=True, capture_output=True)
            print("  SSL certificates generated successfully")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  Warning: Could not generate SSL certs: {e}")
            print("  Falling back to HTTP (webcam may not work)")
            return None, None

        return str(cert_file), str(key_file)

    def run(self, open_browser: bool = True):
        """
        Run the server.

        Args:
            open_browser: Automatically open browser to the dashboard
        """
        # Setup SSL for HTTPS (required for WebGazer webcam access)
        ssl_certfile = self.config.ssl_certfile
        ssl_keyfile = self.config.ssl_keyfile

        if self.config.ssl_enabled and not (ssl_certfile and ssl_keyfile):
            ssl_certfile, ssl_keyfile = self._generate_ssl_certs()

        use_https = ssl_certfile and ssl_keyfile
        protocol = "https" if use_https else "http"

        if open_browser:
            import webbrowser
            url = f"{protocol}://{self.config.host}:{self.config.port}/dashboard"
            # Open browser after a short delay to allow server to start
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        print(f"\n{'='*60}")
        print(f"  Web HCI Collector Server")
        print(f"{'='*60}")
        if use_https:
            print(f"  Mode:         HTTPS (required for webcam access)")
        else:
            print(f"  Mode:         HTTP (webcam may not work!)")
        print(f"  Dashboard:    {protocol}://{self.config.host}:{self.config.port}/dashboard")
        print(f"  Experiment:   {protocol}://{self.config.host}:{self.config.port}/")
        print(f"  Calibration:  {protocol}://{self.config.host}:{self.config.port}/calibration")
        print(f"  Analytics:    {protocol}://{self.config.host}:{self.config.port}/analytics")
        print(f"  Output:       {self.config.output_dir}")
        if use_https:
            print(f"\n  NOTE: You may need to accept the self-signed certificate")
            print(f"        in your browser when first accessing the page.")
        print(f"{'='*60}\n")

        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info" if self.config.debug else "warning",
            ssl_certfile=ssl_certfile if use_https else None,
            ssl_keyfile=ssl_keyfile if use_https else None
        )


def main():
    """Run the Web HCI Collector Server."""
    server = WebHCICollectorServer()
    server.run()


if __name__ == "__main__":
    main()
