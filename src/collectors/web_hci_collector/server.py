"""
Web HCI Collector Server

FastAPI server with WebSocket support for real-time HCI data collection.
Collects: eye gaze, face mesh, cognitive states, mouse, keyboard events.
"""

import asyncio
import json
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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

from .session_manager import SessionManager, Session
from .data_processor import DataProcessor
from .emotion_detector import EmotionDetector, AsyncEmotionDetector


@dataclass
class ServerConfig:
    """Configuration for the Web HCI Collector Server."""
    host: str = "127.0.0.1"
    port: int = 8000
    output_dir: str = "data/raw/web_hci"
    enable_emotion_detection: bool = True
    save_interval_seconds: int = 30
    debug: bool = False
    # SSL/HTTPS settings for WebGazer (requires HTTPS for webcam access)
    ssl_enabled: bool = True
    ssl_certfile: Optional[str] = None  # Path to cert.pem
    ssl_keyfile: Optional[str] = None   # Path to key.pem


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

        # Initialize components
        self.session_manager = SessionManager()
        self.data_processor = DataProcessor(output_dir=self.config.output_dir)

        # Initialize emotion detector
        if self.config.enable_emotion_detection:
            model_path = Path(__file__).parent / "models" / "denseattnet.onnx"
            self.emotion_detector = EmotionDetector(
                model_path=str(model_path) if model_path.exists() else None,
                backend="onnx" if model_path.exists() else "demo"
            )
            self.async_emotion_detector = AsyncEmotionDetector(self.emotion_detector)
            self.async_emotion_detector.start()
        else:
            self.emotion_detector = None
            self.async_emotion_detector = None

        # Track connected clients for broadcasting
        self.connected_clients: Dict[str, WebSocket] = {}
        self.dashboard_clients: List[WebSocket] = []

        # Background task for emotion detection broadcasting
        self._emotion_broadcast_task = None

        # Setup routes
        self._setup_routes()
        self._setup_static_files()

        # Output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

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

        @self.app.get("/api/sessions")
        async def get_sessions():
            """Get all active sessions."""
            return {
                "sessions": [
                    asdict(s) for s in self.session_manager.get_active_sessions()
                ]
            }

        @self.app.get("/api/session/{session_id}")
        async def get_session(session_id: str):
            """Get session details."""
            session = self.session_manager.get_session(session_id)
            if session:
                return asdict(session)
            return {"error": "Session not found"}

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

        @self.app.post("/api/session/{session_id}/save-video")
        async def save_video(session_id: str):
            """Save video from request body."""
            # Video is saved via the video chunks during collection
            # This endpoint confirms the save location
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
                self.session_manager.end_session(session_id)
            except Exception as e:
                print(f"WebSocket error: {e}")
                self.connected_clients.pop(session_id, None)

        @self.app.websocket("/ws/dashboard")
        async def websocket_dashboard(websocket: WebSocket):
            """WebSocket endpoint for real-time dashboard."""
            await websocket.accept()
            self.dashboard_clients.append(websocket)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Dashboard client connected")

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

        # Add to session data
        self.data_processor.add_data(
            session_id=session_id,
            data_type=data_type,
            timestamp=timestamp,
            data=payload
        )

        # Handle video chunks - save to file
        if data_type == "video_chunk" and payload.get("data"):
            await self._save_video_chunk(session_id, payload)

        # Handle session end - finalize video file
        if data_type == "session" and payload.get("event") == "end":
            await self._finalize_video(session_id)

        # Process face mesh for emotion detection
        if data_type == "face_mesh" and self.async_emotion_detector:
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
