"""
Data Processor for Web HCI Collector

Handles data storage, synchronization, and export.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
import threading

import pandas as pd
import numpy as np


@dataclass
class DataBuffer:
    """Buffer for storing session data."""
    gaze: List[Dict] = field(default_factory=list)
    face_mesh: List[Dict] = field(default_factory=list)
    emotion: List[Dict] = field(default_factory=list)
    mouse: List[Dict] = field(default_factory=list)
    keyboard: List[Dict] = field(default_factory=list)

    def clear(self):
        self.gaze.clear()
        self.face_mesh.clear()
        self.emotion.clear()
        self.mouse.clear()
        self.keyboard.clear()


class DataProcessor:
    """
    Processes and stores HCI data from web clients.

    Features:
    - Buffered storage for performance
    - Automatic periodic saving
    - Multiple export formats (CSV, Parquet, JSON)
    - Data synchronization across streams
    """

    def __init__(self, output_dir: str = "data/raw/web_hci"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data buffers per session
        self.buffers: Dict[str, DataBuffer] = defaultdict(DataBuffer)
        self._lock = threading.Lock()

    def add_data(self, session_id: str, data_type: str, timestamp: float, data: Dict[str, Any]):
        """
        Add data to the buffer.

        Args:
            session_id: Session identifier
            data_type: Type of data (gaze, face_mesh, emotion, mouse, keyboard)
            timestamp: Client timestamp in milliseconds
            data: Data payload
        """
        with self._lock:
            buffer = self.buffers[session_id]

            # Add common fields
            record = {
                "timestamp": timestamp,
                "server_timestamp": datetime.now().timestamp() * 1000,
                **data
            }

            # Add to appropriate buffer
            if data_type == "gaze":
                buffer.gaze.append(record)
            elif data_type == "face_mesh":
                buffer.face_mesh.append(record)
            elif data_type == "emotion":
                buffer.emotion.append(record)
            elif data_type == "mouse":
                buffer.mouse.append(record)
            elif data_type == "keyboard":
                buffer.keyboard.append(record)

    def get_session_data(self, session_id: str) -> Dict[str, List[Dict]]:
        """Get all data for a session."""
        with self._lock:
            buffer = self.buffers.get(session_id, DataBuffer())
            return {
                "gaze": list(buffer.gaze),
                "face_mesh": list(buffer.face_mesh),
                "emotion": list(buffer.emotion),
                "mouse": list(buffer.mouse),
                "keyboard": list(buffer.keyboard),
            }

    def get_latest_data(self, session_id: str, n: int = 100) -> Dict[str, List[Dict]]:
        """Get the latest n records for each data type."""
        with self._lock:
            buffer = self.buffers.get(session_id, DataBuffer())
            return {
                "gaze": list(buffer.gaze[-n:]),
                "face_mesh": list(buffer.face_mesh[-n:]),
                "emotion": list(buffer.emotion[-n:]),
                "mouse": list(buffer.mouse[-n:]),
                "keyboard": list(buffer.keyboard[-n:]),
            }

    def export_session(self, session_id: str, format: str = "csv") -> Optional[Path]:
        """
        Export session data to file.

        Args:
            session_id: Session identifier
            format: Output format (csv, parquet, json)

        Returns:
            Path to exported file(s)
        """
        data = self.get_session_data(session_id)

        if not any(data.values()):
            return None

        # Create session directory
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exported_files = []

        for data_type, records in data.items():
            if not records:
                continue

            df = pd.DataFrame(records)
            filename = f"{data_type}_{timestamp}"

            if format == "csv":
                filepath = session_dir / f"{filename}.csv"
                df.to_csv(filepath, index=False)
            elif format == "parquet":
                filepath = session_dir / f"{filename}.parquet"
                df.to_parquet(filepath, index=False)
            elif format == "json":
                filepath = session_dir / f"{filename}.json"
                df.to_json(filepath, orient="records", indent=2)
            else:
                raise ValueError(f"Unknown format: {format}")

            exported_files.append(filepath)

        # Also save combined/synchronized data
        self._export_synchronized(session_id, session_dir, timestamp, format)

        return session_dir

    def _export_synchronized(self, session_id: str, session_dir: Path, timestamp: str, format: str):
        """Export synchronized data combining all streams."""
        data = self.get_session_data(session_id)

        # Create a unified timeline
        all_timestamps = set()
        for records in data.values():
            for r in records:
                all_timestamps.add(r.get("timestamp"))

        if not all_timestamps:
            return

        # For now, just save metadata about the session
        metadata = {
            "session_id": session_id,
            "export_timestamp": timestamp,
            "data_counts": {
                "gaze": len(data["gaze"]),
                "face_mesh": len(data["face_mesh"]),
                "emotion": len(data["emotion"]),
                "mouse": len(data["mouse"]),
                "keyboard": len(data["keyboard"]),
            },
            "time_range": {
                "start": min(all_timestamps) if all_timestamps else None,
                "end": max(all_timestamps) if all_timestamps else None,
            }
        }

        metadata_path = session_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def clear_session(self, session_id: str):
        """Clear all data for a session."""
        with self._lock:
            if session_id in self.buffers:
                self.buffers[session_id].clear()

    def get_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        data = self.get_session_data(session_id)

        stats = {}
        for data_type, records in data.items():
            if not records:
                stats[data_type] = {"count": 0}
                continue

            timestamps = [r.get("timestamp", 0) for r in records]
            stats[data_type] = {
                "count": len(records),
                "duration_ms": max(timestamps) - min(timestamps) if timestamps else 0,
                "rate_hz": len(records) / ((max(timestamps) - min(timestamps)) / 1000) if len(timestamps) > 1 else 0,
            }

        return stats
