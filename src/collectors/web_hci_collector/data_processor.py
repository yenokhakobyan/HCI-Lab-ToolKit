"""
Data Processor for Web HCI Collector

Handles data storage, synchronization, and export.
"""

import csv
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
    l2cs_gaze: List[Dict] = field(default_factory=list)
    face_mesh: List[Dict] = field(default_factory=list)
    emotion: List[Dict] = field(default_factory=list)
    mouse: List[Dict] = field(default_factory=list)
    keyboard: List[Dict] = field(default_factory=list)
    experiment_event: List[Dict] = field(default_factory=list)
    answer: List[Dict] = field(default_factory=list)
    hover: List[Dict] = field(default_factory=list)
    calibration_click: List[Dict] = field(default_factory=list)
    calibration_validation: List[Dict] = field(default_factory=list)

    def clear(self):
        self.gaze.clear()
        self.l2cs_gaze.clear()
        self.face_mesh.clear()
        self.emotion.clear()
        self.mouse.clear()
        self.keyboard.clear()
        self.experiment_event.clear()
        self.answer.clear()
        self.hover.clear()
        self.calibration_click.clear()
        self.calibration_validation.clear()


# All data types stored in the buffer
_DATA_TYPES = [
    "gaze", "l2cs_gaze", "face_mesh", "emotion",
    "mouse", "keyboard", "experiment_event", "answer", "hover",
    "calibration_click", "calibration_validation",
]


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

        # Periodic save tracking
        self._flush_indices: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._save_timer: Optional[threading.Timer] = None
        self._save_running = False

    def add_data(self, session_id: str, data_type: str, timestamp: float, data: Dict[str, Any]):
        """
        Add data to the buffer.

        Args:
            session_id: Session identifier
            data_type: Type of data (gaze, face_mesh, emotion, mouse, keyboard, answer, hover, etc.)
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
            buf_list = getattr(buffer, data_type, None)
            if buf_list is not None:
                buf_list.append(record)

    def get_session_data(self, session_id: str) -> Dict[str, List[Dict]]:
        """Get all data for a session."""
        with self._lock:
            buffer = self.buffers.get(session_id, DataBuffer())
            return {dt: list(getattr(buffer, dt)) for dt in _DATA_TYPES}

    def get_latest_data(self, session_id: str, n: int = 100) -> Dict[str, List[Dict]]:
        """Get the latest n records for each data type."""
        with self._lock:
            buffer = self.buffers.get(session_id, DataBuffer())
            return {dt: list(getattr(buffer, dt)[-n:]) for dt in _DATA_TYPES}

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

        # Save metadata about the session
        metadata = {
            "session_id": session_id,
            "export_timestamp": timestamp,
            "data_counts": {dt: len(data[dt]) for dt in _DATA_TYPES},
            "time_range": {
                "start": min(all_timestamps) if all_timestamps else None,
                "end": max(all_timestamps) if all_timestamps else None,
            }
        }

        metadata_path = session_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # --- Periodic auto-save ---

    def start_periodic_save(self, interval_seconds: int = 30):
        """Start a background thread that flushes buffers to disk periodically."""
        if self._save_running:
            return
        self._save_running = True
        self._schedule_save(interval_seconds)
        print(f"Periodic data save started (every {interval_seconds}s)")

    def stop_periodic_save(self):
        """Stop the periodic save background thread."""
        self._save_running = False
        if self._save_timer:
            self._save_timer.cancel()
            self._save_timer = None

    def _schedule_save(self, interval: int):
        """Schedule the next save."""
        if not self._save_running:
            return
        self._save_timer = threading.Timer(interval, self._periodic_save_tick, args=[interval])
        self._save_timer.daemon = True
        self._save_timer.start()

    def _periodic_save_tick(self, interval: int):
        """Execute one save tick, then reschedule."""
        try:
            with self._lock:
                session_ids = list(self.buffers.keys())

            for sid in session_ids:
                self.flush_session_to_disk(sid)
        except Exception as e:
            print(f"Periodic save error: {e}")
        finally:
            self._schedule_save(interval)

    def flush_session_to_disk(self, session_id: str):
        """
        Append new records (since last flush) to incremental CSV files on disk.

        This does NOT clear the in-memory buffer — the buffer is still needed
        for live dashboard streaming and final export.
        """
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            buffer = self.buffers.get(session_id)
            if not buffer:
                return

            indices = self._flush_indices[session_id]

            for data_type in _DATA_TYPES:
                buf_list = getattr(buffer, data_type, [])
                last_idx = indices[data_type]

                if last_idx >= len(buf_list):
                    continue

                new_records = buf_list[last_idx:]
                indices[data_type] = len(buf_list)

                if not new_records:
                    continue

                filepath = session_dir / f"{data_type}_live.csv"
                file_exists = filepath.exists()

                try:
                    # Flatten nested dicts for CSV (skip complex nested objects like landmarks)
                    flat_records = []
                    for rec in new_records:
                        flat = {}
                        for k, v in rec.items():
                            if isinstance(v, (dict, list)):
                                flat[k] = json.dumps(v)
                            else:
                                flat[k] = v
                        flat_records.append(flat)

                    if flat_records:
                        fieldnames = list(flat_records[0].keys())
                        # Collect all possible fieldnames from all records
                        for rec in flat_records[1:]:
                            for k in rec:
                                if k not in fieldnames:
                                    fieldnames.append(k)

                        with open(filepath, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                            if not file_exists:
                                writer.writeheader()
                            writer.writerows(flat_records)
                except Exception as e:
                    print(f"Flush error ({data_type} for {session_id}): {e}")

    def clear_session(self, session_id: str):
        """Clear all data for a session."""
        with self._lock:
            if session_id in self.buffers:
                self.buffers[session_id].clear()
            if session_id in self._flush_indices:
                del self._flush_indices[session_id]

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
