"""
Data Processor for Web HCI Collector

Handles data storage, synchronization, and export.
"""

import atexit
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
    drift_sample: List[Dict] = field(default_factory=list)

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
        self.drift_sample.clear()


# All data types stored in the buffer
_DATA_TYPES = [
    "gaze", "l2cs_gaze", "face_mesh", "emotion",
    "mouse", "keyboard", "experiment_event", "answer", "hover",
    "calibration_click", "calibration_validation", "drift_sample",
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

    def __init__(self, output_dir: str = "/Users/yenokhakobyan/HCI Lab ToolKit/data/raw/web_hci"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data buffers per session
        self.buffers: Dict[str, DataBuffer] = defaultdict(DataBuffer)
        self._lock = threading.Lock()

        # Periodic save tracking
        self._flush_indices: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Track known CSV fieldnames per session/data_type to prevent schema drift
        self._known_fields: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self._save_timer: Optional[threading.Timer] = None
        self._save_running = False

        # Defense-in-depth: flush all data on process exit (complements server shutdown)
        atexit.register(self._flush_all_on_exit)

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
        # Flush any pending in-memory data before export so nothing is missed
        self.flush_session_to_disk(session_id)

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

    def _flush_all_on_exit(self):
        """Flush all session buffers on process exit (atexit handler)."""
        self.stop_periodic_save()
        for sid in list(self.buffers.keys()):
            try:
                self.flush_session_to_disk(sid)
            except Exception as e:
                print(f"Exit flush error for {sid}: {e}")

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
                        # Collect fieldnames from this batch
                        batch_fields = list(flat_records[0].keys())
                        for rec in flat_records[1:]:
                            for k in rec:
                                if k not in batch_fields:
                                    batch_fields.append(k)

                        # Merge with known fields to maintain stable schema
                        known = self._known_fields[session_id][data_type]
                        if not known:
                            # First flush — establish the schema
                            known.extend(batch_fields)
                        else:
                            # Subsequent flushes — append any new fields
                            for f_name in batch_fields:
                                if f_name not in known:
                                    known.append(f_name)
                                    print(f"CSV schema expanded: {data_type} gained field '{f_name}' mid-session")

                        with open(filepath, "a", newline="") as f:
                            writer = csv.DictWriter(
                                f, fieldnames=known,
                                extrasaction="ignore", restval=""
                            )
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
            if session_id in self._known_fields:
                del self._known_fields[session_id]

    def correct_drift_offline(self, session_id: str) -> Optional[pd.DataFrame]:
        """
        Post-hoc drift correction using mouse-click/gaze offset as anchor points.

        Uses drift_sample data (collected from implicit mouse-click recalibration)
        to estimate and subtract a rolling bias from gaze coordinates.
        The correction interpolates drift linearly between click anchor points.

        Returns:
            A corrected gaze DataFrame with 'corrected_x' and 'corrected_y' columns,
            or None if no corrections can be made (no gaze or drift data).
        """
        data = self.get_session_data(session_id)
        gaze_records = data.get("gaze", [])
        drift_records = data.get("drift_sample", [])

        if not gaze_records or not drift_records:
            return None

        gaze_df = pd.DataFrame(gaze_records)
        drift_df = pd.DataFrame(drift_records)

        # Validate required columns exist
        for col in ("timestamp",):
            if col not in gaze_df.columns or col not in drift_df.columns:
                return None
        for col in ("click_x", "click_y", "gaze_x", "gaze_y"):
            if col not in drift_df.columns:
                return None

        # Sort by timestamp
        gaze_df = gaze_df.sort_values("timestamp").reset_index(drop=True)
        drift_df = drift_df.sort_values("timestamp").reset_index(drop=True)

        # Compute drift vectors at each anchor point (gaze - click = error to subtract)
        anchors_t = drift_df["timestamp"].values
        drift_x = (drift_df["gaze_x"] - drift_df["click_x"]).values
        drift_y = (drift_df["gaze_y"] - drift_df["click_y"]).values

        if len(anchors_t) < 2:
            # Single anchor: apply uniform correction
            gaze_df["corrected_x"] = gaze_df["x"] - float(drift_x[0])
            gaze_df["corrected_y"] = gaze_df["y"] - float(drift_y[0])
        else:
            # Interpolate drift between anchor points (extrapolate at edges)
            gaze_df["corrected_x"] = gaze_df["x"] - np.interp(
                gaze_df["timestamp"].values, anchors_t, drift_x
            )
            gaze_df["corrected_y"] = gaze_df["y"] - np.interp(
                gaze_df["timestamp"].values, anchors_t, drift_y
            )

        gaze_df["drift_corrected"] = True

        return gaze_df

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
