"""
Tobii Eye Tracker Data Collector

Collects gaze data, pupil diameter, fixations, and other eye tracking metrics
from Tobii Pro eye trackers.
"""

import sys
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any

import pandas as pd

# Add parent directory to path to import tobii_research
sys.path.insert(0, str(Path(__file__).parent.parent))
import tobii_research as tr


class TobiiCollector:
    """
    Collects data from Tobii Pro eye trackers.

    Supports:
    - Gaze point (x, y coordinates on screen)
    - Pupil diameter (left and right)
    - Gaze origin (3D position of eyes)
    - Eye openness
    - Timestamps for synchronization

    Example:
        collector = TobiiCollector()
        collector.connect()
        collector.start_recording()
        time.sleep(10)  # Record for 10 seconds
        collector.stop_recording()
        collector.save_data("my_recording.csv")
    """

    def __init__(self, output_dir: str = "data/raw/tobii"):
        """
        Initialize the Tobii collector.

        Args:
            output_dir: Directory to save recorded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.eyetracker: Optional[tr.EyeTracker] = None
        self.is_recording = False
        self.gaze_data: List[Dict[str, Any]] = []
        self.eye_openness_data: List[Dict[str, Any]] = []

        self._lock = threading.Lock()
        self._recording_start_time: Optional[float] = None

    def find_eyetrackers(self) -> List[Dict[str, str]]:
        """
        Find all connected Tobii eye trackers.

        Returns:
            List of dictionaries with eye tracker info (address, name, model, serial)
        """
        found = tr.find_all_eyetrackers()
        trackers = []
        for tracker in found:
            trackers.append({
                "address": tracker.address,
                "name": tracker.device_name,
                "model": tracker.model,
                "serial_number": tracker.serial_number,
                "firmware_version": tracker.firmware_version
            })
        return trackers

    def connect(self, address: Optional[str] = None) -> bool:
        """
        Connect to a Tobii eye tracker.

        Args:
            address: Specific eye tracker address (URI). If None, connects to first found.

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            if address:
                self.eyetracker = tr.EyeTracker(address)
            else:
                trackers = tr.find_all_eyetrackers()
                if not trackers:
                    print("No eye trackers found!")
                    return False
                self.eyetracker = trackers[0]

            print(f"Connected to: {self.eyetracker.device_name}")
            print(f"  Model: {self.eyetracker.model}")
            print(f"  Serial: {self.eyetracker.serial_number}")
            print(f"  Address: {self.eyetracker.address}")
            print(f"  Frequency: {self.eyetracker.get_gaze_output_frequency()} Hz")
            return True

        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from the eye tracker."""
        if self.is_recording:
            self.stop_recording()
        self.eyetracker = None
        print("Disconnected from eye tracker")

    def get_available_frequencies(self) -> tuple:
        """Get available sampling frequencies for the connected eye tracker."""
        if not self.eyetracker:
            raise RuntimeError("Not connected to an eye tracker")
        return self.eyetracker.get_all_gaze_output_frequencies()

    def set_frequency(self, frequency: float):
        """
        Set the sampling frequency.

        Args:
            frequency: Desired frequency in Hz (must be supported by device)
        """
        if not self.eyetracker:
            raise RuntimeError("Not connected to an eye tracker")
        self.eyetracker.set_gaze_output_frequency(frequency)
        print(f"Frequency set to {frequency} Hz")

    def _gaze_callback(self, gaze_data):
        """Internal callback for gaze data."""
        if not self.is_recording:
            return

        with self._lock:
            # Extract gaze data
            data = {
                # Timestamps
                "device_timestamp": gaze_data.device_time_stamp,
                "system_timestamp": gaze_data.system_time_stamp,
                "recording_timestamp": time.time() - self._recording_start_time,

                # Left eye gaze point (normalized 0-1)
                "left_gaze_x": gaze_data.left_eye.gaze_point.position_on_display_area[0],
                "left_gaze_y": gaze_data.left_eye.gaze_point.position_on_display_area[1],
                "left_gaze_valid": gaze_data.left_eye.gaze_point.validity,

                # Right eye gaze point (normalized 0-1)
                "right_gaze_x": gaze_data.right_eye.gaze_point.position_on_display_area[0],
                "right_gaze_y": gaze_data.right_eye.gaze_point.position_on_display_area[1],
                "right_gaze_valid": gaze_data.right_eye.gaze_point.validity,

                # Left eye pupil
                "left_pupil_diameter": gaze_data.left_eye.pupil.diameter,
                "left_pupil_valid": gaze_data.left_eye.pupil.validity,

                # Right eye pupil
                "right_pupil_diameter": gaze_data.right_eye.pupil.diameter,
                "right_pupil_valid": gaze_data.right_eye.pupil.validity,

                # Left eye gaze origin (3D position in user coordinate system)
                "left_origin_x": gaze_data.left_eye.gaze_origin.position_in_user_coordinates[0],
                "left_origin_y": gaze_data.left_eye.gaze_origin.position_in_user_coordinates[1],
                "left_origin_z": gaze_data.left_eye.gaze_origin.position_in_user_coordinates[2],
                "left_origin_valid": gaze_data.left_eye.gaze_origin.validity,

                # Right eye gaze origin (3D position in user coordinate system)
                "right_origin_x": gaze_data.right_eye.gaze_origin.position_in_user_coordinates[0],
                "right_origin_y": gaze_data.right_eye.gaze_origin.position_in_user_coordinates[1],
                "right_origin_z": gaze_data.right_eye.gaze_origin.position_in_user_coordinates[2],
                "right_origin_valid": gaze_data.right_eye.gaze_origin.validity,
            }

            self.gaze_data.append(data)

    def _eye_openness_callback(self, openness_data):
        """Internal callback for eye openness data."""
        if not self.is_recording:
            return

        with self._lock:
            data = {
                "device_timestamp": openness_data.device_time_stamp,
                "system_timestamp": openness_data.system_time_stamp,
                "recording_timestamp": time.time() - self._recording_start_time,
                "left_eye_openness": openness_data.left_eye.eye_openness_value,
                "left_eye_openness_valid": openness_data.left_eye.validity,
                "right_eye_openness": openness_data.right_eye.eye_openness_value,
                "right_eye_openness_valid": openness_data.right_eye.validity,
            }
            self.eye_openness_data.append(data)

    def start_recording(self, record_eye_openness: bool = True):
        """
        Start recording eye tracking data.

        Args:
            record_eye_openness: Also record eye openness if supported
        """
        if not self.eyetracker:
            raise RuntimeError("Not connected to an eye tracker")

        if self.is_recording:
            print("Already recording!")
            return

        # Clear previous data
        self.gaze_data = []
        self.eye_openness_data = []
        self._recording_start_time = time.time()
        self.is_recording = True

        # Subscribe to gaze data
        self.eyetracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA,
            self._gaze_callback
        )

        # Subscribe to eye openness if supported and requested
        if record_eye_openness:
            if tr.CAPABILITY_HAS_EYE_OPENNESS_DATA in self.eyetracker.device_capabilities:
                self.eyetracker.subscribe_to(
                    tr.EYETRACKER_EYE_OPENNESS_DATA,
                    self._eye_openness_callback
                )

        print(f"Recording started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def stop_recording(self) -> Dict[str, int]:
        """
        Stop recording eye tracking data.

        Returns:
            Dictionary with count of recorded samples
        """
        if not self.is_recording:
            print("Not recording!")
            return {"gaze_samples": 0, "openness_samples": 0}

        self.is_recording = False

        # Unsubscribe from data streams
        self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA)

        if tr.CAPABILITY_HAS_EYE_OPENNESS_DATA in self.eyetracker.device_capabilities:
            try:
                self.eyetracker.unsubscribe_from(tr.EYETRACKER_EYE_OPENNESS_DATA)
            except:
                pass

        duration = time.time() - self._recording_start_time
        print(f"Recording stopped. Duration: {duration:.2f}s")
        print(f"  Gaze samples: {len(self.gaze_data)}")
        print(f"  Eye openness samples: {len(self.eye_openness_data)}")

        return {
            "gaze_samples": len(self.gaze_data),
            "openness_samples": len(self.eye_openness_data),
            "duration": duration
        }

    def get_gaze_dataframe(self) -> pd.DataFrame:
        """
        Get recorded gaze data as a pandas DataFrame.

        Returns:
            DataFrame with gaze data
        """
        if not self.gaze_data:
            return pd.DataFrame()
        return pd.DataFrame(self.gaze_data)

    def get_eye_openness_dataframe(self) -> pd.DataFrame:
        """
        Get recorded eye openness data as a pandas DataFrame.

        Returns:
            DataFrame with eye openness data
        """
        if not self.eye_openness_data:
            return pd.DataFrame()
        return pd.DataFrame(self.eye_openness_data)

    def save_data(self, filename: Optional[str] = None, format: str = "csv") -> Path:
        """
        Save recorded data to file.

        Args:
            filename: Output filename (without extension). If None, uses timestamp.
            format: Output format ("csv" or "parquet")

        Returns:
            Path to saved file
        """
        if not self.gaze_data:
            print("No data to save!")
            return None

        if filename is None:
            filename = f"tobii_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        df = self.get_gaze_dataframe()

        # Add eye openness data if available
        if self.eye_openness_data:
            openness_df = self.get_eye_openness_dataframe()
            # Merge on device_timestamp
            df = pd.merge_asof(
                df.sort_values("device_timestamp"),
                openness_df.sort_values("device_timestamp"),
                on="device_timestamp",
                direction="nearest",
                suffixes=("", "_openness")
            )

        if format == "csv":
            filepath = self.output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        elif format == "parquet":
            filepath = self.output_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Data saved to: {filepath}")
        return filepath

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the connected eye tracker."""
        if not self.eyetracker:
            return {}

        return {
            "address": self.eyetracker.address,
            "device_name": self.eyetracker.device_name,
            "model": self.eyetracker.model,
            "serial_number": self.eyetracker.serial_number,
            "firmware_version": self.eyetracker.firmware_version,
            "runtime_version": self.eyetracker.runtime_version,
            "gaze_output_frequency": self.eyetracker.get_gaze_output_frequency(),
            "available_frequencies": self.eyetracker.get_all_gaze_output_frequencies(),
            "capabilities": self.eyetracker.device_capabilities,
        }


def main():
    """Example usage of TobiiCollector."""
    print("Tobii Eye Tracker Data Collector")
    print("=" * 40)

    collector = TobiiCollector()

    # Find available eye trackers
    print("\nSearching for eye trackers...")
    trackers = collector.find_eyetrackers()

    if not trackers:
        print("No eye trackers found. Make sure your Tobii device is connected.")
        return

    print(f"\nFound {len(trackers)} eye tracker(s):")
    for i, t in enumerate(trackers):
        print(f"  [{i}] {t['name']} ({t['model']}) - {t['serial_number']}")

    # Connect to first tracker
    if collector.connect():
        print("\nDevice info:")
        info = collector.get_device_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Record for 5 seconds
        print("\nRecording for 5 seconds...")
        collector.start_recording()
        time.sleep(5)
        stats = collector.stop_recording()

        # Save data
        filepath = collector.save_data()

        # Show sample of data
        df = collector.get_gaze_dataframe()
        if not df.empty:
            print("\nSample data (first 5 rows):")
            print(df.head())

        collector.disconnect()


if __name__ == "__main__":
    main()
