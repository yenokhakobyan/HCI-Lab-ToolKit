"""
Tobii Eye Tracking Data Analyzer

Provides fixation/saccade detection, heatmap generation, scanpath visualization,
AOI (Area of Interest) analysis, and advanced metrics inspired by Tobii Pro Lab.

Advanced Features:
- Ambient/Focal attention analysis (K coefficient)
- Scanpath similarity and entropy analysis
- AOI transition matrix and sequence analysis
- Cognitive load indicators (ICA, LHIPA, TEPR)
- Microsaccade detection
- Reading behavior analysis
- Smooth pursuit detection
- Statistical comparison tools
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from scipy import signal, stats
from scipy.ndimage import gaussian_filter
from collections import Counter
import warnings


@dataclass
class Fixation:
    """Represents a single fixation event."""
    start_time: float
    end_time: float
    duration: float
    x: float  # centroid x
    y: float  # centroid y
    dispersion: float
    samples: int


@dataclass
class Saccade:
    """Represents a single saccade event."""
    start_time: float
    end_time: float
    duration: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    amplitude: float  # in screen units
    velocity: float  # average velocity
    peak_velocity: float = 0.0  # peak velocity during saccade
    direction: float = 0.0  # direction in degrees (0-360)


@dataclass
class Microsaccade:
    """Represents a microsaccade event (small fixational eye movement)."""
    start_time: float
    end_time: float
    duration: float
    amplitude: float  # typically < 1 degree
    peak_velocity: float
    direction: float  # in degrees
    start_x: float
    start_y: float
    end_x: float
    end_y: float


@dataclass
class SmoothPursuit:
    """Represents a smooth pursuit eye movement."""
    start_time: float
    end_time: float
    duration: float
    mean_velocity: float
    gain: float  # ratio of eye velocity to target velocity
    positions: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class GlissadeEvent:
    """Represents a glissade (post-saccadic drift)."""
    start_time: float
    end_time: float
    duration: float
    amplitude: float
    associated_saccade_index: int


class TobiiAnalyzer:
    """
    Analyzes Tobii eye tracking data with advanced metrics inspired by Tobii Pro Lab.

    Basic Features:
    - Fixation detection (I-DT and I-VT algorithms)
    - Saccade detection
    - Heatmap generation
    - Scanpath visualization
    - AOI (Area of Interest) analysis
    - Pupil analysis

    Advanced Features (Tobii Pro Lab inspired):
    - Ambient/Focal attention coefficient (K coefficient)
    - Scanpath similarity metrics (Levenshtein, ScanMatch, MultiMatch)
    - Stationary entropy and transition entropy
    - AOI transition matrix analysis
    - Cognitive load indicators:
        * Index of Cognitive Activity (ICA)
        * Low/High Index of Pupillary Activity (LHIPA)
        * Task-Evoked Pupillary Response (TEPR)
    - Microsaccade detection and analysis
    - Reading behavior metrics (regressions, skips, refixations)
    - Smooth pursuit detection
    - Glissade detection
    - Time to First Fixation (TTFF) analysis
    - Revisit analysis
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the analyzer.

        Args:
            data: DataFrame with Tobii gaze data (from TobiiCollector)
        """
        self.data = data
        self.fixations: List[Fixation] = []
        self.saccades: List[Saccade] = []
        self.microsaccades: List[Microsaccade] = []
        self.smooth_pursuits: List[SmoothPursuit] = []
        self.glissades: List[GlissadeEvent] = []
        self.screen_size: Tuple[int, int] = (1920, 1080)  # default
        self.sampling_rate: float = 120.0  # Hz, will be auto-detected if possible
        self._velocity_data: Optional[pd.DataFrame] = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load eye tracking data from file.

        Args:
            filepath: Path to CSV or Parquet file

        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        if filepath.suffix == ".csv":
            self.data = pd.read_csv(filepath)
        elif filepath.suffix == ".parquet":
            self.data = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        print(f"Loaded {len(self.data)} samples from {filepath}")
        return self.data

    def set_screen_size(self, width: int, height: int):
        """Set screen size for coordinate conversion."""
        self.screen_size = (width, height)

    def preprocess(self, interpolate_missing: bool = True,
                   filter_invalid: bool = True) -> pd.DataFrame:
        """
        Preprocess gaze data.

        Args:
            interpolate_missing: Interpolate short gaps in data
            filter_invalid: Remove samples with invalid gaze

        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data.copy()

        # Calculate combined gaze point (average of both eyes)
        df["gaze_x"] = df[["left_gaze_x", "right_gaze_x"]].mean(axis=1)
        df["gaze_y"] = df[["left_gaze_y", "right_gaze_y"]].mean(axis=1)

        # Use single eye if other is invalid
        left_valid = df["left_gaze_valid"] == 1
        right_valid = df["right_gaze_valid"] == 1

        # Only left valid
        mask = left_valid & ~right_valid
        df.loc[mask, "gaze_x"] = df.loc[mask, "left_gaze_x"]
        df.loc[mask, "gaze_y"] = df.loc[mask, "left_gaze_y"]

        # Only right valid
        mask = ~left_valid & right_valid
        df.loc[mask, "gaze_x"] = df.loc[mask, "right_gaze_x"]
        df.loc[mask, "gaze_y"] = df.loc[mask, "right_gaze_y"]

        # Mark validity
        df["gaze_valid"] = left_valid | right_valid

        # Combined pupil diameter
        df["pupil_diameter"] = df[["left_pupil_diameter", "right_pupil_diameter"]].mean(axis=1)

        if filter_invalid:
            df = df[df["gaze_valid"]].reset_index(drop=True)

        if interpolate_missing and len(df) > 0:
            # Interpolate short gaps (< 75ms typically)
            df["gaze_x"] = df["gaze_x"].interpolate(method="linear", limit=5)
            df["gaze_y"] = df["gaze_y"].interpolate(method="linear", limit=5)

        # Convert to pixel coordinates
        df["gaze_x_px"] = df["gaze_x"] * self.screen_size[0]
        df["gaze_y_px"] = df["gaze_y"] * self.screen_size[1]

        self.data = df
        return df

    def detect_fixations_idt(self, dispersion_threshold: float = 0.02,
                              duration_threshold: float = 0.1) -> List[Fixation]:
        """
        Detect fixations using I-DT (Dispersion-Threshold) algorithm.

        Args:
            dispersion_threshold: Maximum dispersion (normalized, 0-1 range)
            duration_threshold: Minimum fixation duration in seconds

        Returns:
            List of Fixation objects
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data
        if "gaze_x" not in df.columns:
            df = self.preprocess()

        fixations = []
        i = 0
        n = len(df)

        while i < n:
            # Start a new potential fixation window
            window_start = i
            window_end = i

            while window_end < n:
                window = df.iloc[window_start:window_end + 1]

                # Calculate dispersion (max - min for x and y)
                dispersion_x = window["gaze_x"].max() - window["gaze_x"].min()
                dispersion_y = window["gaze_y"].max() - window["gaze_y"].min()
                dispersion = max(dispersion_x, dispersion_y)

                if dispersion <= dispersion_threshold:
                    window_end += 1
                else:
                    break

            # Check if window meets duration threshold
            if window_end > window_start:
                window = df.iloc[window_start:window_end]

                if "recording_timestamp" in window.columns:
                    duration = window["recording_timestamp"].iloc[-1] - window["recording_timestamp"].iloc[0]
                    start_time = window["recording_timestamp"].iloc[0]
                    end_time = window["recording_timestamp"].iloc[-1]
                else:
                    # Estimate from sample count and typical frequency
                    duration = len(window) / 120  # Assume 120 Hz
                    start_time = window_start / 120
                    end_time = window_end / 120

                if duration >= duration_threshold:
                    fixation = Fixation(
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        x=window["gaze_x"].mean(),
                        y=window["gaze_y"].mean(),
                        dispersion=dispersion,
                        samples=len(window)
                    )
                    fixations.append(fixation)
                    i = window_end
                else:
                    i += 1
            else:
                i += 1

        self.fixations = fixations
        print(f"Detected {len(fixations)} fixations")
        return fixations

    def detect_fixations_ivt(self, velocity_threshold: float = 0.5,
                              duration_threshold: float = 0.1) -> List[Fixation]:
        """
        Detect fixations using I-VT (Velocity-Threshold) algorithm.

        Args:
            velocity_threshold: Maximum velocity (normalized units per second)
            duration_threshold: Minimum fixation duration in seconds

        Returns:
            List of Fixation objects
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data.copy()
        if "gaze_x" not in df.columns:
            df = self.preprocess()

        # Calculate velocity
        if "recording_timestamp" in df.columns:
            dt = df["recording_timestamp"].diff()
        else:
            dt = pd.Series([1/120] * len(df))  # Assume 120 Hz

        dx = df["gaze_x"].diff()
        dy = df["gaze_y"].diff()

        velocity = np.sqrt(dx**2 + dy**2) / dt
        df["velocity"] = velocity.fillna(0)

        # Classify samples as fixation or saccade
        df["is_fixation"] = df["velocity"] < velocity_threshold

        # Group consecutive fixation samples
        df["fixation_group"] = (df["is_fixation"] != df["is_fixation"].shift()).cumsum()

        fixations = []
        for group_id, group in df[df["is_fixation"]].groupby("fixation_group"):
            if "recording_timestamp" in group.columns:
                duration = group["recording_timestamp"].iloc[-1] - group["recording_timestamp"].iloc[0]
                start_time = group["recording_timestamp"].iloc[0]
                end_time = group["recording_timestamp"].iloc[-1]
            else:
                duration = len(group) / 120
                start_time = group.index[0] / 120
                end_time = group.index[-1] / 120

            if duration >= duration_threshold:
                dispersion_x = group["gaze_x"].max() - group["gaze_x"].min()
                dispersion_y = group["gaze_y"].max() - group["gaze_y"].min()

                fixation = Fixation(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    x=group["gaze_x"].mean(),
                    y=group["gaze_y"].mean(),
                    dispersion=max(dispersion_x, dispersion_y),
                    samples=len(group)
                )
                fixations.append(fixation)

        self.fixations = fixations
        print(f"Detected {len(fixations)} fixations (I-VT)")
        return fixations

    def detect_saccades(self) -> List[Saccade]:
        """
        Detect saccades (rapid eye movements between fixations).

        Returns:
            List of Saccade objects
        """
        if not self.fixations:
            self.detect_fixations_idt()

        saccades = []
        for i in range(len(self.fixations) - 1):
            f1 = self.fixations[i]
            f2 = self.fixations[i + 1]

            start_time = f1.end_time
            end_time = f2.start_time
            duration = end_time - start_time

            if duration > 0:
                amplitude = np.sqrt((f2.x - f1.x)**2 + (f2.y - f1.y)**2)
                velocity = amplitude / duration if duration > 0 else 0

                saccade = Saccade(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    start_x=f1.x,
                    start_y=f1.y,
                    end_x=f2.x,
                    end_y=f2.y,
                    amplitude=amplitude,
                    velocity=velocity
                )
                saccades.append(saccade)

        self.saccades = saccades
        print(f"Detected {len(saccades)} saccades")
        return saccades

    def generate_heatmap(self, resolution: Tuple[int, int] = (100, 100),
                         sigma: float = 2.0,
                         use_fixations: bool = True) -> np.ndarray:
        """
        Generate a gaze heatmap.

        Args:
            resolution: Heatmap resolution (width, height)
            sigma: Gaussian smoothing sigma
            use_fixations: Weight by fixation duration, else use raw gaze

        Returns:
            2D numpy array with heatmap values
        """
        from scipy.ndimage import gaussian_filter

        heatmap = np.zeros(resolution[::-1])  # (height, width)

        if use_fixations and self.fixations:
            for fix in self.fixations:
                x = int(fix.x * resolution[0])
                y = int(fix.y * resolution[1])
                if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                    heatmap[y, x] += fix.duration
        else:
            if self.data is None:
                raise ValueError("No data loaded")
            df = self.data
            if "gaze_x" not in df.columns:
                df = self.preprocess()

            for _, row in df.iterrows():
                x = int(row["gaze_x"] * resolution[0])
                y = int(row["gaze_y"] * resolution[1])
                if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                    heatmap[y, x] += 1

        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=sigma)

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def get_fixation_stats(self) -> Dict[str, float]:
        """
        Calculate fixation statistics.

        Returns:
            Dictionary with fixation metrics
        """
        if not self.fixations:
            return {}

        durations = [f.duration for f in self.fixations]

        return {
            "fixation_count": len(self.fixations),
            "mean_duration": np.mean(durations),
            "std_duration": np.std(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "total_fixation_time": np.sum(durations),
        }

    def get_saccade_stats(self) -> Dict[str, float]:
        """
        Calculate saccade statistics.

        Returns:
            Dictionary with saccade metrics
        """
        if not self.saccades:
            return {}

        amplitudes = [s.amplitude for s in self.saccades]
        velocities = [s.velocity for s in self.saccades]
        durations = [s.duration for s in self.saccades]

        return {
            "saccade_count": len(self.saccades),
            "mean_amplitude": np.mean(amplitudes),
            "std_amplitude": np.std(amplitudes),
            "mean_velocity": np.mean(velocities),
            "mean_duration": np.mean(durations),
        }

    def get_pupil_stats(self) -> Dict[str, float]:
        """
        Calculate pupil diameter statistics.

        Returns:
            Dictionary with pupil metrics
        """
        if self.data is None:
            return {}

        df = self.data

        # Get valid pupil data
        left_valid = df[df["left_pupil_valid"] == 1]["left_pupil_diameter"]
        right_valid = df[df["right_pupil_valid"] == 1]["right_pupil_diameter"]

        stats = {}

        if len(left_valid) > 0:
            stats["left_pupil_mean"] = left_valid.mean()
            stats["left_pupil_std"] = left_valid.std()

        if len(right_valid) > 0:
            stats["right_pupil_mean"] = right_valid.mean()
            stats["right_pupil_std"] = right_valid.std()

        if "pupil_diameter" in df.columns:
            valid_pupil = df["pupil_diameter"].dropna()
            if len(valid_pupil) > 0:
                stats["combined_pupil_mean"] = valid_pupil.mean()
                stats["combined_pupil_std"] = valid_pupil.std()

        return stats

    def analyze_aoi(self, aois: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, Dict]:
        """
        Analyze gaze within Areas of Interest (AOIs).

        Args:
            aois: Dictionary mapping AOI names to (x1, y1, x2, y2) bounds (normalized 0-1)

        Returns:
            Dictionary with AOI metrics for each region
        """
        if not self.fixations:
            self.detect_fixations_idt()

        results = {}

        for aoi_name, (x1, y1, x2, y2) in aois.items():
            aoi_fixations = [
                f for f in self.fixations
                if x1 <= f.x <= x2 and y1 <= f.y <= y2
            ]

            if aoi_fixations:
                durations = [f.duration for f in aoi_fixations]
                results[aoi_name] = {
                    "fixation_count": len(aoi_fixations),
                    "total_dwell_time": sum(durations),
                    "mean_fixation_duration": np.mean(durations),
                    "first_fixation_time": aoi_fixations[0].start_time if aoi_fixations else None,
                    "percentage_of_fixations": len(aoi_fixations) / len(self.fixations) * 100,
                }
            else:
                results[aoi_name] = {
                    "fixation_count": 0,
                    "total_dwell_time": 0,
                    "mean_fixation_duration": 0,
                    "first_fixation_time": None,
                    "percentage_of_fixations": 0,
                }

        return results

    def get_fixations_dataframe(self) -> pd.DataFrame:
        """Convert fixations to DataFrame."""
        if not self.fixations:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "start_time": f.start_time,
                "end_time": f.end_time,
                "duration": f.duration,
                "x": f.x,
                "y": f.y,
                "x_px": f.x * self.screen_size[0],
                "y_px": f.y * self.screen_size[1],
                "dispersion": f.dispersion,
                "samples": f.samples,
            }
            for f in self.fixations
        ])

    def get_saccades_dataframe(self) -> pd.DataFrame:
        """Convert saccades to DataFrame."""
        if not self.saccades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.duration,
                "start_x": s.start_x,
                "start_y": s.start_y,
                "end_x": s.end_x,
                "end_y": s.end_y,
                "amplitude": s.amplitude,
                "velocity": s.velocity,
                "peak_velocity": s.peak_velocity,
                "direction": s.direction,
            }
            for s in self.saccades
        ])

    # =========================================================================
    # ADVANCED METRICS - Tobii Pro Lab Inspired
    # =========================================================================

    def _estimate_sampling_rate(self) -> float:
        """Estimate the sampling rate from the data."""
        if self.data is None or "recording_timestamp" not in self.data.columns:
            return self.sampling_rate

        timestamps = self.data["recording_timestamp"].dropna()
        if len(timestamps) < 2:
            return self.sampling_rate

        dt = timestamps.diff().dropna()
        median_dt = dt.median()
        if median_dt > 0:
            self.sampling_rate = 1.0 / median_dt
        return self.sampling_rate

    def _compute_velocity(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Compute point-to-point velocity for gaze data.

        Returns:
            DataFrame with velocity components added
        """
        if self._velocity_data is not None and not force_recompute:
            return self._velocity_data

        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data.copy()
        if "gaze_x" not in df.columns:
            df = self.preprocess()

        self._estimate_sampling_rate()

        # Time differences
        if "recording_timestamp" in df.columns:
            dt = df["recording_timestamp"].diff()
        else:
            dt = pd.Series([1.0 / self.sampling_rate] * len(df))

        # Position differences
        dx = df["gaze_x"].diff()
        dy = df["gaze_y"].diff()

        # Velocity magnitude
        df["velocity"] = np.sqrt(dx**2 + dy**2) / dt.replace(0, np.nan)
        df["velocity_x"] = dx / dt.replace(0, np.nan)
        df["velocity_y"] = dy / dt.replace(0, np.nan)

        # Acceleration
        df["acceleration"] = df["velocity"].diff() / dt.replace(0, np.nan)

        # Direction (degrees, 0 = right, 90 = up)
        df["direction"] = np.degrees(np.arctan2(-dy, dx)) % 360

        self._velocity_data = df
        return df

    # -------------------------------------------------------------------------
    # Ambient/Focal Attention Analysis (K Coefficient)
    # -------------------------------------------------------------------------

    def compute_k_coefficient(self) -> Dict[str, float]:
        """
        Compute the K coefficient for ambient vs focal attention mode.

        The K coefficient is based on the ratio of fixation duration to saccade amplitude.
        - K > 0: Focal/detailed processing (long fixations, short saccades)
        - K < 0: Ambient/global processing (short fixations, long saccades)

        Based on: Krejtz et al. (2016) - "Eye tracking cognitive load using pupil
        diameter and microsaccades with fixed gaze"

        Returns:
            Dictionary with K coefficient and related metrics
        """
        if not self.fixations or not self.saccades:
            if not self.fixations:
                self.detect_fixations_idt()
            if not self.saccades:
                self.detect_saccades()

        if len(self.fixations) < 2 or len(self.saccades) < 1:
            return {"k_coefficient": np.nan, "attention_mode": "insufficient_data"}

        # Get fixation durations and following saccade amplitudes
        fix_durations = []
        sacc_amplitudes = []

        for i, (fix, sacc) in enumerate(zip(self.fixations[:-1], self.saccades)):
            fix_durations.append(fix.duration)
            sacc_amplitudes.append(sacc.amplitude)

        fix_durations = np.array(fix_durations)
        sacc_amplitudes = np.array(sacc_amplitudes)

        # Z-score normalization
        fix_z = stats.zscore(fix_durations) if np.std(fix_durations) > 0 else np.zeros_like(fix_durations)
        sacc_z = stats.zscore(sacc_amplitudes) if np.std(sacc_amplitudes) > 0 else np.zeros_like(sacc_amplitudes)

        # K coefficient: difference of z-scores
        k_values = fix_z - sacc_z
        k_coefficient = np.mean(k_values)

        # Determine attention mode
        if k_coefficient > 0.5:
            attention_mode = "focal"
        elif k_coefficient < -0.5:
            attention_mode = "ambient"
        else:
            attention_mode = "mixed"

        return {
            "k_coefficient": k_coefficient,
            "k_std": np.std(k_values),
            "attention_mode": attention_mode,
            "focal_ratio": np.mean(k_values > 0),
            "ambient_ratio": np.mean(k_values < 0),
            "k_values": k_values.tolist(),
        }

    def compute_k_coefficient_over_time(self, window_size: float = 5.0,
                                         step_size: float = 1.0) -> pd.DataFrame:
        """
        Compute K coefficient over sliding time windows.

        Args:
            window_size: Window size in seconds
            step_size: Step size in seconds

        Returns:
            DataFrame with time-varying K coefficient
        """
        if not self.fixations or not self.saccades:
            return pd.DataFrame()

        results = []
        start_time = self.fixations[0].start_time
        end_time = self.fixations[-1].end_time

        current_time = start_time
        while current_time + window_size <= end_time:
            window_end = current_time + window_size

            # Get fixations and saccades in window
            window_fix = [f for f in self.fixations
                          if current_time <= f.start_time < window_end]
            window_sacc = [s for s in self.saccades
                           if current_time <= s.start_time < window_end]

            if len(window_fix) >= 2 and len(window_sacc) >= 1:
                fix_dur = [f.duration for f in window_fix[:-1]]
                sacc_amp = [s.amplitude for s in window_sacc[:len(fix_dur)]]

                if len(fix_dur) == len(sacc_amp) and len(fix_dur) > 0:
                    fix_z = stats.zscore(fix_dur) if np.std(fix_dur) > 0 else np.zeros(len(fix_dur))
                    sacc_z = stats.zscore(sacc_amp) if np.std(sacc_amp) > 0 else np.zeros(len(sacc_amp))
                    k = np.mean(fix_z - sacc_z)
                else:
                    k = np.nan
            else:
                k = np.nan

            results.append({
                "time": current_time + window_size / 2,
                "k_coefficient": k,
                "n_fixations": len(window_fix),
                "n_saccades": len(window_sacc),
            })

            current_time += step_size

        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Scanpath Analysis
    # -------------------------------------------------------------------------

    def compute_scanpath_entropy(self, aois: Dict[str, Tuple[float, float, float, float]],
                                  entropy_type: str = "both") -> Dict[str, float]:
        """
        Compute scanpath entropy metrics.

        Args:
            aois: Dictionary mapping AOI names to bounds
            entropy_type: "stationary", "transition", or "both"

        Returns:
            Dictionary with entropy metrics
        """
        if not self.fixations:
            self.detect_fixations_idt()

        # Map fixations to AOIs
        aoi_sequence = []
        for fix in self.fixations:
            assigned_aoi = None
            for aoi_name, (x1, y1, x2, y2) in aois.items():
                if x1 <= fix.x <= x2 and y1 <= fix.y <= y2:
                    assigned_aoi = aoi_name
                    break
            aoi_sequence.append(assigned_aoi if assigned_aoi else "outside")

        results = {}

        # Stationary entropy (distribution of fixations across AOIs)
        if entropy_type in ["stationary", "both"]:
            aoi_counts = Counter(aoi_sequence)
            total = sum(aoi_counts.values())
            probabilities = [count / total for count in aoi_counts.values()]
            stationary_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(len(aois) + 1)  # +1 for "outside"
            normalized_stationary = stationary_entropy / max_entropy if max_entropy > 0 else 0

            results["stationary_entropy"] = stationary_entropy
            results["normalized_stationary_entropy"] = normalized_stationary

        # Transition entropy (predictability of transitions)
        if entropy_type in ["transition", "both"]:
            if len(aoi_sequence) > 1:
                transitions = [(aoi_sequence[i], aoi_sequence[i+1])
                               for i in range(len(aoi_sequence) - 1)]
                transition_counts = Counter(transitions)
                total_transitions = sum(transition_counts.values())

                transition_entropy = 0
                for (from_aoi, to_aoi), count in transition_counts.items():
                    p = count / total_transitions
                    transition_entropy -= p * np.log2(p) if p > 0 else 0

                max_transition_entropy = np.log2(len(transition_counts))
                normalized_transition = (transition_entropy / max_transition_entropy
                                          if max_transition_entropy > 0 else 0)

                results["transition_entropy"] = transition_entropy
                results["normalized_transition_entropy"] = normalized_transition
            else:
                results["transition_entropy"] = 0
                results["normalized_transition_entropy"] = 0

        results["aoi_sequence"] = aoi_sequence
        return results

    def compute_aoi_transition_matrix(self, aois: Dict[str, Tuple[float, float, float, float]],
                                       normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Compute transition matrix between AOIs.

        Args:
            aois: Dictionary mapping AOI names to bounds
            normalize: Whether to normalize by row (probabilities)

        Returns:
            Tuple of (transition matrix, AOI labels including "outside")
        """
        if not self.fixations:
            self.detect_fixations_idt()

        # Map fixations to AOIs
        aoi_names = list(aois.keys()) + ["outside"]
        aoi_to_idx = {name: i for i, name in enumerate(aoi_names)}

        aoi_sequence = []
        for fix in self.fixations:
            assigned_aoi = "outside"
            for aoi_name, (x1, y1, x2, y2) in aois.items():
                if x1 <= fix.x <= x2 and y1 <= fix.y <= y2:
                    assigned_aoi = aoi_name
                    break
            aoi_sequence.append(assigned_aoi)

        # Build transition matrix
        n_aois = len(aoi_names)
        transition_matrix = np.zeros((n_aois, n_aois))

        for i in range(len(aoi_sequence) - 1):
            from_idx = aoi_to_idx[aoi_sequence[i]]
            to_idx = aoi_to_idx[aoi_sequence[i + 1]]
            transition_matrix[from_idx, to_idx] += 1

        if normalize:
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transition_matrix = transition_matrix / row_sums

        return transition_matrix, aoi_names

    def compute_scanpath_similarity(self, other_fixations: List[Fixation],
                                     method: str = "levenshtein",
                                     aois: Optional[Dict[str, Tuple]] = None) -> float:
        """
        Compute similarity between two scanpaths.

        Args:
            other_fixations: Fixations from another recording
            method: "levenshtein" (string-edit distance) or "euclidean"
            aois: AOIs for string-based methods

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if not self.fixations:
            self.detect_fixations_idt()

        if method == "levenshtein" and aois:
            # Convert to AOI strings
            def to_aoi_string(fixations):
                result = []
                for fix in fixations:
                    for aoi_name, (x1, y1, x2, y2) in aois.items():
                        if x1 <= fix.x <= x2 and y1 <= fix.y <= y2:
                            result.append(aoi_name[0])
                            break
                    else:
                        result.append("O")
                return "".join(result)

            str1 = to_aoi_string(self.fixations)
            str2 = to_aoi_string(other_fixations)

            # Levenshtein distance
            m, n = len(str1), len(str2)
            if m == 0 or n == 0:
                return 0.0

            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if str1[i-1] == str2[j-1] else 1
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

            max_len = max(m, n)
            similarity = 1 - (dp[m][n] / max_len)
            return similarity

        elif method == "euclidean":
            # Simple point-wise comparison (DTW-like)
            min_len = min(len(self.fixations), len(other_fixations))
            if min_len == 0:
                return 0.0

            total_dist = 0
            for i in range(min_len):
                f1 = self.fixations[i]
                f2 = other_fixations[i]
                dist = np.sqrt((f1.x - f2.x)**2 + (f1.y - f2.y)**2)
                total_dist += dist

            # Normalize by max possible distance (diagonal = sqrt(2))
            max_dist = np.sqrt(2) * min_len
            similarity = 1 - (total_dist / max_dist)
            return max(0, similarity)

        return 0.0

    # -------------------------------------------------------------------------
    # AOI Advanced Metrics
    # -------------------------------------------------------------------------

    def analyze_aoi_advanced(self, aois: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, Dict]:
        """
        Advanced AOI analysis with Tobii Pro Lab-style metrics.

        Includes:
        - Time to First Fixation (TTFF)
        - First Fixation Duration
        - Revisits count
        - Visit duration stats
        - Entry time (first and all)
        - Glances (visits shorter than threshold)

        Args:
            aois: Dictionary mapping AOI names to bounds

        Returns:
            Dictionary with comprehensive AOI metrics
        """
        if not self.fixations:
            self.detect_fixations_idt()

        results = {}
        recording_start = self.fixations[0].start_time if self.fixations else 0

        for aoi_name, (x1, y1, x2, y2) in aois.items():
            aoi_fixations = []
            visits = []  # List of (start_time, end_time, fixations_in_visit)
            current_visit = []

            for i, fix in enumerate(self.fixations):
                in_aoi = x1 <= fix.x <= x2 and y1 <= fix.y <= y2

                if in_aoi:
                    aoi_fixations.append((i, fix))
                    current_visit.append(fix)
                else:
                    if current_visit:
                        visits.append({
                            "start": current_visit[0].start_time,
                            "end": current_visit[-1].end_time,
                            "duration": current_visit[-1].end_time - current_visit[0].start_time,
                            "fixation_count": len(current_visit),
                        })
                        current_visit = []

            # Don't forget last visit
            if current_visit:
                visits.append({
                    "start": current_visit[0].start_time,
                    "end": current_visit[-1].end_time,
                    "duration": current_visit[-1].end_time - current_visit[0].start_time,
                    "fixation_count": len(current_visit),
                })

            # Calculate metrics
            durations = [f.duration for _, f in aoi_fixations]
            visit_durations = [v["duration"] for v in visits]

            results[aoi_name] = {
                # Basic metrics
                "fixation_count": len(aoi_fixations),
                "total_dwell_time": sum(durations) if durations else 0,
                "mean_fixation_duration": np.mean(durations) if durations else 0,
                "std_fixation_duration": np.std(durations) if durations else 0,

                # Time to First Fixation
                "time_to_first_fixation": (aoi_fixations[0][1].start_time - recording_start
                                            if aoi_fixations else None),
                "first_fixation_duration": aoi_fixations[0][1].duration if aoi_fixations else None,

                # Visit metrics
                "visit_count": len(visits),
                "revisit_count": max(0, len(visits) - 1),
                "mean_visit_duration": np.mean(visit_durations) if visit_durations else 0,
                "total_visit_duration": sum(visit_durations) if visit_durations else 0,

                # Glances (short visits < 150ms)
                "glance_count": sum(1 for v in visits if v["duration"] < 0.15),

                # Entry times
                "first_entry_time": visits[0]["start"] if visits else None,
                "all_entry_times": [v["start"] for v in visits],

                # Sequence info
                "first_fixation_index": aoi_fixations[0][0] if aoi_fixations else None,
                "percentage_of_fixations": (len(aoi_fixations) / len(self.fixations) * 100
                                             if self.fixations else 0),
            }

        return results

    # -------------------------------------------------------------------------
    # Cognitive Load Indicators
    # -------------------------------------------------------------------------

    def compute_index_of_cognitive_activity(self, window_size: float = 1.0) -> Dict[str, Any]:
        """
        Compute Index of Cognitive Activity (ICA) based on pupil diameter fluctuations.

        ICA measures rapid small fluctuations in pupil size that correlate with
        cognitive workload. Based on Marshall (2002).

        Args:
            window_size: Analysis window in seconds

        Returns:
            Dictionary with ICA metrics
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data.copy()
        if "pupil_diameter" not in df.columns:
            df = self.preprocess()

        self._estimate_sampling_rate()

        pupil = df["pupil_diameter"].dropna().values
        if len(pupil) < 10:
            return {"ica": np.nan, "ica_left": np.nan, "ica_right": np.nan}

        # High-pass filter to isolate rapid fluctuations
        # Butterworth filter, cutoff ~0.5 Hz
        nyquist = self.sampling_rate / 2
        cutoff = min(0.5, nyquist * 0.8)
        b, a = signal.butter(2, cutoff / nyquist, btype='high')

        # Apply filter
        try:
            filtered = signal.filtfilt(b, a, pupil)
        except ValueError:
            return {"ica": np.nan}

        # ICA = mean of absolute values of rapid changes
        ica = np.mean(np.abs(filtered))

        # Per-eye ICA if available
        results = {"ica": ica}

        for eye in ["left", "right"]:
            col = f"{eye}_pupil_diameter"
            if col in df.columns:
                eye_pupil = df[col].dropna().values
                if len(eye_pupil) > 10:
                    try:
                        filtered_eye = signal.filtfilt(b, a, eye_pupil)
                        results[f"ica_{eye}"] = np.mean(np.abs(filtered_eye))
                    except ValueError:
                        results[f"ica_{eye}"] = np.nan

        return results

    def compute_pupillary_response(self, baseline_duration: float = 1.0,
                                    event_times: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compute Task-Evoked Pupillary Response (TEPR) metrics.

        Args:
            baseline_duration: Duration of baseline period in seconds
            event_times: Optional list of event onset times

        Returns:
            Dictionary with TEPR metrics
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data.copy()
        if "pupil_diameter" not in df.columns:
            df = self.preprocess()

        timestamps = df["recording_timestamp"].values
        pupil = df["pupil_diameter"].values

        # Baseline (first N seconds)
        baseline_mask = timestamps < baseline_duration
        baseline_pupil = pupil[baseline_mask]
        baseline_mean = np.nanmean(baseline_pupil)
        baseline_std = np.nanstd(baseline_pupil)

        # Overall TEPR metrics
        post_baseline = pupil[~baseline_mask]
        if len(post_baseline) == 0:
            return {"baseline_mean": baseline_mean, "baseline_std": baseline_std}

        tepr = (post_baseline - baseline_mean) / baseline_mean * 100  # Percent change

        results = {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "mean_tepr": np.nanmean(tepr),
            "max_tepr": np.nanmax(tepr),
            "min_tepr": np.nanmin(tepr),
            "tepr_std": np.nanstd(tepr),
            "peak_dilation_time": timestamps[~baseline_mask][np.nanargmax(tepr)]
                                   if len(tepr) > 0 else None,
        }

        # Event-locked responses if provided
        if event_times:
            event_responses = []
            for event_time in event_times:
                # Get pupil in window around event (-0.5s to +2s)
                mask = (timestamps >= event_time - 0.5) & (timestamps <= event_time + 2.0)
                event_pupil = pupil[mask]
                if len(event_pupil) > 0:
                    pre_event = pupil[(timestamps >= event_time - 0.5) &
                                       (timestamps < event_time)]
                    pre_mean = np.nanmean(pre_event) if len(pre_event) > 0 else baseline_mean
                    event_tepr = (event_pupil - pre_mean) / pre_mean * 100
                    event_responses.append({
                        "event_time": event_time,
                        "peak_response": np.nanmax(event_tepr),
                        "mean_response": np.nanmean(event_tepr),
                    })
            results["event_responses"] = event_responses

        return results

    def compute_lhipa(self, low_freq_range: Tuple[float, float] = (0.04, 0.15),
                      high_freq_range: Tuple[float, float] = (0.15, 0.5)) -> Dict[str, float]:
        """
        Compute Low/High Index of Pupillary Activity (LHIPA).

        LHIPA uses spectral analysis of pupil fluctuations to assess cognitive load.
        Low frequency (0.04-0.15 Hz) reflects parasympathetic activity.
        High frequency (0.15-0.5 Hz) reflects cognitive/task demands.

        Args:
            low_freq_range: Low frequency band (Hz)
            high_freq_range: High frequency band (Hz)

        Returns:
            Dictionary with LHIPA metrics
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data.copy()
        if "pupil_diameter" not in df.columns:
            df = self.preprocess()

        self._estimate_sampling_rate()

        pupil = df["pupil_diameter"].dropna().values
        if len(pupil) < int(self.sampling_rate * 10):  # Need at least 10 seconds
            return {"lhipa": np.nan, "lipa": np.nan, "hipa": np.nan}

        # Compute power spectral density
        freqs, psd = signal.welch(pupil, fs=self.sampling_rate, nperseg=min(256, len(pupil)))

        # Low frequency power
        low_mask = (freqs >= low_freq_range[0]) & (freqs <= low_freq_range[1])
        lipa = np.trapz(psd[low_mask], freqs[low_mask]) if np.any(low_mask) else 0

        # High frequency power
        high_mask = (freqs >= high_freq_range[0]) & (freqs <= high_freq_range[1])
        hipa = np.trapz(psd[high_mask], freqs[high_mask]) if np.any(high_mask) else 0

        # LHIPA ratio
        lhipa = lipa / hipa if hipa > 0 else np.nan

        return {
            "lhipa": lhipa,
            "lipa": lipa,
            "hipa": hipa,
            "total_power": np.trapz(psd, freqs),
        }

    # -------------------------------------------------------------------------
    # Microsaccade Detection
    # -------------------------------------------------------------------------

    def detect_microsaccades(self, velocity_threshold: float = 6.0,
                              min_duration: float = 0.006,
                              max_amplitude: float = 0.01) -> List[Microsaccade]:
        """
        Detect microsaccades during fixations.

        Uses the Engbert & Kliegl (2003) algorithm for microsaccade detection.

        Args:
            velocity_threshold: Velocity threshold in median-based units (lambda)
            min_duration: Minimum duration in seconds
            max_amplitude: Maximum amplitude (normalized, typically < 1 degree)

        Returns:
            List of Microsaccade objects
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self._compute_velocity()
        self._estimate_sampling_rate()

        # Compute velocity threshold using Engbert & Kliegl method
        vx = df["velocity_x"].dropna().values
        vy = df["velocity_y"].dropna().values

        if len(vx) < 10:
            return []

        # Median-based threshold
        median_vx = np.median(np.abs(vx - np.median(vx)))
        median_vy = np.median(np.abs(vy - np.median(vy)))

        threshold_x = velocity_threshold * median_vx
        threshold_y = velocity_threshold * median_vy

        # Elliptic threshold
        microsaccades = []
        is_microsaccade = ((vx / threshold_x)**2 + (vy / threshold_y)**2) > 1

        # Find microsaccade events
        in_microsaccade = False
        start_idx = 0

        for i, is_ms in enumerate(is_microsaccade):
            if is_ms and not in_microsaccade:
                in_microsaccade = True
                start_idx = i
            elif not is_ms and in_microsaccade:
                in_microsaccade = False
                end_idx = i - 1

                # Calculate microsaccade properties
                if "recording_timestamp" in df.columns:
                    duration = (df.iloc[end_idx]["recording_timestamp"] -
                                df.iloc[start_idx]["recording_timestamp"])
                    start_time = df.iloc[start_idx]["recording_timestamp"]
                    end_time = df.iloc[end_idx]["recording_timestamp"]
                else:
                    duration = (end_idx - start_idx) / self.sampling_rate
                    start_time = start_idx / self.sampling_rate
                    end_time = end_idx / self.sampling_rate

                if duration >= min_duration:
                    start_x = df.iloc[start_idx]["gaze_x"]
                    start_y = df.iloc[start_idx]["gaze_y"]
                    end_x = df.iloc[end_idx]["gaze_x"]
                    end_y = df.iloc[end_idx]["gaze_y"]

                    amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

                    if amplitude <= max_amplitude:
                        direction = np.degrees(np.arctan2(-(end_y - start_y),
                                                           end_x - start_x)) % 360
                        peak_vel = df.iloc[start_idx:end_idx+1]["velocity"].max()

                        microsaccades.append(Microsaccade(
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            amplitude=amplitude,
                            peak_velocity=peak_vel,
                            direction=direction,
                            start_x=start_x,
                            start_y=start_y,
                            end_x=end_x,
                            end_y=end_y,
                        ))

        self.microsaccades = microsaccades
        return microsaccades

    def get_microsaccade_stats(self) -> Dict[str, float]:
        """Get statistics for detected microsaccades."""
        if not self.microsaccades:
            return {}

        amplitudes = [m.amplitude for m in self.microsaccades]
        durations = [m.duration for m in self.microsaccades]
        velocities = [m.peak_velocity for m in self.microsaccades]

        # Rate per second
        if self.data is not None and "recording_timestamp" in self.data.columns:
            total_time = (self.data["recording_timestamp"].max() -
                          self.data["recording_timestamp"].min())
        else:
            total_time = len(self.data) / self.sampling_rate if self.data is not None else 1

        return {
            "microsaccade_count": len(self.microsaccades),
            "rate_per_second": len(self.microsaccades) / total_time if total_time > 0 else 0,
            "mean_amplitude": np.mean(amplitudes),
            "std_amplitude": np.std(amplitudes),
            "mean_duration": np.mean(durations),
            "mean_peak_velocity": np.mean(velocities),
        }

    # -------------------------------------------------------------------------
    # Reading Analysis
    # -------------------------------------------------------------------------

    def analyze_reading_behavior(self, reading_direction: str = "ltr",
                                  line_height: float = 0.05) -> Dict[str, Any]:
        """
        Analyze reading-specific eye movement patterns.

        Detects:
        - Forward saccades (progressive)
        - Regressions (backward saccades)
        - Line returns
        - Refixations
        - Word skips

        Args:
            reading_direction: "ltr" (left-to-right) or "rtl" (right-to-left)
            line_height: Approximate line height in normalized coordinates

        Returns:
            Dictionary with reading metrics
        """
        if not self.fixations or not self.saccades:
            if not self.fixations:
                self.detect_fixations_idt()
            if not self.saccades:
                self.detect_saccades()

        forward_saccades = []
        regressions = []
        line_returns = []
        refixations = []

        is_ltr = reading_direction.lower() == "ltr"

        for i, sacc in enumerate(self.saccades):
            dx = sacc.end_x - sacc.start_x
            dy = sacc.end_y - sacc.start_y

            # Line return detection (large backward + downward movement)
            if abs(dy) > line_height * 0.5:
                if (is_ltr and dx < -0.1) or (not is_ltr and dx > 0.1):
                    line_returns.append(i)
                    continue

            # Progressive vs regressive
            if is_ltr:
                if dx > 0:
                    forward_saccades.append(i)
                elif dx < -0.01:  # Small threshold for noise
                    regressions.append(i)
            else:
                if dx < 0:
                    forward_saccades.append(i)
                elif dx > 0.01:
                    regressions.append(i)

        # Refixations (consecutive fixations very close together)
        for i in range(len(self.fixations) - 1):
            f1 = self.fixations[i]
            f2 = self.fixations[i + 1]
            dist = np.sqrt((f2.x - f1.x)**2 + (f2.y - f1.y)**2)
            if dist < 0.02:  # Very close fixations
                refixations.append(i)

        # Calculate rates
        total_saccades = len(self.saccades)

        return {
            "forward_saccade_count": len(forward_saccades),
            "regression_count": len(regressions),
            "regression_rate": len(regressions) / total_saccades if total_saccades > 0 else 0,
            "line_return_count": len(line_returns),
            "refixation_count": len(refixations),
            "refixation_rate": len(refixations) / len(self.fixations) if self.fixations else 0,

            # Reading efficiency (higher = more efficient)
            "reading_efficiency": len(forward_saccades) / total_saccades if total_saccades > 0 else 0,

            # Detailed indices
            "forward_saccade_indices": forward_saccades,
            "regression_indices": regressions,
            "line_return_indices": line_returns,
            "refixation_indices": refixations,
        }

    # -------------------------------------------------------------------------
    # Smooth Pursuit Detection
    # -------------------------------------------------------------------------

    def detect_smooth_pursuit(self, velocity_range: Tuple[float, float] = (0.02, 0.3),
                               min_duration: float = 0.1) -> List[SmoothPursuit]:
        """
        Detect smooth pursuit eye movements.

        Smooth pursuits have relatively constant velocity between saccade and fixation speeds.

        Args:
            velocity_range: (min, max) velocity for pursuit detection
            min_duration: Minimum duration in seconds

        Returns:
            List of SmoothPursuit objects
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self._compute_velocity()
        self._estimate_sampling_rate()

        # Identify potential pursuit segments
        velocity = df["velocity"].values
        is_pursuit = (velocity >= velocity_range[0]) & (velocity <= velocity_range[1])

        pursuits = []
        in_pursuit = False
        start_idx = 0

        for i, is_p in enumerate(is_pursuit):
            if is_p and not in_pursuit:
                in_pursuit = True
                start_idx = i
            elif not is_p and in_pursuit:
                in_pursuit = False
                end_idx = i - 1

                # Calculate duration
                if "recording_timestamp" in df.columns:
                    duration = (df.iloc[end_idx]["recording_timestamp"] -
                                df.iloc[start_idx]["recording_timestamp"])
                    start_time = df.iloc[start_idx]["recording_timestamp"]
                    end_time = df.iloc[end_idx]["recording_timestamp"]
                else:
                    duration = (end_idx - start_idx) / self.sampling_rate
                    start_time = start_idx / self.sampling_rate
                    end_time = end_idx / self.sampling_rate

                if duration >= min_duration:
                    segment = df.iloc[start_idx:end_idx+1]
                    positions = list(zip(segment["gaze_x"], segment["gaze_y"]))

                    pursuits.append(SmoothPursuit(
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        mean_velocity=segment["velocity"].mean(),
                        gain=1.0,  # Would need target velocity for true gain
                        positions=positions,
                    ))

        self.smooth_pursuits = pursuits
        return pursuits

    # -------------------------------------------------------------------------
    # Gaze Quality Metrics
    # -------------------------------------------------------------------------

    def compute_data_quality_metrics(self) -> Dict[str, float]:
        """
        Compute data quality metrics for the recording.

        Returns:
            Dictionary with quality metrics
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data
        total_samples = len(df)

        # Validity rates
        left_valid = df["left_gaze_valid"].sum() / total_samples if "left_gaze_valid" in df.columns else np.nan
        right_valid = df["right_gaze_valid"].sum() / total_samples if "right_gaze_valid" in df.columns else np.nan
        both_valid = ((df["left_gaze_valid"] == 1) & (df["right_gaze_valid"] == 1)).sum() / total_samples

        # Gap analysis (consecutive invalid samples)
        if "gaze_valid" in df.columns:
            invalid = ~df["gaze_valid"]
        else:
            invalid = (df["left_gaze_valid"] == 0) & (df["right_gaze_valid"] == 0)

        gap_lengths = []
        current_gap = 0
        for is_invalid in invalid:
            if is_invalid:
                current_gap += 1
            elif current_gap > 0:
                gap_lengths.append(current_gap)
                current_gap = 0

        # Sampling regularity
        self._estimate_sampling_rate()
        if "recording_timestamp" in df.columns:
            dt = df["recording_timestamp"].diff().dropna()
            expected_dt = 1.0 / self.sampling_rate
            sampling_jitter = dt.std() / expected_dt if expected_dt > 0 else np.nan
        else:
            sampling_jitter = np.nan

        # Precision (RMS of sample-to-sample noise during fixations)
        precision = np.nan
        if self.fixations:
            rms_values = []
            for fix in self.fixations[:10]:  # Sample first 10 fixations
                fix_samples = df[(df["recording_timestamp"] >= fix.start_time) &
                                  (df["recording_timestamp"] <= fix.end_time)]
                if len(fix_samples) > 1 and "gaze_x" in fix_samples.columns:
                    dx = fix_samples["gaze_x"].diff().dropna()
                    dy = fix_samples["gaze_y"].diff().dropna()
                    rms = np.sqrt(np.mean(dx**2 + dy**2))
                    rms_values.append(rms)
            if rms_values:
                precision = np.mean(rms_values)

        return {
            "total_samples": total_samples,
            "left_eye_validity": left_valid,
            "right_eye_validity": right_valid,
            "binocular_validity": both_valid,
            "overall_validity": max(left_valid, right_valid) if not np.isnan(left_valid) else np.nan,
            "gap_count": len(gap_lengths),
            "mean_gap_length": np.mean(gap_lengths) if gap_lengths else 0,
            "max_gap_length": max(gap_lengths) if gap_lengths else 0,
            "sampling_rate": self.sampling_rate,
            "sampling_jitter": sampling_jitter,
            "precision_rms": precision,
        }

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def get_microsaccades_dataframe(self) -> pd.DataFrame:
        """Convert microsaccades to DataFrame."""
        if not self.microsaccades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "start_time": m.start_time,
                "end_time": m.end_time,
                "duration": m.duration,
                "amplitude": m.amplitude,
                "peak_velocity": m.peak_velocity,
                "direction": m.direction,
                "start_x": m.start_x,
                "start_y": m.start_y,
                "end_x": m.end_x,
                "end_y": m.end_y,
            }
            for m in self.microsaccades
        ])

    def export_all_events(self, output_dir: str) -> Dict[str, str]:
        """
        Export all detected events to CSV files.

        Args:
            output_dir: Directory to save files

        Returns:
            Dictionary mapping event type to file path
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported = {}

        # Fixations
        fix_df = self.get_fixations_dataframe()
        if not fix_df.empty:
            path = output_path / "fixations.csv"
            fix_df.to_csv(path, index=False)
            exported["fixations"] = str(path)

        # Saccades
        sacc_df = self.get_saccades_dataframe()
        if not sacc_df.empty:
            path = output_path / "saccades.csv"
            sacc_df.to_csv(path, index=False)
            exported["saccades"] = str(path)

        # Microsaccades
        ms_df = self.get_microsaccades_dataframe()
        if not ms_df.empty:
            path = output_path / "microsaccades.csv"
            ms_df.to_csv(path, index=False)
            exported["microsaccades"] = str(path)

        return exported

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all analyses.

        Returns:
            Dictionary with all computed metrics
        """
        report = {
            "data_quality": self.compute_data_quality_metrics() if self.data is not None else {},
            "fixation_stats": self.get_fixation_stats(),
            "saccade_stats": self.get_saccade_stats(),
            "pupil_stats": self.get_pupil_stats(),
        }

        # Add advanced metrics if computed
        if self.microsaccades:
            report["microsaccade_stats"] = self.get_microsaccade_stats()

        if self.fixations and self.saccades:
            report["k_coefficient"] = self.compute_k_coefficient()

        return report

    # =========================================================================
    # STATE-OF-THE-ART ANALYSIS METHODS
    # =========================================================================

    # -------------------------------------------------------------------------
    # MultiMatch Scanpath Comparison
    # -------------------------------------------------------------------------

    def compute_multimatch(self, other_fixations: List[Fixation],
                           screen_size: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
        """
        Compute MultiMatch scanpath similarity metrics.

        MultiMatch compares scanpaths on 5 dimensions:
        1. Vector similarity (shape)
        2. Direction similarity
        3. Length similarity (saccade amplitudes)
        4. Position similarity
        5. Duration similarity (fixation durations)

        Based on: Dewhurst et al. (2012) - "It depends on how you look at it:
        Scanpath comparison in multiple dimensions with MultiMatch"

        Args:
            other_fixations: Fixations from another scanpath to compare
            screen_size: Screen dimensions for normalization (uses self.screen_size if None)

        Returns:
            Dictionary with similarity scores for each dimension (0-1, higher = more similar)
        """
        if not self.fixations:
            self.detect_fixations_idt()

        if len(self.fixations) < 2 or len(other_fixations) < 2:
            return {
                "vector_similarity": np.nan,
                "direction_similarity": np.nan,
                "length_similarity": np.nan,
                "position_similarity": np.nan,
                "duration_similarity": np.nan,
                "overall_similarity": np.nan,
            }

        screen = screen_size or self.screen_size

        # Extract scanpath vectors
        def get_vectors(fixations):
            vectors = []
            for i in range(len(fixations) - 1):
                f1, f2 = fixations[i], fixations[i + 1]
                dx = (f2.x - f1.x) * screen[0]
                dy = (f2.y - f1.y) * screen[1]
                vectors.append((dx, dy))
            return vectors

        def get_positions(fixations):
            return [(f.x * screen[0], f.y * screen[1]) for f in fixations]

        def get_durations(fixations):
            return [f.duration for f in fixations]

        vec1 = get_vectors(self.fixations)
        vec2 = get_vectors(other_fixations)
        pos1 = get_positions(self.fixations)
        pos2 = get_positions(other_fixations)
        dur1 = get_durations(self.fixations)
        dur2 = get_durations(other_fixations)

        # Simplify scanpaths to same length using linear interpolation indices
        min_len = min(len(vec1), len(vec2))
        if min_len < 1:
            return {k: np.nan for k in ["vector_similarity", "direction_similarity",
                                         "length_similarity", "position_similarity",
                                         "duration_similarity", "overall_similarity"]}

        # Resample to match lengths
        idx1 = np.linspace(0, len(vec1) - 1, min_len).astype(int)
        idx2 = np.linspace(0, len(vec2) - 1, min_len).astype(int)

        vec1_resampled = [vec1[i] for i in idx1]
        vec2_resampled = [vec2[i] for i in idx2]

        idx1_pos = np.linspace(0, len(pos1) - 1, min_len + 1).astype(int)
        idx2_pos = np.linspace(0, len(pos2) - 1, min_len + 1).astype(int)
        pos1_resampled = [pos1[i] for i in idx1_pos]
        pos2_resampled = [pos2[i] for i in idx2_pos]

        idx1_dur = np.linspace(0, len(dur1) - 1, min_len + 1).astype(int)
        idx2_dur = np.linspace(0, len(dur2) - 1, min_len + 1).astype(int)
        dur1_resampled = [dur1[i] for i in idx1_dur]
        dur2_resampled = [dur2[i] for i in idx2_dur]

        # 1. Vector similarity (shape) - normalized dot product
        vector_sims = []
        for v1, v2 in zip(vec1_resampled, vec2_resampled):
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if mag1 > 0 and mag2 > 0:
                dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)
                vector_sims.append((dot + 1) / 2)  # Normalize to 0-1
        vector_similarity = np.mean(vector_sims) if vector_sims else np.nan

        # 2. Direction similarity - angular difference
        direction_sims = []
        for v1, v2 in zip(vec1_resampled, vec2_resampled):
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            diff = abs(angle1 - angle2)
            diff = min(diff, 2*np.pi - diff)  # Shortest angular distance
            direction_sims.append(1 - diff / np.pi)
        direction_similarity = np.mean(direction_sims) if direction_sims else np.nan

        # 3. Length similarity - saccade amplitude comparison
        length_sims = []
        for v1, v2 in zip(vec1_resampled, vec2_resampled):
            len1 = np.sqrt(v1[0]**2 + v1[1]**2)
            len2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if max(len1, len2) > 0:
                length_sims.append(min(len1, len2) / max(len1, len2))
        length_similarity = np.mean(length_sims) if length_sims else np.nan

        # 4. Position similarity - Euclidean distance between fixation positions
        position_dists = []
        max_dist = np.sqrt(screen[0]**2 + screen[1]**2)
        for p1, p2 in zip(pos1_resampled, pos2_resampled):
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            position_dists.append(1 - dist / max_dist)
        position_similarity = np.mean(position_dists) if position_dists else np.nan

        # 5. Duration similarity - fixation duration comparison
        duration_sims = []
        for d1, d2 in zip(dur1_resampled, dur2_resampled):
            if max(d1, d2) > 0:
                duration_sims.append(min(d1, d2) / max(d1, d2))
        duration_similarity = np.mean(duration_sims) if duration_sims else np.nan

        # Overall similarity (average of all dimensions)
        valid_scores = [s for s in [vector_similarity, direction_similarity,
                                     length_similarity, position_similarity,
                                     duration_similarity] if not np.isnan(s)]
        overall_similarity = np.mean(valid_scores) if valid_scores else np.nan

        return {
            "vector_similarity": vector_similarity,
            "direction_similarity": direction_similarity,
            "length_similarity": length_similarity,
            "position_similarity": position_similarity,
            "duration_similarity": duration_similarity,
            "overall_similarity": overall_similarity,
        }

    def compute_dtw_similarity(self, other_fixations: List[Fixation],
                                use_duration: bool = True) -> Dict[str, float]:
        """
        Compute Dynamic Time Warping (DTW) distance between scanpaths.

        DTW finds optimal alignment between two temporal sequences,
        allowing for variations in timing and speed.

        Args:
            other_fixations: Fixations from another scanpath
            use_duration: Include fixation duration in distance calculation

        Returns:
            Dictionary with DTW distance and normalized similarity
        """
        if not self.fixations:
            self.detect_fixations_idt()

        if len(self.fixations) < 1 or len(other_fixations) < 1:
            return {"dtw_distance": np.nan, "dtw_similarity": np.nan}

        # Convert fixations to feature vectors
        def fix_to_features(fix, use_dur):
            if use_dur:
                return np.array([fix.x, fix.y, fix.duration])
            return np.array([fix.x, fix.y])

        seq1 = np.array([fix_to_features(f, use_duration) for f in self.fixations])
        seq2 = np.array([fix_to_features(f, use_duration) for f in other_fixations])

        n, m = len(seq1), len(seq2)

        # DTW distance matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )

        dtw_distance = dtw_matrix[n, m]

        # Normalize by path length
        path_length = n + m
        normalized_distance = dtw_distance / path_length if path_length > 0 else 0

        # Convert to similarity (assuming max distance is sqrt(2) for normalized coords)
        max_expected_dist = np.sqrt(2) * path_length
        similarity = 1 - (dtw_distance / max_expected_dist) if max_expected_dist > 0 else 0
        similarity = max(0, min(1, similarity))

        # Compute warping ratio (how much warping vs diagonal path)
        warping_ratio = path_length / (2 * min(n, m)) if min(n, m) > 0 else 1.0

        # Backtrack to find warping path
        warping_path = []
        i, j = n, m
        while i > 0 and j > 0:
            warping_path.append((i - 1, j - 1))
            candidates = [
                (dtw_matrix[i-1, j-1], i-1, j-1),
                (dtw_matrix[i-1, j], i-1, j),
                (dtw_matrix[i, j-1], i, j-1)
            ]
            _, i, j = min(candidates, key=lambda x: x[0])
        warping_path.reverse()

        return {
            "dtw_distance": dtw_distance,
            "normalized_distance": normalized_distance,
            "dtw_normalized_distance": normalized_distance,  # alias for compatibility
            "dtw_similarity": similarity,
            "path_length": path_length,
            "warping_ratio": warping_ratio,
            "warping_path": warping_path,
        }

    # -------------------------------------------------------------------------
    # Saliency Map Comparison Metrics
    # -------------------------------------------------------------------------

    def compare_to_saliency_map(self, saliency_map: np.ndarray,
                                 use_fixations: bool = True) -> Dict[str, float]:
        """
        Compare gaze data to a saliency/attention map.

        Computes standard metrics used in saliency model evaluation:
        - NSS (Normalized Scanpath Saliency)
        - AUC-Judd (Area Under ROC Curve)
        - KL Divergence
        - CC (Correlation Coefficient)
        - SIM (Similarity/histogram intersection)

        Args:
            saliency_map: 2D numpy array with predicted saliency (normalized 0-1)
            use_fixations: Use fixation locations (True) or raw gaze (False)

        Returns:
            Dictionary with comparison metrics
        """
        if self.data is None:
            raise ValueError("No data loaded")

        if saliency_map.max() == 0:
            return {k: np.nan for k in ["nss", "auc_judd", "kl_divergence",
                                         "correlation", "similarity"]}

        # Normalize saliency map
        sal_map = saliency_map.astype(float)
        sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)

        h, w = sal_map.shape

        # Get gaze points
        if use_fixations:
            if not self.fixations:
                self.detect_fixations_idt()
            points = [(int(f.x * w), int(f.y * h)) for f in self.fixations]
            weights = [f.duration for f in self.fixations]
        else:
            df = self.data
            if "gaze_x" not in df.columns:
                df = self.preprocess()
            points = [(int(x * w), int(y * h))
                      for x, y in zip(df["gaze_x"], df["gaze_y"])
                      if 0 <= x <= 1 and 0 <= y <= 1]
            weights = [1] * len(points)

        if not points:
            return {k: np.nan for k in ["nss", "auc_judd", "kl_divergence",
                                         "correlation", "similarity"]}

        # Create fixation map
        fixation_map = np.zeros_like(sal_map)
        for (x, y), w_val in zip(points, weights):
            if 0 <= x < sal_map.shape[1] and 0 <= y < sal_map.shape[0]:
                fixation_map[y, x] += w_val

        # Normalize fixation map
        if fixation_map.sum() > 0:
            fixation_map = fixation_map / fixation_map.sum()

        # 1. NSS (Normalized Scanpath Saliency)
        # Mean saliency at fixation locations, z-scored
        sal_mean = sal_map.mean()
        sal_std = sal_map.std()
        if sal_std > 0:
            sal_normalized = (sal_map - sal_mean) / sal_std
            nss_values = [sal_normalized[min(y, h-1), min(x, w-1)]
                          for x, y in points
                          if 0 <= x < w and 0 <= y < h]
            nss = np.mean(nss_values) if nss_values else np.nan
        else:
            nss = np.nan

        # 2. AUC-Judd (Area Under ROC Curve)
        # Threshold saliency map and compute true/false positive rates
        sal_at_fixations = [sal_map[min(y, h-1), min(x, w-1)]
                            for x, y in points
                            if 0 <= x < w and 0 <= y < h]

        if sal_at_fixations:
            thresholds = np.linspace(0, 1, 100)
            tpr = []  # True positive rate
            fpr = []  # False positive rate

            total_fixations = len(sal_at_fixations)
            total_non_fixation = sal_map.size - total_fixations

            for thresh in thresholds:
                # True positives: fixations above threshold
                tp = sum(1 for s in sal_at_fixations if s >= thresh)
                # False positives: non-fixation pixels above threshold
                fp = np.sum(sal_map >= thresh) - tp

                tpr.append(tp / total_fixations if total_fixations > 0 else 0)
                fpr.append(fp / total_non_fixation if total_non_fixation > 0 else 0)

            # Compute AUC using trapezoidal rule
            auc_judd = np.trapz(tpr, fpr)
            auc_judd = abs(auc_judd)  # Ensure positive
        else:
            auc_judd = np.nan

        # 3. KL Divergence
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        sal_prob = sal_map / (sal_map.sum() + eps)
        fix_prob = fixation_map + eps
        fix_prob = fix_prob / fix_prob.sum()

        kl_div = np.sum(fix_prob * np.log(fix_prob / (sal_prob + eps) + eps))

        # 4. Correlation Coefficient (CC)
        correlation = np.corrcoef(sal_map.flatten(), fixation_map.flatten())[0, 1]

        # 5. Similarity (SIM) - histogram intersection
        sal_hist = sal_map / (sal_map.sum() + eps)
        fix_hist = fixation_map / (fixation_map.sum() + eps)
        similarity = np.sum(np.minimum(sal_hist, fix_hist))

        return {
            "nss": nss,
            "auc_judd": auc_judd,
            "kl_divergence": kl_div,
            "correlation": correlation,
            "similarity": similarity,
        }

    # -------------------------------------------------------------------------
    # Blink Detection and Analysis
    # -------------------------------------------------------------------------

    def detect_blinks(self, min_duration: float = 0.05,
                      max_duration: float = 0.5) -> List[Dict[str, float]]:
        """
        Detect blinks from pupil data.

        Blinks are detected as periods where both eyes lose tracking
        or pupil diameter drops to zero/invalid.

        Args:
            min_duration: Minimum blink duration in seconds
            max_duration: Maximum blink duration in seconds

        Returns:
            List of blink events with timing information
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data
        self._estimate_sampling_rate()

        # Detect invalid pupil samples (blink candidates)
        if "left_pupil_valid" in df.columns and "right_pupil_valid" in df.columns:
            invalid = (df["left_pupil_valid"] == 0) | (df["right_pupil_valid"] == 0)
        elif "left_pupil_diameter" in df.columns:
            invalid = (df["left_pupil_diameter"] == 0) | (df["right_pupil_diameter"] == 0)
        else:
            return []

        blinks = []
        in_blink = False
        blink_start_idx = 0

        for i, is_invalid in enumerate(invalid):
            if is_invalid and not in_blink:
                in_blink = True
                blink_start_idx = i
            elif not is_invalid and in_blink:
                in_blink = False
                blink_end_idx = i - 1

                # Calculate duration
                if "recording_timestamp" in df.columns:
                    start_time = df.iloc[blink_start_idx]["recording_timestamp"]
                    end_time = df.iloc[blink_end_idx]["recording_timestamp"]
                    duration = end_time - start_time
                else:
                    duration = (blink_end_idx - blink_start_idx) / self.sampling_rate
                    start_time = blink_start_idx / self.sampling_rate
                    end_time = blink_end_idx / self.sampling_rate

                # Filter by duration
                if min_duration <= duration <= max_duration:
                    blinks.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        "start_index": blink_start_idx,
                        "end_index": blink_end_idx,
                    })

        return blinks

    def get_blink_stats(self, blinks: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Compute blink statistics.

        Args:
            blinks: List of blink events (from detect_blinks). If None, detects blinks first.

        Returns:
            Dictionary with blink metrics
        """
        if blinks is None:
            blinks = self.detect_blinks()

        if not blinks:
            return {
                "blink_count": 0,
                "blink_rate": 0,
                "mean_duration": np.nan,
                "std_duration": np.nan,
            }

        durations = [b["duration"] for b in blinks]

        # Calculate recording duration
        if self.data is not None and "recording_timestamp" in self.data.columns:
            total_time = (self.data["recording_timestamp"].max() -
                          self.data["recording_timestamp"].min())
        else:
            total_time = len(self.data) / self.sampling_rate if self.data is not None else 1

        # Blink rate per minute
        blink_rate = (len(blinks) / total_time) * 60 if total_time > 0 else 0

        return {
            "blink_count": len(blinks),
            "blink_rate_per_minute": blink_rate,
            "mean_duration": np.mean(durations),
            "std_duration": np.std(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "total_blink_time": np.sum(durations),
            "blink_percentage": (np.sum(durations) / total_time * 100) if total_time > 0 else 0,
        }

    # -------------------------------------------------------------------------
    # Wavelet-based Pupil Analysis
    # -------------------------------------------------------------------------

    def compute_pupil_wavelet_analysis(self, wavelet: str = "morl",
                                        scales: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform wavelet analysis on pupil diameter signal.

        Wavelet analysis reveals time-frequency characteristics of pupil
        oscillations at multiple scales, useful for cognitive load assessment.

        Args:
            wavelet: Wavelet type ('morl' for Morlet, 'mexh' for Mexican hat)
            scales: Array of scales to analyze (if None, auto-computed)

        Returns:
            Dictionary with wavelet coefficients and derived metrics
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data.copy()
        if "pupil_diameter" not in df.columns:
            df = self.preprocess()

        self._estimate_sampling_rate()

        pupil = df["pupil_diameter"].dropna().values
        if len(pupil) < 100:
            return {"error": "Insufficient data for wavelet analysis"}

        # Remove mean and handle NaN
        pupil = pupil - np.nanmean(pupil)
        pupil = np.nan_to_num(pupil)

        # Define scales if not provided
        if scales is None:
            # Scales corresponding to frequencies of interest (0.01 - 4 Hz)
            min_freq = 0.01
            max_freq = min(4.0, self.sampling_rate / 4)
            n_scales = 50
            freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_scales)
            # Convert frequencies to scales (for Morlet wavelet)
            scales = self.sampling_rate / (2 * freqs * np.pi)

        try:
            # Continuous Wavelet Transform
            coefficients, frequencies = self._cwt(pupil, scales, wavelet)

            # Compute power at different frequency bands
            power = np.abs(coefficients) ** 2
            mean_power = np.mean(power, axis=1)

            # Define frequency bands
            bands = {
                "very_low": (0.01, 0.04),    # VLF - autonomic/hormonal
                "low": (0.04, 0.15),          # LF - sympathetic
                "high": (0.15, 0.5),          # HF - parasympathetic/cognitive
                "very_high": (0.5, 2.0),      # VHF - rapid fluctuations
            }

            band_power = {}
            for band_name, (f_low, f_high) in bands.items():
                mask = (frequencies >= f_low) & (frequencies <= f_high)
                if np.any(mask):
                    band_power[f"{band_name}_power"] = np.mean(mean_power[mask])
                else:
                    band_power[f"{band_name}_power"] = np.nan

            # Time-varying power in cognitive load band (HF)
            hf_mask = (frequencies >= 0.15) & (frequencies <= 0.5)
            if np.any(hf_mask):
                hf_power_time = np.mean(power[hf_mask, :], axis=0)
            else:
                hf_power_time = np.array([])

            return {
                "scales": scales.tolist(),
                "frequencies": frequencies.tolist(),
                "mean_power_spectrum": mean_power.tolist(),
                **band_power,
                "total_power": np.sum(mean_power),
                "dominant_frequency": frequencies[np.argmax(mean_power)],
                "hf_power_timeseries": hf_power_time.tolist() if len(hf_power_time) > 0 else [],
            }

        except Exception as e:
            return {"error": str(e)}

    def _cwt(self, data: np.ndarray, scales: np.ndarray,
             wavelet: str = "morl") -> Tuple[np.ndarray, np.ndarray]:
        """
        Continuous Wavelet Transform implementation.

        Args:
            data: Input signal
            scales: Array of scales
            wavelet: Wavelet type

        Returns:
            Tuple of (coefficients, frequencies)
        """
        n = len(data)
        coefficients = np.zeros((len(scales), n))

        for i, scale in enumerate(scales):
            # Generate wavelet
            wavelet_length = min(10 * int(scale), n)
            t = np.arange(-wavelet_length // 2, wavelet_length // 2) / scale

            if wavelet == "morl":
                # Morlet wavelet
                psi = np.exp(-t**2 / 2) * np.cos(5 * t)
            elif wavelet == "mexh":
                # Mexican hat wavelet
                psi = (1 - t**2) * np.exp(-t**2 / 2)
            else:
                psi = np.exp(-t**2 / 2) * np.cos(5 * t)

            psi = psi / np.sqrt(scale)

            # Convolve
            conv = np.convolve(data, psi, mode='same')
            coefficients[i, :] = conv

        # Convert scales to frequencies
        if wavelet == "morl":
            central_freq = 5 / (2 * np.pi)  # Morlet central frequency
        else:
            central_freq = 0.25  # Approximate for Mexican hat

        frequencies = central_freq * self.sampling_rate / scales

        return coefficients, frequencies

    # -------------------------------------------------------------------------
    # Main Sequence Analysis
    # -------------------------------------------------------------------------

    def compute_main_sequence(self, fit_model: bool = True) -> Dict[str, Any]:
        """
        Analyze the main sequence relationship between saccade amplitude and peak velocity.

        The main sequence is a fundamental property of saccades showing
        a consistent relationship: peak_velocity = A * amplitude^B

        Args:
            fit_model: Whether to fit a power law model

        Returns:
            Dictionary with main sequence analysis results
        """
        if not self.saccades:
            self.detect_saccades()

        if len(self.saccades) < 5:
            return {"error": "Insufficient saccades for main sequence analysis"}

        amplitudes = np.array([s.amplitude for s in self.saccades])
        velocities = np.array([s.velocity for s in self.saccades])

        # Filter out invalid values
        valid = (amplitudes > 0) & (velocities > 0) & np.isfinite(amplitudes) & np.isfinite(velocities)
        amplitudes = amplitudes[valid]
        velocities = velocities[valid]

        if len(amplitudes) < 5:
            return {"error": "Insufficient valid saccades"}

        results = {
            "n_saccades": len(amplitudes),
            "amplitude_range": (float(np.min(amplitudes)), float(np.max(amplitudes))),
            "velocity_range": (float(np.min(velocities)), float(np.max(velocities))),
            "correlation": float(np.corrcoef(amplitudes, velocities)[0, 1]),
        }

        if fit_model:
            try:
                # Fit power law: V = A * amp^B (linear in log space)
                log_amp = np.log(amplitudes)
                log_vel = np.log(velocities)

                # Linear regression in log space
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_amp, log_vel)

                results["power_law"] = {
                    "coefficient_A": float(np.exp(intercept)),
                    "exponent_B": float(slope),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "std_error": float(std_err),
                }

                # Also fit linear model for comparison
                lin_slope, lin_intercept, lin_r, lin_p, lin_std = stats.linregress(amplitudes, velocities)
                results["linear_model"] = {
                    "slope": float(lin_slope),
                    "intercept": float(lin_intercept),
                    "r_squared": float(lin_r ** 2),
                }

            except Exception as e:
                results["fit_error"] = str(e)

        # Detect outliers (saccades that deviate from main sequence)
        if "power_law" in results:
            predicted = results["power_law"]["coefficient_A"] * amplitudes ** results["power_law"]["exponent_B"]
            residuals = velocities - predicted
            std_residual = np.std(residuals)
            outlier_mask = np.abs(residuals) > 2 * std_residual
            results["outlier_count"] = int(np.sum(outlier_mask))
            results["outlier_percentage"] = float(np.mean(outlier_mask) * 100)

        return results

    # -------------------------------------------------------------------------
    # Fixation Dispersion Analysis
    # -------------------------------------------------------------------------

    def compute_fixation_stability(self) -> Dict[str, float]:
        """
        Analyze fixation stability metrics.

        Computes measures of how stable gaze is during fixations,
        which can indicate attention quality and neurological health.

        Returns:
            Dictionary with fixation stability metrics
        """
        if not self.fixations:
            self.detect_fixations_idt()

        if self.data is None or len(self.fixations) < 1:
            return {}

        df = self.data
        if "gaze_x" not in df.columns:
            df = self.preprocess()

        bcea_values = []  # Bivariate Contour Ellipse Area
        rms_values = []
        std_values = []

        for fix in self.fixations:
            # Get samples during this fixation
            if "recording_timestamp" in df.columns:
                mask = ((df["recording_timestamp"] >= fix.start_time) &
                        (df["recording_timestamp"] <= fix.end_time))
            else:
                continue

            fix_data = df[mask]
            if len(fix_data) < 3:
                continue

            x = fix_data["gaze_x"].values
            y = fix_data["gaze_y"].values

            # Remove NaN
            valid = ~(np.isnan(x) | np.isnan(y))
            x, y = x[valid], y[valid]

            if len(x) < 3:
                continue

            # Standard deviation
            std_x = np.std(x)
            std_y = np.std(y)
            std_values.append(np.sqrt(std_x**2 + std_y**2))

            # RMS (sample-to-sample)
            dx = np.diff(x)
            dy = np.diff(y)
            rms = np.sqrt(np.mean(dx**2 + dy**2))
            rms_values.append(rms)

            # BCEA - Bivariate Contour Ellipse Area
            # Area containing 68% of fixation points
            if std_x > 0 and std_y > 0:
                correlation = np.corrcoef(x, y)[0, 1] if len(x) > 2 else 0
                correlation = 0 if np.isnan(correlation) else correlation
                k = 1.14  # For 68% confidence
                bcea = 2 * k * np.pi * std_x * std_y * np.sqrt(1 - correlation**2)
                bcea_values.append(bcea)

        if not bcea_values:
            return {}

        return {
            "mean_bcea": float(np.mean(bcea_values)),
            "std_bcea": float(np.std(bcea_values)),
            "mean_rms": float(np.mean(rms_values)) if rms_values else np.nan,
            "mean_std": float(np.mean(std_values)) if std_values else np.nan,
            "stability_index": float(1 / (np.mean(bcea_values) + 0.001)),  # Higher = more stable
            "n_analyzed_fixations": len(bcea_values),
        }

    # -------------------------------------------------------------------------
    # Gaze Dispersion Entropy
    # -------------------------------------------------------------------------

    def compute_gaze_dispersion_entropy(self, grid_size: Tuple[int, int] = (10, 10)) -> Dict[str, float]:
        """
        Compute spatial dispersion entropy of gaze distribution.

        Higher entropy indicates more distributed/exploratory viewing,
        lower entropy indicates more focused viewing.

        Args:
            grid_size: Grid dimensions for spatial binning

        Returns:
            Dictionary with entropy metrics
        """
        if self.data is None:
            raise ValueError("No data loaded")

        df = self.data
        if "gaze_x" not in df.columns:
            df = self.preprocess()

        # Create spatial histogram
        x = df["gaze_x"].dropna().values
        y = df["gaze_y"].dropna().values

        # Filter to valid range
        valid = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        x, y = x[valid], y[valid]

        if len(x) < 10:
            return {"spatial_entropy": np.nan, "normalized_entropy": np.nan}

        # 2D histogram
        hist, _, _ = np.histogram2d(x, y, bins=grid_size, range=[[0, 1], [0, 1]])

        # Normalize to probability distribution
        prob = hist / hist.sum()
        prob = prob[prob > 0]  # Remove zeros

        # Shannon entropy
        entropy = -np.sum(prob * np.log2(prob))

        # Maximum possible entropy (uniform distribution)
        max_entropy = np.log2(grid_size[0] * grid_size[1])

        return {
            "spatial_entropy": float(entropy),
            "normalized_entropy": float(entropy / max_entropy) if max_entropy > 0 else 0,
            "max_entropy": float(max_entropy),
            "n_occupied_cells": int(np.sum(hist > 0)),
            "total_cells": int(grid_size[0] * grid_size[1]),
            "coverage_ratio": float(np.sum(hist > 0) / (grid_size[0] * grid_size[1])),
        }

    # =========================================================================
    # VISUALIZATION INTEGRATION
    # =========================================================================

    def create_visualizer(self) -> 'TobiiVisualizer':
        """
        Create a TobiiVisualizer instance configured with this analyzer's data.

        Returns:
            Configured TobiiVisualizer instance
        """
        from ..visualization.tobii_viz import TobiiVisualizer
        viz = TobiiVisualizer(screen_size=self.screen_size)
        return viz

    def visualize_all(self, output_dir: Optional[str] = None,
                      aois: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
                      show_plots: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive visualizations for all available analyses.

        This convenience method runs analyses and creates visualizations
        for fixations, saccades, heatmaps, cognitive load, and more.

        Args:
            output_dir: Directory to save figures (None = don't save)
            aois: Areas of Interest for AOI-based visualizations
            show_plots: Whether to display plots (set False for batch processing)

        Returns:
            Dictionary containing all generated figures and analysis results
        """
        import matplotlib.pyplot as plt
        from ..visualization.tobii_viz import TobiiVisualizer

        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Create visualizer
        viz = TobiiVisualizer(screen_size=self.screen_size)

        results = {"figures": {}, "analyses": {}}
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        # Ensure data is preprocessed
        if "gaze_x" not in self.data.columns:
            self.preprocess()

        # 1. Detect events if not already done
        if not self.fixations:
            self.detect_fixations_idt()
        if not self.saccades:
            self.detect_saccades()

        # 2. Generate heatmap
        heatmap = self.generate_heatmap(resolution=(100, 100))
        fig_heatmap = viz.plot_heatmap(
            heatmap, title="Gaze Heatmap",
            save_path=str(output_path / "heatmap.png") if output_path else None
        )
        results["figures"]["heatmap"] = fig_heatmap

        # 3. Scanpath visualization
        fig_scanpath = viz.plot_scanpath(
            self.fixations, self.saccades, title="Scanpath",
            save_path=str(output_path / "scanpath.png") if output_path else None
        )
        results["figures"]["scanpath"] = fig_scanpath

        # 4. Fixation duration histogram
        fig_fix_hist = viz.plot_fixation_duration_histogram(
            self.fixations, title="Fixation Duration Distribution",
            save_path=str(output_path / "fixation_histogram.png") if output_path else None
        )
        results["figures"]["fixation_histogram"] = fig_fix_hist

        # 5. Pupil timeseries
        fig_pupil = viz.plot_pupil_timeseries(
            self.data, title="Pupil Diameter Over Time",
            save_path=str(output_path / "pupil_timeseries.png") if output_path else None
        )
        results["figures"]["pupil_timeseries"] = fig_pupil

        # 6. Saccade analysis
        fig_main_seq = viz.plot_main_sequence(
            self.saccades, title="Saccade Main Sequence",
            save_path=str(output_path / "main_sequence.png") if output_path else None
        )
        results["figures"]["main_sequence"] = fig_main_seq

        fig_sacc_polar = viz.plot_saccade_polar(
            self.saccades, title="Saccade Directions",
            save_path=str(output_path / "saccade_polar.png") if output_path else None
        )
        results["figures"]["saccade_polar"] = fig_sacc_polar

        # 7. Data quality dashboard
        quality_metrics = self.compute_data_quality_metrics()
        results["analyses"]["data_quality"] = quality_metrics
        fig_quality = viz.plot_data_quality_dashboard(
            quality_metrics, self.data, title="Data Quality Report",
            save_path=str(output_path / "data_quality.png") if output_path else None
        )
        results["figures"]["data_quality"] = fig_quality

        # 8. Cognitive load analysis
        try:
            ica_result = self.compute_index_of_cognitive_activity()
            lhipa_result = self.compute_lhipa()
            tepr_result = self.compute_pupillary_response()

            results["analyses"]["ica"] = ica_result
            results["analyses"]["lhipa"] = lhipa_result
            results["analyses"]["tepr"] = tepr_result

            fig_cognitive = viz.plot_cognitive_load_dashboard(
                self.data, ica_result, lhipa_result, tepr_result,
                title="Cognitive Load Analysis",
                save_path=str(output_path / "cognitive_load.png") if output_path else None
            )
            results["figures"]["cognitive_load"] = fig_cognitive
        except Exception as e:
            print(f"Cognitive load analysis skipped: {e}")

        # 9. K coefficient over time
        try:
            k_data = self.compute_k_coefficient_over_time()
            if len(k_data) > 0:
                fig_k = viz.plot_k_coefficient_timeline(
                    k_data, title="Ambient/Focal Attention",
                    save_path=str(output_path / "k_coefficient.png") if output_path else None
                )
                results["figures"]["k_coefficient"] = fig_k
                results["analyses"]["k_coefficient"] = self.compute_k_coefficient()
        except Exception as e:
            print(f"K coefficient analysis skipped: {e}")

        # 10. AOI analysis (if AOIs provided)
        if aois:
            aoi_results = self.analyze_aoi_advanced(aois)
            results["analyses"]["aoi"] = aoi_results

            fig_aoi = viz.plot_aoi_overlay(
                aois, aoi_results, title="Areas of Interest",
                save_path=str(output_path / "aoi_overlay.png") if output_path else None
            )
            results["figures"]["aoi_overlay"] = fig_aoi

            # Transition matrix
            trans_matrix, labels = self.compute_aoi_transition_matrix(aois)
            fig_trans = viz.plot_aoi_transition_matrix(
                trans_matrix, labels, title="AOI Transition Matrix",
                save_path=str(output_path / "aoi_transitions.png") if output_path else None
            )
            results["figures"]["aoi_transitions"] = fig_trans
            results["analyses"]["aoi_transition_matrix"] = trans_matrix.tolist()

        # 11. Blink analysis
        try:
            blinks = self.detect_blinks()
            blink_stats = self.get_blink_stats(blinks)
            results["analyses"]["blinks"] = blink_stats

            if blinks:
                fig_blinks = viz.plot_blink_timeline(
                    blinks, self.data, title="Blink Detection",
                    save_path=str(output_path / "blinks.png") if output_path else None
                )
                results["figures"]["blinks"] = fig_blinks
        except Exception as e:
            print(f"Blink analysis skipped: {e}")

        # 12. Summary stats
        results["analyses"]["fixation_stats"] = self.get_fixation_stats()
        results["analyses"]["saccade_stats"] = self.get_saccade_stats()
        results["analyses"]["pupil_stats"] = self.get_pupil_stats()

        # Summary dashboard
        stats = {**self.get_fixation_stats(), **self.get_saccade_stats()}
        fig_dashboard = viz.create_summary_dashboard(
            self.data, self.fixations, heatmap, stats,
            save_path=str(output_path / "summary_dashboard.png") if output_path else None
        )
        results["figures"]["summary_dashboard"] = fig_dashboard

        if not show_plots:
            plt.close("all")

        if output_path:
            print(f"\nVisualization complete! Figures saved to: {output_path}")

        return results

    def quick_report(self) -> str:
        """
        Generate a quick text report of key metrics.

        Returns:
            Formatted string with analysis summary
        """
        if self.data is None:
            return "No data loaded."

        # Ensure analyses are run
        if not self.fixations:
            self.detect_fixations_idt()
        if not self.saccades:
            self.detect_saccades()

        fix_stats = self.get_fixation_stats()
        sacc_stats = self.get_saccade_stats()
        pupil_stats = self.get_pupil_stats()
        quality = self.compute_data_quality_metrics()

        report = []
        report.append("=" * 60)
        report.append("TOBII EYE TRACKING ANALYSIS REPORT")
        report.append("=" * 60)

        report.append("\n📊 DATA QUALITY")
        report.append("-" * 40)
        report.append(f"  Total samples:     {quality.get('total_samples', 'N/A'):,}")
        report.append(f"  Sampling rate:     {quality.get('sampling_rate', 'N/A'):.1f} Hz")
        report.append(f"  Overall validity:  {quality.get('overall_validity', 0)*100:.1f}%")
        report.append(f"  Data gaps:         {quality.get('gap_count', 0)}")

        report.append("\n👁️ FIXATIONS")
        report.append("-" * 40)
        report.append(f"  Count:             {fix_stats.get('fixation_count', 0)}")
        report.append(f"  Mean duration:     {fix_stats.get('mean_duration', 0)*1000:.1f} ms")
        report.append(f"  Std duration:      {fix_stats.get('std_duration', 0)*1000:.1f} ms")
        report.append(f"  Total time:        {fix_stats.get('total_fixation_time', 0):.2f} s")

        report.append("\n⚡ SACCADES")
        report.append("-" * 40)
        report.append(f"  Count:             {sacc_stats.get('saccade_count', 0)}")
        report.append(f"  Mean amplitude:    {sacc_stats.get('mean_amplitude', 0)*100:.2f}% screen")
        report.append(f"  Mean velocity:     {sacc_stats.get('mean_velocity', 0)*100:.2f}%/s")

        report.append("\n👀 PUPIL")
        report.append("-" * 40)
        if "combined_pupil_mean" in pupil_stats:
            report.append(f"  Mean diameter:     {pupil_stats.get('combined_pupil_mean', 0):.2f} mm")
            report.append(f"  Std diameter:      {pupil_stats.get('combined_pupil_std', 0):.2f} mm")

        # K coefficient if available
        try:
            k_result = self.compute_k_coefficient()
            report.append("\n🎯 ATTENTION MODE")
            report.append("-" * 40)
            report.append(f"  K coefficient:     {k_result.get('k_coefficient', 0):.3f}")
            report.append(f"  Attention mode:    {k_result.get('attention_mode', 'N/A')}")
        except Exception:
            pass

        report.append("\n" + "=" * 60)

        return "\n".join(report)
