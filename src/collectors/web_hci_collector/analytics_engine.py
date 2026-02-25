"""
HCI Analytics Engine — State-of-the-art multimodal data analysis.

Computes comprehensive HCI metrics from collected session data:
- Gaze: fixations, saccades, heatmap, scanpath, dispersion
- Attention: K coefficient, gaze entropy, cognitive load
- Behavioral: mouse dynamics, keystroke dynamics, gaze-mouse coordination
- Emotion: timeline, distribution, facial features, head pose
- Temporal: all metrics on synchronized timeline
- Comparison: multi-session side-by-side

Reuses algorithms from src/analysis/dashboard.py where applicable.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixation / Saccade detection (adapted from dashboard.py)
# ---------------------------------------------------------------------------

def detect_fixations_idt(
    gaze_df: pd.DataFrame,
    dispersion_threshold: float = 150.0,
    min_duration_ms: float = 100.0,
) -> pd.DataFrame:
    """I-DT (Dispersion-Threshold) fixation detection for webcam gaze data."""
    if gaze_df is None or len(gaze_df) < 3:
        return pd.DataFrame()

    df = gaze_df[["x", "y", "timestamp"]].dropna().reset_index(drop=True)
    if len(df) < 3:
        return pd.DataFrame()

    fixations: List[Dict] = []
    fid = 0
    i = 0
    while i < len(df):
        j = i + 1
        while j < len(df):
            window = df.iloc[i : j + 1]
            disp = max(window["x"].max() - window["x"].min(),
                       window["y"].max() - window["y"].min())
            if disp <= dispersion_threshold:
                j += 1
            else:
                break

        window = df.iloc[i:j]
        duration = float(window["timestamp"].iloc[-1] - window["timestamp"].iloc[0])
        if duration >= min_duration_ms and len(window) >= 2:
            fid += 1
            fixations.append({
                "fixation_id": fid,
                "x": float(window["x"].mean()),
                "y": float(window["y"].mean()),
                "duration": duration,
                "start_time": float(window["timestamp"].iloc[0]),
                "end_time": float(window["timestamp"].iloc[-1]),
                "point_count": len(window),
                "dispersion": float(max(
                    window["x"].max() - window["x"].min(),
                    window["y"].max() - window["y"].min(),
                )),
                "std_x": float(window["x"].std()) if len(window) > 1 else 0.0,
                "std_y": float(window["y"].std()) if len(window) > 1 else 0.0,
            })
            i = j
        else:
            i += 1

    return pd.DataFrame(fixations)


def detect_saccades(fixations_df: pd.DataFrame) -> pd.DataFrame:
    """Detect saccades between consecutive fixations."""
    if fixations_df is None or len(fixations_df) < 2:
        return pd.DataFrame()

    saccades: List[Dict] = []
    for idx in range(len(fixations_df) - 1):
        f1 = fixations_df.iloc[idx]
        f2 = fixations_df.iloc[idx + 1]
        amp = float(np.sqrt((f2["x"] - f1["x"]) ** 2 + (f2["y"] - f1["y"]) ** 2))
        dur = float(f2["start_time"] - f1["end_time"])
        if dur > 0:
            vel = amp / dur
            direction = float(np.degrees(np.arctan2(
                f2["y"] - f1["y"], f2["x"] - f1["x"]
            )))
            saccades.append({
                "saccade_id": idx + 1,
                "start_x": float(f1["x"]),
                "start_y": float(f1["y"]),
                "end_x": float(f2["x"]),
                "end_y": float(f2["y"]),
                "amplitude": amp,
                "duration": dur,
                "velocity": vel,
                "direction": direction,
            })
    return pd.DataFrame(saccades)


# ---------------------------------------------------------------------------
# Analytics Engine
# ---------------------------------------------------------------------------

class AnalyticsEngine:
    """Compute comprehensive HCI metrics for a session."""

    def __init__(self, data_processor):
        self.data_processor = data_processor

    # ---- master entry point ------------------------------------------------

    def analyze_session(self, session_id: str) -> Dict[str, Any]:
        """Compute all analytics for a session. Returns JSON-serializable dict."""
        data = self.data_processor.get_session_data(session_id)
        if not data:
            return {"error": "Session not found or no data"}

        # Build DataFrames
        gaze_df = self._to_df(data.get("gaze", []))
        mouse_df = self._to_df(data.get("mouse", []))
        keyboard_df = self._to_df(data.get("keyboard", []))
        emotion_df = self._to_df(data.get("emotion", []))
        face_mesh_records = data.get("face_mesh", [])
        hover_df = self._to_df(data.get("hover", []))
        experiment_events = data.get("experiment_event", [])

        # Load metadata
        session_dir = Path(self.data_processor.output_dir) / session_id
        metadata = self._load_metadata(session_dir)

        # Compute each category
        overview = self.compute_session_overview(data, metadata, gaze_df, emotion_df)
        gaze = self.compute_gaze_metrics(gaze_df)
        attention = self.compute_attention_metrics(
            gaze_df,
            gaze.get("fixations_df"),
            gaze.get("saccades_df"),
            emotion_df,
        )
        behavioral = self.compute_behavioral_metrics(mouse_df, keyboard_df, gaze_df)
        emotion = self.compute_emotion_metrics(emotion_df, face_mesh_records)
        temporal = self.compute_temporal_patterns(
            gaze_df, mouse_df, keyboard_df, emotion_df,
            gaze.get("fixations_df"),
            experiment_events,
        )

        # Strip internal DataFrames before returning JSON
        gaze.pop("fixations_df", None)
        gaze.pop("saccades_df", None)

        return {
            "session_id": session_id,
            "overview": overview,
            "gaze": gaze,
            "attention": attention,
            "behavioral": behavioral,
            "emotion": emotion,
            "temporal": temporal,
        }

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _to_df(records: List[Dict]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    @staticmethod
    def _load_metadata(session_dir: Path) -> Dict:
        meta_path = session_dir / "session_metadata.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except Exception:
                pass
        return {}

    @staticmethod
    def _safe_float(v, default=0.0):
        try:
            f = float(v)
            return default if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return default

    def _make_serializable(self, obj):
        """Recursively convert numpy/pandas types to JSON-safe Python types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return self._make_serializable(obj.tolist())
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    # ---- 1. Session Overview -----------------------------------------------

    def compute_session_overview(
        self,
        data: Dict,
        metadata: Dict,
        gaze_df: pd.DataFrame,
        emotion_df: pd.DataFrame,
    ) -> Dict:
        # Duration
        duration_ms = metadata.get("duration_ms", 0)
        if not duration_ms and not gaze_df.empty:
            duration_ms = float(gaze_df["timestamp"].max() - gaze_df["timestamp"].min())

        # Sample counts
        stream_counts: Dict[str, int] = {}
        for key in [
            "gaze", "mouse", "keyboard", "emotion", "face_mesh",
            "hover", "experiment_event", "answer", "calibration_click",
            "drift_sample", "window_resize",
        ]:
            stream_counts[key] = len(data.get(key, []))

        total_samples = sum(stream_counts.values())

        # Data quality
        gaze_tracking_rate = 0.0
        if not gaze_df.empty and "face_detected" in gaze_df.columns:
            gaze_tracking_rate = float(gaze_df["face_detected"].astype(bool).mean()) * 100

        emotion_confidence = 0.0
        if not emotion_df.empty and "confidence" in emotion_df.columns:
            emotion_confidence = float(emotion_df["confidence"].mean()) * 100

        # Completeness: which streams have meaningful data
        completeness = {k: (v > 0) for k, v in stream_counts.items()}

        # Quality score (0-100): weighted average of tracking rate + coverage
        active_streams = sum(1 for v in completeness.values() if v)
        quality_score = (
            0.5 * gaze_tracking_rate
            + 0.3 * min(100, active_streams / 5 * 100)
            + 0.2 * emotion_confidence
        )

        return self._make_serializable({
            "duration_ms": duration_ms,
            "duration_formatted": self._fmt_duration(duration_ms),
            "total_samples": total_samples,
            "stream_counts": stream_counts,
            "completeness": completeness,
            "gaze_tracking_rate": round(gaze_tracking_rate, 1),
            "emotion_confidence": round(emotion_confidence, 1),
            "quality_score": round(quality_score, 1),
            "status": metadata.get("status", "unknown"),
        })

    @staticmethod
    def _fmt_duration(ms):
        s = int(ms / 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    # ---- 2. Gaze Metrics ---------------------------------------------------

    def compute_gaze_metrics(self, gaze_df: pd.DataFrame) -> Dict:
        if gaze_df.empty or "x" not in gaze_df.columns:
            return {"available": False, "fixations_df": pd.DataFrame(), "saccades_df": pd.DataFrame()}

        # Fixation detection
        fixations_df = detect_fixations_idt(gaze_df)
        saccades_df = detect_saccades(fixations_df)

        # Fixation stats
        fix_stats: Dict[str, Any] = {}
        if not fixations_df.empty:
            fix_stats = {
                "count": len(fixations_df),
                "duration_mean": float(fixations_df["duration"].mean()),
                "duration_median": float(fixations_df["duration"].median()),
                "duration_std": float(fixations_df["duration"].std()),
                "duration_min": float(fixations_df["duration"].min()),
                "duration_max": float(fixations_df["duration"].max()),
                "total_fixation_time": float(fixations_df["duration"].sum()),
                "dispersion_mean": float(fixations_df["dispersion"].mean()),
            }

        # Saccade stats
        sac_stats: Dict[str, Any] = {}
        if not saccades_df.empty:
            sac_stats = {
                "count": len(saccades_df),
                "amplitude_mean": float(saccades_df["amplitude"].mean()),
                "amplitude_std": float(saccades_df["amplitude"].std()),
                "velocity_mean": float(saccades_df["velocity"].mean()),
                "velocity_max": float(saccades_df["velocity"].max()),
                "duration_mean": float(saccades_df["duration"].mean()),
                "directions": saccades_df["direction"].tolist(),
                "amplitudes": saccades_df["amplitude"].tolist(),
            }

        # Heatmap: 2D histogram
        heatmap = self._compute_heatmap(gaze_df)

        # Scanpath: fixation centroids + durations
        scanpath = []
        if not fixations_df.empty:
            for _, f in fixations_df.iterrows():
                scanpath.append({
                    "x": float(f["x"]),
                    "y": float(f["y"]),
                    "duration": float(f["duration"]),
                    "id": int(f["fixation_id"]),
                })

        # Gaze dispersion over time (5s sliding window)
        dispersion_timeline = self._gaze_dispersion_timeline(gaze_df, window_ms=5000)

        # Gaze velocity over time
        velocity_timeline = self._gaze_velocity_timeline(gaze_df)

        # Fixation duration histogram bins
        fix_dur_hist = []
        if not fixations_df.empty:
            counts, edges = np.histogram(fixations_df["duration"], bins=20)
            fix_dur_hist = [
                {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(counts[i])}
                for i in range(len(counts))
            ]

        # Saccade amplitude histogram
        sac_amp_hist = []
        if not saccades_df.empty:
            counts, edges = np.histogram(saccades_df["amplitude"], bins=20)
            sac_amp_hist = [
                {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(counts[i])}
                for i in range(len(counts))
            ]

        return self._make_serializable({
            "available": True,
            "fixation_stats": fix_stats,
            "saccade_stats": sac_stats,
            "heatmap": heatmap,
            "scanpath": scanpath,
            "dispersion_timeline": dispersion_timeline,
            "velocity_timeline": velocity_timeline,
            "fixation_duration_histogram": fix_dur_hist,
            "saccade_amplitude_histogram": sac_amp_hist,
            # internal - stripped before JSON return
            "fixations_df": fixations_df,
            "saccades_df": saccades_df,
        })

    def _compute_heatmap(self, gaze_df: pd.DataFrame, bins: int = 50) -> Dict:
        x = gaze_df["x"].values
        y = gaze_df["y"].values
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1

        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
        return {
            "z": H.T.tolist(),
            "x_edges": xedges.tolist(),
            "y_edges": yedges.tolist(),
            "x_range": [x_min, x_max],
            "y_range": [y_min, y_max],
        }

    def _gaze_dispersion_timeline(self, gaze_df: pd.DataFrame, window_ms: float = 5000) -> List[Dict]:
        if len(gaze_df) < 10:
            return []
        ts = gaze_df["timestamp"].values
        xs = gaze_df["x"].values
        ys = gaze_df["y"].values
        t0 = ts[0]
        timeline = []
        step = window_ms / 2  # 50% overlap
        t = t0
        while t < ts[-1]:
            mask = (ts >= t) & (ts < t + window_ms)
            if mask.sum() >= 3:
                std_x = float(np.std(xs[mask]))
                std_y = float(np.std(ys[mask]))
                disp = float(np.sqrt(std_x ** 2 + std_y ** 2))
                timeline.append({"time": float(t - t0), "value": disp})
            t += step
        return timeline

    def _gaze_velocity_timeline(self, gaze_df: pd.DataFrame) -> List[Dict]:
        if len(gaze_df) < 2:
            return []
        ts = gaze_df["timestamp"].values
        xs = gaze_df["x"].values
        ys = gaze_df["y"].values
        dt = np.diff(ts)
        dt[dt == 0] = 1
        dx = np.diff(xs)
        dy = np.diff(ys)
        vel = np.sqrt(dx ** 2 + dy ** 2) / dt

        # Downsample to ~200 points for chart performance
        step = max(1, len(vel) // 200)
        t0 = ts[0]
        timeline = []
        for i in range(0, len(vel), step):
            chunk = vel[i : i + step]
            timeline.append({
                "time": float(ts[i + 1] - t0),
                "value": float(np.mean(chunk)),
            })
        return timeline

    # ---- 3. Attention Metrics ----------------------------------------------

    def compute_attention_metrics(
        self,
        gaze_df: pd.DataFrame,
        fixations_df: Optional[pd.DataFrame],
        saccades_df: Optional[pd.DataFrame],
        emotion_df: pd.DataFrame,
    ) -> Dict:
        if gaze_df.empty:
            return {"available": False}

        if fixations_df is None or fixations_df.empty:
            fixations_df = pd.DataFrame()
        if saccades_df is None or saccades_df.empty:
            saccades_df = pd.DataFrame()

        t0 = float(gaze_df["timestamp"].iloc[0])

        # K coefficient over time (sliding windows)
        k_timeline = self._k_coefficient_timeline(fixations_df, saccades_df, t0)

        # Gaze entropy over time
        entropy_timeline = self._gaze_entropy_timeline(gaze_df, t0)

        # Fixation stability (rolling std of fixation duration)
        fixation_stability = []
        if len(fixations_df) >= 5:
            durations = fixations_df["duration"].values
            starts = fixations_df["start_time"].values
            window = 5
            for i in range(window, len(durations)):
                chunk = durations[i - window : i]
                fixation_stability.append({
                    "time": float(starts[i] - t0),
                    "value": float(np.std(chunk)),
                })

        # Cognitive load index (composite)
        cognitive_load = self._cognitive_load_timeline(
            gaze_df, fixations_df, saccades_df, t0
        )

        # Engagement from emotion data
        engagement_timeline = []
        if not emotion_df.empty and "engagement" in emotion_df.columns:
            step = max(1, len(emotion_df) // 300)
            et0 = float(emotion_df["timestamp"].iloc[0])
            for i in range(0, len(emotion_df), step):
                row = emotion_df.iloc[i]
                engagement_timeline.append({
                    "time": float(row["timestamp"] - et0),
                    "value": self._safe_float(row["engagement"]),
                })

        return self._make_serializable({
            "available": True,
            "k_coefficient_timeline": k_timeline,
            "gaze_entropy_timeline": entropy_timeline,
            "fixation_stability_timeline": fixation_stability,
            "cognitive_load_timeline": cognitive_load,
            "engagement_timeline": engagement_timeline,
        })

    def _k_coefficient_timeline(
        self, fixations_df: pd.DataFrame, saccades_df: pd.DataFrame, t0: float,
        window_ms: float = 10000, step_ms: float = 5000,
    ) -> List[Dict]:
        if fixations_df.empty or saccades_df.empty:
            return []
        timeline = []
        t_start = min(
            float(fixations_df["start_time"].iloc[0]),
            float(saccades_df.iloc[0].get("start_x", fixations_df["start_time"].iloc[0])),
        )
        t_end = max(
            float(fixations_df["end_time"].iloc[-1]),
            float(fixations_df["end_time"].iloc[-1]),
        )
        t = t_start
        while t < t_end:
            n_fix = int(((fixations_df["start_time"] >= t) & (fixations_df["start_time"] < t + window_ms)).sum())
            # Count saccades in window by checking saccade start times
            sac_times = (fixations_df["end_time"].values[:-1] + fixations_df["start_time"].values[1:]) / 2
            n_sac = int(((sac_times >= t) & (sac_times < t + window_ms)).sum()) if len(sac_times) > 0 else 0
            denom = n_fix + n_sac
            if denom > 0:
                k = (n_sac - n_fix) / denom
                timeline.append({"time": float(t - t0), "value": float(k)})
            t += step_ms
        return timeline

    def _gaze_entropy_timeline(
        self, gaze_df: pd.DataFrame, t0: float,
        grid_size: int = 10, window_ms: float = 10000, step_ms: float = 5000,
    ) -> List[Dict]:
        ts = gaze_df["timestamp"].values
        xs = gaze_df["x"].values
        ys = gaze_df["y"].values
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        timeline = []
        t = ts[0]
        while t < ts[-1]:
            mask = (ts >= t) & (ts < t + window_ms)
            if mask.sum() >= 5:
                gx = ((xs[mask] - x_min) / x_range * (grid_size - 1)).astype(int)
                gy = ((ys[mask] - y_min) / y_range * (grid_size - 1)).astype(int)
                gx = np.clip(gx, 0, grid_size - 1)
                gy = np.clip(gy, 0, grid_size - 1)
                cells = gx * grid_size + gy
                _, counts = np.unique(cells, return_counts=True)
                probs = counts / counts.sum()
                entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
                timeline.append({"time": float(t - t0), "value": entropy})
            t += step_ms
        return timeline

    def _cognitive_load_timeline(
        self,
        gaze_df: pd.DataFrame,
        fixations_df: pd.DataFrame,
        saccades_df: pd.DataFrame,
        t0: float,
        window_ms: float = 10000,
        step_ms: float = 5000,
    ) -> List[Dict]:
        """Composite cognitive load index: fixation duration variance + saccade rate + dispersion."""
        ts = gaze_df["timestamp"].values
        timeline = []
        t = ts[0]
        while t < ts[-1]:
            scores = []

            # Fixation duration variability (normalized)
            if not fixations_df.empty:
                mask = (fixations_df["start_time"] >= t) & (fixations_df["start_time"] < t + window_ms)
                win_fix = fixations_df[mask]
                if len(win_fix) >= 2:
                    cv = float(win_fix["duration"].std() / (win_fix["duration"].mean() + 1e-9))
                    scores.append(min(cv, 2.0) / 2.0)  # normalize to 0-1

            # Saccade rate
            if not saccades_df.empty:
                # approximate saccade times
                n_sac = len(saccades_df[
                    (saccades_df.get("duration", pd.Series(dtype=float)).index < len(saccades_df))
                ]) if len(saccades_df) > 0 else 0
                # Simple: count saccades per second
                total_dur = float(ts[-1] - ts[0])
                if total_dur > 0:
                    sac_rate = len(saccades_df) / (total_dur / 1000)
                    scores.append(min(sac_rate / 5.0, 1.0))

            # Gaze dispersion
            g_mask = (ts >= t) & (ts < t + window_ms)
            if g_mask.sum() >= 3:
                std = float(np.std(gaze_df["x"].values[g_mask]) + np.std(gaze_df["y"].values[g_mask]))
                scores.append(min(std / 500, 1.0))

            if scores:
                load = float(np.mean(scores))
                timeline.append({"time": float(t - t0), "value": load})
            t += step_ms
        return timeline

    # ---- 4. Behavioral Metrics ---------------------------------------------

    def compute_behavioral_metrics(
        self,
        mouse_df: pd.DataFrame,
        keyboard_df: pd.DataFrame,
        gaze_df: pd.DataFrame,
    ) -> Dict:
        result: Dict[str, Any] = {"available": False}

        if mouse_df.empty and keyboard_df.empty:
            return result
        result["available"] = True

        # Mouse velocity
        mouse_velocity = self._mouse_velocity_timeline(mouse_df)
        result["mouse_velocity_timeline"] = mouse_velocity

        # Click patterns
        result["click_patterns"] = self._click_patterns(mouse_df)

        # Keystroke dynamics
        result["keystroke_dynamics"] = self._keystroke_dynamics(keyboard_df)

        # Gaze-mouse coordination
        result["gaze_mouse_coordination"] = self._gaze_mouse_coordination(gaze_df, mouse_df)

        # Idle periods
        result["idle_periods"] = self._detect_idle(gaze_df, mouse_df, keyboard_df)

        return self._make_serializable(result)

    def _mouse_velocity_timeline(self, mouse_df: pd.DataFrame) -> List[Dict]:
        if mouse_df.empty or "x" not in mouse_df.columns:
            return []
        df = mouse_df.dropna(subset=["x", "y", "timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if len(df) < 2:
            return []
        ts = df["timestamp"].values
        xs = df["x"].values
        ys = df["y"].values
        dt = np.diff(ts)
        dt[dt == 0] = 1
        vel = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / dt

        t0 = ts[0]
        step = max(1, len(vel) // 200)
        timeline = []
        for i in range(0, len(vel), step):
            chunk = vel[i : i + step]
            timeline.append({"time": float(ts[i + 1] - t0), "value": float(np.mean(chunk))})
        return timeline

    def _click_patterns(self, mouse_df: pd.DataFrame) -> Dict:
        if mouse_df.empty or "event" not in mouse_df.columns:
            return {"total_clicks": 0}
        clicks = mouse_df[mouse_df["event"].isin(["click", "mousedown"])].copy()
        if clicks.empty:
            # Try without event filter
            return {"total_clicks": 0}
        clicks = clicks.sort_values("timestamp").reset_index(drop=True)
        intervals = np.diff(clicks["timestamp"].values)

        return {
            "total_clicks": len(clicks),
            "click_times": [float(t - clicks["timestamp"].iloc[0]) for t in clicks["timestamp"]],
            "click_x": clicks["x"].tolist() if "x" in clicks.columns else [],
            "click_y": clicks["y"].tolist() if "y" in clicks.columns else [],
            "inter_click_intervals": intervals.tolist() if len(intervals) > 0 else [],
            "mean_interval": float(np.mean(intervals)) if len(intervals) > 0 else 0,
        }

    def _keystroke_dynamics(self, keyboard_df: pd.DataFrame) -> Dict:
        if keyboard_df.empty:
            return {"total_events": 0}

        df = keyboard_df.sort_values("timestamp").reset_index(drop=True)
        total = len(df)

        # Inter-key intervals
        intervals = np.diff(df["timestamp"].values)
        intervals = intervals[intervals > 0]

        # Typing speed over time (chars per minute in 10s windows)
        typing_speed = []
        if total > 2:
            t0 = df["timestamp"].iloc[0]
            t_end = df["timestamp"].iloc[-1]
            window = 10000
            t = t0
            while t < t_end:
                mask = (df["timestamp"] >= t) & (df["timestamp"] < t + window)
                n = mask.sum()
                cpm = n / (window / 60000)
                typing_speed.append({"time": float(t - t0), "value": float(cpm)})
                t += window / 2

        return {
            "total_events": total,
            "inter_key_intervals": intervals[:100].tolist() if len(intervals) > 0 else [],
            "mean_interval": float(np.mean(intervals)) if len(intervals) > 0 else 0,
            "std_interval": float(np.std(intervals)) if len(intervals) > 0 else 0,
            "typing_speed_timeline": typing_speed,
        }

    def _gaze_mouse_coordination(self, gaze_df: pd.DataFrame, mouse_df: pd.DataFrame) -> Dict:
        if gaze_df.empty or mouse_df.empty or "x" not in gaze_df.columns or "x" not in mouse_df.columns:
            return {"available": False}

        # Align by nearest timestamp
        g_ts = gaze_df["timestamp"].values
        m_ts = mouse_df["timestamp"].values
        if len(g_ts) < 10 or len(m_ts) < 10:
            return {"available": False}

        # For each gaze sample, find nearest mouse sample
        distances = []
        t0 = g_ts[0]
        timeline = []
        m_idx = 0
        step = max(1, len(g_ts) // 200)
        for i in range(0, len(g_ts), step):
            gt = g_ts[i]
            # Advance mouse index to nearest
            while m_idx < len(m_ts) - 1 and abs(m_ts[m_idx + 1] - gt) < abs(m_ts[m_idx] - gt):
                m_idx += 1
            if abs(m_ts[m_idx] - gt) < 500:  # within 500ms
                gx, gy = float(gaze_df.iloc[i]["x"]), float(gaze_df.iloc[i]["y"])
                mx, my = float(mouse_df.iloc[m_idx]["x"]), float(mouse_df.iloc[m_idx]["y"])
                d = float(np.sqrt((gx - mx) ** 2 + (gy - my) ** 2))
                distances.append(d)
                timeline.append({"time": float(gt - t0), "value": d})

        correlation = 0.0
        if len(distances) > 10:
            try:
                gx_arr = gaze_df["x"].values[:len(distances)]
                mx_arr = mouse_df["x"].values[:len(distances)]
                correlation = float(np.corrcoef(gx_arr[:min(len(gx_arr), len(mx_arr))],
                                               mx_arr[:min(len(gx_arr), len(mx_arr))])[0, 1])
            except Exception:
                pass

        return {
            "available": True,
            "mean_distance": float(np.mean(distances)) if distances else 0,
            "std_distance": float(np.std(distances)) if distances else 0,
            "correlation": self._safe_float(correlation),
            "distance_timeline": timeline,
        }

    def _detect_idle(
        self, gaze_df: pd.DataFrame, mouse_df: pd.DataFrame, keyboard_df: pd.DataFrame,
        threshold_ms: float = 3000,
    ) -> List[Dict]:
        """Detect idle periods (>threshold_ms with no activity)."""
        all_ts = []
        for df in [gaze_df, mouse_df, keyboard_df]:
            if not df.empty and "timestamp" in df.columns:
                all_ts.extend(df["timestamp"].tolist())
        if not all_ts:
            return []

        all_ts = sorted(all_ts)
        t0 = all_ts[0]
        idle_periods = []
        for i in range(1, len(all_ts)):
            gap = all_ts[i] - all_ts[i - 1]
            if gap >= threshold_ms:
                idle_periods.append({
                    "start_time": float(all_ts[i - 1] - t0),
                    "end_time": float(all_ts[i] - t0),
                    "duration": float(gap),
                })
        return idle_periods[:50]  # limit

    # ---- 5. Emotion Metrics ------------------------------------------------

    def compute_emotion_metrics(
        self, emotion_df: pd.DataFrame, face_mesh_records: List[Dict]
    ) -> Dict:
        result: Dict[str, Any] = {"available": False}
        if emotion_df.empty:
            return result

        result["available"] = True
        states = ["confusion", "engagement", "boredom", "frustration"]
        available_states = [s for s in states if s in emotion_df.columns]

        if not available_states:
            return result

        # Timeline (downsampled)
        t0 = float(emotion_df["timestamp"].iloc[0])
        step = max(1, len(emotion_df) // 300)
        timeline: Dict[str, List[Dict]] = {s: [] for s in available_states}
        for i in range(0, len(emotion_df), step):
            row = emotion_df.iloc[i]
            t = float(row["timestamp"] - t0)
            for s in available_states:
                timeline[s].append({"time": t, "value": self._safe_float(row.get(s, 0))})
        result["timeline"] = timeline

        # Distribution: fraction of time each emotion is dominant
        if len(available_states) >= 2:
            state_vals = emotion_df[available_states].values
            dominant = np.argmax(state_vals, axis=1)
            distribution = {}
            for idx, s in enumerate(available_states):
                distribution[s] = float((dominant == idx).mean())
            result["distribution"] = distribution

        # Summary stats per state
        summary = {}
        for s in available_states:
            vals = pd.to_numeric(emotion_df[s], errors="coerce").dropna()
            if len(vals) > 0:
                summary[s] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
        result["summary"] = summary

        # Head pose stability (from face mesh)
        result["head_pose_stability"] = self._head_pose_stability(face_mesh_records)

        # Facial feature trends
        result["facial_features"] = self._facial_feature_trends(face_mesh_records)

        return self._make_serializable(result)

    def _head_pose_stability(self, face_mesh_records: List[Dict]) -> Dict:
        yaws, pitches, rolls, times = [], [], [], []
        for r in face_mesh_records:
            hp = r.get("head_pose")
            if hp and isinstance(hp, dict):
                ts = r.get("timestamp", 0)
                y = hp.get("yaw")
                p = hp.get("pitch")
                rl = hp.get("roll")
                if y is not None and p is not None:
                    yaws.append(float(y))
                    pitches.append(float(p))
                    rolls.append(float(rl) if rl is not None else 0)
                    times.append(float(ts))
            elif isinstance(hp, str):
                try:
                    hp_dict = json.loads(hp)
                    ts = r.get("timestamp", 0)
                    yaws.append(float(hp_dict.get("yaw", 0)))
                    pitches.append(float(hp_dict.get("pitch", 0)))
                    rolls.append(float(hp_dict.get("roll", 0)))
                    times.append(float(ts))
                except Exception:
                    pass

        if not times:
            return {"available": False}

        t0 = times[0]
        step = max(1, len(times) // 200)
        timeline = []
        for i in range(0, len(times), step):
            timeline.append({
                "time": float(times[i] - t0),
                "yaw": yaws[i],
                "pitch": pitches[i],
                "roll": rolls[i],
            })

        return {
            "available": True,
            "yaw_std": float(np.std(yaws)),
            "pitch_std": float(np.std(pitches)),
            "roll_std": float(np.std(rolls)),
            "timeline": timeline,
        }

    def _facial_feature_trends(self, face_mesh_records: List[Dict]) -> Dict:
        """Extract eye openness and brow position from face mesh landmarks."""
        if not face_mesh_records:
            return {"available": False}

        eye_openness = []
        times = []
        for r in face_mesh_records:
            kp = r.get("key_points")
            if kp and isinstance(kp, dict):
                # Use eye landmarks if available
                left_eye = kp.get("left_eye_openness")
                right_eye = kp.get("right_eye_openness")
                if left_eye is not None and right_eye is not None:
                    eye_openness.append((float(left_eye) + float(right_eye)) / 2)
                    times.append(float(r.get("timestamp", 0)))
            elif isinstance(kp, str):
                try:
                    kp_dict = json.loads(kp)
                    le = kp_dict.get("left_eye_openness")
                    re = kp_dict.get("right_eye_openness")
                    if le is not None and re is not None:
                        eye_openness.append((float(le) + float(re)) / 2)
                        times.append(float(r.get("timestamp", 0)))
                except Exception:
                    pass

        if not times:
            return {"available": False}

        t0 = times[0]
        step = max(1, len(times) // 200)
        timeline = []
        for i in range(0, len(times), step):
            timeline.append({
                "time": float(times[i] - t0),
                "eye_openness": eye_openness[i] if i < len(eye_openness) else 0,
            })

        return {"available": True, "eye_openness_timeline": timeline}

    # ---- 6. Temporal Patterns ----------------------------------------------

    def compute_temporal_patterns(
        self,
        gaze_df: pd.DataFrame,
        mouse_df: pd.DataFrame,
        keyboard_df: pd.DataFrame,
        emotion_df: pd.DataFrame,
        fixations_df: Optional[pd.DataFrame],
        experiment_events: List[Dict],
    ) -> Dict:
        """All key metrics on a unified timeline + event markers."""
        # Event markers
        markers = []
        for ev in (experiment_events or []):
            markers.append({
                "time": float(ev.get("timestamp", 0)),
                "type": ev.get("event", "event"),
                "label": ev.get("label", ev.get("event", "")),
            })

        # Normalize marker times
        all_t0 = float("inf")
        for df in [gaze_df, mouse_df, keyboard_df, emotion_df]:
            if not df.empty and "timestamp" in df.columns:
                all_t0 = min(all_t0, float(df["timestamp"].iloc[0]))
        if all_t0 == float("inf"):
            all_t0 = 0

        for m in markers:
            m["time"] = m["time"] - all_t0 if m["time"] > all_t0 else m["time"]

        # Activity rate over time (events per second, 5s windows)
        activity = self._activity_rate(gaze_df, mouse_df, keyboard_df, all_t0)

        return self._make_serializable({
            "event_markers": markers[:200],
            "activity_rate": activity,
        })

    def _activity_rate(
        self, gaze_df, mouse_df, keyboard_df, t0, window_ms=5000
    ) -> Dict[str, List[Dict]]:
        result = {}
        for name, df in [("gaze", gaze_df), ("mouse", mouse_df), ("keyboard", keyboard_df)]:
            if df.empty or "timestamp" not in df.columns:
                continue
            ts = df["timestamp"].values
            timeline = []
            t = ts[0]
            while t < ts[-1]:
                mask = (ts >= t) & (ts < t + window_ms)
                rate = float(mask.sum()) / (window_ms / 1000)
                timeline.append({"time": float(t - t0), "value": rate})
                t += window_ms / 2
            result[name] = timeline
        return result

    # ---- 7. Compare Sessions -----------------------------------------------

    def compare_sessions(self, session_ids: List[str]) -> Dict:
        """Compare metrics across multiple sessions."""
        results = {}
        for sid in session_ids:
            try:
                analytics = self.analyze_session(sid)
                # Extract key metrics for comparison
                overview = analytics.get("overview", {})
                gaze = analytics.get("gaze", {})
                fix_stats = gaze.get("fixation_stats", {})
                sac_stats = gaze.get("saccade_stats", {})
                behavioral = analytics.get("behavioral", {})

                results[sid] = {
                    "duration": overview.get("duration_formatted", "N/A"),
                    "duration_ms": overview.get("duration_ms", 0),
                    "quality_score": overview.get("quality_score", 0),
                    "total_samples": overview.get("total_samples", 0),
                    "fixation_count": fix_stats.get("count", 0),
                    "fixation_duration_mean": fix_stats.get("duration_mean", 0),
                    "saccade_count": sac_stats.get("count", 0),
                    "saccade_amplitude_mean": sac_stats.get("amplitude_mean", 0),
                    "total_clicks": behavioral.get("click_patterns", {}).get("total_clicks", 0),
                    "total_keystrokes": behavioral.get("keystroke_dynamics", {}).get("total_events", 0),
                }
            except Exception as e:
                results[sid] = {"error": str(e)}

        return {"sessions": results}
