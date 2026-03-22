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
        face_mesh_records_all = data.get("face_mesh", [])
        # Sample face_mesh to avoid OOM/hang on large sessions (keep every Nth record)
        _FM_MAX = 2000
        if len(face_mesh_records_all) > _FM_MAX:
            step = len(face_mesh_records_all) // _FM_MAX
            face_mesh_records = face_mesh_records_all[::step][:_FM_MAX]
        else:
            face_mesh_records = face_mesh_records_all
        hover_df = self._to_df(data.get("hover", []))
        answer_records = data.get("answer", [])
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

        # Cross-stream correlations (emotion × gaze × mouse)
        coordination = self.compute_cross_stream_metrics(
            gaze_df, mouse_df, emotion_df, gaze.get("fixations_df"),
        )

        # Per-question analysis, answer tracking, survey
        questions = self.analyze_per_question(
            answer_records, experiment_events,
            gaze_df, mouse_df, keyboard_df, emotion_df, face_mesh_records, hover_df,
        )
        answers = self.extract_answers(answer_records, experiment_events)
        survey = self.parse_survey_data(answer_records, experiment_events)

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
            "coordination": coordination,
            "questions": questions,
            "answers": answers,
            "survey": survey,
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
        if not duration_ms and not gaze_df.empty and len(gaze_df) > 1:
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

        # Mouse trajectory analysis
        result["mouse_trajectory"] = self._mouse_trajectory(mouse_df)

        # Mouse path stats
        result["mouse_path_stats"] = self._mouse_path_stats(mouse_df)

        # Behavioral patterns (hesitations, jitter, backtracking)
        result["behavioral_patterns"] = self._detect_behavioral_patterns(mouse_df)

        # Click patterns
        result["click_patterns"] = self._click_patterns(mouse_df)

        # Keystroke dynamics
        result["keystroke_dynamics"] = self._keystroke_dynamics(keyboard_df)

        # Gaze-mouse coordination (enhanced with lag detection)
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

    def _mouse_trajectory(self, mouse_df: pd.DataFrame, max_points: int = 500) -> Dict:
        """Downsampled mouse trajectory for visualization."""
        if mouse_df.empty or "x" not in mouse_df.columns:
            return {"available": False}
        df = mouse_df.dropna(subset=["x", "y", "timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if len(df) < 2:
            return {"available": False}
        step = max(1, len(df) // max_points)
        t0 = df["timestamp"].iloc[0]
        points = []
        for i in range(0, len(df), step):
            points.append({
                "x": float(df.iloc[i]["x"]),
                "y": float(df.iloc[i]["y"]),
                "time": float(df.iloc[i]["timestamp"] - t0),
            })
        return {"available": True, "points": points}

    def _mouse_path_stats(self, mouse_df: pd.DataFrame) -> Dict:
        """Compute mouse path statistics: distance, direction changes, speed."""
        if mouse_df.empty or "x" not in mouse_df.columns:
            return {}
        df = mouse_df.dropna(subset=["x", "y", "timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if len(df) < 3:
            return {}

        xs = df["x"].values.astype(float)
        ys = df["y"].values.astype(float)
        ts = df["timestamp"].values.astype(float)

        dx = np.diff(xs)
        dy = np.diff(ys)
        seg_dist = np.sqrt(dx ** 2 + dy ** 2)
        total_distance = float(np.sum(seg_dist))

        dt = np.diff(ts)
        dt[dt == 0] = 1
        speeds = seg_dist / dt
        avg_speed = float(np.mean(speeds))
        peak_speed = float(np.max(speeds))

        # Direction changes (>45 degrees)
        angles = np.arctan2(dy, dx)
        angle_diffs = np.abs(np.diff(angles))
        # Wrap around
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        direction_changes = int(np.sum(angle_diffs > np.pi / 4))

        duration_s = (ts[-1] - ts[0]) / 1000
        return {
            "total_distance_px": round(total_distance),
            "avg_speed_px_ms": round(avg_speed, 3),
            "peak_speed_px_ms": round(peak_speed, 3),
            "direction_changes": direction_changes,
            "duration_s": round(duration_s, 1),
            "total_samples": len(df),
        }

    def _detect_behavioral_patterns(self, mouse_df: pd.DataFrame) -> Dict:
        """Detect hesitations, jitter, and backtracking in mouse movement."""
        result = {"hesitations": [], "jitter_events": 0, "backtrack_count": 0,
                  "rapid_scans": 0, "hesitation_count": 0, "jitter_timeline": []}

        if mouse_df.empty or "x" not in mouse_df.columns:
            return result

        df = mouse_df.dropna(subset=["x", "y", "timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if len(df) < 10:
            return result

        xs = df["x"].values.astype(float)
        ys = df["y"].values.astype(float)
        ts = df["timestamp"].values.astype(float)
        t0 = ts[0]

        # --- Hesitations: cursor stays within 5px for >300ms ---
        hesitations = []
        i = 0
        while i < len(df):
            j = i + 1
            while j < len(df):
                dist = np.sqrt((xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2)
                if dist > 5:
                    break
                j += 1
            duration = ts[min(j, len(ts) - 1)] - ts[i]
            if duration >= 300:
                hesitations.append({
                    "time": float(ts[i] - t0),
                    "duration_ms": float(duration),
                    "x": float(xs[i]),
                    "y": float(ys[i]),
                })
            i = max(j, i + 1)
        result["hesitations"] = hesitations[:100]
        result["hesitation_count"] = len(hesitations)

        # --- Jitter: rapid direction changes in short windows ---
        window_ms = 500
        jitter_events = 0
        jitter_timeline = []
        wi = 0
        for wi_start in range(0, len(df) - 5, 5):
            t_start = ts[wi_start]
            # Find end of window
            wi_end = wi_start
            while wi_end < len(df) and ts[wi_end] - t_start < window_ms:
                wi_end += 1
            if wi_end - wi_start < 4:
                continue
            wdx = np.diff(xs[wi_start:wi_end])
            wdy = np.diff(ys[wi_start:wi_end])
            wangles = np.arctan2(wdy, wdx)
            wangle_diffs = np.abs(np.diff(wangles))
            wangle_diffs = np.minimum(wangle_diffs, 2 * np.pi - wangle_diffs)
            dir_changes = int(np.sum(wangle_diffs > np.pi / 4))
            if dir_changes >= 4:
                jitter_events += 1
            jitter_timeline.append({"time": float(t_start - t0), "value": dir_changes})
        result["jitter_events"] = jitter_events
        result["jitter_timeline"] = jitter_timeline[::max(1, len(jitter_timeline) // 200)]

        # --- Backtracking: cursor returns near a position visited >2s ago ---
        backtrack_count = 0
        check_step = max(1, len(df) // 500)
        for i in range(check_step * 10, len(df), check_step):
            # Look back at positions from >2s ago
            lookback_end = i
            while lookback_end > 0 and ts[i] - ts[lookback_end] < 2000:
                lookback_end -= 1
            if lookback_end <= 0:
                continue
            # Sample a few past positions
            for k in range(0, lookback_end, max(1, lookback_end // 5)):
                dist = np.sqrt((xs[i] - xs[k]) ** 2 + (ys[i] - ys[k]) ** 2)
                if dist < 30:
                    backtrack_count += 1
                    break
        result["backtrack_count"] = backtrack_count

        # --- Rapid scans: high velocity sweeps ---
        dt = np.diff(ts)
        dt[dt == 0] = 1
        speeds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / dt
        rapid_threshold = np.percentile(speeds, 95) if len(speeds) > 10 else 2.0
        result["rapid_scans"] = int(np.sum(speeds > rapid_threshold))

        return result

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

        # Coordination score: 1 - normalized distance
        max_screen_dist = 2000  # approximate diagonal
        coord_score = 1 - min(1.0, float(np.mean(distances)) / max_screen_dist) if distances else 0

        # Lag detection via cross-correlation
        lag_analysis = self._compute_gaze_mouse_lag(gaze_df, mouse_df)

        # Coordination score timeline (sliding window)
        coord_timeline = []
        if len(timeline) > 5:
            window = max(5, len(timeline) // 20)
            for i in range(0, len(timeline) - window, max(1, window // 2)):
                chunk = timeline[i:i + window]
                mean_d = np.mean([c["value"] for c in chunk])
                sc = 1 - min(1.0, mean_d / max_screen_dist)
                coord_timeline.append({"time": chunk[0]["time"], "value": float(sc)})

        return {
            "available": True,
            "mean_distance": float(np.mean(distances)) if distances else 0,
            "std_distance": float(np.std(distances)) if distances else 0,
            "correlation": self._safe_float(correlation),
            "coordination_score": self._safe_float(coord_score),
            "distance_timeline": timeline,
            "coordination_timeline": coord_timeline,
            "lag_analysis": lag_analysis,
        }

    def _compute_gaze_mouse_lag(
        self, gaze_df: pd.DataFrame, mouse_df: pd.DataFrame, max_lag: int = 10,
    ) -> Dict:
        """Cross-correlate gaze and mouse X positions at different lags."""
        if gaze_df.empty or mouse_df.empty or "x" not in gaze_df.columns:
            return {"available": False}

        # Downsample to aligned arrays
        g_x = gaze_df["x"].values.astype(float)
        m_x = mouse_df["x"].values.astype(float)
        min_len = min(len(g_x), len(m_x), 500)
        if min_len < 20:
            return {"available": False}

        # Subsample to same length
        g_step = max(1, len(g_x) // min_len)
        m_step = max(1, len(m_x) // min_len)
        g_x = g_x[::g_step][:min_len]
        m_x = m_x[::m_step][:min_len]

        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                g = g_x[lag:]
                m = m_x[:len(g)]
            else:
                m = m_x[-lag:]
                g = g_x[:len(m)]
            n = min(len(g), len(m))
            if n < 10:
                continue
            try:
                r, _ = stats.pearsonr(g[:n], m[:n])
                correlations.append({"lag": lag, "correlation": self._safe_float(r)})
            except Exception:
                correlations.append({"lag": lag, "correlation": 0})

        if not correlations:
            return {"available": False}

        best = max(correlations, key=lambda x: abs(x["correlation"]))
        gaze_leads = best["lag"] > 0
        return {
            "available": True,
            "correlations": correlations,
            "best_lag": best["lag"],
            "best_correlation": best["correlation"],
            "gaze_leads_mouse": gaze_leads,
            "interpretation": (
                f"Gaze leads mouse by ~{best['lag']} samples" if gaze_leads
                else f"Mouse leads gaze by ~{abs(best['lag'])} samples" if best["lag"] < 0
                else "Gaze and mouse are synchronized"
            ),
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

    # ---- 5b. Cross-Stream Correlations --------------------------------------

    def compute_cross_stream_metrics(
        self,
        gaze_df: pd.DataFrame,
        mouse_df: pd.DataFrame,
        emotion_df: pd.DataFrame,
        fixations_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Compute correlations between emotion, gaze, and mouse streams."""
        result: Dict[str, Any] = {"available": False}

        emotion_cols = ["confusion", "engagement", "boredom", "frustration"]
        present_emotions = [c for c in emotion_cols if not emotion_df.empty and c in emotion_df.columns]

        if not present_emotions or (gaze_df.empty and mouse_df.empty):
            return result

        result["available"] = True

        # --- Emotion × Gaze correlation matrix ---
        eg_matrix = self._emotion_gaze_correlation(emotion_df, gaze_df, fixations_df, present_emotions)
        result["emotion_gaze"] = eg_matrix

        # --- Emotion × Mouse correlation matrix ---
        em_matrix = self._emotion_mouse_correlation(emotion_df, mouse_df, present_emotions)
        result["emotion_mouse"] = em_matrix

        # --- Combined state timeline ---
        result["combined_timeline"] = self._combined_state_timeline(
            gaze_df, mouse_df, emotion_df, present_emotions,
        )

        return self._make_serializable(result)

    def _emotion_gaze_correlation(
        self, emotion_df, gaze_df, fixations_df, emotion_cols, window_ms=10000,
    ) -> Dict:
        """Correlate each emotion with gaze metrics in time windows."""
        if emotion_df.empty or gaze_df.empty or "timestamp" not in emotion_df.columns:
            return {"available": False}

        gaze_metrics = ["dispersion", "velocity"]
        matrix = {em: {gm: 0.0 for gm in gaze_metrics} for em in emotion_cols}

        e_ts = emotion_df["timestamp"].values.astype(float)
        g_ts = gaze_df["timestamp"].values.astype(float)
        t_start = max(e_ts[0], g_ts[0])
        t_end = min(e_ts[-1], g_ts[-1])

        if t_end - t_start < window_ms * 2:
            return {"available": False, "matrix": matrix, "labels_x": gaze_metrics, "labels_y": emotion_cols}

        # Build windowed signals
        windows = np.arange(t_start, t_end, window_ms / 2)
        emotion_signals = {em: [] for em in emotion_cols}
        gaze_signals = {gm: [] for gm in gaze_metrics}

        for w_start in windows:
            w_end = w_start + window_ms

            # Emotion means in window
            e_mask = (e_ts >= w_start) & (e_ts < w_end)
            e_win = emotion_df[e_mask]
            for em in emotion_cols:
                if len(e_win) > 0:
                    vals = pd.to_numeric(e_win[em], errors="coerce").dropna()
                    emotion_signals[em].append(float(vals.mean()) if len(vals) > 0 else 0)
                else:
                    emotion_signals[em].append(0)

            # Gaze metrics in window
            g_mask = (g_ts >= w_start) & (g_ts < w_end)
            g_win = gaze_df[g_mask]
            if len(g_win) > 2 and "x" in g_win.columns:
                gaze_signals["dispersion"].append(
                    float(g_win["x"].std() + g_win["y"].std()) / 2)
                dx = np.diff(g_win["x"].values.astype(float))
                dy = np.diff(g_win["y"].values.astype(float))
                dt = np.diff(g_win["timestamp"].values.astype(float))
                dt[dt == 0] = 1
                gaze_signals["velocity"].append(float(np.mean(np.sqrt(dx**2 + dy**2) / dt)))
            else:
                gaze_signals["dispersion"].append(0)
                gaze_signals["velocity"].append(0)

        # Compute correlations
        for em in emotion_cols:
            for gm in gaze_metrics:
                es = np.array(emotion_signals[em])
                gs = np.array(gaze_signals[gm])
                if len(es) > 5 and np.std(es) > 0 and np.std(gs) > 0:
                    try:
                        r, _ = stats.pearsonr(es, gs)
                        matrix[em][gm] = self._safe_float(r)
                    except Exception:
                        pass

        return {
            "available": True,
            "matrix": matrix,
            "labels_x": gaze_metrics,
            "labels_y": emotion_cols,
        }

    def _emotion_mouse_correlation(
        self, emotion_df, mouse_df, emotion_cols, window_ms=10000,
    ) -> Dict:
        """Correlate each emotion with mouse metrics in time windows."""
        if emotion_df.empty or mouse_df.empty or "timestamp" not in emotion_df.columns:
            return {"available": False}

        mouse_metrics = ["velocity", "click_rate"]
        matrix = {em: {mm: 0.0 for mm in mouse_metrics} for em in emotion_cols}

        e_ts = emotion_df["timestamp"].values.astype(float)
        m_ts = mouse_df["timestamp"].values.astype(float)
        t_start = max(e_ts[0], m_ts[0])
        t_end = min(e_ts[-1], m_ts[-1])

        if t_end - t_start < window_ms * 2:
            return {"available": False, "matrix": matrix, "labels_x": mouse_metrics, "labels_y": emotion_cols}

        windows = np.arange(t_start, t_end, window_ms / 2)
        emotion_signals = {em: [] for em in emotion_cols}
        mouse_signals = {mm: [] for mm in mouse_metrics}

        has_event = "event" in mouse_df.columns

        for w_start in windows:
            w_end = w_start + window_ms

            # Emotion means
            e_mask = (e_ts >= w_start) & (e_ts < w_end)
            e_win = emotion_df[e_mask]
            for em in emotion_cols:
                if len(e_win) > 0:
                    vals = pd.to_numeric(e_win[em], errors="coerce").dropna()
                    emotion_signals[em].append(float(vals.mean()) if len(vals) > 0 else 0)
                else:
                    emotion_signals[em].append(0)

            # Mouse metrics
            m_mask = (m_ts >= w_start) & (m_ts < w_end)
            m_win = mouse_df[m_mask]
            if len(m_win) > 2 and "x" in m_win.columns:
                dx = np.diff(m_win["x"].values.astype(float))
                dy = np.diff(m_win["y"].values.astype(float))
                dt = np.diff(m_win["timestamp"].values.astype(float))
                dt[dt == 0] = 1
                mouse_signals["velocity"].append(float(np.mean(np.sqrt(dx**2 + dy**2) / dt)))
                if has_event:
                    clicks = m_win[m_win["event"].isin(["click", "mousedown"])]
                    mouse_signals["click_rate"].append(len(clicks) / (window_ms / 1000))
                else:
                    mouse_signals["click_rate"].append(0)
            else:
                mouse_signals["velocity"].append(0)
                mouse_signals["click_rate"].append(0)

        # Compute correlations
        for em in emotion_cols:
            for mm in mouse_metrics:
                es = np.array(emotion_signals[em])
                ms = np.array(mouse_signals[mm])
                if len(es) > 5 and np.std(es) > 0 and np.std(ms) > 0:
                    try:
                        r, _ = stats.pearsonr(es, ms)
                        matrix[em][mm] = self._safe_float(r)
                    except Exception:
                        pass

        return {
            "available": True,
            "matrix": matrix,
            "labels_x": mouse_metrics,
            "labels_y": emotion_cols,
        }

    def _combined_state_timeline(
        self, gaze_df, mouse_df, emotion_df, emotion_cols, window_ms=10000,
    ) -> Dict:
        """Build a combined timeline: coordination score + cognitive load + dominant emotion."""
        all_ts = []
        for df in [gaze_df, mouse_df, emotion_df]:
            if not df.empty and "timestamp" in df.columns:
                all_ts.extend(df["timestamp"].tolist())
        if not all_ts:
            return {"available": False}

        all_ts = sorted(all_ts)
        t_start = all_ts[0]
        t_end = all_ts[-1]
        if t_end - t_start < window_ms:
            return {"available": False}

        g_ts = gaze_df["timestamp"].values.astype(float) if not gaze_df.empty and "timestamp" in gaze_df.columns else np.array([])
        m_ts = mouse_df["timestamp"].values.astype(float) if not mouse_df.empty and "timestamp" in mouse_df.columns else np.array([])
        e_ts = emotion_df["timestamp"].values.astype(float) if not emotion_df.empty and "timestamp" in emotion_df.columns else np.array([])

        windows = np.arange(t_start, t_end, window_ms / 2)
        coordination = []
        cognitive_load = []
        dominant_emotions = []
        times = []

        for w_start in windows:
            w_end = w_start + window_ms
            times.append(float(w_start - t_start))

            # Coordination: gaze-mouse distance
            if len(g_ts) > 0 and len(m_ts) > 0:
                g_mask = (g_ts >= w_start) & (g_ts < w_end)
                m_mask = (m_ts >= w_start) & (m_ts < w_end)
                g_win = gaze_df[g_mask]
                m_win = mouse_df[m_mask]
                if len(g_win) > 0 and len(m_win) > 0 and "x" in g_win.columns and "x" in m_win.columns:
                    gx = g_win["x"].mean()
                    gy = g_win["y"].mean()
                    mx = m_win["x"].mean()
                    my = m_win["y"].mean()
                    d = float(np.sqrt((gx - mx) ** 2 + (gy - my) ** 2))
                    coordination.append(1 - min(1.0, d / 2000))
                else:
                    coordination.append(None)
            else:
                coordination.append(None)

            # Cognitive load proxy: gaze dispersion
            if len(g_ts) > 0:
                g_mask = (g_ts >= w_start) & (g_ts < w_end)
                g_win = gaze_df[g_mask]
                if len(g_win) > 2 and "x" in g_win.columns:
                    disp = float(g_win["x"].std() + g_win["y"].std()) / 2
                    cognitive_load.append(min(1.0, disp / 300))
                else:
                    cognitive_load.append(None)
            else:
                cognitive_load.append(None)

            # Dominant emotion
            if len(e_ts) > 0 and emotion_cols:
                e_mask = (e_ts >= w_start) & (e_ts < w_end)
                e_win = emotion_df[e_mask]
                if len(e_win) > 0:
                    means = {}
                    for em in emotion_cols:
                        if em in e_win.columns:
                            vals = pd.to_numeric(e_win[em], errors="coerce").dropna()
                            means[em] = float(vals.mean()) if len(vals) > 0 else 0
                    if means:
                        dominant_emotions.append(max(means, key=means.get))
                    else:
                        dominant_emotions.append("unknown")
                else:
                    dominant_emotions.append("unknown")
            else:
                dominant_emotions.append("unknown")

        return {
            "available": True,
            "times": times,
            "coordination": coordination,
            "cognitive_load": cognitive_load,
            "dominant_emotions": dominant_emotions,
        }

    # ---- 6. Temporal Patterns -----------------------------------------------

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

    # ---- 8. Per-Question Analysis -------------------------------------------

    # Survey question labels from green_energy_v2.1.html
    SURVEY_LABELS = {
        "q1": "How difficult was the material to understand?",
        "q2": "How difficult were the questions to answer?",
        "q3": "How much mental effort did you invest?",
        "q4": "How often did you use the written text (left panel)?",
        "q5": "How often did you use the charts and graphics (middle panel)?",
        "q6": "How often did you use the calculator?",
        "q7": "I prefer making decisions based on analysis rather than intuition.",
        "q8": "I prefer to gather all information before making a decision.",
        "q9": "I am comfortable making quick decisions under uncertainty.",
        "q10": "How engaged did you feel during the study?",
        "q11": "How confident are you in your answers?",
        "q12": "Prior knowledge of solar energy before this study?",
        "q13": "How comfortable are you with math calculations?",
        "q14": "I learn better from visual information than written text.",
        "q15": "I prefer working independently rather than in groups.",
        "q16": "Gender",
        "q17": "Age group",
        "q18": "Student level",
        "q19": "Field of study",
    }

    SURVEY_SECTIONS = {
        "difficulty_effort": {"label": "Difficulty & Effort", "keys": ["q1", "q2", "q3"]},
        "information_use": {"label": "Information Use", "keys": ["q4", "q5", "q6"]},
        "decision_style": {"label": "Decision Style", "keys": ["q7", "q8", "q9"]},
        "experience": {"label": "Experience", "keys": ["q10", "q11", "q12"]},
        "about_you": {"label": "About You", "keys": ["q13", "q14", "q15"]},
        "demographics": {"label": "Demographics", "keys": ["q16", "q17", "q18", "q19"]},
    }

    QUESTION_TITLES = {
        "q1": "Understanding Energy Units",
        "q2": "Solar Panel Basics",
        "q3": "Solar Cost & Tax Credit",
        "q4": "Solar Payback Period",
        "q5": "Daily Energy Production",
        "q6": "Monthly Electricity Bill",
        "q7": "Wind Energy Output",
        "q8": "Battery Storage Sizing",
        "q9": "Carbon Footprint Reduction",
        "q10": "Heat Pump Efficiency",
        "q11": "LED vs Incandescent Savings",
        "q12": "Home Energy Audit",
    }

    REFOCUS_DURATION_MS = 5000  # 5-second refocus break between questions

    def detect_question_boundaries(
        self, answer_records: List[Dict], experiment_events: List[Dict],
    ) -> List[Dict]:
        """
        Detect question time boundaries from answer records and experiment events.
        Returns sorted list of question dicts with start/end timestamps.
        """
        questions = []

        # Strategy 1: Use answer records (most reliable)
        answer_entries = []
        for rec in answer_records:
            qid = rec.get("question_id", "")
            if qid and qid.startswith("q") and qid != "survey":
                ts = rec.get("timestamp") or rec.get("response_time_ms")
                if ts is not None:
                    answer_entries.append({
                        "question_id": qid,
                        "answer_timestamp": float(ts),
                        "selected_answer": rec.get("selected_answer", ""),
                        "response_time_ms": float(rec.get("response_time_ms", 0)),
                        "calc_used": bool(rec.get("calc_used", False)),
                    })

        # Strategy 2: Fallback to experiment_events answer_submit
        if not answer_entries:
            for ev in experiment_events:
                ev_type = ev.get("type", "")
                if ev_type == "answer_submit":
                    qnum = ev.get("question")
                    ts = ev.get("timestamp") or ev.get("ts")
                    if qnum is not None and ts is not None:
                        answer_entries.append({
                            "question_id": f"q{qnum}",
                            "answer_timestamp": float(ts),
                            "selected_answer": ev.get("answer", ""),
                            "response_time_ms": float(ev.get("timestamp", ts)),
                            "calc_used": bool(ev.get("calcUsed", False)),
                        })

        if not answer_entries:
            return []

        # Sort by timestamp
        answer_entries.sort(key=lambda x: x["answer_timestamp"])

        # Detect screen_view events for precise start times
        screen_views = {}
        for ev in experiment_events:
            if ev.get("type") == "screen_view":
                screen = ev.get("screen", "")
                ts = ev.get("timestamp") or ev.get("ts")
                if screen and ts is not None:
                    screen_views[screen] = float(ts)

        # Build question boundaries
        for i, entry in enumerate(answer_entries):
            qid = entry["question_id"]

            # Start time: from screen_view if available, else infer
            if qid in screen_views:
                start_ms = screen_views[qid]
            elif i == 0:
                start_ms = 0  # First question starts at session start
            else:
                # After previous answer + refocus break
                start_ms = answer_entries[i - 1]["answer_timestamp"] + self.REFOCUS_DURATION_MS

            end_ms = entry["answer_timestamp"]

            # Extract question number
            try:
                qnum = int(qid.replace("q", ""))
            except ValueError:
                qnum = i + 1

            questions.append({
                "question_id": qid,
                "question_number": qnum,
                "title": self.QUESTION_TITLES.get(qid, f"Question {qnum}"),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": max(0, end_ms - start_ms),
                "selected_answer": entry["selected_answer"],
                "response_time_ms": entry["response_time_ms"],
                "calc_used": entry["calc_used"],
            })

        return questions

    def _slice_df(self, df: pd.DataFrame, start_ms: float, end_ms: float) -> pd.DataFrame:
        """Slice a DataFrame by timestamp range."""
        if df is None or df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()
        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)
        return df[mask].reset_index(drop=True)

    def _compute_question_metrics(
        self, q: Dict,
        gaze_df: pd.DataFrame, mouse_df: pd.DataFrame,
        keyboard_df: pd.DataFrame, emotion_df: pd.DataFrame,
        hover_df: pd.DataFrame,
    ) -> Dict:
        """Compute summary metrics for a single question time slice."""
        start = q["start_ms"]
        end = q["end_ms"]

        # Slice all streams
        g = self._slice_df(gaze_df, start, end)
        m = self._slice_df(mouse_df, start, end)
        k = self._slice_df(keyboard_df, start, end)
        em = self._slice_df(emotion_df, start, end)
        h = self._slice_df(hover_df, start, end)

        metrics = {
            "question_id": q["question_id"],
            "question_number": q["question_number"],
            "title": q["title"],
            "start_ms": q["start_ms"],
            "end_ms": q["end_ms"],
            "duration_ms": q["duration_ms"],
            "duration_formatted": f"{q['duration_ms'] / 1000:.1f}s",
            "selected_answer": q["selected_answer"],
            "response_time_ms": q["response_time_ms"],
            "calc_used": q["calc_used"],
        }

        # Gaze metrics
        if not g.empty and "x" in g.columns:
            fixations = detect_fixations_idt(g)
            metrics["gaze_samples"] = len(g)
            metrics["fixation_count"] = len(fixations)
            metrics["fixation_duration_mean"] = self._safe_float(
                fixations["duration"].mean()) if not fixations.empty else 0
            metrics["fixation_duration_total"] = self._safe_float(
                fixations["duration"].sum()) if not fixations.empty else 0

            saccades = detect_saccades(fixations) if not fixations.empty else pd.DataFrame()
            metrics["saccade_count"] = len(saccades)
            metrics["saccade_amplitude_mean"] = self._safe_float(
                saccades["amplitude"].mean()) if not saccades.empty else 0

            # Gaze dispersion
            metrics["gaze_dispersion_x"] = self._safe_float(g["x"].std())
            metrics["gaze_dispersion_y"] = self._safe_float(g["y"].std())

            # K coefficient (single value for the question)
            fc = len(fixations)
            sc = len(saccades)
            if fc + sc > 0:
                metrics["k_coefficient"] = self._safe_float((sc - fc) / (sc + fc))
            else:
                metrics["k_coefficient"] = 0

            # Cognitive load (simplified)
            fix_dur_var = self._safe_float(fixations["duration"].var()) if not fixations.empty else 0
            sac_rate = sc / max(1, q["duration_ms"] / 1000)
            disp = (metrics["gaze_dispersion_x"] + metrics["gaze_dispersion_y"]) / 2
            # Normalize each component to [0,1] range with reasonable defaults
            norm_var = min(1.0, fix_dur_var / 50000) if fix_dur_var > 0 else 0
            norm_sac = min(1.0, sac_rate / 5) if sac_rate > 0 else 0
            norm_disp = min(1.0, disp / 300) if disp > 0 else 0
            metrics["cognitive_load"] = self._safe_float(
                (norm_var + norm_sac + norm_disp) / 3)

            # Heatmap for this question
            heatmap = self._compute_heatmap(g)
            metrics["heatmap"] = heatmap
        else:
            metrics["gaze_samples"] = 0
            metrics["fixation_count"] = 0
            metrics["fixation_duration_mean"] = 0
            metrics["saccade_count"] = 0
            metrics["k_coefficient"] = 0
            metrics["cognitive_load"] = 0
            metrics["heatmap"] = None

        # Mouse metrics
        if not m.empty:
            metrics["mouse_events"] = len(m)
            if "event" in m.columns:
                clicks = m[m["event"].isin(["click", "mousedown"])]
                metrics["click_count"] = len(clicks)
            else:
                metrics["click_count"] = 0

            # Mouse path stats per question
            path = self._mouse_path_stats(m)
            metrics["mouse_path_distance"] = path.get("total_distance_px", 0)
            metrics["mouse_direction_changes"] = path.get("direction_changes", 0)

            # Hesitation count per question
            patterns = self._detect_behavioral_patterns(m)
            metrics["hesitation_count"] = patterns.get("hesitation_count", 0)
            metrics["jitter_events"] = patterns.get("jitter_events", 0)

            # Gaze-mouse coordination per question
            if not g.empty and "x" in g.columns:
                coord = self._gaze_mouse_coordination(g, m)
                metrics["coordination_score"] = coord.get("coordination_score", 0)
            else:
                metrics["coordination_score"] = 0
        else:
            metrics["mouse_events"] = 0
            metrics["click_count"] = 0
            metrics["mouse_path_distance"] = 0
            metrics["hesitation_count"] = 0
            metrics["jitter_events"] = 0
            metrics["coordination_score"] = 0

        # Keyboard metrics
        if not k.empty:
            metrics["key_events"] = len(k)
        else:
            metrics["key_events"] = 0

        # Emotion metrics
        emotion_cols = ["confusion", "engagement", "boredom", "frustration"]
        if not em.empty:
            metrics["emotion_samples"] = len(em)
            present_cols = [c for c in emotion_cols if c in em.columns]
            emotion_means = {}
            for col in present_cols:
                vals = pd.to_numeric(em[col], errors="coerce").dropna()
                emotion_means[col] = self._safe_float(vals.mean())
            metrics["emotion_means"] = emotion_means

            # Dominant emotion
            if emotion_means:
                metrics["dominant_emotion"] = max(emotion_means, key=emotion_means.get)
            else:
                metrics["dominant_emotion"] = "unknown"
        else:
            metrics["emotion_samples"] = 0
            metrics["emotion_means"] = {}
            metrics["dominant_emotion"] = "unknown"

        # AOI dwell from hover data
        if not h.empty and "aoi" in h.columns:
            qid = q["question_id"]
            prefix = qid + "-"
            q_hover = h[h["aoi"].str.startswith(prefix, na=False)]
            aoi_dwell = {}
            for aoi_name in q_hover["aoi"].unique():
                aoi_rows = q_hover[q_hover["aoi"] == aoi_name]
                if "dwell_time_ms" in aoi_rows.columns:
                    total_dwell = pd.to_numeric(
                        aoi_rows["dwell_time_ms"], errors="coerce"
                    ).sum()
                    short_name = aoi_name.replace(prefix, "")
                    aoi_dwell[short_name] = self._safe_float(total_dwell)
            metrics["aoi_dwell"] = aoi_dwell
        else:
            metrics["aoi_dwell"] = {}

        return metrics

    def analyze_per_question(
        self,
        answer_records: List[Dict],
        experiment_events: List[Dict],
        gaze_df: pd.DataFrame,
        mouse_df: pd.DataFrame,
        keyboard_df: pd.DataFrame,
        emotion_df: pd.DataFrame,
        face_mesh_records: List[Dict],
        hover_df: pd.DataFrame,
    ) -> Dict:
        """
        Segment session by questions and compute metrics for each.
        Returns per-question metrics + summary comparison data.
        """
        boundaries = self.detect_question_boundaries(answer_records, experiment_events)

        if not boundaries:
            return {"available": False, "questions": [], "comparison": {}}

        question_metrics = []
        for q in boundaries:
            m = self._compute_question_metrics(
                q, gaze_df, mouse_df, keyboard_df, emotion_df, hover_df,
            )
            question_metrics.append(self._make_serializable(m))

        # Build comparison charts data
        comparison = {
            "question_ids": [q["question_id"] for q in question_metrics],
            "titles": [q["title"] for q in question_metrics],
            "duration_ms": [q["duration_ms"] for q in question_metrics],
            "fixation_count": [q["fixation_count"] for q in question_metrics],
            "cognitive_load": [q["cognitive_load"] for q in question_metrics],
            "click_count": [q["click_count"] for q in question_metrics],
            "k_coefficient": [q.get("k_coefficient", 0) for q in question_metrics],
            "coordination_score": [q.get("coordination_score", 0) for q in question_metrics],
            "hesitation_count": [q.get("hesitation_count", 0) for q in question_metrics],
            "mouse_path_distance": [q.get("mouse_path_distance", 0) for q in question_metrics],
        }

        # Add dominant emotion per question
        comparison["dominant_emotions"] = [
            q.get("dominant_emotion", "unknown") for q in question_metrics
        ]

        # Emotion means per state across questions
        emotion_cols = ["confusion", "engagement", "boredom", "frustration"]
        for col in emotion_cols:
            comparison[f"mean_{col}"] = [
                q.get("emotion_means", {}).get(col, 0) for q in question_metrics
            ]

        return {
            "available": True,
            "count": len(question_metrics),
            "questions": question_metrics,
            "comparison": comparison,
        }

    # ---- 9. Answer Tracking ------------------------------------------------

    def extract_answers(
        self, answer_records: List[Dict], experiment_events: List[Dict],
    ) -> Dict:
        """Extract answer data for display — selected answers, response times, calc usage."""
        answers = []

        # From answer records
        for rec in answer_records:
            qid = rec.get("question_id", "")
            if qid and qid.startswith("q") and qid != "survey":
                try:
                    qnum = int(qid.replace("q", ""))
                except ValueError:
                    continue
                answers.append({
                    "question_id": qid,
                    "question_number": qnum,
                    "title": self.QUESTION_TITLES.get(qid, f"Question {qnum}"),
                    "selected_answer": rec.get("selected_answer", ""),
                    "response_time_ms": self._safe_float(rec.get("response_time_ms", 0)),
                    "calc_used": bool(rec.get("calc_used", False)),
                    "timestamp": self._safe_float(rec.get("timestamp", 0)),
                })

        # Fallback: from experiment_events
        if not answers:
            for ev in experiment_events:
                if ev.get("type") == "answer_submit":
                    qnum = ev.get("question")
                    if qnum is not None:
                        qid = f"q{qnum}"
                        answers.append({
                            "question_id": qid,
                            "question_number": int(qnum),
                            "title": self.QUESTION_TITLES.get(qid, f"Question {qnum}"),
                            "selected_answer": ev.get("answer", ""),
                            "response_time_ms": self._safe_float(ev.get("timestamp", 0)),
                            "calc_used": bool(ev.get("calcUsed", False)),
                            "timestamp": self._safe_float(ev.get("ts", 0)),
                        })

        # Deduplicate by question_id, keep latest
        seen = {}
        for a in answers:
            seen[a["question_id"]] = a
        answers = sorted(seen.values(), key=lambda x: x["question_number"])

        if not answers:
            return {"available": False, "answers": [], "summary": {}}

        # Compute summary stats
        response_times = [a["response_time_ms"] for a in answers if a["response_time_ms"] > 0]
        calc_count = sum(1 for a in answers if a["calc_used"])

        return {
            "available": True,
            "total_answered": len(answers),
            "total_questions": 12,
            "calc_usage_count": calc_count,
            "calc_usage_rate": round(calc_count / max(1, len(answers)) * 100),
            "avg_response_time_ms": self._safe_float(
                np.mean(response_times)) if response_times else 0,
            "avg_response_time_formatted": f"{np.mean(response_times) / 1000:.1f}s" if response_times else "N/A",
            "response_times": [
                {"question_id": a["question_id"], "time_ms": a["response_time_ms"]}
                for a in answers
            ],
            "answers": answers,
            "summary": {
                "answered": len(answers),
                "skipped": 12 - len(answers),
                "calc_used": calc_count,
            },
        }

    # ---- 10. Survey Data ---------------------------------------------------

    def parse_survey_data(
        self, answer_records: List[Dict], experiment_events: List[Dict],
    ) -> Dict:
        """Parse post-study survey responses from answer or experiment_event data."""
        survey_raw = {}

        # Strategy 1: From answer records with question_id='survey'
        for rec in answer_records:
            if rec.get("question_id") == "survey":
                sd = rec.get("survey_data", {})
                # CSV round-trip serializes dicts to JSON strings — deserialize if needed
                if isinstance(sd, str):
                    try:
                        sd = json.loads(sd)
                    except (ValueError, TypeError):
                        sd = {}
                if isinstance(sd, dict) and sd:
                    survey_raw = sd
                break

        # Strategy 2: From experiment_events survey_complete
        if not survey_raw:
            for ev in experiment_events:
                if ev.get("type") == "survey_complete":
                    sd = ev.get("surveyData", {})
                    # CSV round-trip serializes dicts to JSON strings — deserialize if needed
                    if isinstance(sd, str):
                        try:
                            sd = json.loads(sd)
                        except (ValueError, TypeError):
                            sd = {}
                    if isinstance(sd, dict) and sd:
                        survey_raw = sd
                    break

        if not survey_raw:
            return {"available": False}

        # Parse responses into structured format
        ratings = {}  # q1-q15 (numeric 1-5)
        demographics = {}  # q16-q19 (text values)

        for key, val in survey_raw.items():
            if not key.startswith("q"):
                continue
            try:
                qnum = int(key.replace("q", ""))
            except ValueError:
                continue

            if qnum <= 15:
                try:
                    ratings[key] = int(val)
                except (ValueError, TypeError):
                    ratings[key] = val
            else:
                demographics[key] = str(val)

        # Group by section
        sections = {}
        for sec_id, sec_info in self.SURVEY_SECTIONS.items():
            sec_data = {}
            for k in sec_info["keys"]:
                if k in ratings:
                    sec_data[k] = ratings[k]
                elif k in demographics:
                    sec_data[k] = demographics[k]
            if sec_data:
                sections[sec_id] = {
                    "label": sec_info["label"],
                    "responses": sec_data,
                    "labels": {k: self.SURVEY_LABELS.get(k, k) for k in sec_data},
                }

        # Compute section averages (for rating sections only)
        section_averages = {}
        for sec_id in ["difficulty_effort", "information_use", "decision_style",
                        "experience", "about_you"]:
            sec = sections.get(sec_id, {})
            vals = [v for v in sec.get("responses", {}).values() if isinstance(v, (int, float))]
            if vals:
                section_averages[sec_id] = {
                    "label": self.SURVEY_SECTIONS[sec_id]["label"],
                    "mean": round(float(np.mean(vals)), 2),
                    "values": vals,
                }

        return {
            "available": True,
            "ratings": ratings,
            "demographics": demographics,
            "sections": sections,
            "section_averages": section_averages,
            "labels": self.SURVEY_LABELS,
        }
