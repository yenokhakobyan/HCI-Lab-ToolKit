"""
HCI Lab Session Analysis Dashboard

A Streamlit-based dashboard for analyzing web HCI session data including:
- Video playback with synchronized timeline
- Eye tracking heatmaps and fixation analysis
- Mouse movement trajectories and click heatmaps
- Areas of Interest (AOI) analysis
- Attention metrics
- Gaze-Mouse coordination
- Cognitive load indicators
- Behavioral patterns
- Emotion timeline integration
- Comparative analysis
- Export & reporting
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from scipy import ndimage
from scipy.stats import pearsonr
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from io import BytesIO


# Page configuration
st.set_page_config(
    page_title="HCI Session Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "web_hci"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AOI:
    """Area of Interest definition."""
    name: str
    x: int
    y: int
    width: int
    height: int
    color: str = "blue"

    def contains(self, px: float, py: float) -> bool:
        """Check if point is within AOI."""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)


# =============================================================================
# Data Loading Functions
# =============================================================================

def get_available_sessions() -> List[str]:
    """Get list of available session IDs."""
    if not DATA_DIR.exists():
        return []
    return [d.name for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]


def load_session_data(session_id: str) -> Dict:
    """Load all data for a session."""
    session_dir = DATA_DIR / session_id
    data = {
        "gaze": None,
        "mouse": None,
        "face_mesh": None,
        "emotion": None,
        "experiment_event": None,
        "metadata": None,
        "timeline": None,
        "video_path": None
    }

    # Load CSV files
    for csv_file in session_dir.glob("*.csv"):
        # Extract data type from filename like "gaze_20251209_145319.csv" -> "gaze"
        stem = csv_file.stem
        parts = stem.split('_')
        data_type = None
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                data_type = '_'.join(parts[:i])
                break
        if data_type is None:
            data_type = stem

        try:
            df = pd.read_csv(csv_file)
            data[data_type] = df
        except Exception as e:
            st.warning(f"Failed to load {csv_file.name}: {e}")

    # Load metadata
    for json_file in session_dir.glob("metadata_*.json"):
        try:
            with open(json_file) as f:
                data["metadata"] = json.load(f)
        except Exception as e:
            st.warning(f"Failed to load metadata: {e}")

    # Load timeline
    for json_file in session_dir.glob("timeline_*.json"):
        try:
            with open(json_file) as f:
                data["timeline"] = json.load(f)
        except Exception as e:
            st.warning(f"Failed to load timeline: {e}")

    # Check for video
    video_file = session_dir / "recording.webm"
    if video_file.exists():
        data["video_path"] = video_file

    return data


def load_multiple_sessions(session_ids: List[str]) -> Dict[str, Dict]:
    """Load data for multiple sessions."""
    return {sid: load_session_data(sid) for sid in session_ids}


# =============================================================================
# AOI Analysis Functions
# =============================================================================

def analyze_aoi(gaze_df: pd.DataFrame, mouse_df: pd.DataFrame, aois: List[AOI]) -> pd.DataFrame:
    """Analyze gaze and mouse data for Areas of Interest."""
    if not aois:
        return pd.DataFrame()

    results = []

    for aoi in aois:
        aoi_data = {
            'aoi_name': aoi.name,
            'gaze_points': 0,
            'gaze_duration_ms': 0,
            'gaze_entries': 0,
            'first_gaze_time': None,
            'mouse_points': 0,
            'mouse_clicks': 0,
            'mouse_entries': 0,
            'dwell_time_ms': 0,
            'hover_time_ms': 0
        }

        # Analyze gaze data
        if gaze_df is not None and not gaze_df.empty:
            in_aoi = gaze_df.apply(lambda r: aoi.contains(r['x'], r['y']), axis=1)
            aoi_gaze = gaze_df[in_aoi]

            aoi_data['gaze_points'] = len(aoi_gaze)

            if not aoi_gaze.empty:
                # Calculate duration (sum of time differences while in AOI)
                time_diffs = aoi_gaze['timestamp'].diff().fillna(0)
                aoi_data['gaze_duration_ms'] = time_diffs.sum()

                # Count entries (transitions from outside to inside)
                aoi_data['gaze_entries'] = (in_aoi & ~in_aoi.shift(1).fillna(False)).sum()

                # First fixation time
                aoi_data['first_gaze_time'] = aoi_gaze['timestamp'].min()

        # Analyze mouse data
        if mouse_df is not None and not mouse_df.empty:
            in_aoi_mouse = mouse_df.apply(lambda r: aoi.contains(r['x'], r['y']), axis=1)
            aoi_mouse = mouse_df[in_aoi_mouse]

            aoi_data['mouse_points'] = len(aoi_mouse)

            if 'event' in mouse_df.columns:
                aoi_data['mouse_clicks'] = len(aoi_mouse[aoi_mouse['event'] == 'click'])

            aoi_data['mouse_entries'] = (in_aoi_mouse & ~in_aoi_mouse.shift(1).fillna(False)).sum()

            if not aoi_mouse.empty:
                time_diffs = aoi_mouse['timestamp'].diff().fillna(0)
                aoi_data['hover_time_ms'] = time_diffs.sum()

        results.append(aoi_data)

    return pd.DataFrame(results)


def create_aoi_visualization(gaze_df: pd.DataFrame, aois: List[AOI],
                            width: int = 1920, height: int = 1080) -> go.Figure:
    """Create visualization with AOI overlays."""
    fig = go.Figure()

    # Add gaze points
    if gaze_df is not None and not gaze_df.empty:
        fig.add_trace(go.Scatter(
            x=gaze_df['x'],
            y=gaze_df['y'],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.3),
            name='Gaze Points'
        ))

    # Add AOI rectangles
    for aoi in aois:
        fig.add_shape(
            type="rect",
            x0=aoi.x, y0=aoi.y,
            x1=aoi.x + aoi.width, y1=aoi.y + aoi.height,
            line=dict(color=aoi.color, width=3),
            fillcolor=aoi.color,
            opacity=0.2,
            name=aoi.name
        )
        # Add label
        fig.add_annotation(
            x=aoi.x + aoi.width/2,
            y=aoi.y,
            text=aoi.name,
            showarrow=False,
            font=dict(size=12, color=aoi.color)
        )

    fig.update_layout(
        title="AOI Analysis",
        xaxis=dict(range=[0, width], title="X"),
        yaxis=dict(range=[height, 0], title="Y"),
        height=600
    )

    return fig


def create_aoi_transition_matrix(gaze_df: pd.DataFrame, aois: List[AOI]) -> pd.DataFrame:
    """Create transition matrix between AOIs."""
    if gaze_df is None or gaze_df.empty or not aois:
        return pd.DataFrame()

    # Assign each gaze point to an AOI (or "Outside")
    def get_aoi_name(x, y):
        for aoi in aois:
            if aoi.contains(x, y):
                return aoi.name
        return "Outside"

    gaze_df = gaze_df.copy()
    gaze_df['aoi'] = gaze_df.apply(lambda r: get_aoi_name(r['x'], r['y']), axis=1)

    # Count transitions
    aoi_names = [aoi.name for aoi in aois] + ["Outside"]
    transition_matrix = pd.DataFrame(0, index=aoi_names, columns=aoi_names)

    for i in range(1, len(gaze_df)):
        from_aoi = gaze_df.iloc[i-1]['aoi']
        to_aoi = gaze_df.iloc[i]['aoi']
        if from_aoi != to_aoi:
            transition_matrix.loc[from_aoi, to_aoi] += 1

    return transition_matrix


# =============================================================================
# Attention Metrics Functions
# =============================================================================

def calculate_attention_metrics(gaze_df: pd.DataFrame, fixations_df: pd.DataFrame,
                               aois: List[AOI] = None) -> Dict:
    """Calculate comprehensive attention metrics."""
    metrics = {
        'total_gaze_points': 0,
        'total_duration_ms': 0,
        'total_fixations': 0,
        'mean_fixation_duration_ms': 0,
        'fixation_rate': 0,  # fixations per second
        'saccade_count': 0,
        'mean_saccade_amplitude': 0,
        'gaze_dispersion': 0,  # Standard deviation of gaze positions
        'aoi_metrics': {}
    }

    if gaze_df is None or gaze_df.empty:
        return metrics

    metrics['total_gaze_points'] = len(gaze_df)
    metrics['total_duration_ms'] = gaze_df['timestamp'].max() - gaze_df['timestamp'].min()

    # Fixation metrics
    if fixations_df is not None and not fixations_df.empty:
        metrics['total_fixations'] = len(fixations_df)
        metrics['mean_fixation_duration_ms'] = fixations_df['duration'].mean()

        if metrics['total_duration_ms'] > 0:
            metrics['fixation_rate'] = metrics['total_fixations'] / (metrics['total_duration_ms'] / 1000)

    # Saccade analysis (movements between fixations)
    if fixations_df is not None and len(fixations_df) > 1:
        dx = fixations_df['x'].diff().dropna()
        dy = fixations_df['y'].diff().dropna()
        saccade_amplitudes = np.sqrt(dx**2 + dy**2)
        metrics['saccade_count'] = len(saccade_amplitudes)
        metrics['mean_saccade_amplitude'] = saccade_amplitudes.mean()

    # Gaze dispersion
    metrics['gaze_dispersion'] = np.sqrt(gaze_df['x'].var() + gaze_df['y'].var())

    # AOI-specific metrics (Time to First Fixation)
    if aois and fixations_df is not None and not fixations_df.empty:
        session_start = gaze_df['timestamp'].min()
        for aoi in aois:
            aoi_fixations = fixations_df[
                fixations_df.apply(lambda r: aoi.contains(r['x'], r['y']), axis=1)
            ]
            if not aoi_fixations.empty:
                ttff = aoi_fixations['start_time'].min() - session_start
                total_dwell = aoi_fixations['duration'].sum()
                revisits = len(aoi_fixations) - 1 if len(aoi_fixations) > 0 else 0

                metrics['aoi_metrics'][aoi.name] = {
                    'ttff_ms': ttff,
                    'total_dwell_ms': total_dwell,
                    'fixation_count': len(aoi_fixations),
                    'revisits': revisits
                }

    return metrics


# =============================================================================
# Gaze-Mouse Coordination Functions
# =============================================================================

def analyze_gaze_mouse_coordination(gaze_df: pd.DataFrame, mouse_df: pd.DataFrame) -> Dict:
    """Analyze coordination between gaze and mouse movements."""
    results = {
        'correlation_x': None,
        'correlation_y': None,
        'mean_distance': None,
        'coordination_score': None,
        'gaze_leads_mouse': 0,
        'mouse_leads_gaze': 0,
        'synchronized': 0,
        'lag_analysis': []
    }

    if gaze_df is None or mouse_df is None or gaze_df.empty or mouse_df.empty:
        return results

    # Interpolate to align timestamps
    gaze = gaze_df.copy()
    mouse = mouse_df.copy()

    # Find common time range
    start_time = max(gaze['timestamp'].min(), mouse['timestamp'].min())
    end_time = min(gaze['timestamp'].max(), mouse['timestamp'].max())

    # Filter to common range
    gaze = gaze[(gaze['timestamp'] >= start_time) & (gaze['timestamp'] <= end_time)]
    mouse = mouse[(mouse['timestamp'] >= start_time) & (mouse['timestamp'] <= end_time)]

    if len(gaze) < 10 or len(mouse) < 10:
        return results

    # Create common timestamps for comparison
    common_times = np.linspace(start_time, end_time, min(len(gaze), len(mouse), 500))

    # Interpolate positions
    gaze_x_interp = np.interp(common_times, gaze['timestamp'], gaze['x'])
    gaze_y_interp = np.interp(common_times, gaze['timestamp'], gaze['y'])
    mouse_x_interp = np.interp(common_times, mouse['timestamp'], mouse['x'])
    mouse_y_interp = np.interp(common_times, mouse['timestamp'], mouse['y'])

    # Calculate correlations
    if len(gaze_x_interp) > 2:
        results['correlation_x'], _ = pearsonr(gaze_x_interp, mouse_x_interp)
        results['correlation_y'], _ = pearsonr(gaze_y_interp, mouse_y_interp)

    # Calculate mean distance between gaze and mouse
    distances = np.sqrt((gaze_x_interp - mouse_x_interp)**2 + (gaze_y_interp - mouse_y_interp)**2)
    results['mean_distance'] = np.mean(distances)

    # Coordination score (0-1, higher = better coordination)
    max_screen_dist = np.sqrt(1920**2 + 1080**2)  # Approximate max distance
    results['coordination_score'] = 1 - (results['mean_distance'] / max_screen_dist)

    # Analyze temporal relationship (who leads whom)
    # Cross-correlation to find lag
    for lag in range(-10, 11):  # Check lags from -10 to +10 samples
        if lag < 0:
            g_x = gaze_x_interp[-lag:]
            m_x = mouse_x_interp[:lag]
        elif lag > 0:
            g_x = gaze_x_interp[:-lag]
            m_x = mouse_x_interp[lag:]
        else:
            g_x = gaze_x_interp
            m_x = mouse_x_interp

        if len(g_x) > 2:
            corr, _ = pearsonr(g_x, m_x)
            results['lag_analysis'].append({'lag': lag, 'correlation': corr})

    # Determine who typically leads
    if results['lag_analysis']:
        best_lag = max(results['lag_analysis'], key=lambda x: x['correlation'])
        if best_lag['lag'] < 0:
            results['gaze_leads_mouse'] = abs(best_lag['lag'])
        elif best_lag['lag'] > 0:
            results['mouse_leads_gaze'] = best_lag['lag']
        else:
            results['synchronized'] = 1

    return results


def create_gaze_mouse_comparison_plot(gaze_df: pd.DataFrame, mouse_df: pd.DataFrame) -> go.Figure:
    """Create visualization comparing gaze and mouse positions."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=("X Position Over Time", "Y Position Over Time"))

    if gaze_df is not None and not gaze_df.empty:
        fig.add_trace(go.Scatter(
            x=gaze_df['timestamp']/1000, y=gaze_df['x'],
            mode='lines', name='Gaze X', line=dict(color='blue')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=gaze_df['timestamp']/1000, y=gaze_df['y'],
            mode='lines', name='Gaze Y', line=dict(color='blue')
        ), row=2, col=1)

    if mouse_df is not None and not mouse_df.empty:
        fig.add_trace(go.Scatter(
            x=mouse_df['timestamp']/1000, y=mouse_df['x'],
            mode='lines', name='Mouse X', line=dict(color='red')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=mouse_df['timestamp']/1000, y=mouse_df['y'],
            mode='lines', name='Mouse Y', line=dict(color='red')
        ), row=2, col=1)

    fig.update_layout(height=500, title="Gaze vs Mouse Position Over Time")
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)

    return fig


# =============================================================================
# Cognitive Load Indicators Functions
# =============================================================================

def calculate_cognitive_load_indicators(gaze_df: pd.DataFrame, fixations_df: pd.DataFrame,
                                       face_mesh_df: pd.DataFrame = None) -> Dict:
    """Calculate indicators of cognitive load."""
    indicators = {
        'fixation_duration_trend': None,  # Increasing = higher load
        'pupil_dilation_mean': None,
        'blink_rate': None,  # Blinks per minute
        'saccade_velocity_mean': None,
        'cognitive_load_score': 0,  # Composite score 0-100
        'time_series': []
    }

    if gaze_df is None or gaze_df.empty:
        return indicators

    gaze = gaze_df.copy()

    # Analyze fixation duration trend (longer fixations = higher cognitive load)
    if fixations_df is not None and not fixations_df.empty:
        fixations = fixations_df.copy()
        fixations['time_bin'] = pd.cut(fixations['start_time'], bins=10, labels=False)

        bin_means = fixations.groupby('time_bin')['duration'].mean()
        if len(bin_means) > 1:
            # Calculate trend (positive = increasing duration = higher load)
            x = np.arange(len(bin_means))
            y = bin_means.values
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                indicators['fixation_duration_trend'] = slope

    # Saccade velocity (slower = higher load)
    gaze['dx'] = gaze['x'].diff()
    gaze['dy'] = gaze['y'].diff()
    gaze['dt'] = gaze['timestamp'].diff()
    gaze['velocity'] = np.sqrt(gaze['dx']**2 + gaze['dy']**2) / (gaze['dt'] + 1)
    indicators['saccade_velocity_mean'] = gaze['velocity'].mean()

    # Blink detection from face mesh (if available)
    if face_mesh_df is not None and not face_mesh_df.empty:
        # Estimate blink rate from face mesh data gaps or eye landmarks
        duration_minutes = (face_mesh_df['timestamp'].max() - face_mesh_df['timestamp'].min()) / 60000
        if duration_minutes > 0:
            # Simple heuristic: count large gaps in face detection as potential blinks
            time_gaps = face_mesh_df['timestamp'].diff()
            potential_blinks = (time_gaps > 100) & (time_gaps < 400)  # 100-400ms gaps
            indicators['blink_rate'] = potential_blinks.sum() / duration_minutes

    # Calculate composite cognitive load score
    load_components = []

    if indicators['fixation_duration_trend'] is not None:
        # Normalize trend to 0-100
        trend_score = min(100, max(0, indicators['fixation_duration_trend'] * 10 + 50))
        load_components.append(trend_score)

    if indicators['saccade_velocity_mean'] is not None:
        # Lower velocity = higher load
        velocity_score = min(100, max(0, 100 - indicators['saccade_velocity_mean'] * 0.5))
        load_components.append(velocity_score)

    if load_components:
        indicators['cognitive_load_score'] = np.mean(load_components)

    # Time series of cognitive load
    window_size = len(gaze) // 20 if len(gaze) > 20 else 1
    if window_size > 0:
        for i in range(0, len(gaze) - window_size, window_size):
            window = gaze.iloc[i:i+window_size]
            avg_velocity = window['velocity'].mean()
            time_point = window['timestamp'].mean()

            # Simple load estimate from velocity
            load_estimate = min(100, max(0, 100 - avg_velocity * 0.5))
            indicators['time_series'].append({
                'time': time_point,
                'load': load_estimate
            })

    return indicators


def create_cognitive_load_timeline(indicators: Dict) -> go.Figure:
    """Create timeline visualization of cognitive load."""
    fig = go.Figure()

    if indicators['time_series']:
        times = [p['time']/1000 for p in indicators['time_series']]
        loads = [p['load'] for p in indicators['time_series']]

        fig.add_trace(go.Scatter(
            x=times, y=loads,
            mode='lines+markers',
            name='Cognitive Load',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)'
        ))

    fig.update_layout(
        title="Cognitive Load Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Cognitive Load Score (0-100)",
        yaxis=dict(range=[0, 100]),
        height=400
    )

    return fig


# =============================================================================
# Behavioral Patterns Functions
# =============================================================================

def detect_behavioral_patterns(gaze_df: pd.DataFrame, mouse_df: pd.DataFrame,
                              fixations_df: pd.DataFrame) -> Dict:
    """Detect various behavioral patterns in the data."""
    patterns = {
        'hesitations': [],  # Mouse hovers without clicks
        'reading_pattern': None,  # F-pattern, Z-pattern, etc.
        'rapid_scanning': 0,  # Quick eye movements count
        'focused_attention': [],  # Prolonged fixations
        'mouse_jitter': 0,  # Nervous mouse movements
        'backtracking': 0,  # Returning to previous positions
        'gaze_trajectory': [],  # For visualization
        'pattern_summary': {}
    }

    # Detect hesitations (mouse stays in area without clicking)
    if mouse_df is not None and not mouse_df.empty:
        mouse = mouse_df.copy()
        mouse['dx'] = mouse['x'].diff().abs().fillna(0)
        mouse['dy'] = mouse['y'].diff().abs().fillna(0)
        mouse['dt'] = mouse['timestamp'].diff().fillna(1)

        # Find periods of low movement (hesitation)
        # Use 30px threshold - more realistic for mouse movement
        mouse['movement'] = mouse['dx'] + mouse['dy']
        mouse['is_stationary'] = mouse['movement'] < 30

        # Group consecutive stationary periods
        mouse['stationary_group'] = (~mouse['is_stationary']).cumsum()

        for group_id, group in mouse[mouse['is_stationary']].groupby('stationary_group'):
            if len(group) < 2:
                continue
            duration = group['timestamp'].max() - group['timestamp'].min()
            if duration > 300:  # More than 300ms (lowered from 500)
                # Check if no click occurred
                has_click = 'event' in group.columns and (group['event'] == 'click').any()
                if not has_click:
                    patterns['hesitations'].append({
                        'x': group['x'].mean(),
                        'y': group['y'].mean(),
                        'duration_ms': duration,
                        'time': group['timestamp'].min()
                    })

        # Detect mouse jitter (rapid small movements)
        small_rapid = (mouse['movement'] > 5) & (mouse['movement'] < 50) & (mouse['dt'] < 100)
        patterns['mouse_jitter'] = int(small_rapid.sum())

    # Detect reading patterns from gaze
    if gaze_df is not None and not gaze_df.empty:
        gaze = gaze_df.copy()

        # Analyze horizontal vs vertical movement
        gaze['dx'] = gaze['x'].diff().fillna(0)
        gaze['dy'] = gaze['y'].diff().fillna(0)
        gaze['dt'] = gaze['timestamp'].diff().fillna(1)

        # Filter out very small movements (noise)
        significant_moves = gaze[(gaze['dx'].abs() > 20) | (gaze['dy'].abs() > 20)]

        if len(significant_moves) > 5:
            # F-pattern: mostly left-to-right, then down, repeat
            # Z-pattern: diagonal movements
            horizontal_moves = (significant_moves['dx'].abs() > significant_moves['dy'].abs()).sum()
            vertical_moves = (significant_moves['dy'].abs() > significant_moves['dx'].abs()).sum()

            total_moves = horizontal_moves + vertical_moves
            if total_moves > 0:
                h_ratio = horizontal_moves / total_moves
                v_ratio = vertical_moves / total_moves

                if h_ratio > 0.6:
                    patterns['reading_pattern'] = 'F-pattern (horizontal)'
                elif v_ratio > 0.6:
                    patterns['reading_pattern'] = 'Vertical scanning'
                else:
                    # Check for diagonal patterns
                    diagonal_moves = ((significant_moves['dx'].abs() > 30) &
                                     (significant_moves['dy'].abs() > 30)).sum()
                    if diagonal_moves > total_moves * 0.2:
                        patterns['reading_pattern'] = 'Z-pattern (diagonal)'
                    else:
                        patterns['reading_pattern'] = 'Mixed pattern'
            else:
                patterns['reading_pattern'] = 'Insufficient data'
        else:
            patterns['reading_pattern'] = 'Insufficient data'

        # Detect rapid scanning (quick saccades)
        # Calculate velocity and find high-velocity movements
        gaze['velocity'] = np.sqrt(gaze['dx']**2 + gaze['dy']**2) / (gaze['dt'].replace(0, 1))

        # Use percentile-based threshold for rapid movements
        if len(gaze) > 10:
            velocity_threshold = gaze['velocity'].quantile(0.75)
            rapid_saccades = gaze[gaze['velocity'] > velocity_threshold]
            patterns['rapid_scanning'] = len(rapid_saccades)
        else:
            patterns['rapid_scanning'] = 0

        # Detect backtracking - returns to previous positions
        # Use larger threshold (200px) for webcam tracking noise
        threshold = 200
        backtrack_count = 0
        step = max(1, len(gaze) // 100)  # Sample to avoid O(n^2) for large datasets

        for i in range(20, len(gaze), step):
            current = gaze.iloc[i]
            # Look back 5-20 samples ago (not immediate neighbors)
            recent = gaze.iloc[max(0, i-20):i-5]
            if len(recent) > 0:
                distances = np.sqrt((recent['x'] - current['x'])**2 +
                                   (recent['y'] - current['y'])**2)
                if (distances < threshold).any():
                    backtrack_count += 1

        patterns['backtracking'] = backtrack_count

        # Store gaze trajectory for visualization
        if len(gaze) > 0:
            sample_rate = max(1, len(gaze) // 200)
            patterns['gaze_trajectory'] = gaze[['x', 'y', 'timestamp']].iloc[::sample_rate].to_dict('records')

    # Detect focused attention (long fixations)
    if fixations_df is not None and not fixations_df.empty:
        # Lower threshold to 300ms for focused attention
        long_fixations = fixations_df[fixations_df['duration'] > 300]
        patterns['focused_attention'] = long_fixations.to_dict('records')
    else:
        # If no fixations provided, detect from gaze directly using simple clustering
        if gaze_df is not None and not gaze_df.empty:
            gaze = gaze_df.copy()
            gaze['dx'] = gaze['x'].diff().abs().fillna(0)
            gaze['dy'] = gaze['y'].diff().abs().fillna(0)
            gaze['movement'] = gaze['dx'] + gaze['dy']
            gaze['is_still'] = gaze['movement'] < 50  # Low movement = attention
            gaze['still_group'] = (~gaze['is_still']).cumsum()

            for group_id, group in gaze[gaze['is_still']].groupby('still_group'):
                if len(group) >= 3:
                    duration = group['timestamp'].max() - group['timestamp'].min()
                    if duration > 200:
                        patterns['focused_attention'].append({
                            'x': group['x'].mean(),
                            'y': group['y'].mean(),
                            'duration': duration,
                            'point_count': len(group)
                        })

    # Summary
    patterns['pattern_summary'] = {
        'hesitation_count': len(patterns['hesitations']),
        'reading_pattern': patterns['reading_pattern'] or 'Unknown',
        'rapid_scanning_events': patterns['rapid_scanning'],
        'focused_attention_count': len(patterns['focused_attention']),
        'mouse_jitter_events': patterns['mouse_jitter'],
        'backtracking_events': patterns['backtracking']
    }

    return patterns


def create_behavioral_patterns_visualization(patterns: Dict, gaze_df: pd.DataFrame,
                                            mouse_df: pd.DataFrame) -> go.Figure:
    """Create visualization of detected behavioral patterns."""
    fig = go.Figure()

    # Plot gaze trajectory as background
    if patterns.get('gaze_trajectory'):
        traj = patterns['gaze_trajectory']
        fig.add_trace(go.Scatter(
            x=[p['x'] for p in traj],
            y=[p['y'] for p in traj],
            mode='lines',
            line=dict(color='lightblue', width=1),
            opacity=0.4,
            name='Gaze Path',
            hoverinfo='skip'
        ))
    elif gaze_df is not None and not gaze_df.empty:
        # Fallback: sample gaze data directly
        sample_rate = max(1, len(gaze_df) // 200)
        sampled = gaze_df.iloc[::sample_rate]
        fig.add_trace(go.Scatter(
            x=sampled['x'],
            y=sampled['y'],
            mode='lines',
            line=dict(color='lightblue', width=1),
            opacity=0.4,
            name='Gaze Path',
            hoverinfo='skip'
        ))

    # Plot hesitations as orange circles
    if patterns.get('hesitations'):
        hesitations = patterns['hesitations']
        sizes = [max(10, min(50, h['duration_ms']/30)) for h in hesitations]
        fig.add_trace(go.Scatter(
            x=[h['x'] for h in hesitations],
            y=[h['y'] for h in hesitations],
            mode='markers',
            marker=dict(
                size=sizes,
                color='orange',
                opacity=0.7,
                line=dict(color='darkorange', width=2)
            ),
            name=f'Hesitations ({len(hesitations)})',
            text=[f"Hesitation<br>Duration: {h['duration_ms']:.0f}ms" for h in hesitations],
            hoverinfo='text'
        ))

    # Plot focused attention as green circles
    if patterns.get('focused_attention'):
        focused = patterns['focused_attention']
        sizes = [max(10, min(50, f['duration']/30)) for f in focused]
        fig.add_trace(go.Scatter(
            x=[f['x'] for f in focused],
            y=[f['y'] for f in focused],
            mode='markers',
            marker=dict(
                size=sizes,
                color='green',
                opacity=0.7,
                line=dict(color='darkgreen', width=2)
            ),
            name=f'Focused Attention ({len(focused)})',
            text=[f"Focus Area<br>Duration: {f['duration']:.0f}ms" for f in focused],
            hoverinfo='text'
        ))

    # Add mouse click positions if available
    if mouse_df is not None and not mouse_df.empty and 'event' in mouse_df.columns:
        clicks = mouse_df[mouse_df['event'] == 'click']
        if not clicks.empty:
            fig.add_trace(go.Scatter(
                x=clicks['x'],
                y=clicks['y'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x'),
                name=f'Clicks ({len(clicks)})',
                text=[f"Click at ({x:.0f}, {y:.0f})" for x, y in zip(clicks['x'], clicks['y'])],
                hoverinfo='text'
            ))

    fig.update_layout(
        title="Behavioral Patterns Map",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        yaxis=dict(autorange="reversed"),
        height=550,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


# =============================================================================
# Emotion Timeline Functions - Face Mesh Based Analysis
# =============================================================================

# MediaPipe Face Mesh landmark indices for emotion analysis
FACE_LANDMARKS = {
    # Eyes
    'left_eye_top': 159,
    'left_eye_bottom': 145,
    'left_eye_inner': 133,
    'left_eye_outer': 33,
    'right_eye_top': 386,
    'right_eye_bottom': 374,
    'right_eye_inner': 362,
    'right_eye_outer': 263,

    # Eyebrows
    'left_eyebrow_inner': 107,
    'left_eyebrow_outer': 70,
    'left_eyebrow_top': 105,
    'right_eyebrow_inner': 336,
    'right_eyebrow_outer': 300,
    'right_eyebrow_top': 334,

    # Mouth
    'mouth_left': 61,
    'mouth_right': 291,
    'mouth_top': 13,
    'mouth_bottom': 14,
    'upper_lip_top': 0,
    'lower_lip_bottom': 17,

    # Nose
    'nose_tip': 4,

    # Face contour
    'chin': 152,
    'forehead': 10,
}


def parse_landmarks(landmarks_str: str) -> List[Dict]:
    """Parse landmarks from string format to list of dicts."""
    try:
        if isinstance(landmarks_str, str):
            import ast
            return ast.literal_eval(landmarks_str)
        return landmarks_str
    except:
        return []


def get_landmark_point(landmarks: List[Dict], idx: int) -> Optional[Dict]:
    """Get a specific landmark point by index."""
    if landmarks and idx < len(landmarks):
        return landmarks[idx]
    return None


def calculate_distance(p1: Dict, p2: Dict) -> float:
    """Calculate 3D distance between two landmarks."""
    if not p1 or not p2:
        return 0
    return np.sqrt(
        (p1['x'] - p2['x'])**2 +
        (p1['y'] - p2['y'])**2 +
        (p1.get('z', 0) - p2.get('z', 0))**2
    )


def analyze_facial_features(landmarks: List[Dict]) -> Dict:
    """
    Analyze facial features from landmarks to detect emotions.
    Uses geometric ratios and distances similar to Facial Action Coding System (FACS).
    """
    features = {
        'eye_aspect_ratio_left': 0,
        'eye_aspect_ratio_right': 0,
        'mouth_aspect_ratio': 0,
        'eyebrow_raise_left': 0,
        'eyebrow_raise_right': 0,
        'mouth_width': 0,
        'brow_furrow': 0,
        'smile_ratio': 0,
        'valid': False
    }

    if not landmarks or len(landmarks) < 400:
        return features

    features['valid'] = True

    # Eye Aspect Ratio (EAR) - detects blinks and eye openness
    # EAR = (vertical distance) / (horizontal distance)
    left_eye_v = calculate_distance(
        get_landmark_point(landmarks, FACE_LANDMARKS['left_eye_top']),
        get_landmark_point(landmarks, FACE_LANDMARKS['left_eye_bottom'])
    )
    left_eye_h = calculate_distance(
        get_landmark_point(landmarks, FACE_LANDMARKS['left_eye_inner']),
        get_landmark_point(landmarks, FACE_LANDMARKS['left_eye_outer'])
    )
    features['eye_aspect_ratio_left'] = left_eye_v / left_eye_h if left_eye_h > 0 else 0

    right_eye_v = calculate_distance(
        get_landmark_point(landmarks, FACE_LANDMARKS['right_eye_top']),
        get_landmark_point(landmarks, FACE_LANDMARKS['right_eye_bottom'])
    )
    right_eye_h = calculate_distance(
        get_landmark_point(landmarks, FACE_LANDMARKS['right_eye_inner']),
        get_landmark_point(landmarks, FACE_LANDMARKS['right_eye_outer'])
    )
    features['eye_aspect_ratio_right'] = right_eye_v / right_eye_h if right_eye_h > 0 else 0

    # Mouth Aspect Ratio (MAR) - detects mouth openness
    mouth_v = calculate_distance(
        get_landmark_point(landmarks, FACE_LANDMARKS['mouth_top']),
        get_landmark_point(landmarks, FACE_LANDMARKS['mouth_bottom'])
    )
    mouth_h = calculate_distance(
        get_landmark_point(landmarks, FACE_LANDMARKS['mouth_left']),
        get_landmark_point(landmarks, FACE_LANDMARKS['mouth_right'])
    )
    features['mouth_aspect_ratio'] = mouth_v / mouth_h if mouth_h > 0 else 0
    features['mouth_width'] = mouth_h

    # Eyebrow raise - distance from eyebrow to eye
    nose_tip = get_landmark_point(landmarks, FACE_LANDMARKS['nose_tip'])

    left_brow = get_landmark_point(landmarks, FACE_LANDMARKS['left_eyebrow_top'])
    left_eye = get_landmark_point(landmarks, FACE_LANDMARKS['left_eye_top'])
    if left_brow and left_eye:
        features['eyebrow_raise_left'] = left_eye['y'] - left_brow['y']

    right_brow = get_landmark_point(landmarks, FACE_LANDMARKS['right_eyebrow_top'])
    right_eye = get_landmark_point(landmarks, FACE_LANDMARKS['right_eye_top'])
    if right_brow and right_eye:
        features['eyebrow_raise_right'] = right_eye['y'] - right_brow['y']

    # Brow furrow - distance between inner eyebrows
    left_inner = get_landmark_point(landmarks, FACE_LANDMARKS['left_eyebrow_inner'])
    right_inner = get_landmark_point(landmarks, FACE_LANDMARKS['right_eyebrow_inner'])
    if left_inner and right_inner:
        features['brow_furrow'] = calculate_distance(left_inner, right_inner)

    # Smile ratio - mouth corners relative to center
    mouth_left = get_landmark_point(landmarks, FACE_LANDMARKS['mouth_left'])
    mouth_right = get_landmark_point(landmarks, FACE_LANDMARKS['mouth_right'])
    mouth_center_y = get_landmark_point(landmarks, FACE_LANDMARKS['mouth_top'])
    if mouth_left and mouth_right and mouth_center_y:
        corner_avg_y = (mouth_left['y'] + mouth_right['y']) / 2
        features['smile_ratio'] = mouth_center_y['y'] - corner_avg_y

    return features


def classify_emotion(features: Dict, baseline: Dict = None) -> Dict:
    """
    Classify emotion based on facial features.
    Returns probabilities for each emotion.
    """
    emotions = {
        'neutral': 0.5,
        'happy': 0.0,
        'surprised': 0.0,
        'confused': 0.0,
        'focused': 0.0,
        'frustrated': 0.0,
        'tired': 0.0
    }

    if not features.get('valid', False):
        return emotions

    # Baseline normalization (if provided)
    ear_avg = (features['eye_aspect_ratio_left'] + features['eye_aspect_ratio_right']) / 2
    mar = features['mouth_aspect_ratio']
    brow_raise = (features['eyebrow_raise_left'] + features['eyebrow_raise_right']) / 2
    smile = features['smile_ratio']
    brow_furrow = features['brow_furrow']

    # Happy detection (smile + raised mouth corners)
    if smile > 0.01 and mar < 0.3:
        emotions['happy'] = min(1.0, smile * 20 + 0.3)
        emotions['neutral'] -= 0.3

    # Surprised (raised eyebrows + open mouth + wide eyes)
    if brow_raise > 0.03 and mar > 0.2 and ear_avg > 0.25:
        emotions['surprised'] = min(1.0, brow_raise * 10 + mar * 2)
        emotions['neutral'] -= 0.3

    # Confused (furrowed brow + slight head tilt indicator)
    if brow_furrow < 0.05 and brow_raise < 0.02:
        emotions['confused'] = 0.4
        emotions['neutral'] -= 0.2

    # Focused (slightly narrowed eyes + neutral mouth)
    if ear_avg < 0.2 and mar < 0.15:
        emotions['focused'] = 0.5
        emotions['neutral'] -= 0.2

    # Frustrated (furrowed brow + tense mouth)
    if brow_furrow < 0.04 and smile < -0.01:
        emotions['frustrated'] = 0.4
        emotions['neutral'] -= 0.2

    # Tired (droopy eyes)
    if ear_avg < 0.15:
        emotions['tired'] = min(1.0, (0.2 - ear_avg) * 5)
        emotions['neutral'] -= 0.2

    # Normalize
    emotions['neutral'] = max(0, emotions['neutral'])
    total = sum(emotions.values())
    if total > 0:
        emotions = {k: v/total for k, v in emotions.items()}

    return emotions


def analyze_emotion_timeline(face_mesh_df: pd.DataFrame, timeline_data: Dict,
                            mouse_df: pd.DataFrame) -> Dict:
    """
    Comprehensive emotion analysis using facial landmarks and behavioral signals.
    """
    analysis = {
        'emotion_timeline': [],
        'feature_timeline': [],
        'dominant_emotions': {},
        'emotion_events': [],
        'blink_rate': 0,
        'blink_events': [],
        'engagement_timeline': [],
        'frustration_indicators': [],
        'valence_timeline': [],  # Positive/Negative
        'arousal_timeline': [],  # High/Low energy
        'emotion_summary': {}
    }

    # Analyze face mesh for emotions
    if face_mesh_df is not None and not face_mesh_df.empty:
        fm = face_mesh_df.copy()

        # Parse landmarks and analyze each frame
        emotion_counts = {}
        prev_ear = None
        blink_count = 0

        for idx, row in fm.iterrows():
            timestamp = row['timestamp']

            # Parse landmarks
            landmarks = parse_landmarks(row.get('landmarks', '[]'))

            if landmarks:
                # Extract facial features
                features = analyze_facial_features(landmarks)
                analysis['feature_timeline'].append({
                    'time': timestamp,
                    **features
                })

                # Classify emotion
                emotions = classify_emotion(features)
                dominant = max(emotions, key=emotions.get)

                analysis['emotion_timeline'].append({
                    'time': timestamp,
                    'dominant': dominant,
                    'confidence': emotions[dominant],
                    **emotions
                })

                # Count dominant emotions
                emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1

                # Calculate valence (positive-negative) and arousal (energy level)
                valence = emotions['happy'] - emotions['frustrated'] - emotions['tired']
                arousal = emotions['surprised'] + emotions['happy'] * 0.5 - emotions['tired']

                analysis['valence_timeline'].append({
                    'time': timestamp,
                    'valence': valence
                })
                analysis['arousal_timeline'].append({
                    'time': timestamp,
                    'arousal': arousal
                })

                # Blink detection using EAR
                ear = (features['eye_aspect_ratio_left'] + features['eye_aspect_ratio_right']) / 2
                if prev_ear is not None and prev_ear > 0.2 and ear < 0.15:
                    blink_count += 1
                    analysis['blink_events'].append({'time': timestamp})
                prev_ear = ear

        # Calculate blink rate (per minute)
        if fm['timestamp'].max() > fm['timestamp'].min():
            duration_minutes = (fm['timestamp'].max() - fm['timestamp'].min()) / 60000
            analysis['blink_rate'] = blink_count / duration_minutes if duration_minutes > 0 else 0

        # Dominant emotions summary
        total_frames = sum(emotion_counts.values())
        analysis['dominant_emotions'] = {
            k: v / total_frames * 100 for k, v in emotion_counts.items()
        } if total_frames > 0 else {}

        # Engagement from face visibility and attention
        window_size = max(1, len(fm) // 30)
        for i in range(0, len(fm) - window_size, window_size):
            window_emotions = analysis['emotion_timeline'][i:i+window_size]
            if window_emotions:
                # Engagement = focused + happy - tired - confused
                engagement = np.mean([
                    e.get('focused', 0) + e.get('happy', 0) * 0.5 -
                    e.get('tired', 0) - e.get('confused', 0) * 0.3
                    for e in window_emotions
                ])
                analysis['engagement_timeline'].append({
                    'time': window_emotions[len(window_emotions)//2]['time'],
                    'score': (engagement + 0.5) * 100  # Scale to 0-100
                })

    # Analyze mouse behavior for frustration
    if mouse_df is not None and not mouse_df.empty:
        mouse = mouse_df.copy()
        mouse['dx'] = mouse['x'].diff()
        mouse['dy'] = mouse['y'].diff()
        mouse['dt'] = mouse['timestamp'].diff().fillna(1)
        mouse['velocity'] = np.sqrt(mouse['dx']**2 + mouse['dy']**2) / (mouse['dt'] + 1)
        mouse['direction_change'] = (mouse['dx'] * mouse['dx'].shift(1) < 0) | \
                                   (mouse['dy'] * mouse['dy'].shift(1) < 0)

        # Sliding window for frustration detection
        window = 15
        for i in range(window, len(mouse) - window):
            segment = mouse.iloc[i-window:i+window]
            rapid_changes = segment['direction_change'].sum()
            high_velocity = (segment['velocity'] > segment['velocity'].quantile(0.75)).sum()

            if rapid_changes > 8 and high_velocity > 8:
                analysis['frustration_indicators'].append({
                    'time': segment['timestamp'].mean(),
                    'x': segment['x'].mean(),
                    'y': segment['y'].mean(),
                    'intensity': (rapid_changes + high_velocity) / 2,
                    'source': 'mouse_behavior'
                })

    # Add timeline events
    if timeline_data and 'clicks' in timeline_data:
        for click in timeline_data['clicks']:
            analysis['emotion_events'].append({
                'time': click['time'],
                'type': 'click',
                'x': click.get('x'),
                'y': click.get('y')
            })

    # Summary statistics
    if analysis['emotion_timeline']:
        analysis['emotion_summary'] = {
            'dominant_emotions': analysis['dominant_emotions'],
            'blink_rate_per_min': analysis['blink_rate'],
            'total_blinks': len(analysis['blink_events']),
            'frustration_events': len(analysis['frustration_indicators']),
            'avg_valence': np.mean([v['valence'] for v in analysis['valence_timeline']]) if analysis['valence_timeline'] else 0,
            'avg_arousal': np.mean([a['arousal'] for a in analysis['arousal_timeline']]) if analysis['arousal_timeline'] else 0,
            'avg_engagement': np.mean([e['score'] for e in analysis['engagement_timeline']]) if analysis['engagement_timeline'] else 0
        }
    else:
        analysis['emotion_summary'] = {
            'dominant_emotions': {},
            'blink_rate_per_min': 0,
            'total_blinks': 0,
            'frustration_events': len(analysis['frustration_indicators']),
            'avg_valence': 0,
            'avg_arousal': 0,
            'avg_engagement': 0
        }

    return analysis


def create_emotion_timeline_plot(emotion_analysis: Dict) -> go.Figure:
    """Create comprehensive emotion timeline visualization."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Emotion Probabilities Over Time",
            "Valence (Positive/Negative) & Arousal (Energy)",
            "Engagement Score",
            "Blinks & Frustration Events"
        ),
        row_heights=[0.35, 0.25, 0.2, 0.2]
    )

    colors = {
        'neutral': 'gray',
        'happy': 'green',
        'surprised': 'yellow',
        'confused': 'purple',
        'focused': 'blue',
        'frustrated': 'red',
        'tired': 'brown'
    }

    # Row 1: Emotion probabilities
    if emotion_analysis.get('emotion_timeline'):
        times = [e['time']/1000 for e in emotion_analysis['emotion_timeline']]

        for emotion in ['happy', 'focused', 'neutral', 'confused', 'frustrated', 'tired', 'surprised']:
            values = [e.get(emotion, 0) for e in emotion_analysis['emotion_timeline']]
            fig.add_trace(go.Scatter(
                x=times, y=values,
                mode='lines',
                name=emotion.capitalize(),
                line=dict(color=colors.get(emotion, 'gray'), width=1.5),
                stackgroup='emotions'
            ), row=1, col=1)

    # Row 2: Valence and Arousal
    if emotion_analysis.get('valence_timeline'):
        times = [v['time']/1000 for v in emotion_analysis['valence_timeline']]
        valences = [v['valence'] for v in emotion_analysis['valence_timeline']]
        fig.add_trace(go.Scatter(
            x=times, y=valences,
            mode='lines',
            name='Valence',
            line=dict(color='green', width=2)
        ), row=2, col=1)

    if emotion_analysis.get('arousal_timeline'):
        times = [a['time']/1000 for a in emotion_analysis['arousal_timeline']]
        arousals = [a['arousal'] for a in emotion_analysis['arousal_timeline']]
        fig.add_trace(go.Scatter(
            x=times, y=arousals,
            mode='lines',
            name='Arousal',
            line=dict(color='orange', width=2)
        ), row=2, col=1)

    # Row 3: Engagement
    if emotion_analysis.get('engagement_timeline'):
        times = [e['time']/1000 for e in emotion_analysis['engagement_timeline']]
        scores = [e['score'] for e in emotion_analysis['engagement_timeline']]
        fig.add_trace(go.Scatter(
            x=times, y=scores,
            mode='lines',
            name='Engagement',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,0,255,0.1)'
        ), row=3, col=1)

    # Row 4: Blinks and Frustration
    if emotion_analysis.get('blink_events'):
        blink_times = [b['time']/1000 for b in emotion_analysis['blink_events']]
        fig.add_trace(go.Scatter(
            x=blink_times,
            y=[1] * len(blink_times),
            mode='markers',
            name='Blinks',
            marker=dict(color='cyan', size=6, symbol='line-ns')
        ), row=4, col=1)

    if emotion_analysis.get('frustration_indicators'):
        frust_times = [f['time']/1000 for f in emotion_analysis['frustration_indicators']]
        frust_intensities = [f['intensity']/20 for f in emotion_analysis['frustration_indicators']]
        fig.add_trace(go.Scatter(
            x=frust_times,
            y=frust_intensities,
            mode='markers',
            name='Frustration',
            marker=dict(color='red', size=10, symbol='x')
        ), row=4, col=1)

    fig.update_layout(
        height=700,
        title="Emotion Analysis Timeline",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="Engagement %", row=3, col=1)
    fig.update_yaxes(title_text="Events", row=4, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=4, col=1)

    return fig


def create_emotion_distribution_chart(emotion_analysis: Dict) -> go.Figure:
    """Create pie chart of dominant emotions distribution."""
    if not emotion_analysis.get('dominant_emotions'):
        return None

    emotions = emotion_analysis['dominant_emotions']

    colors = {
        'neutral': 'gray',
        'happy': 'green',
        'surprised': 'yellow',
        'confused': 'purple',
        'focused': 'blue',
        'frustrated': 'red',
        'tired': 'brown'
    }

    fig = go.Figure(data=[go.Pie(
        labels=list(emotions.keys()),
        values=list(emotions.values()),
        marker_colors=[colors.get(e, 'gray') for e in emotions.keys()],
        textinfo='label+percent',
        hole=0.4
    )])

    fig.update_layout(
        title="Emotion Distribution",
        height=350
    )

    return fig


def create_valence_arousal_scatter(emotion_analysis: Dict) -> go.Figure:
    """Create valence-arousal scatter plot (circumplex model of affect)."""
    if not emotion_analysis.get('valence_timeline') or not emotion_analysis.get('arousal_timeline'):
        return None

    valences = [v['valence'] for v in emotion_analysis['valence_timeline']]
    arousals = [a['arousal'] for a in emotion_analysis['arousal_timeline']]
    times = [v['time']/1000 for v in emotion_analysis['valence_timeline']]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=valences,
        y=arousals,
        mode='markers',
        marker=dict(
            color=times,
            colorscale='Viridis',
            size=6,
            showscale=True,
            colorbar=dict(title="Time (s)")
        ),
        text=[f"Time: {t:.1f}s" for t in times],
        hoverinfo='text'
    ))

    # Add quadrant labels
    fig.add_annotation(x=0.5, y=0.5, text="Excited/Happy", showarrow=False, font=dict(size=10, color='green'))
    fig.add_annotation(x=-0.5, y=0.5, text="Angry/Stressed", showarrow=False, font=dict(size=10, color='red'))
    fig.add_annotation(x=0.5, y=-0.5, text="Calm/Relaxed", showarrow=False, font=dict(size=10, color='blue'))
    fig.add_annotation(x=-0.5, y=-0.5, text="Sad/Bored", showarrow=False, font=dict(size=10, color='gray'))

    # Add axes
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Valence-Arousal Space (Circumplex Model)",
        xaxis_title="Valence (Negative ← → Positive)",
        yaxis_title="Arousal (Low ← → High)",
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        height=400
    )

    return fig


def create_facial_features_timeline(emotion_analysis: Dict) -> go.Figure:
    """Create timeline of raw facial features."""
    if not emotion_analysis.get('feature_timeline'):
        return None

    features = emotion_analysis['feature_timeline']
    times = [f['time']/1000 for f in features]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       subplot_titles=("Eye Aspect Ratio", "Mouth Aspect Ratio", "Eyebrow Position"))

    # EAR
    ear_left = [f.get('eye_aspect_ratio_left', 0) for f in features]
    ear_right = [f.get('eye_aspect_ratio_right', 0) for f in features]
    fig.add_trace(go.Scatter(x=times, y=ear_left, name='Left Eye', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=ear_right, name='Right Eye', line=dict(color='red')), row=1, col=1)

    # MAR
    mar = [f.get('mouth_aspect_ratio', 0) for f in features]
    fig.add_trace(go.Scatter(x=times, y=mar, name='Mouth', line=dict(color='green')), row=2, col=1)

    # Eyebrow
    brow_left = [f.get('eyebrow_raise_left', 0) for f in features]
    brow_right = [f.get('eyebrow_raise_right', 0) for f in features]
    fig.add_trace(go.Scatter(x=times, y=brow_left, name='Left Brow', line=dict(color='purple')), row=3, col=1)
    fig.add_trace(go.Scatter(x=times, y=brow_right, name='Right Brow', line=dict(color='orange')), row=3, col=1)

    fig.update_layout(height=500, title="Facial Feature Metrics Over Time")
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)

    return fig


# =============================================================================
# Comparative Analysis Functions
# =============================================================================

def compare_sessions(sessions_data: Dict[str, Dict]) -> pd.DataFrame:
    """Compare metrics across multiple sessions."""
    comparison = []

    for session_id, data in sessions_data.items():
        metrics = {
            'session_id': session_id,
            'gaze_points': len(data['gaze']) if data['gaze'] is not None else 0,
            'mouse_events': len(data['mouse']) if data['mouse'] is not None else 0,
            'duration_ms': 0,
            'clicks': 0,
            'avg_gaze_x': None,
            'avg_gaze_y': None,
            'mouse_distance': 0
        }

        if data['metadata']:
            time_range = data['metadata'].get('time_range', {})
            metrics['duration_ms'] = time_range.get('end', 0) - time_range.get('start', 0)

        if data['gaze'] is not None and not data['gaze'].empty:
            metrics['avg_gaze_x'] = data['gaze']['x'].mean()
            metrics['avg_gaze_y'] = data['gaze']['y'].mean()

        if data['mouse'] is not None and not data['mouse'].empty:
            if 'event' in data['mouse'].columns:
                metrics['clicks'] = len(data['mouse'][data['mouse']['event'] == 'click'])

            dx = data['mouse']['x'].diff()
            dy = data['mouse']['y'].diff()
            metrics['mouse_distance'] = np.sqrt(dx**2 + dy**2).sum()

        comparison.append(metrics)

    return pd.DataFrame(comparison)


def create_comparison_charts(comparison_df: pd.DataFrame) -> List[go.Figure]:
    """Create charts comparing sessions."""
    figures = []

    # Duration comparison
    fig1 = go.Figure(data=[
        go.Bar(x=comparison_df['session_id'], y=comparison_df['duration_ms']/1000,
               name='Duration (s)')
    ])
    fig1.update_layout(title="Session Duration Comparison", height=300)
    figures.append(fig1)

    # Activity comparison
    fig2 = go.Figure(data=[
        go.Bar(x=comparison_df['session_id'], y=comparison_df['gaze_points'],
               name='Gaze Points'),
        go.Bar(x=comparison_df['session_id'], y=comparison_df['mouse_events'],
               name='Mouse Events'),
        go.Bar(x=comparison_df['session_id'], y=comparison_df['clicks']*100,
               name='Clicks (x100)')
    ])
    fig2.update_layout(title="Activity Comparison", barmode='group', height=300)
    figures.append(fig2)

    return figures


def create_aggregate_heatmap(sessions_data: Dict[str, Dict], width: int = 1920,
                            height: int = 1080) -> go.Figure:
    """Create aggregate heatmap from multiple sessions."""
    all_gaze = []

    for session_id, data in sessions_data.items():
        if data['gaze'] is not None and not data['gaze'].empty:
            all_gaze.append(data['gaze'][['x', 'y']])

    if not all_gaze:
        return None

    combined = pd.concat(all_gaze, ignore_index=True)

    bins_x, bins_y = 50, 30
    heatmap, _, _ = np.histogram2d(
        combined['x'].clip(0, width),
        combined['y'].clip(0, height),
        bins=[bins_x, bins_y],
        range=[[0, width], [0, height]]
    )
    heatmap = ndimage.gaussian_filter(heatmap, sigma=2)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap.T,
        x=np.linspace(0, width, bins_x),
        y=np.linspace(0, height, bins_y),
        colorscale='Hot',
        showscale=True
    ))

    fig.update_layout(
        title=f"Aggregate Gaze Heatmap ({len(sessions_data)} sessions)",
        yaxis=dict(autorange="reversed"),
        height=500
    )

    return fig


# =============================================================================
# Export & Reporting Functions
# =============================================================================

def generate_report_data(data: Dict, fixations: pd.DataFrame,
                        attention_metrics: Dict, patterns: Dict) -> Dict:
    """Generate comprehensive report data."""
    report = {
        'session_info': {
            'session_id': data.get('metadata', {}).get('session_id', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'duration_ms': 0
        },
        'summary_statistics': {},
        'attention_metrics': attention_metrics,
        'behavioral_patterns': patterns.get('pattern_summary', {}),
        'fixation_summary': {},
        'recommendations': []
    }

    # Duration
    if data['metadata']:
        time_range = data['metadata'].get('time_range', {})
        report['session_info']['duration_ms'] = time_range.get('end', 0) - time_range.get('start', 0)

    # Summary statistics
    if data['gaze'] is not None:
        report['summary_statistics']['total_gaze_points'] = len(data['gaze'])
        report['summary_statistics']['gaze_coverage'] = {
            'x_range': [float(data['gaze']['x'].min()), float(data['gaze']['x'].max())],
            'y_range': [float(data['gaze']['y'].min()), float(data['gaze']['y'].max())]
        }

    if data['mouse'] is not None:
        report['summary_statistics']['total_mouse_events'] = len(data['mouse'])
        if 'event' in data['mouse'].columns:
            report['summary_statistics']['total_clicks'] = len(data['mouse'][data['mouse']['event'] == 'click'])

    # Fixation summary
    if fixations is not None and not fixations.empty:
        report['fixation_summary'] = {
            'total_fixations': len(fixations),
            'mean_duration_ms': float(fixations['duration'].mean()),
            'max_duration_ms': float(fixations['duration'].max()),
            'min_duration_ms': float(fixations['duration'].min())
        }

    # Generate recommendations
    if patterns.get('pattern_summary', {}).get('hesitation_count', 0) > 5:
        report['recommendations'].append("High hesitation count detected - consider simplifying the interface")

    if patterns.get('pattern_summary', {}).get('backtracking_events', 0) > 10:
        report['recommendations'].append("Frequent backtracking observed - users may be having difficulty finding information")

    if attention_metrics.get('cognitive_load_score', 0) > 70:
        report['recommendations'].append("High cognitive load detected - consider reducing complexity")

    return report


def export_to_csv(data: Dict, prefix: str = "export") -> Dict[str, str]:
    """Export all data to CSV strings."""
    exports = {}

    for key, df in data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            exports[f"{prefix}_{key}.csv"] = df.to_csv(index=False)

    return exports


# =============================================================================
# Original Visualization Functions
# =============================================================================

def create_gaze_heatmap(gaze_df: pd.DataFrame, width: int = 1920, height: int = 1080,
                        time_start: float = None, time_end: float = None) -> go.Figure:
    """Create a heatmap from gaze data."""
    if gaze_df is None or gaze_df.empty:
        return None

    df = gaze_df.copy()

    if time_start is not None:
        df = df[df['timestamp'] >= time_start]
    if time_end is not None:
        df = df[df['timestamp'] <= time_end]

    if df.empty:
        return None

    bins_x, bins_y = 50, 30
    heatmap, xedges, yedges = np.histogram2d(
        df['x'].clip(0, width), df['y'].clip(0, height),
        bins=[bins_x, bins_y], range=[[0, width], [0, height]]
    )
    heatmap = ndimage.gaussian_filter(heatmap, sigma=2)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap.T,
        x=np.linspace(0, width, bins_x),
        y=np.linspace(0, height, bins_y),
        colorscale='Hot',
        showscale=True,
        colorbar=dict(title="Fixation Density")
    ))

    fig.update_layout(
        title="Gaze Heatmap",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        yaxis=dict(scaleanchor="x", scaleratio=height/width, autorange="reversed"),
        height=500
    )

    return fig


def create_mouse_trajectory(mouse_df: pd.DataFrame,
                           time_start: float = None, time_end: float = None) -> go.Figure:
    """Create mouse movement trajectory visualization."""
    if mouse_df is None or mouse_df.empty:
        return None

    df = mouse_df.copy()

    if time_start is not None:
        df = df[df['timestamp'] >= time_start]
    if time_end is not None:
        df = df[df['timestamp'] <= time_end]

    if df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['y'],
        mode='lines',
        line=dict(color='blue', width=1),
        name='Mouse Path',
        opacity=0.6
    ))

    if 'event' in df.columns:
        clicks = df[df['event'] == 'click']
        if not clicks.empty:
            fig.add_trace(go.Scatter(
                x=clicks['x'], y=clicks['y'],
                mode='markers',
                marker=dict(color='red', size=12, symbol='circle'),
                name='Clicks'
            ))

    fig.update_layout(
        title="Mouse Movement Trajectory",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        yaxis=dict(autorange="reversed"),
        height=500,
        showlegend=True
    )

    return fig


def create_click_heatmap(mouse_df: pd.DataFrame, width: int = 1920, height: int = 1080) -> go.Figure:
    """Create heatmap of click positions."""
    if mouse_df is None or mouse_df.empty:
        return None

    if 'event' in mouse_df.columns:
        clicks = mouse_df[mouse_df['event'] == 'click']
    else:
        return None

    if clicks.empty:
        return None

    fig = go.Figure(data=go.Scatter(
        x=clicks['x'], y=clicks['y'],
        mode='markers',
        marker=dict(size=20, color='red', opacity=0.6, symbol='circle'),
        text=[f"Click at ({x:.0f}, {y:.0f})" for x, y in zip(clicks['x'], clicks['y'])],
        hoverinfo='text'
    ))

    fig.update_layout(
        title="Click Positions",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        xaxis=dict(range=[0, width]),
        yaxis=dict(range=[height, 0]),
        height=500
    )

    return fig


def create_timeline_chart(data: Dict) -> go.Figure:
    """Create a synchronized timeline showing all data streams."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Gaze Velocity", "Mouse Activity", "Face Detection", "Events"),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    if data["gaze"] is not None and not data["gaze"].empty:
        gaze = data["gaze"].copy()
        gaze['time_sec'] = gaze['timestamp'] / 1000
        gaze['dx'] = gaze['x'].diff()
        gaze['dy'] = gaze['y'].diff()
        gaze['velocity'] = np.sqrt(gaze['dx']**2 + gaze['dy']**2)

        fig.add_trace(
            go.Scatter(x=gaze['time_sec'], y=gaze['velocity'],
                      mode='lines', name='Gaze Velocity', line=dict(color='blue')),
            row=1, col=1
        )

    if data["mouse"] is not None and not data["mouse"].empty:
        mouse = data["mouse"].copy()
        mouse['time_sec'] = mouse['timestamp'] / 1000
        mouse['activity'] = 1
        if 'event' in mouse.columns:
            mouse.loc[mouse['event'] == 'click', 'activity'] = 2

        fig.add_trace(
            go.Scatter(x=mouse['time_sec'], y=mouse['activity'],
                      mode='markers', name='Mouse Activity',
                      marker=dict(color='green', size=4)),
            row=2, col=1
        )

    if data["face_mesh"] is not None and not data["face_mesh"].empty:
        fm = data["face_mesh"].copy()
        fm['time_sec'] = fm['timestamp'] / 1000
        fig.add_trace(
            go.Scatter(x=fm['time_sec'], y=[1]*len(fm),
                      mode='markers', name='Face Detected',
                      marker=dict(color='purple', size=3)),
            row=3, col=1
        )

    if data["timeline"] is not None and "clicks" in data["timeline"]:
        clicks = data["timeline"]["clicks"]
        if clicks:
            click_times = [c['time']/1000 for c in clicks]
            fig.add_trace(
                go.Scatter(x=click_times, y=[1]*len(click_times),
                          mode='markers', name='User Clicks',
                          marker=dict(color='red', size=10, symbol='star')),
                row=4, col=1
            )

    fig.update_layout(height=600, title="Session Timeline", showlegend=True)
    fig.update_xaxes(title_text="Time (seconds)", row=4, col=1)

    return fig


def create_gaze_scanpath(gaze_df: pd.DataFrame, max_points: int = 200) -> go.Figure:
    """Create scanpath visualization showing gaze sequence."""
    if gaze_df is None or gaze_df.empty:
        return None

    df = gaze_df.head(max_points).copy()
    colors = np.linspace(0, 1, len(df))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['y'],
        mode='lines',
        line=dict(color='gray', width=1),
        opacity=0.5,
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df['x'], y=df['y'],
        mode='markers',
        marker=dict(
            color=colors, colorscale='Viridis', size=8,
            showscale=True, colorbar=dict(title="Time Progress")
        ),
        name='Gaze Points'
    ))

    fig.update_layout(
        title=f"Gaze Scanpath (First {len(df)} points)",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        yaxis=dict(autorange="reversed"),
        height=500
    )

    return fig


def calculate_fixations(gaze_df: pd.DataFrame, velocity_threshold: float = 50,
                       min_duration: float = 100, dispersion_threshold: float = 150) -> pd.DataFrame:
    """
    Detect fixations using I-DT (Dispersion-Threshold) algorithm - better for webcam eye tracking.

    This uses a sliding window approach that's more robust to noise than velocity-based methods.

    Args:
        gaze_df: DataFrame with gaze data (x, y, timestamp)
        velocity_threshold: Maximum velocity (px/ms) - used as secondary filter
        min_duration: Minimum fixation duration in ms
        dispersion_threshold: Maximum dispersion (px) within a fixation window
    """
    if gaze_df is None or gaze_df.empty:
        return pd.DataFrame()

    df = gaze_df.copy().reset_index(drop=True)

    if len(df) < 3:
        return pd.DataFrame()

    fixations = []
    fixation_id = 0

    # I-DT Algorithm: Sliding window based on dispersion
    i = 0
    while i < len(df):
        # Start a new potential fixation window
        window_start = i
        window_end = i + 1

        # Expand window while dispersion is below threshold
        while window_end < len(df):
            window = df.iloc[window_start:window_end + 1]

            # Calculate dispersion (max distance in x or y)
            disp_x = window['x'].max() - window['x'].min()
            disp_y = window['y'].max() - window['y'].min()
            dispersion = max(disp_x, disp_y)

            if dispersion <= dispersion_threshold:
                window_end += 1
            else:
                break

        # Check if window meets minimum duration
        window = df.iloc[window_start:window_end]
        duration = window['timestamp'].max() - window['timestamp'].min()

        if duration >= min_duration and len(window) >= 2:
            disp_x = window['x'].max() - window['x'].min()
            disp_y = window['y'].max() - window['y'].min()
            dispersion = max(disp_x, disp_y)

            fixation_id += 1
            fixations.append({
                'fixation_id': fixation_id,
                'x': window['x'].mean(),
                'y': window['y'].mean(),
                'duration': duration,
                'start_time': window['timestamp'].min(),
                'end_time': window['timestamp'].max(),
                'point_count': len(window),
                'dispersion': dispersion,
                'std_x': window['x'].std() if len(window) > 1 else 0,
                'std_y': window['y'].std() if len(window) > 1 else 0
            })

            # Move past this fixation
            i = window_end
        else:
            # No fixation found, move to next point
            i += 1

    return pd.DataFrame(fixations)


def calculate_saccades(gaze_df: pd.DataFrame, fixations_df: pd.DataFrame,
                      velocity_threshold: float = 50) -> pd.DataFrame:
    """
    Detect saccades (rapid eye movements between fixations).

    Args:
        gaze_df: DataFrame with gaze data
        fixations_df: DataFrame with detected fixations
        velocity_threshold: Minimum velocity to be considered a saccade
    """
    if gaze_df is None or gaze_df.empty:
        return pd.DataFrame()

    if fixations_df is None or fixations_df.empty or len(fixations_df) < 2:
        return pd.DataFrame()

    saccades = []

    # Calculate saccades between consecutive fixations
    for i in range(len(fixations_df) - 1):
        fix1 = fixations_df.iloc[i]
        fix2 = fixations_df.iloc[i + 1]

        # Saccade properties
        amplitude = np.sqrt((fix2['x'] - fix1['x'])**2 + (fix2['y'] - fix1['y'])**2)
        duration = fix2['start_time'] - fix1['end_time']

        if duration > 0:
            velocity = amplitude / duration

            # Direction in degrees (0 = right, 90 = down)
            direction = np.degrees(np.arctan2(fix2['y'] - fix1['y'], fix2['x'] - fix1['x']))

            saccades.append({
                'saccade_id': i + 1,
                'start_x': fix1['x'],
                'start_y': fix1['y'],
                'end_x': fix2['x'],
                'end_y': fix2['y'],
                'amplitude': amplitude,
                'duration': duration,
                'velocity': velocity,
                'direction': direction,
                'start_time': fix1['end_time'],
                'end_time': fix2['start_time'],
                'from_fixation': fix1['fixation_id'] if 'fixation_id' in fix1 else i + 1,
                'to_fixation': fix2['fixation_id'] if 'fixation_id' in fix2 else i + 2
            })

    return pd.DataFrame(saccades)


def create_saccade_visualization(fixations_df: pd.DataFrame, saccades_df: pd.DataFrame,
                                width: int = 1920, height: int = 1080) -> go.Figure:
    """Create visualization of fixations and saccades."""
    fig = go.Figure()

    # Draw saccades as arrows/lines
    if saccades_df is not None and not saccades_df.empty:
        for _, saccade in saccades_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[saccade['start_x'], saccade['end_x']],
                y=[saccade['start_y'], saccade['end_y']],
                mode='lines',
                line=dict(color='red', width=1),
                opacity=0.5,
                showlegend=False,
                hoverinfo='text',
                text=f"Saccade {saccade['saccade_id']}<br>Amplitude: {saccade['amplitude']:.0f}px<br>Duration: {saccade['duration']:.0f}ms"
            ))

    # Draw fixations as circles
    if fixations_df is not None and not fixations_df.empty:
        fig.add_trace(go.Scatter(
            x=fixations_df['x'],
            y=fixations_df['y'],
            mode='markers+text',
            marker=dict(
                size=fixations_df['duration'] / 30 + 10,
                color=fixations_df['duration'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Duration (ms)"),
                line=dict(color='white', width=1)
            ),
            text=fixations_df['fixation_id'] if 'fixation_id' in fixations_df.columns else None,
            textposition='middle center',
            textfont=dict(size=8, color='white'),
            name='Fixations',
            hoverinfo='text',
            hovertext=[f"Fixation {row.get('fixation_id', i+1)}<br>Duration: {row['duration']:.0f}ms<br>Position: ({row['x']:.0f}, {row['y']:.0f})"
                      for i, row in fixations_df.iterrows()]
        ))

    fig.update_layout(
        title="Fixations & Saccades (Scanpath)",
        xaxis=dict(range=[0, width], title="X Position"),
        yaxis=dict(range=[height, 0], title="Y Position"),
        height=600,
        showlegend=True
    )

    return fig


def create_saccade_amplitude_histogram(saccades_df: pd.DataFrame) -> go.Figure:
    """Create histogram of saccade amplitudes."""
    if saccades_df is None or saccades_df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=saccades_df['amplitude'],
        nbinsx=20,
        name='Saccade Amplitude',
        marker_color='red'
    ))

    fig.update_layout(
        title="Saccade Amplitude Distribution",
        xaxis_title="Amplitude (pixels)",
        yaxis_title="Count",
        height=300
    )

    return fig


def create_saccade_direction_polar(saccades_df: pd.DataFrame) -> go.Figure:
    """Create polar plot of saccade directions."""
    if saccades_df is None or saccades_df.empty:
        return None

    # Bin directions into 12 sectors (30 degrees each)
    bins = np.linspace(-180, 180, 13)
    hist, _ = np.histogram(saccades_df['direction'], bins=bins)

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=hist,
        theta=np.linspace(0, 360, 12, endpoint=False),
        width=30,
        marker_color='blue',
        opacity=0.7,
        name='Saccade Direction'
    ))

    fig.update_layout(
        title="Saccade Direction Distribution",
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(direction='clockwise', rotation=90)
        ),
        height=400
    )

    return fig


def display_session_stats(data: Dict):
    """Display session statistics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Gaze Points", len(data["gaze"]) if data["gaze"] is not None else 0)
    with col2:
        st.metric("Mouse Events", len(data["mouse"]) if data["mouse"] is not None else 0)
    with col3:
        if data["metadata"]:
            time_range = data["metadata"].get("time_range", {})
            duration = (time_range.get("end", 0) - time_range.get("start", 0)) / 1000
            st.metric("Duration", f"{duration:.1f}s")
        else:
            st.metric("Duration", "N/A")
    with col4:
        st.metric("Face Mesh Frames", len(data["face_mesh"]) if data["face_mesh"] is not None else 0)


# =============================================================================
# Main Application
# =============================================================================

def main():
    st.title("HCI Session Analysis Dashboard")
    st.markdown("Comprehensive analysis of eye tracking, mouse movement, and behavioral data.")

    # Sidebar
    st.sidebar.header("Session Selection")
    sessions = get_available_sessions()

    if not sessions:
        st.error(f"No sessions found in {DATA_DIR}")
        st.info("Run an HCI experiment first to generate data.")
        return

    selected_session = st.sidebar.selectbox(
        "Select Session", sessions, format_func=lambda x: f"Session: {x}"
    )

    # Load session data
    with st.spinner("Loading session data..."):
        data = load_session_data(selected_session)

    # Calculate fixations for use across tabs
    fixations = calculate_fixations(data["gaze"]) if data["gaze"] is not None else pd.DataFrame()

    # Display session info
    st.header(f"Session: {selected_session}")
    display_session_stats(data)

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Video & Timeline",
        "Eye Tracking",
        "Mouse Analysis",
        "Fixations & Saccades",
        "AOI Analysis",
        "Attention Metrics",
        "Gaze-Mouse",
        "Behavior & Emotion",
        "Compare & Export"
    ])

    # Tab 1: Video & Timeline
    with tab1:
        st.subheader("Session Recording")
        col1, col2 = st.columns([2, 1])

        with col1:
            if data["video_path"] and data["video_path"].exists():
                with open(data["video_path"], "rb") as f:
                    st.video(f.read())
            else:
                st.warning("No video recording available for this session.")

        with col2:
            st.markdown("### Session Metadata")
            if data["metadata"]:
                st.json(data["metadata"])

        st.subheader("Synchronized Timeline")
        st.plotly_chart(create_timeline_chart(data), use_container_width=True)

    # Tab 2: Eye Tracking
    with tab2:
        st.subheader("Eye Tracking Analysis")

        if data["gaze"] is not None and not data["gaze"].empty:
            gaze_df = data["gaze"]
            min_time, max_time = gaze_df['timestamp'].min(), gaze_df['timestamp'].max()

            time_range = st.slider(
                "Time Range (ms)", float(min_time), float(max_time),
                (float(min_time), float(max_time)), key="gaze_time"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Gaze Heatmap")
                fig = create_gaze_heatmap(gaze_df, time_start=time_range[0], time_end=time_range[1])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Gaze Scanpath")
                fig = create_gaze_scanpath(gaze_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Gaze Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean X", f"{gaze_df['x'].mean():.1f}")
            with col2:
                st.metric("Mean Y", f"{gaze_df['y'].mean():.1f}")
            with col3:
                if len(gaze_df) > 1:
                    rate = 1000 / gaze_df['timestamp'].diff().mean()
                    st.metric("Sampling Rate", f"{rate:.1f} Hz")
        else:
            st.info("No gaze data available.")

    # Tab 3: Mouse Analysis
    with tab3:
        st.subheader("Mouse Movement Analysis")

        if data["mouse"] is not None and not data["mouse"].empty:
            mouse_df = data["mouse"]
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Movement Trajectory")
                fig = create_mouse_trajectory(mouse_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Click Positions")
                fig = create_click_heatmap(mouse_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No click data available.")

            st.markdown("### Mouse Statistics")
            total_events = len(mouse_df)
            clicks = len(mouse_df[mouse_df['event'] == 'click']) if 'event' in mouse_df.columns else 0
            moves = total_events - clicks

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Events", total_events)
            with col2:
                st.metric("Clicks", clicks)
            with col3:
                st.metric("Movements", moves)
            with col4:
                dx, dy = mouse_df['x'].diff(), mouse_df['y'].diff()
                st.metric("Total Distance", f"{np.sqrt(dx**2 + dy**2).sum():.0f} px")
        else:
            st.info("No mouse data available.")

    # Tab 4: Fixations & Saccades
    with tab4:
        st.subheader("Fixation & Saccade Analysis")

        if data["gaze"] is not None and not data["gaze"].empty:
            st.markdown("### Detection Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                velocity_threshold = st.slider("Velocity Threshold (px/ms)", 1, 200, 50,
                    help="Lower = stricter fixation detection")
            with col2:
                min_duration = st.slider("Min Fixation Duration (ms)", 50, 500, 100,
                    help="Minimum time to count as fixation")
            with col3:
                dispersion_threshold = st.slider("Max Dispersion (px)", 50, 400, 150,
                    help="Maximum spread within a fixation. Higher = more lenient (better for webcam tracking)")

            # Calculate fixations and saccades
            fixations = calculate_fixations(data["gaze"], velocity_threshold, min_duration, dispersion_threshold)
            saccades = calculate_saccades(data["gaze"], fixations, velocity_threshold)

            # Summary metrics
            st.markdown("### Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Fixations", len(fixations))
            with col2:
                st.metric("Saccades", len(saccades))
            with col3:
                if not fixations.empty:
                    st.metric("Mean Fix Duration", f"{fixations['duration'].mean():.0f} ms")
                else:
                    st.metric("Mean Fix Duration", "N/A")
            with col4:
                if not saccades.empty:
                    st.metric("Mean Saccade Amp", f"{saccades['amplitude'].mean():.0f} px")
                else:
                    st.metric("Mean Saccade Amp", "N/A")
            with col5:
                if not fixations.empty and not saccades.empty:
                    fix_time = fixations['duration'].sum()
                    total_time = data["gaze"]['timestamp'].max() - data["gaze"]['timestamp'].min()
                    fix_ratio = (fix_time / total_time * 100) if total_time > 0 else 0
                    st.metric("Fixation Ratio", f"{fix_ratio:.1f}%")
                else:
                    st.metric("Fixation Ratio", "N/A")

            # Scanpath visualization
            st.markdown("### Scanpath Visualization (Fixations + Saccades)")
            if not fixations.empty:
                scanpath_fig = create_saccade_visualization(fixations, saccades)
                st.plotly_chart(scanpath_fig, use_container_width=True)
            else:
                st.warning("No fixations detected. Try adjusting thresholds.")

            # Fixation Analysis
            st.markdown("### Fixation Analysis")
            if not fixations.empty:
                col1, col2 = st.columns(2)

                with col1:
                    # Fixation duration histogram
                    fig_fix_hist = go.Figure()
                    fig_fix_hist.add_trace(go.Histogram(
                        x=fixations['duration'],
                        nbinsx=20,
                        name='Fixation Duration',
                        marker_color='blue'
                    ))
                    fig_fix_hist.update_layout(
                        title="Fixation Duration Distribution",
                        xaxis_title="Duration (ms)",
                        yaxis_title="Count",
                        height=300
                    )
                    st.plotly_chart(fig_fix_hist, use_container_width=True)

                with col2:
                    # Fixation position heatmap
                    fig_fix_map = go.Figure()
                    fig_fix_map.add_trace(go.Scatter(
                        x=fixations['x'], y=fixations['y'],
                        mode='markers',
                        marker=dict(
                            size=fixations['duration']/20 + 5,
                            color=fixations['duration'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Duration (ms)")
                        ),
                        text=[f"#{row['fixation_id']}: {row['duration']:.0f}ms"
                              for _, row in fixations.iterrows()],
                        hoverinfo='text'
                    ))
                    fig_fix_map.update_layout(
                        title="Fixation Positions",
                        yaxis=dict(autorange="reversed"),
                        height=300
                    )
                    st.plotly_chart(fig_fix_map, use_container_width=True)

                # Fixation statistics table
                st.markdown("#### Fixation Statistics")
                fix_stats = {
                    'Metric': ['Total Fixations', 'Mean Duration', 'Std Duration', 'Min Duration', 'Max Duration', 'Mean Dispersion'],
                    'Value': [
                        len(fixations),
                        f"{fixations['duration'].mean():.1f} ms",
                        f"{fixations['duration'].std():.1f} ms",
                        f"{fixations['duration'].min():.1f} ms",
                        f"{fixations['duration'].max():.1f} ms",
                        f"{fixations['dispersion'].mean():.1f} px"
                    ]
                }
                st.dataframe(pd.DataFrame(fix_stats), hide_index=True)

                with st.expander("View Fixation Data"):
                    st.dataframe(fixations.round(2))
            else:
                st.info("No fixations detected with current parameters.")

            # Saccade Analysis
            st.markdown("### Saccade Analysis")
            if not saccades.empty:
                col1, col2 = st.columns(2)

                with col1:
                    # Saccade amplitude histogram
                    amp_fig = create_saccade_amplitude_histogram(saccades)
                    if amp_fig:
                        st.plotly_chart(amp_fig, use_container_width=True)

                with col2:
                    # Saccade direction polar plot
                    dir_fig = create_saccade_direction_polar(saccades)
                    if dir_fig:
                        st.plotly_chart(dir_fig, use_container_width=True)

                # Saccade statistics table
                st.markdown("#### Saccade Statistics")
                sacc_stats = {
                    'Metric': ['Total Saccades', 'Mean Amplitude', 'Std Amplitude', 'Min Amplitude', 'Max Amplitude',
                              'Mean Velocity', 'Mean Duration'],
                    'Value': [
                        len(saccades),
                        f"{saccades['amplitude'].mean():.1f} px",
                        f"{saccades['amplitude'].std():.1f} px",
                        f"{saccades['amplitude'].min():.1f} px",
                        f"{saccades['amplitude'].max():.1f} px",
                        f"{saccades['velocity'].mean():.2f} px/ms",
                        f"{saccades['duration'].mean():.1f} ms"
                    ]
                }
                st.dataframe(pd.DataFrame(sacc_stats), hide_index=True)

                with st.expander("View Saccade Data"):
                    st.dataframe(saccades.round(2))
            else:
                st.info("No saccades detected (need at least 2 fixations).")

        else:
            st.info("No gaze data available.")

    # Tab 5: AOI Analysis
    with tab5:
        st.subheader("Areas of Interest (AOI) Analysis")

        st.markdown("### Define AOIs")
        st.info("Define rectangular areas of interest to analyze gaze and mouse behavior.")

        # AOI definition interface
        num_aois = st.number_input("Number of AOIs", min_value=1, max_value=10, value=3)

        aois = []
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']

        cols = st.columns(min(num_aois, 3))
        for i in range(num_aois):
            with cols[i % 3]:
                st.markdown(f"**AOI {i+1}**")
                name = st.text_input(f"Name", value=f"AOI_{i+1}", key=f"aoi_name_{i}")
                x = st.number_input(f"X", value=100 + i*300, key=f"aoi_x_{i}")
                y = st.number_input(f"Y", value=100, key=f"aoi_y_{i}")
                w = st.number_input(f"Width", value=200, key=f"aoi_w_{i}")
                h = st.number_input(f"Height", value=200, key=f"aoi_h_{i}")
                aois.append(AOI(name=name, x=int(x), y=int(y), width=int(w), height=int(h), color=colors[i]))

        if st.button("Analyze AOIs"):
            # Visualization
            st.markdown("### AOI Visualization")
            fig = create_aoi_visualization(data["gaze"], aois)
            st.plotly_chart(fig, use_container_width=True)

            # Analysis results
            st.markdown("### AOI Metrics")
            aoi_results = analyze_aoi(data["gaze"], data["mouse"], aois)
            if not aoi_results.empty:
                st.dataframe(aoi_results)

            # Transition matrix
            st.markdown("### AOI Transition Matrix")
            transitions = create_aoi_transition_matrix(data["gaze"], aois)
            if not transitions.empty:
                fig = px.imshow(transitions, text_auto=True, title="Transitions Between AOIs")
                st.plotly_chart(fig, use_container_width=True)

    # Tab 6: Attention Metrics
    with tab6:
        st.subheader("Attention Metrics")

        # Define AOIs for attention analysis (reuse from session state or defaults)
        default_aois = [
            AOI("Top-Left", 0, 0, 640, 360, "red"),
            AOI("Top-Right", 640, 0, 640, 360, "blue"),
            AOI("Center", 320, 180, 640, 360, "green")
        ]

        metrics = calculate_attention_metrics(data["gaze"], fixations, default_aois)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fixations", metrics['total_fixations'])
        with col2:
            st.metric("Mean Fixation Duration", f"{metrics['mean_fixation_duration_ms']:.0f} ms")
        with col3:
            st.metric("Fixation Rate", f"{metrics['fixation_rate']:.2f} /sec")
        with col4:
            st.metric("Gaze Dispersion", f"{metrics['gaze_dispersion']:.0f} px")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Saccade Count", metrics['saccade_count'])
        with col2:
            st.metric("Mean Saccade Amplitude", f"{metrics['mean_saccade_amplitude']:.0f} px")

        if metrics['aoi_metrics']:
            st.markdown("### Time to First Fixation (TTFF) by AOI")
            ttff_data = []
            for aoi_name, aoi_metrics in metrics['aoi_metrics'].items():
                ttff_data.append({
                    'AOI': aoi_name,
                    'TTFF (ms)': aoi_metrics['ttff_ms'],
                    'Total Dwell (ms)': aoi_metrics['total_dwell_ms'],
                    'Fixations': aoi_metrics['fixation_count'],
                    'Revisits': aoi_metrics['revisits']
                })
            st.dataframe(pd.DataFrame(ttff_data))

    # Tab 7: Gaze-Mouse Coordination
    with tab7:
        st.subheader("Gaze-Mouse Coordination Analysis")

        coordination = analyze_gaze_mouse_coordination(data["gaze"], data["mouse"])

        col1, col2, col3 = st.columns(3)
        with col1:
            if coordination['correlation_x'] is not None:
                st.metric("X Correlation", f"{coordination['correlation_x']:.3f}")
            else:
                st.metric("X Correlation", "N/A")
        with col2:
            if coordination['correlation_y'] is not None:
                st.metric("Y Correlation", f"{coordination['correlation_y']:.3f}")
            else:
                st.metric("Y Correlation", "N/A")
        with col3:
            if coordination['coordination_score'] is not None:
                st.metric("Coordination Score", f"{coordination['coordination_score']:.2f}")
            else:
                st.metric("Coordination Score", "N/A")

        col1, col2, col3 = st.columns(3)
        with col1:
            if coordination['mean_distance'] is not None:
                st.metric("Mean Gaze-Mouse Distance", f"{coordination['mean_distance']:.0f} px")
        with col2:
            if coordination['gaze_leads_mouse']:
                st.metric("Gaze Leads Mouse By", f"{coordination['gaze_leads_mouse']} samples")
            elif coordination['mouse_leads_gaze']:
                st.metric("Mouse Leads Gaze By", f"{coordination['mouse_leads_gaze']} samples")
            else:
                st.metric("Temporal Relationship", "Synchronized")

        st.markdown("### Position Comparison Over Time")
        fig = create_gaze_mouse_comparison_plot(data["gaze"], data["mouse"])
        st.plotly_chart(fig, use_container_width=True)

        if coordination['lag_analysis']:
            st.markdown("### Lag Analysis")
            lag_df = pd.DataFrame(coordination['lag_analysis'])
            fig = px.line(lag_df, x='lag', y='correlation',
                         title="Cross-correlation by Lag (positive lag = mouse leads)")
            st.plotly_chart(fig, use_container_width=True)

    # Tab 8: Behavior & Emotion
    with tab8:
        st.subheader("Behavioral Patterns & Emotion Analysis")

        patterns = detect_behavioral_patterns(data["gaze"], data["mouse"], fixations)

        # Emotion analysis with progress indicator (can be slow for large datasets)
        with st.spinner("Analyzing facial expressions..."):
            emotion = analyze_emotion_timeline(data["face_mesh"], data["timeline"], data["mouse"])

        # === BEHAVIORAL PATTERNS SECTION ===
        st.markdown("## Behavioral Patterns")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Reading Pattern", patterns['pattern_summary'].get('reading_pattern', 'N/A'))
        with col2:
            st.metric("Hesitations Detected", patterns['pattern_summary'].get('hesitation_count', 0))
        with col3:
            st.metric("Backtracking Events", patterns['pattern_summary'].get('backtracking_events', 0))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rapid Scanning Events", patterns['pattern_summary'].get('rapid_scanning_events', 0))
        with col2:
            st.metric("Focused Attention Areas", patterns['pattern_summary'].get('focused_attention_count', 0))
        with col3:
            st.metric("Mouse Jitter Events", patterns['pattern_summary'].get('mouse_jitter_events', 0))

        st.markdown("### Behavioral Patterns Map")
        fig = create_behavioral_patterns_visualization(patterns, data["gaze"], data["mouse"])
        st.plotly_chart(fig, use_container_width=True)

        # === EMOTION ANALYSIS SECTION ===
        st.markdown("---")
        st.markdown("## Emotion Analysis (Face Mesh Based)")

        if emotion.get('emotion_timeline'):
            # Summary metrics
            st.markdown("### Summary Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Avg Valence",
                         f"{emotion['emotion_summary'].get('avg_valence', 0):.2f}",
                         help="Positive = happy, Negative = frustrated/tired")
            with col2:
                st.metric("Avg Arousal",
                         f"{emotion['emotion_summary'].get('avg_arousal', 0):.2f}",
                         help="High = excited/alert, Low = calm/tired")
            with col3:
                st.metric("Avg Engagement",
                         f"{emotion['emotion_summary'].get('avg_engagement', 0):.1f}%")
            with col4:
                st.metric("Blink Rate",
                         f"{emotion['emotion_summary'].get('blink_rate_per_min', 0):.1f}/min",
                         help="Normal: 15-20 blinks/min")
            with col5:
                st.metric("Frustration Events",
                         emotion['emotion_summary'].get('frustration_events', 0))

            # Emotion distribution
            st.markdown("### Emotion Distribution")
            col1, col2 = st.columns(2)

            with col1:
                dist_fig = create_emotion_distribution_chart(emotion)
                if dist_fig:
                    st.plotly_chart(dist_fig, use_container_width=True)
                else:
                    st.info("No emotion distribution data available")

            with col2:
                # Dominant emotions table
                if emotion.get('dominant_emotions'):
                    st.markdown("#### Dominant Emotions (%)")
                    emotions_df = pd.DataFrame([
                        {'Emotion': k.capitalize(), 'Percentage': f"{v:.1f}%"}
                        for k, v in sorted(emotion['dominant_emotions'].items(),
                                          key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(emotions_df, hide_index=True)

            # Valence-Arousal scatter (Circumplex Model)
            st.markdown("### Valence-Arousal Space (Circumplex Model of Affect)")
            va_fig = create_valence_arousal_scatter(emotion)
            if va_fig:
                st.plotly_chart(va_fig, use_container_width=True)
                st.caption("""
                The circumplex model represents emotions in 2D space:
                - **Valence** (x-axis): Positive (right) vs Negative (left)
                - **Arousal** (y-axis): High energy (top) vs Low energy (bottom)
                """)

            # Main emotion timeline
            st.markdown("### Emotion Timeline")
            timeline_fig = create_emotion_timeline_plot(emotion)
            st.plotly_chart(timeline_fig, use_container_width=True)

            # Facial features timeline (expandable)
            with st.expander("View Raw Facial Feature Metrics"):
                features_fig = create_facial_features_timeline(emotion)
                if features_fig:
                    st.plotly_chart(features_fig, use_container_width=True)
                    st.markdown("""
                    **Metrics Explanation:**
                    - **Eye Aspect Ratio (EAR)**: Ratio of eye height to width. Lower = more closed eyes (tired/blink)
                    - **Mouth Aspect Ratio (MAR)**: Ratio of mouth height to width. Higher = more open mouth (surprise/yawn)
                    - **Eyebrow Position**: Distance from eyebrow to eye. Higher = raised eyebrows (surprise)
                    """)

            # Blink analysis
            if emotion.get('blink_events'):
                st.markdown("### Blink Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Blinks Detected", len(emotion['blink_events']))
                with col2:
                    blink_rate = emotion['emotion_summary'].get('blink_rate_per_min', 0)
                    if blink_rate < 10:
                        interpretation = "Low (focused/staring)"
                    elif blink_rate < 20:
                        interpretation = "Normal"
                    else:
                        interpretation = "High (fatigue/stress)"
                    st.metric("Blink Rate Interpretation", interpretation)

        else:
            st.warning("No face mesh data available for emotion analysis.")
            st.info("""
            **Emotion analysis requires face mesh data.**

            Make sure:
            1. The webcam was enabled during the session
            2. Face detection was active
            3. The face_mesh CSV file exists in the session folder
            """)

    # Tab 9: Compare & Export
    with tab9:
        st.subheader("Comparative Analysis & Export")

        st.markdown("### Compare Multiple Sessions")
        selected_sessions = st.multiselect(
            "Select sessions to compare",
            sessions,
            default=[selected_session]
        )

        if len(selected_sessions) > 1 and st.button("Compare Sessions"):
            with st.spinner("Loading sessions..."):
                multi_data = load_multiple_sessions(selected_sessions)

            comparison = compare_sessions(multi_data)
            st.dataframe(comparison)

            charts = create_comparison_charts(comparison)
            for chart in charts:
                st.plotly_chart(chart, use_container_width=True)

            st.markdown("### Aggregate Heatmap")
            agg_heatmap = create_aggregate_heatmap(multi_data)
            if agg_heatmap:
                st.plotly_chart(agg_heatmap, use_container_width=True)

        st.markdown("---")
        st.markdown("### Export Report")

        if st.button("Generate Report"):
            patterns = detect_behavioral_patterns(data["gaze"], data["mouse"], fixations)
            attention = calculate_attention_metrics(data["gaze"], fixations)
            report = generate_report_data(data, fixations, attention, patterns)

            st.json(report)

            # Download as JSON
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                "Download Report (JSON)",
                report_json,
                f"report_{selected_session}.json",
                "application/json"
            )

        st.markdown("### Export Raw Data")
        data_type = st.selectbox(
            "Select Data Type",
            ["gaze", "mouse", "face_mesh", "experiment_event", "fixations"]
        )

        if data_type == "fixations":
            export_df = fixations
        elif data[data_type] is not None:
            export_df = data[data_type]
        else:
            export_df = None

        if export_df is not None and not export_df.empty:
            st.write(f"Shape: {export_df.shape}")
            st.dataframe(export_df.head(100))

            csv = export_df.to_csv(index=False)
            st.download_button(
                f"Download {data_type}.csv",
                csv,
                f"{selected_session}_{data_type}.csv",
                "text/csv"
            )
        else:
            st.info(f"No {data_type} data available.")


if __name__ == "__main__":
    main()
