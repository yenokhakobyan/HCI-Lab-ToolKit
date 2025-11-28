"""
Tobii Eye Tracking Visualization

Provides visualization tools for eye tracking data including:
- Heatmaps
- Scanpaths
- Fixation plots
- Pupil timeseries
- AOI overlays
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class TobiiVisualizer:
    """
    Visualizes Tobii eye tracking data.

    Supports both matplotlib (static) and plotly (interactive) visualizations.
    """

    def __init__(self, screen_size: Tuple[int, int] = (1920, 1080)):
        """
        Initialize visualizer.

        Args:
            screen_size: Screen resolution (width, height)
        """
        self.screen_size = screen_size
        self.background_image: Optional[np.ndarray] = None

    def set_background(self, image_path: str):
        """
        Set background image for overlays (e.g., stimulus screenshot).

        Args:
            image_path: Path to image file
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow required for background images: pip install pillow")

        self.background_image = np.array(Image.open(image_path))
        print(f"Background set: {self.background_image.shape}")

    def plot_heatmap(self, heatmap: np.ndarray,
                     title: str = "Gaze Heatmap",
                     cmap: str = "hot",
                     alpha: float = 0.6,
                     show_colorbar: bool = True,
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot a gaze heatmap.

        Args:
            heatmap: 2D numpy array from TobiiAnalyzer.generate_heatmap()
            title: Plot title
            cmap: Colormap name
            alpha: Transparency for overlay
            show_colorbar: Show colorbar
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Show background if available
        if self.background_image is not None:
            ax.imshow(self.background_image)
            im = ax.imshow(heatmap, cmap=cmap, alpha=alpha,
                          extent=[0, self.screen_size[0], self.screen_size[1], 0])
        else:
            im = ax.imshow(heatmap, cmap=cmap,
                          extent=[0, self.screen_size[0], self.screen_size[1], 0])

        if show_colorbar:
            plt.colorbar(im, ax=ax, label="Gaze Density")

        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        return fig

    def plot_scanpath(self, fixations: List,
                      saccades: Optional[List] = None,
                      title: str = "Scanpath",
                      show_order: bool = True,
                      fixation_color: str = "blue",
                      saccade_color: str = "gray",
                      duration_scale: float = 500,
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot scanpath (fixations connected by saccades).

        Args:
            fixations: List of Fixation objects
            saccades: List of Saccade objects (optional)
            title: Plot title
            show_order: Show fixation numbers
            fixation_color: Color for fixation circles
            saccade_color: Color for saccade lines
            duration_scale: Scale factor for fixation circle size
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Show background if available
        if self.background_image is not None:
            ax.imshow(self.background_image)

        # Convert to pixel coordinates
        fix_x = [f.x * self.screen_size[0] for f in fixations]
        fix_y = [f.y * self.screen_size[1] for f in fixations]
        fix_dur = [f.duration for f in fixations]

        # Draw saccades (lines connecting fixations)
        if len(fixations) > 1:
            for i in range(len(fixations) - 1):
                ax.plot([fix_x[i], fix_x[i+1]], [fix_y[i], fix_y[i+1]],
                       color=saccade_color, linewidth=1, alpha=0.5, zorder=1)

        # Draw fixations (circles sized by duration)
        sizes = [d * duration_scale for d in fix_dur]
        scatter = ax.scatter(fix_x, fix_y, s=sizes, c=fixation_color,
                            alpha=0.6, edgecolors="white", linewidths=1, zorder=2)

        # Add fixation order numbers
        if show_order:
            for i, (x, y) in enumerate(zip(fix_x, fix_y)):
                ax.annotate(str(i+1), (x, y), ha="center", va="center",
                           fontsize=8, color="white", fontweight="bold", zorder=3)

        ax.set_xlim(0, self.screen_size[0])
        ax.set_ylim(self.screen_size[1], 0)  # Invert Y axis
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        return fig

    def plot_fixation_duration_histogram(self, fixations: List,
                                          bins: int = 30,
                                          title: str = "Fixation Duration Distribution",
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot histogram of fixation durations.

        Args:
            fixations: List of Fixation objects
            bins: Number of histogram bins
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        durations = [f.duration * 1000 for f in fixations]  # Convert to ms

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(durations, bins=bins, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(durations), color="red", linestyle="--",
                  label=f"Mean: {np.mean(durations):.1f} ms")
        ax.axvline(np.median(durations), color="green", linestyle="--",
                  label=f"Median: {np.median(durations):.1f} ms")

        ax.set_xlabel("Fixation Duration (ms)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_pupil_timeseries(self, data: pd.DataFrame,
                               title: str = "Pupil Diameter Over Time",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot pupil diameter over time.

        Args:
            data: DataFrame with pupil data
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 5))

        time_col = "recording_timestamp" if "recording_timestamp" in data.columns else data.index

        # Plot both eyes
        if "left_pupil_diameter" in data.columns:
            valid = data["left_pupil_valid"] == 1 if "left_pupil_valid" in data.columns else True
            ax.plot(data.loc[valid, "recording_timestamp"] if "recording_timestamp" in data.columns else data[valid].index,
                   data.loc[valid, "left_pupil_diameter"],
                   label="Left Eye", alpha=0.7, linewidth=0.5)

        if "right_pupil_diameter" in data.columns:
            valid = data["right_pupil_valid"] == 1 if "right_pupil_valid" in data.columns else True
            ax.plot(data.loc[valid, "recording_timestamp"] if "recording_timestamp" in data.columns else data[valid].index,
                   data.loc[valid, "right_pupil_diameter"],
                   label="Right Eye", alpha=0.7, linewidth=0.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pupil Diameter (mm)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_gaze_trajectory(self, data: pd.DataFrame,
                              title: str = "Gaze Trajectory",
                              colorby: str = "time",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot raw gaze trajectory colored by time or other variable.

        Args:
            data: DataFrame with gaze data
            title: Plot title
            colorby: Variable to color by ("time", "velocity", "pupil")
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        if self.background_image is not None:
            ax.imshow(self.background_image)

        # Get coordinates
        if "gaze_x_px" in data.columns:
            x = data["gaze_x_px"].values
            y = data["gaze_y_px"].values
        else:
            x = data["gaze_x"].values * self.screen_size[0] if "gaze_x" in data.columns else \
                data["left_gaze_x"].values * self.screen_size[0]
            y = data["gaze_y"].values * self.screen_size[1] if "gaze_y" in data.columns else \
                data["left_gaze_y"].values * self.screen_size[1]

        # Create color values
        if colorby == "time":
            colors = np.arange(len(x))
            cmap = "viridis"
            label = "Time (samples)"
        elif colorby == "pupil" and "pupil_diameter" in data.columns:
            colors = data["pupil_diameter"].values
            cmap = "coolwarm"
            label = "Pupil Diameter (mm)"
        else:
            colors = np.arange(len(x))
            cmap = "viridis"
            label = "Time (samples)"

        # Create line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Remove invalid segments
        valid = ~np.isnan(segments).any(axis=(1, 2))
        segments = segments[valid]
        colors = colors[:-1][valid]

        lc = LineCollection(segments, cmap=cmap, array=colors, linewidth=1, alpha=0.7)
        ax.add_collection(lc)

        plt.colorbar(lc, ax=ax, label=label)

        ax.set_xlim(0, self.screen_size[0])
        ax.set_ylim(self.screen_size[1], 0)
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_aoi_overlay(self, aois: Dict[str, Tuple[float, float, float, float]],
                          aoi_results: Optional[Dict] = None,
                          title: str = "Areas of Interest",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot AOI regions with optional metrics.

        Args:
            aois: Dict mapping names to (x1, y1, x2, y2) normalized bounds
            aoi_results: Results from TobiiAnalyzer.analyze_aoi()
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        if self.background_image is not None:
            ax.imshow(self.background_image)

        colors = plt.cm.Set1(np.linspace(0, 1, len(aois)))

        for (name, (x1, y1, x2, y2)), color in zip(aois.items(), colors):
            # Convert to pixels
            px1, py1 = x1 * self.screen_size[0], y1 * self.screen_size[1]
            px2, py2 = x2 * self.screen_size[0], y2 * self.screen_size[1]
            width, height = px2 - px1, py2 - py1

            rect = patches.Rectangle((px1, py1), width, height,
                                     linewidth=2, edgecolor=color,
                                     facecolor=color, alpha=0.2)
            ax.add_patch(rect)

            # Add label
            label = name
            if aoi_results and name in aoi_results:
                r = aoi_results[name]
                label += f"\nFix: {r['fixation_count']}, Dwell: {r['total_dwell_time']:.2f}s"

            ax.text(px1 + 5, py1 + 20, label, fontsize=9,
                   color="white", backgroundcolor=(*color[:3], 0.7))

        ax.set_xlim(0, self.screen_size[0])
        ax.set_ylim(self.screen_size[1], 0)
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_interactive_heatmap(self, heatmap: np.ndarray,
                                  title: str = "Interactive Gaze Heatmap") -> 'go.Figure':
        """
        Create an interactive heatmap using Plotly.

        Args:
            heatmap: 2D numpy array
            title: Plot title

        Returns:
            Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required: pip install plotly")

        fig = go.Figure(data=go.Heatmap(
            z=heatmap,
            colorscale="Hot",
            showscale=True
        ))

        fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        return fig

    def plot_interactive_scanpath(self, fixations: List,
                                   title: str = "Interactive Scanpath") -> 'go.Figure':
        """
        Create an interactive scanpath using Plotly.

        Args:
            fixations: List of Fixation objects
            title: Plot title

        Returns:
            Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required: pip install plotly")

        fix_x = [f.x * self.screen_size[0] for f in fixations]
        fix_y = [f.y * self.screen_size[1] for f in fixations]
        fix_dur = [f.duration * 1000 for f in fixations]  # ms
        fix_order = list(range(1, len(fixations) + 1))

        fig = go.Figure()

        # Saccade lines
        fig.add_trace(go.Scatter(
            x=fix_x, y=fix_y,
            mode="lines",
            line=dict(color="gray", width=1),
            name="Saccades",
            hoverinfo="skip"
        ))

        # Fixation points
        fig.add_trace(go.Scatter(
            x=fix_x, y=fix_y,
            mode="markers+text",
            marker=dict(
                size=[d/10 for d in fix_dur],
                color=fix_order,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Order")
            ),
            text=fix_order,
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            hovertemplate="Fixation %{text}<br>X: %{x:.0f}<br>Y: %{y:.0f}<br>Duration: %{customdata:.0f}ms",
            customdata=fix_dur,
            name="Fixations"
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(title="X (pixels)", range=[0, self.screen_size[0]]),
            yaxis=dict(title="Y (pixels)", range=[self.screen_size[1], 0]),
            showlegend=True
        )

        return fig

    def create_summary_dashboard(self, data: pd.DataFrame,
                                  fixations: List,
                                  heatmap: np.ndarray,
                                  stats: Dict,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a multi-panel summary dashboard.

        Args:
            data: Raw gaze DataFrame
            fixations: List of Fixation objects
            heatmap: Heatmap array
            stats: Dictionary of statistics
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))

        # Heatmap
        ax1 = fig.add_subplot(2, 2, 1)
        if self.background_image is not None:
            ax1.imshow(self.background_image)
            ax1.imshow(heatmap, cmap="hot", alpha=0.6,
                      extent=[0, self.screen_size[0], self.screen_size[1], 0])
        else:
            ax1.imshow(heatmap, cmap="hot",
                      extent=[0, self.screen_size[0], self.screen_size[1], 0])
        ax1.set_title("Gaze Heatmap")

        # Scanpath
        ax2 = fig.add_subplot(2, 2, 2)
        if self.background_image is not None:
            ax2.imshow(self.background_image)
        fix_x = [f.x * self.screen_size[0] for f in fixations]
        fix_y = [f.y * self.screen_size[1] for f in fixations]
        ax2.plot(fix_x, fix_y, "b-", alpha=0.5, linewidth=1)
        ax2.scatter(fix_x, fix_y, c=range(len(fixations)), cmap="viridis", s=50)
        ax2.set_xlim(0, self.screen_size[0])
        ax2.set_ylim(self.screen_size[1], 0)
        ax2.set_title("Scanpath")

        # Fixation duration histogram
        ax3 = fig.add_subplot(2, 2, 3)
        durations = [f.duration * 1000 for f in fixations]
        ax3.hist(durations, bins=20, edgecolor="black", alpha=0.7)
        ax3.axvline(np.mean(durations), color="red", linestyle="--")
        ax3.set_xlabel("Duration (ms)")
        ax3.set_ylabel("Count")
        ax3.set_title("Fixation Duration Distribution")

        # Stats text
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        stats_text = "Summary Statistics\n" + "=" * 30 + "\n\n"
        for key, value in stats.items():
            if isinstance(value, float):
                stats_text += f"{key}: {value:.3f}\n"
            else:
                stats_text += f"{key}: {value}\n"
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Dashboard saved: {save_path}")

        return fig

    # =========================================================================
    # COGNITIVE LOAD VISUALIZATIONS
    # =========================================================================

    def plot_k_coefficient_timeline(self, k_data: pd.DataFrame,
                                     title: str = "Ambient/Focal Attention Over Time",
                                     save_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Plot K coefficient over time showing ambient vs focal attention modes.

        Args:
            k_data: DataFrame from TobiiAnalyzer.compute_k_coefficient_over_time()
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        time = k_data["time"].values
        k_values = k_data["k_coefficient"].values

        # Fill regions for ambient (K < 0) and focal (K > 0)
        ax.fill_between(time, k_values, 0, where=(k_values >= 0),
                       color="steelblue", alpha=0.4, label="Focal (K > 0)")
        ax.fill_between(time, k_values, 0, where=(k_values < 0),
                       color="coral", alpha=0.4, label="Ambient (K < 0)")

        # Plot the K coefficient line
        ax.plot(time, k_values, color="black", linewidth=1.5, zorder=3)

        # Add threshold lines
        ax.axhline(y=0.5, color="steelblue", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=-0.5, color="coral", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.8, linewidth=1)

        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("K Coefficient", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Add annotations
        ax.annotate("Focal Processing", xy=(0.02, 0.95), xycoords="axes fraction",
                   fontsize=9, color="steelblue", style="italic")
        ax.annotate("Ambient Processing", xy=(0.02, 0.05), xycoords="axes fraction",
                   fontsize=9, color="coral", style="italic")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_cognitive_load_dashboard(self, data: pd.DataFrame,
                                       ica_result: Dict,
                                       lhipa_result: Dict,
                                       tepr_result: Dict,
                                       title: str = "Cognitive Load Analysis",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive cognitive load visualization dashboard.

        Args:
            data: Raw gaze DataFrame with pupil data
            ica_result: Results from compute_index_of_cognitive_activity()
            lhipa_result: Results from compute_lhipa()
            tepr_result: Results from compute_pupillary_response()
            title: Overall title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # 1. Pupil diameter timeseries (top)
        ax1 = fig.add_subplot(3, 2, (1, 2))
        if "recording_timestamp" in data.columns and "pupil_diameter" in data.columns:
            time = data["recording_timestamp"].values
            pupil = data["pupil_diameter"].values
            valid = ~np.isnan(pupil)
            ax1.plot(time[valid], pupil[valid], linewidth=0.5, alpha=0.8, color="purple")

            # Add baseline indication
            if "baseline_mean" in tepr_result:
                ax1.axhline(tepr_result["baseline_mean"], color="green",
                           linestyle="--", label=f"Baseline: {tepr_result['baseline_mean']:.2f}mm")

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Pupil Diameter (mm)")
            ax1.set_title("Pupil Diameter Over Time")
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)

        # 2. ICA indicator gauge
        ax2 = fig.add_subplot(3, 2, 3)
        ica_value = ica_result.get("ica", 0) if ica_result else 0
        self._plot_gauge(ax2, ica_value, "ICA", max_val=0.1, colormap="YlOrRd")

        # 3. LHIPA ratio gauge
        ax3 = fig.add_subplot(3, 2, 4)
        lhipa_value = lhipa_result.get("lhipa", 1) if lhipa_result else 1
        self._plot_gauge(ax3, lhipa_value, "LHIPA Ratio", max_val=3, colormap="RdYlGn_r")

        # 4. Frequency power distribution (LHIPA bands)
        ax4 = fig.add_subplot(3, 2, 5)
        if lhipa_result and "lipa" in lhipa_result and "hipa" in lhipa_result:
            bands = ["Low Freq\n(0.04-0.15 Hz)", "High Freq\n(0.15-0.5 Hz)"]
            powers = [lhipa_result.get("lipa", 0), lhipa_result.get("hipa", 0)]
            colors = ["lightblue", "salmon"]
            bars = ax4.bar(bands, powers, color=colors, edgecolor="black")
            ax4.set_ylabel("Power")
            ax4.set_title("Pupil Frequency Components")

            # Add value labels
            for bar, val in zip(bars, powers):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        # 5. TEPR summary stats
        ax5 = fig.add_subplot(3, 2, 6)
        ax5.axis("off")
        if tepr_result:
            stats_text = "Task-Evoked Pupillary Response\n" + "=" * 35 + "\n\n"
            stats_text += f"Baseline Mean:     {tepr_result.get('baseline_mean', 'N/A'):.3f} mm\n"
            stats_text += f"Baseline Std:      {tepr_result.get('baseline_std', 'N/A'):.3f} mm\n"
            stats_text += f"Mean TEPR:         {tepr_result.get('mean_tepr', 'N/A'):.2f}%\n"
            stats_text += f"Max TEPR:          {tepr_result.get('max_tepr', 'N/A'):.2f}%\n"
            stats_text += f"Peak Dilation Time: {tepr_result.get('peak_dilation_time', 'N/A'):.2f}s\n"
            ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
                    fontsize=11, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def _plot_gauge(self, ax, value: float, label: str, max_val: float = 1.0,
                    colormap: str = "RdYlGn_r"):
        """Helper to create a gauge-style indicator."""
        # Create a simple bar-style gauge
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, 1)

        # Background bar
        ax.barh(0.5, max_val, height=0.4, color="lightgray", edgecolor="black")

        # Value bar with color based on value
        cmap = plt.cm.get_cmap(colormap)
        color = cmap(min(value / max_val, 1.0))
        ax.barh(0.5, min(value, max_val), height=0.4, color=color, edgecolor="black")

        # Value text
        ax.text(max_val / 2, 0.5, f"{value:.4f}", ha="center", va="center",
               fontsize=12, fontweight="bold")
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    # =========================================================================
    # MICROSACCADE AND READING VISUALIZATIONS
    # =========================================================================

    def plot_microsaccade_polar(self, microsaccades: List,
                                 title: str = "Microsaccade Direction Distribution",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot microsaccade directions on a polar plot.

        Args:
            microsaccades: List of Microsaccade objects
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")

        if not microsaccades:
            ax.set_title("No microsaccades detected")
            return fig

        # Extract directions and amplitudes
        directions = [np.radians(m.direction) for m in microsaccades]
        amplitudes = [m.amplitude * 1000 for m in microsaccades]  # Scale for visibility

        # Create histogram of directions
        n_bins = 36  # 10-degree bins
        counts, bin_edges = np.histogram(directions, bins=n_bins, range=(0, 2*np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot as rose diagram
        width = 2 * np.pi / n_bins
        bars = ax.bar(bin_centers, counts, width=width, bottom=0.0,
                     alpha=0.7, edgecolor="black")

        # Color by count
        max_count = max(counts) if max(counts) > 0 else 1
        for bar, count in zip(bars, counts):
            bar.set_facecolor(plt.cm.Blues(count / max_count))

        ax.set_theta_zero_location("E")  # 0° at right (East)
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

        # Add legend with stats
        stats_text = f"N = {len(microsaccades)}\n"
        stats_text += f"Mean Amp: {np.mean(amplitudes):.1f}\n"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.3))

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_reading_analysis(self, fixations: List, saccades: List,
                               reading_results: Dict,
                               title: str = "Reading Behavior Analysis",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize reading-specific eye movement patterns.

        Args:
            fixations: List of Fixation objects
            saccades: List of Saccade objects
            reading_results: Results from TobiiAnalyzer.analyze_reading_behavior()
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # 1. Scanpath with reading annotations
        ax1 = fig.add_subplot(2, 2, 1)
        if self.background_image is not None:
            ax1.imshow(self.background_image)

        fix_x = [f.x * self.screen_size[0] for f in fixations]
        fix_y = [f.y * self.screen_size[1] for f in fixations]

        # Color-code saccades
        forward_idx = set(reading_results.get("forward_saccade_indices", []))
        regression_idx = set(reading_results.get("regression_indices", []))
        line_return_idx = set(reading_results.get("line_return_indices", []))

        for i in range(len(fixations) - 1):
            if i in regression_idx:
                color, lw = "red", 2
            elif i in line_return_idx:
                color, lw = "orange", 2
            else:
                color, lw = "blue", 1
            ax1.plot([fix_x[i], fix_x[i+1]], [fix_y[i], fix_y[i+1]],
                    color=color, linewidth=lw, alpha=0.7)

        ax1.scatter(fix_x, fix_y, c=range(len(fixations)), cmap="viridis",
                   s=30, zorder=3, edgecolors="white", linewidths=0.5)
        ax1.set_xlim(0, self.screen_size[0])
        ax1.set_ylim(self.screen_size[1], 0)
        ax1.set_title("Scanpath (blue=forward, red=regression, orange=line return)")

        # 2. Saccade type breakdown (pie chart)
        ax2 = fig.add_subplot(2, 2, 2)
        counts = [
            reading_results.get("forward_saccade_count", 0),
            reading_results.get("regression_count", 0),
            reading_results.get("line_return_count", 0),
        ]
        labels = ["Forward", "Regressions", "Line Returns"]
        colors = ["steelblue", "coral", "orange"]
        explode = (0, 0.05, 0)

        if sum(counts) > 0:
            ax2.pie(counts, labels=labels, colors=colors, explode=explode,
                   autopct="%1.1f%%", startangle=90)
        ax2.set_title("Saccade Type Distribution")

        # 3. Reading metrics bar chart
        ax3 = fig.add_subplot(2, 2, 3)
        metrics = {
            "Regression Rate": reading_results.get("regression_rate", 0) * 100,
            "Refixation Rate": reading_results.get("refixation_rate", 0) * 100,
            "Reading Efficiency": reading_results.get("reading_efficiency", 0) * 100,
        }
        bars = ax3.bar(metrics.keys(), metrics.values(), color=["coral", "gold", "green"],
                      edgecolor="black")
        ax3.set_ylabel("Percentage (%)")
        ax3.set_title("Reading Metrics")
        ax3.set_ylim(0, 100)

        for bar, val in zip(bars, metrics.values()):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

        # 4. Statistics summary
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        stats_text = "Reading Statistics\n" + "=" * 30 + "\n\n"
        stats_text += f"Forward Saccades:   {reading_results.get('forward_saccade_count', 0)}\n"
        stats_text += f"Regressions:        {reading_results.get('regression_count', 0)}\n"
        stats_text += f"Line Returns:       {reading_results.get('line_return_count', 0)}\n"
        stats_text += f"Refixations:        {reading_results.get('refixation_count', 0)}\n\n"
        stats_text += f"Regression Rate:    {reading_results.get('regression_rate', 0)*100:.1f}%\n"
        stats_text += f"Reading Efficiency: {reading_results.get('reading_efficiency', 0)*100:.1f}%\n"

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # =========================================================================
    # AOI TRANSITION VISUALIZATION
    # =========================================================================

    def plot_aoi_transition_matrix(self, transition_matrix: np.ndarray,
                                    aoi_labels: List[str],
                                    title: str = "AOI Transition Matrix",
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot AOI transition matrix as a heatmap.

        Args:
            transition_matrix: Transition matrix from TobiiAnalyzer.compute_aoi_transition_matrix()
            aoi_labels: List of AOI names
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(transition_matrix, cmap="YlOrRd")

        # Add labels
        ax.set_xticks(range(len(aoi_labels)))
        ax.set_yticks(range(len(aoi_labels)))
        ax.set_xticklabels(aoi_labels, rotation=45, ha="right")
        ax.set_yticklabels(aoi_labels)

        # Add value annotations
        for i in range(len(aoi_labels)):
            for j in range(len(aoi_labels)):
                value = transition_matrix[i, j]
                text_color = "white" if value > 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                       color=text_color, fontsize=9)

        ax.set_xlabel("To AOI", fontsize=11)
        ax.set_ylabel("From AOI", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Transition Probability")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_aoi_sequence_diagram(self, aoi_sequence: List[str], timestamps: Optional[List[float]] = None,
                                   title: str = "AOI Viewing Sequence",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot AOI viewing sequence as a timeline diagram.

        Args:
            aoi_sequence: List of AOI names in viewing order
            timestamps: Optional list of timestamps for each AOI visit
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 4))

        unique_aois = list(dict.fromkeys(aoi_sequence))  # Preserve order
        aoi_to_y = {aoi: i for i, aoi in enumerate(unique_aois)}
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_aois)))
        aoi_colors = {aoi: colors[i] for i, aoi in enumerate(unique_aois)}

        # Create x-axis values
        if timestamps:
            x_vals = timestamps
            x_label = "Time (s)"
        else:
            x_vals = list(range(len(aoi_sequence)))
            x_label = "Fixation Number"

        # Plot sequence as horizontal bars/segments
        for i, (x, aoi) in enumerate(zip(x_vals, aoi_sequence)):
            y = aoi_to_y[aoi]
            ax.barh(y, 1, left=x, color=aoi_colors[aoi], edgecolor="black", height=0.6)

        # Connect with lines
        for i in range(len(aoi_sequence) - 1):
            y1 = aoi_to_y[aoi_sequence[i]]
            y2 = aoi_to_y[aoi_sequence[i + 1]]
            x = x_vals[i] + 0.5
            ax.plot([x, x + 0.5], [y1, y2], color="gray", linewidth=0.5, alpha=0.5)

        ax.set_yticks(range(len(unique_aois)))
        ax.set_yticklabels(unique_aois)
        ax.set_xlabel(x_label)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # =========================================================================
    # SACCADE ANALYSIS VISUALIZATIONS
    # =========================================================================

    def plot_saccade_polar(self, saccades: List,
                            title: str = "Saccade Direction Distribution",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot saccade directions on a polar plot with amplitude encoding.

        Args:
            saccades: List of Saccade objects
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="polar")

        if not saccades:
            ax.set_title("No saccades detected")
            return fig

        # Calculate directions
        directions = []
        amplitudes = []
        for s in saccades:
            dx = s.end_x - s.start_x
            dy = s.end_y - s.start_y
            direction = np.arctan2(dy, dx)
            amplitude = np.sqrt(dx**2 + dy**2)
            directions.append(direction)
            amplitudes.append(amplitude)

        # Scale amplitudes for visualization
        amp_scaled = np.array(amplitudes) * self.screen_size[0]  # Convert to pixels

        # Scatter plot with amplitude as radius
        scatter = ax.scatter(directions, amp_scaled, c=range(len(saccades)),
                            cmap="viridis", alpha=0.6, s=30)

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

        # Add colorbar for temporal order
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Saccade Order")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_main_sequence(self, saccades: List,
                            title: str = "Saccade Main Sequence",
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot the main sequence (amplitude vs peak velocity relationship).

        The main sequence is a fundamental relationship in oculomotor research.

        Args:
            saccades: List of Saccade objects
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if not saccades:
            ax.set_title("No saccades detected")
            return fig

        amplitudes = [s.amplitude * self.screen_size[0] for s in saccades]  # Convert to pixels
        velocities = [s.velocity * self.screen_size[0] for s in saccades]  # Convert to pixels/s
        durations = [s.duration * 1000 for s in saccades]  # Convert to ms

        # Scatter plot with duration as color
        scatter = ax.scatter(amplitudes, velocities, c=durations, cmap="plasma",
                            alpha=0.6, s=50, edgecolors="white", linewidths=0.5)

        # Fit power law: V = k * A^n (typically n ≈ 0.5-0.7)
        if len(amplitudes) > 3:
            try:
                from scipy.optimize import curve_fit

                def power_law(x, k, n):
                    return k * np.power(x, n)

                amp_arr = np.array(amplitudes)
                vel_arr = np.array(velocities)
                valid = (amp_arr > 0) & (vel_arr > 0)

                if np.sum(valid) > 3:
                    popt, _ = curve_fit(power_law, amp_arr[valid], vel_arr[valid],
                                        p0=[100, 0.5], maxfev=5000)
                    x_fit = np.linspace(min(amp_arr[valid]), max(amp_arr[valid]), 100)
                    y_fit = power_law(x_fit, *popt)
                    ax.plot(x_fit, y_fit, "r--", linewidth=2,
                           label=f"Fit: V = {popt[0]:.1f} × A^{popt[1]:.2f}")
                    ax.legend()
            except Exception:
                pass  # Skip fitting if it fails

        ax.set_xlabel("Saccade Amplitude (pixels)", fontsize=11)
        ax.set_ylabel("Saccade Velocity (pixels/s)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Duration (ms)")

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_saccade_amplitude_histogram(self, saccades: List,
                                          bins: int = 30,
                                          title: str = "Saccade Amplitude Distribution",
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot histogram of saccade amplitudes.

        Args:
            saccades: List of Saccade objects
            bins: Number of histogram bins
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        amplitudes = [s.amplitude * self.screen_size[0] for s in saccades]  # Convert to pixels

        ax.hist(amplitudes, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(np.mean(amplitudes), color="red", linestyle="--",
                  label=f"Mean: {np.mean(amplitudes):.1f} px")
        ax.axvline(np.median(amplitudes), color="green", linestyle="--",
                  label=f"Median: {np.median(amplitudes):.1f} px")

        ax.set_xlabel("Saccade Amplitude (pixels)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # =========================================================================
    # COMPARATIVE SCANPATH VISUALIZATION
    # =========================================================================

    def plot_scanpath_comparison(self, fixations1: List, fixations2: List,
                                  multimatch_results: Optional[Dict] = None,
                                  labels: Tuple[str, str] = ("Scanpath 1", "Scanpath 2"),
                                  title: str = "Scanpath Comparison",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot two scanpaths side by side for comparison.

        Args:
            fixations1: First list of Fixation objects
            fixations2: Second list of Fixation objects
            multimatch_results: Optional MultiMatch similarity results
            labels: Labels for the two scanpaths
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Scanpath 1
        ax1 = fig.add_subplot(1, 2, 1)
        if self.background_image is not None:
            ax1.imshow(self.background_image)

        fix1_x = [f.x * self.screen_size[0] for f in fixations1]
        fix1_y = [f.y * self.screen_size[1] for f in fixations1]
        fix1_dur = [f.duration for f in fixations1]

        ax1.plot(fix1_x, fix1_y, "b-", alpha=0.5, linewidth=1)
        ax1.scatter(fix1_x, fix1_y, s=[d * 500 for d in fix1_dur],
                   c=range(len(fixations1)), cmap="Blues", alpha=0.7,
                   edgecolors="white", linewidths=0.5)
        ax1.set_xlim(0, self.screen_size[0])
        ax1.set_ylim(self.screen_size[1], 0)
        ax1.set_title(labels[0])

        # Scanpath 2
        ax2 = fig.add_subplot(1, 2, 2)
        if self.background_image is not None:
            ax2.imshow(self.background_image)

        fix2_x = [f.x * self.screen_size[0] for f in fixations2]
        fix2_y = [f.y * self.screen_size[1] for f in fixations2]
        fix2_dur = [f.duration for f in fixations2]

        ax2.plot(fix2_x, fix2_y, "r-", alpha=0.5, linewidth=1)
        ax2.scatter(fix2_x, fix2_y, s=[d * 500 for d in fix2_dur],
                   c=range(len(fixations2)), cmap="Reds", alpha=0.7,
                   edgecolors="white", linewidths=0.5)
        ax2.set_xlim(0, self.screen_size[0])
        ax2.set_ylim(self.screen_size[1], 0)
        ax2.set_title(labels[1])

        # Add MultiMatch results if available
        if multimatch_results:
            text = "MultiMatch Similarity\n" + "-" * 25 + "\n"
            for key, value in multimatch_results.items():
                if isinstance(value, float) and not np.isnan(value):
                    text += f"{key}: {value:.3f}\n"
            fig.text(0.5, 0.02, text, ha="center", fontsize=10,
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_dtw_alignment(self, fixations1: List, fixations2: List,
                            dtw_results: Dict,
                            title: str = "DTW Scanpath Alignment",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize Dynamic Time Warping alignment between two scanpaths.

        Args:
            fixations1: First list of Fixation objects
            fixations2: Second list of Fixation objects
            dtw_results: Results from TobiiAnalyzer.compute_dtw_similarity()
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Left: Spatial alignment
        ax1 = axes[0]
        if self.background_image is not None:
            ax1.imshow(self.background_image, alpha=0.3)

        fix1_x = [f.x * self.screen_size[0] for f in fixations1]
        fix1_y = [f.y * self.screen_size[1] for f in fixations1]
        fix2_x = [f.x * self.screen_size[0] for f in fixations2]
        fix2_y = [f.y * self.screen_size[1] for f in fixations2]

        # Draw alignment lines from warping path
        warping_path = dtw_results.get("warping_path", [])
        for i, j in warping_path[::max(1, len(warping_path)//20)]:  # Sample for clarity
            if i < len(fix1_x) and j < len(fix2_x):
                ax1.plot([fix1_x[i], fix2_x[j]], [fix1_y[i], fix2_y[j]],
                        "g-", alpha=0.3, linewidth=0.5)

        # Draw scanpaths
        ax1.plot(fix1_x, fix1_y, "b-", linewidth=2, alpha=0.7, label="Scanpath 1")
        ax1.plot(fix2_x, fix2_y, "r-", linewidth=2, alpha=0.7, label="Scanpath 2")
        ax1.scatter(fix1_x, fix1_y, c="blue", s=50, zorder=3, edgecolors="white")
        ax1.scatter(fix2_x, fix2_y, c="red", s=50, zorder=3, edgecolors="white")

        ax1.set_xlim(0, self.screen_size[0])
        ax1.set_ylim(self.screen_size[1], 0)
        ax1.set_title("Spatial Alignment")
        ax1.legend()

        # Right: DTW cost matrix with path
        ax2 = axes[1]

        n, m = len(fixations1), len(fixations2)
        cost_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                f1, f2 = fixations1[i], fixations2[j]
                cost_matrix[i, j] = np.sqrt((f1.x - f2.x)**2 + (f1.y - f2.y)**2)

        im = ax2.imshow(cost_matrix, cmap="viridis", aspect="auto", origin="lower")

        # Draw warping path
        if warping_path:
            path_i = [p[0] for p in warping_path]
            path_j = [p[1] for p in warping_path]
            ax2.plot(path_j, path_i, "r-", linewidth=2, label="Warping Path")

        ax2.set_xlabel("Scanpath 2 Index")
        ax2.set_ylabel("Scanpath 1 Index")
        ax2.set_title(f"DTW Cost Matrix\nSimilarity: {dtw_results.get('dtw_similarity', 0):.3f}")

        plt.colorbar(im, ax=ax2, label="Distance")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # =========================================================================
    # DATA QUALITY VISUALIZATION
    # =========================================================================

    def plot_data_quality_dashboard(self, quality_metrics: Dict,
                                     data: Optional[pd.DataFrame] = None,
                                     title: str = "Eye Tracking Data Quality",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive data quality visualization dashboard.

        Args:
            quality_metrics: Results from TobiiAnalyzer.compute_data_quality_metrics()
            data: Optional raw data for additional visualizations
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # 1. Validity rates bar chart
        ax1 = fig.add_subplot(2, 3, 1)
        validity_metrics = {
            "Left Eye": quality_metrics.get("left_eye_validity", 0) * 100,
            "Right Eye": quality_metrics.get("right_eye_validity", 0) * 100,
            "Binocular": quality_metrics.get("binocular_validity", 0) * 100,
            "Overall": quality_metrics.get("overall_validity", 0) * 100,
        }
        colors = ["steelblue", "coral", "green", "purple"]
        bars = ax1.bar(validity_metrics.keys(), validity_metrics.values(), color=colors)
        ax1.set_ylabel("Validity (%)")
        ax1.set_title("Gaze Validity Rates")
        ax1.set_ylim(0, 100)
        ax1.axhline(80, color="gray", linestyle="--", alpha=0.5, label="80% threshold")

        for bar, val in zip(bars, validity_metrics.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

        # 2. Quality grade gauge
        ax2 = fig.add_subplot(2, 3, 2)
        overall = quality_metrics.get("overall_validity", 0)
        if overall >= 0.9:
            grade, color = "Excellent", "green"
        elif overall >= 0.8:
            grade, color = "Good", "yellowgreen"
        elif overall >= 0.7:
            grade, color = "Acceptable", "orange"
        else:
            grade, color = "Poor", "red"

        ax2.pie([overall, 1-overall], colors=[color, "lightgray"],
               startangle=90, counterclock=False)
        ax2.text(0, 0, f"{grade}\n{overall*100:.1f}%", ha="center", va="center",
                fontsize=14, fontweight="bold")
        ax2.set_title("Overall Quality Grade")

        # 3. Gap analysis
        ax3 = fig.add_subplot(2, 3, 3)
        gap_data = [
            quality_metrics.get("gap_count", 0),
            quality_metrics.get("mean_gap_length", 0),
            quality_metrics.get("max_gap_length", 0),
        ]
        gap_labels = ["Gap Count", "Mean Gap\n(samples)", "Max Gap\n(samples)"]
        bars = ax3.bar(gap_labels, gap_data, color="salmon", edgecolor="black")
        ax3.set_title("Data Gap Analysis")

        for bar, val in zip(bars, gap_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

        # 4. Sampling rate info
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.axis("off")
        sampling_info = f"""
Sampling Information
{'='*30}

Sampling Rate:    {quality_metrics.get('sampling_rate', 'N/A'):.1f} Hz
Sampling Jitter:  {quality_metrics.get('sampling_jitter', 'N/A'):.4f}
Total Samples:    {quality_metrics.get('total_samples', 'N/A'):,}
Precision (RMS):  {quality_metrics.get('precision_rms', 'N/A'):.6f}
        """
        ax4.text(0.1, 0.9, sampling_info, transform=ax4.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))

        # 5. Validity over time (if data available)
        ax5 = fig.add_subplot(2, 3, 5)
        if data is not None and "recording_timestamp" in data.columns:
            # Calculate rolling validity
            window_size = max(1, len(data) // 100)

            if "gaze_valid" in data.columns:
                validity = data["gaze_valid"].astype(float)
            else:
                validity = ((data.get("left_gaze_valid", 0) == 1) |
                           (data.get("right_gaze_valid", 0) == 1)).astype(float)

            rolling_validity = validity.rolling(window=window_size, min_periods=1).mean() * 100
            time = data["recording_timestamp"].values

            ax5.plot(time, rolling_validity, linewidth=1, color="steelblue")
            ax5.fill_between(time, rolling_validity, alpha=0.3)
            ax5.axhline(80, color="red", linestyle="--", alpha=0.5)
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Validity (%)")
            ax5.set_title("Validity Over Time")
            ax5.set_ylim(0, 100)
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No temporal data available",
                    ha="center", va="center", transform=ax5.transAxes)
            ax5.set_title("Validity Over Time")

        # 6. Summary recommendations
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis("off")

        recommendations = []
        if overall < 0.7:
            recommendations.append("⚠ Low validity: Check calibration and participant positioning")
        if quality_metrics.get("max_gap_length", 0) > 50:
            recommendations.append("⚠ Large gaps detected: Consider interpolation or segment removal")
        if quality_metrics.get("sampling_jitter", 0) > 0.1:
            recommendations.append("⚠ High jitter: Check hardware performance")
        if not recommendations:
            recommendations.append("✓ Data quality is acceptable for analysis")

        rec_text = "Recommendations\n" + "=" * 30 + "\n\n"
        rec_text += "\n".join(recommendations)

        ax6.text(0.1, 0.9, rec_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # =========================================================================
    # ENHANCED HEATMAP VISUALIZATIONS
    # =========================================================================

    def plot_contour_heatmap(self, heatmap: np.ndarray,
                              title: str = "Gaze Density Contours",
                              levels: int = 10,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot heatmap as contour lines for cleaner visualization.

        Args:
            heatmap: 2D numpy array from TobiiAnalyzer.generate_heatmap()
            title: Plot title
            levels: Number of contour levels
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if self.background_image is not None:
            ax.imshow(self.background_image)

        # Create coordinate grids
        y, x = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
        x = x / heatmap.shape[1] * self.screen_size[0]
        y = y / heatmap.shape[0] * self.screen_size[1]

        # Plot contours
        contour = ax.contour(x, y, heatmap, levels=levels, cmap="hot", linewidths=1.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

        # Add filled contours with transparency
        ax.contourf(x, y, heatmap, levels=levels, cmap="hot", alpha=0.3)

        ax.set_xlim(0, self.screen_size[0])
        ax.set_ylim(self.screen_size[1], 0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_3d_heatmap(self, heatmap: np.ndarray,
                         title: str = "3D Gaze Density Surface",
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot heatmap as 3D surface.

        Args:
            heatmap: 2D numpy array from TobiiAnalyzer.generate_heatmap()
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Create coordinate grids
        y, x = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
        x = x / heatmap.shape[1] * self.screen_size[0]
        y = y / heatmap.shape[0] * self.screen_size[1]

        # Plot surface
        surf = ax.plot_surface(x, y, heatmap, cmap="hot", linewidth=0,
                               antialiased=True, alpha=0.8)

        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_zlabel("Gaze Density")
        ax.set_title(title, fontsize=12, fontweight="bold")

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Density")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # =========================================================================
    # BLINK VISUALIZATION
    # =========================================================================

    def plot_blink_timeline(self, blinks: List[Dict], data: pd.DataFrame,
                             title: str = "Blink Detection Timeline",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot blink events on a timeline with pupil data.

        Args:
            blinks: List of blink events from TobiiAnalyzer.detect_blinks()
            data: Raw data DataFrame with pupil information
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 5))

        # Plot pupil diameter
        if "recording_timestamp" in data.columns and "pupil_diameter" in data.columns:
            time = data["recording_timestamp"].values
            pupil = data["pupil_diameter"].values
            valid = ~np.isnan(pupil)
            ax.plot(time[valid], pupil[valid], linewidth=0.5, alpha=0.8,
                   color="purple", label="Pupil Diameter")

        # Highlight blink regions
        for blink in blinks:
            ax.axvspan(blink["start_time"], blink["end_time"],
                      color="red", alpha=0.3, label="Blink" if blink == blinks[0] else "")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pupil Diameter (mm)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Add blink count annotation
        ax.text(0.02, 0.98, f"Total Blinks: {len(blinks)}",
               transform=ax.transAxes, fontsize=10,
               verticalalignment="top",
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
