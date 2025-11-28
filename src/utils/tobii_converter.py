"""
Tobii Data Format Converter

Converts various Tobii export formats to the HCI Lab ToolKit standard format.
Supports:
- Tobii Pro Lab exports (.tsv, .txt)
- Tobii Studio exports
- Tobii Pro SDK raw exports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple


class TobiiConverter:
    """
    Converts Tobii eye tracking data to the HCI Lab ToolKit standard format.

    Standard output columns:
    - device_timestamp: Eye tracker clock (microseconds)
    - system_timestamp: System clock (microseconds)
    - recording_timestamp: Time since start (seconds)
    - left_gaze_x/y: Left eye gaze (normalized 0-1)
    - right_gaze_x/y: Right eye gaze (normalized 0-1)
    - left/right_gaze_valid: Validity flags (0 or 1)
    - left/right_pupil_diameter: Pupil size (mm)
    - left/right_pupil_valid: Validity flags
    - left/right_origin_x/y/z: 3D eye position (mm)
    - left/right_origin_valid: Validity flags
    """

    # Column mapping from Tobii Pro Lab export to standard format
    TOBII_PRO_LAB_MAPPING = {
        # Timestamps
        "Recording timestamp": "recording_timestamp_us",
        "Eyetracker timestamp": "device_timestamp",

        # Left eye gaze (2D normalized)
        "Gaze2d_Left.x": "left_gaze_x_raw",
        "Gaze2d_Left.y": "left_gaze_y_raw",

        # Right eye gaze (2D normalized)
        "Gaze2d_Right.x": "right_gaze_x_raw",
        "Gaze2d_Right.y": "right_gaze_y_raw",

        # Pupil diameter
        "PupilDiam_Left": "left_pupil_diameter",
        "PupilDiam_Right": "right_pupil_diameter",

        # Validity
        "Validity_Left": "left_validity_raw",
        "Validity_Right": "right_validity_raw",

        # 3D eye position (absolute)
        "Eyepos3d_Left.x": "left_origin_x",
        "Eyepos3d_Left.y": "left_origin_y",
        "Eyepos3d_Left.z": "left_origin_z",
        "Eyepos3d_Right.x": "right_origin_x",
        "Eyepos3d_Right.y": "right_origin_y",
        "Eyepos3d_Right.z": "right_origin_z",

        # 3D gaze point
        "Gaze3d_Left.x": "left_gaze3d_x",
        "Gaze3d_Left.y": "left_gaze3d_y",
        "Gaze3d_Left.z": "left_gaze3d_z",
        "Gaze3d_Right.x": "right_gaze3d_x",
        "Gaze3d_Right.y": "right_gaze3d_y",
        "Gaze3d_Right.z": "right_gaze3d_z",

        # Events
        "Event value": "event_value",
        "Event message": "event_message",
    }

    def __init__(self, screen_size: Tuple[int, int] = (1920, 1080)):
        """
        Initialize converter.

        Args:
            screen_size: Screen resolution (width, height) for coordinate conversion
        """
        self.screen_size = screen_size

    def detect_format(self, filepath: str) -> str:
        """
        Detect the Tobii export format.

        Args:
            filepath: Path to data file

        Returns:
            Format identifier string
        """
        filepath = Path(filepath)

        # Read first few lines to detect format
        with open(filepath, 'r') as f:
            first_line = f.readline()

        # Check for tab-separated (Tobii Pro Lab)
        if '\t' in first_line:
            if 'Gaze2d_Left' in first_line:
                return 'tobii_pro_lab'
            elif 'Gaze point X' in first_line and 'Validity left' in first_line:
                return 'tobii_pro_lab_v2'  # Newer Tobii Pro Lab export format
            elif 'Eyetracker timestamp' in first_line:
                return 'tobii_pro_lab'
            elif 'GazePointX' in first_line:
                return 'tobii_studio'

        # Check for comma-separated (SDK export)
        if ',' in first_line:
            if 'left_gaze_x' in first_line:
                return 'hci_toolkit'  # Already in our format
            elif 'device_time_stamp' in first_line:
                return 'tobii_sdk'

        return 'unknown'

    def convert_tobii_pro_lab(self, filepath: str,
                               output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Convert Tobii Pro Lab export (.tsv/.txt) to standard format.

        Args:
            filepath: Path to input file
            output_path: Optional path to save converted file

        Returns:
            Converted DataFrame
        """
        filepath = Path(filepath)

        # Read tab-separated file
        df = pd.read_csv(filepath, sep='\t', low_memory=False)

        print(f"Loaded {len(df)} rows from {filepath.name}")
        print(f"Original columns: {list(df.columns)}")

        # Create output dataframe
        output = pd.DataFrame()

        # Process timestamps
        if 'Recording timestamp' in df.columns:
            # Convert from microseconds to seconds for recording_timestamp
            output['recording_timestamp'] = df['Recording timestamp'] / 1_000_000
            # Normalize to start from 0
            output['recording_timestamp'] = output['recording_timestamp'] - output['recording_timestamp'].iloc[0]

        if 'Eyetracker timestamp' in df.columns:
            output['device_timestamp'] = df['Eyetracker timestamp']
            output['system_timestamp'] = df['Eyetracker timestamp']  # Use same if no separate system timestamp

        # Process gaze data - need to convert from pixel to normalized
        # Tobii Pro Lab exports gaze in pixels, we need 0-1 range
        if 'Gaze2d_Left.x' in df.columns:
            # Check if values are in pixel range or already normalized
            max_gaze_x = df['Gaze2d_Left.x'].replace(0, np.nan).max()

            if max_gaze_x > 1.5:  # Likely pixel coordinates
                output['left_gaze_x'] = df['Gaze2d_Left.x'] / self.screen_size[0]
                output['left_gaze_y'] = df['Gaze2d_Left.y'] / self.screen_size[1]
                output['right_gaze_x'] = df['Gaze2d_Right.x'] / self.screen_size[0]
                output['right_gaze_y'] = df['Gaze2d_Right.y'] / self.screen_size[1]
            else:
                # Already normalized or very small screen
                output['left_gaze_x'] = df['Gaze2d_Left.x']
                output['left_gaze_y'] = df['Gaze2d_Left.y']
                output['right_gaze_x'] = df['Gaze2d_Right.x']
                output['right_gaze_y'] = df['Gaze2d_Right.y']

        # Process validity
        # Tobii uses 0=valid, 4=invalid (or other non-zero values)
        # We use 1=valid, 0=invalid
        if 'Validity_Left' in df.columns:
            output['left_gaze_valid'] = (df['Validity_Left'] == 0).astype(int)
            output['right_gaze_valid'] = (df['Validity_Right'] == 0).astype(int)
            output['left_pupil_valid'] = (df['Validity_Left'] == 0).astype(int)
            output['right_pupil_valid'] = (df['Validity_Right'] == 0).astype(int)
            output['left_origin_valid'] = (df['Validity_Left'] == 0).astype(int)
            output['right_origin_valid'] = (df['Validity_Right'] == 0).astype(int)

        # Pupil diameter
        if 'PupilDiam_Left' in df.columns:
            output['left_pupil_diameter'] = df['PupilDiam_Left']
            output['right_pupil_diameter'] = df['PupilDiam_Right']

        # 3D eye position (origin)
        if 'Eyepos3d_Left.x' in df.columns:
            output['left_origin_x'] = df['Eyepos3d_Left.x']
            output['left_origin_y'] = df['Eyepos3d_Left.y']
            output['left_origin_z'] = df['Eyepos3d_Left.z']
            output['right_origin_x'] = df['Eyepos3d_Right.x']
            output['right_origin_y'] = df['Eyepos3d_Right.y']
            output['right_origin_z'] = df['Eyepos3d_Right.z']

        # Events (if present)
        if 'Event value' in df.columns:
            output['event_value'] = df['Event value']
        if 'Event message' in df.columns:
            output['event_message'] = df['Event message']

        # Clean up invalid samples (set to NaN where validity is 0)
        for col in ['left_gaze_x', 'left_gaze_y', 'left_pupil_diameter']:
            if col in output.columns and 'left_gaze_valid' in output.columns:
                output.loc[output['left_gaze_valid'] == 0, col] = np.nan

        for col in ['right_gaze_x', 'right_gaze_y', 'right_pupil_diameter']:
            if col in output.columns and 'right_gaze_valid' in output.columns:
                output.loc[output['right_gaze_valid'] == 0, col] = np.nan

        print(f"Converted to {len(output)} rows with columns: {list(output.columns)}")

        # Calculate statistics
        valid_left = output['left_gaze_valid'].sum() if 'left_gaze_valid' in output.columns else 0
        valid_right = output['right_gaze_valid'].sum() if 'right_gaze_valid' in output.columns else 0
        print(f"Valid samples - Left: {valid_left} ({100*valid_left/len(output):.1f}%), "
              f"Right: {valid_right} ({100*valid_right/len(output):.1f}%)")

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            if output_path.suffix == '.parquet':
                output.to_parquet(output_path, index=False)
            else:
                output.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")

        return output

    def convert_tobii_pro_lab_v2(self, filepath: str,
                                  output_path: Optional[str] = None,
                                  filter_eye_tracker_only: bool = True) -> pd.DataFrame:
        """
        Convert newer Tobii Pro Lab export format (.tsv) to standard format.

        This format uses columns like:
        - 'Gaze point X', 'Gaze point Y' (pixels)
        - 'Gaze point left X', 'Gaze point left Y' (pixels)
        - 'Validity left', 'Validity right' (text: "Valid"/"Invalid")
        - 'Pupil diameter left', 'Pupil diameter right' (mm)
        - 'Eye position left X (DACSmm)', etc.

        Args:
            filepath: Path to input file
            output_path: Optional path to save converted file
            filter_eye_tracker_only: If True, only keep Eye Tracker sensor rows

        Returns:
            Converted DataFrame
        """
        filepath = Path(filepath)

        # Read tab-separated file
        df = pd.read_csv(filepath, sep='\t', low_memory=False)

        print(f"Loaded {len(df)} rows from {filepath.name}")

        # Filter to only Eye Tracker sensor data if requested
        if filter_eye_tracker_only and 'Sensor' in df.columns:
            df = df[df['Sensor'] == 'Eye Tracker'].copy()
            print(f"Filtered to {len(df)} Eye Tracker rows")

        # Get screen size from data if available
        if 'Recording resolution width' in df.columns:
            screen_width = df['Recording resolution width'].iloc[0]
            screen_height = df['Recording resolution height'].iloc[0]
            if pd.notna(screen_width) and pd.notna(screen_height):
                self.screen_size = (int(screen_width), int(screen_height))
                print(f"Detected screen size: {self.screen_size}")

        # Create output dataframe
        output = pd.DataFrame()

        # Process timestamps
        if 'Recording timestamp' in df.columns:
            # Recording timestamp is in microseconds
            output['recording_timestamp'] = df['Recording timestamp'].astype(float) / 1_000_000
            # Normalize to start from 0
            output['recording_timestamp'] = output['recording_timestamp'] - output['recording_timestamp'].iloc[0]

        if 'Eyetracker timestamp' in df.columns:
            output['device_timestamp'] = pd.to_numeric(df['Eyetracker timestamp'], errors='coerce')
            output['system_timestamp'] = output['device_timestamp']

        # Process gaze data - convert from pixels to normalized (0-1)
        # Combined gaze point
        if 'Gaze point X' in df.columns:
            gaze_x = pd.to_numeric(df['Gaze point X'], errors='coerce')
            gaze_y = pd.to_numeric(df['Gaze point Y'], errors='coerce')

            # Check if already normalized or in pixels
            max_x = gaze_x.max()
            if max_x > 1.5:  # Pixel coordinates
                output['gaze_x'] = gaze_x / self.screen_size[0]
                output['gaze_y'] = gaze_y / self.screen_size[1]
            else:
                output['gaze_x'] = gaze_x
                output['gaze_y'] = gaze_y

        # Left eye gaze
        if 'Gaze point left X' in df.columns:
            left_x = pd.to_numeric(df['Gaze point left X'], errors='coerce')
            left_y = pd.to_numeric(df['Gaze point left Y'], errors='coerce')

            max_left_x = left_x.max()
            if max_left_x > 1.5:  # Pixel coordinates
                output['left_gaze_x'] = left_x / self.screen_size[0]
                output['left_gaze_y'] = left_y / self.screen_size[1]
            else:
                output['left_gaze_x'] = left_x
                output['left_gaze_y'] = left_y

        # Right eye gaze
        if 'Gaze point right X' in df.columns:
            right_x = pd.to_numeric(df['Gaze point right X'], errors='coerce')
            right_y = pd.to_numeric(df['Gaze point right Y'], errors='coerce')

            max_right_x = right_x.max()
            if max_right_x > 1.5:  # Pixel coordinates
                output['right_gaze_x'] = right_x / self.screen_size[0]
                output['right_gaze_y'] = right_y / self.screen_size[1]
            else:
                output['right_gaze_x'] = right_x
                output['right_gaze_y'] = right_y

        # Process validity - convert from "Valid"/"Invalid" text to 1/0
        if 'Validity left' in df.columns:
            output['left_gaze_valid'] = (df['Validity left'] == 'Valid').astype(int)
            output['left_pupil_valid'] = output['left_gaze_valid']
            output['left_origin_valid'] = output['left_gaze_valid']

        if 'Validity right' in df.columns:
            output['right_gaze_valid'] = (df['Validity right'] == 'Valid').astype(int)
            output['right_pupil_valid'] = output['right_gaze_valid']
            output['right_origin_valid'] = output['right_gaze_valid']

        # Pupil diameter (already in mm)
        if 'Pupil diameter left' in df.columns:
            output['left_pupil_diameter'] = pd.to_numeric(df['Pupil diameter left'], errors='coerce')
        if 'Pupil diameter right' in df.columns:
            output['right_pupil_diameter'] = pd.to_numeric(df['Pupil diameter right'], errors='coerce')

        # 3D eye position (origin) - in DACSmm
        if 'Eye position left X (DACSmm)' in df.columns:
            output['left_origin_x'] = pd.to_numeric(df['Eye position left X (DACSmm)'], errors='coerce')
            output['left_origin_y'] = pd.to_numeric(df['Eye position left Y (DACSmm)'], errors='coerce')
            output['left_origin_z'] = pd.to_numeric(df['Eye position left Z (DACSmm)'], errors='coerce')

        if 'Eye position right X (DACSmm)' in df.columns:
            output['right_origin_x'] = pd.to_numeric(df['Eye position right X (DACSmm)'], errors='coerce')
            output['right_origin_y'] = pd.to_numeric(df['Eye position right Y (DACSmm)'], errors='coerce')
            output['right_origin_z'] = pd.to_numeric(df['Eye position right Z (DACSmm)'], errors='coerce')

        # Gaze direction vectors (useful for 3D gaze analysis)
        if 'Gaze direction left X' in df.columns:
            output['left_gaze_direction_x'] = pd.to_numeric(df['Gaze direction left X'], errors='coerce')
            output['left_gaze_direction_y'] = pd.to_numeric(df['Gaze direction left Y'], errors='coerce')
            output['left_gaze_direction_z'] = pd.to_numeric(df['Gaze direction left Z'], errors='coerce')

        if 'Gaze direction right X' in df.columns:
            output['right_gaze_direction_x'] = pd.to_numeric(df['Gaze direction right X'], errors='coerce')
            output['right_gaze_direction_y'] = pd.to_numeric(df['Gaze direction right Y'], errors='coerce')
            output['right_gaze_direction_z'] = pd.to_numeric(df['Gaze direction right Z'], errors='coerce')

        # Eye openness (useful for blink detection)
        if 'Eye openness left' in df.columns:
            output['left_eye_openness'] = pd.to_numeric(df['Eye openness left'], errors='coerce')
        if 'Eye openness right' in df.columns:
            output['right_eye_openness'] = pd.to_numeric(df['Eye openness right'], errors='coerce')

        # Eye movement type (Tobii's classification)
        if 'Eye movement type' in df.columns:
            output['eye_movement_type'] = df['Eye movement type']
        if 'Eye movement type index' in df.columns:
            output['eye_movement_index'] = pd.to_numeric(df['Eye movement type index'], errors='coerce')

        # Fixation data (if available)
        if 'Fixation point X' in df.columns:
            fix_x = pd.to_numeric(df['Fixation point X'], errors='coerce')
            fix_y = pd.to_numeric(df['Fixation point Y'], errors='coerce')
            output['fixation_x'] = fix_x / self.screen_size[0]
            output['fixation_y'] = fix_y / self.screen_size[1]

        # Events
        if 'Event' in df.columns:
            output['event'] = df['Event']
        if 'Event value' in df.columns:
            output['event_value'] = df['Event value']

        # Stimulus info
        if 'Presented Stimulus name' in df.columns:
            output['stimulus_name'] = df['Presented Stimulus name']
        if 'Presented Media name' in df.columns:
            output['media_name'] = df['Presented Media name']

        # Clean up invalid samples (set gaze to NaN where validity is 0)
        if 'left_gaze_valid' in output.columns:
            for col in ['left_gaze_x', 'left_gaze_y', 'left_pupil_diameter']:
                if col in output.columns:
                    output.loc[output['left_gaze_valid'] == 0, col] = np.nan

        if 'right_gaze_valid' in output.columns:
            for col in ['right_gaze_x', 'right_gaze_y', 'right_pupil_diameter']:
                if col in output.columns:
                    output.loc[output['right_gaze_valid'] == 0, col] = np.nan

        print(f"Converted to {len(output)} rows with columns: {list(output.columns)}")

        # Calculate statistics
        valid_left = output['left_gaze_valid'].sum() if 'left_gaze_valid' in output.columns else 0
        valid_right = output['right_gaze_valid'].sum() if 'right_gaze_valid' in output.columns else 0
        total = len(output)
        print(f"Valid samples - Left: {valid_left} ({100*valid_left/total:.1f}%), "
              f"Right: {valid_right} ({100*valid_right/total:.1f}%)")

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix == '.parquet':
                output.to_parquet(output_path, index=False)
            else:
                output.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")

        return output

    def convert(self, filepath: str,
                output_path: Optional[str] = None,
                format: Optional[str] = None) -> pd.DataFrame:
        """
        Auto-detect format and convert to standard format.

        Args:
            filepath: Path to input file
            output_path: Optional path to save converted file
            format: Force specific format (auto-detect if None)

        Returns:
            Converted DataFrame
        """
        filepath = Path(filepath)

        if format is None:
            format = self.detect_format(filepath)
            print(f"Detected format: {format}")

        if format == 'tobii_pro_lab':
            return self.convert_tobii_pro_lab(filepath, output_path)
        elif format == 'tobii_pro_lab_v2':
            return self.convert_tobii_pro_lab_v2(filepath, output_path)
        elif format == 'hci_toolkit':
            print("File is already in HCI ToolKit format")
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported or unknown format: {format}")

    def batch_convert(self, input_dir: str, output_dir: str,
                      pattern: str = "*.txt") -> Dict[str, pd.DataFrame]:
        """
        Convert all files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            pattern: Glob pattern for input files

        Returns:
            Dictionary mapping filenames to DataFrames
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        files = list(input_dir.glob(pattern))

        print(f"Found {len(files)} files to convert")

        for filepath in files:
            try:
                output_path = output_dir / f"{filepath.stem}_converted.csv"
                df = self.convert(filepath, output_path)
                results[filepath.name] = df
                print(f"  ✓ {filepath.name}")
            except Exception as e:
                print(f"  ✗ {filepath.name}: {e}")
                results[filepath.name] = None

        return results


def convert_sceneviewing_data(input_path: str = None, output_path: str = None):
    """
    Convenience function to convert the sceneviewing_tobii example data.

    Args:
        input_path: Path to input file (default: example data)
        output_path: Path to output file (default: data/raw/tobii/)
    """
    from pathlib import Path

    # Default paths
    if input_path is None:
        input_path = Path(__file__).parent.parent.parent / "data/raw/examples/sceneviewing_tobii/tobii_sceneviewing_eyetrack_ascii.txt"

    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "data/raw/tobii/sceneviewing_converted.csv"

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return None

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert
    converter = TobiiConverter(screen_size=(1920, 1080))
    df = converter.convert(str(input_path), str(output_path))

    return df


def convert_tobii_tsv(input_path: str,
                      output_path: Optional[str] = None,
                      screen_size: Tuple[int, int] = (1920, 1080)) -> pd.DataFrame:
    """
    Convenience function to convert Tobii Pro Lab TSV export to HCI ToolKit format.

    This is the main function to use for converting Tobii data exports.

    Args:
        input_path: Path to Tobii TSV file
        output_path: Optional path to save converted CSV file.
                     If None, returns DataFrame without saving.
        screen_size: Screen resolution (width, height) for coordinate conversion.
                     Will be auto-detected from file if available.

    Returns:
        Converted DataFrame ready for use with TobiiAnalyzer

    Example:
        >>> from src.utils.tobii_converter import convert_tobii_tsv
        >>> df = convert_tobii_tsv(
        ...     "data/raw/tobii/recording.tsv",
        ...     "data/raw/tobii/recording_converted.csv"
        ... )
        >>> # Or without saving:
        >>> df = convert_tobii_tsv("data/raw/tobii/recording.tsv")
    """
    converter = TobiiConverter(screen_size=screen_size)
    return converter.convert(input_path, output_path)


if __name__ == "__main__":
    import sys

    # Check if a file path was provided as argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None

        if output_file is None:
            # Generate default output path
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}_converted.csv")

        print(f"Converting: {input_file}")
        print("=" * 60)

        df = convert_tobii_tsv(input_file, output_file)

        if df is not None:
            print("\n" + "=" * 60)
            print("Sample of converted data:")
            print(df.head(10))
            print(f"\nData shape: {df.shape}")
            if 'recording_timestamp' in df.columns:
                print(f"Duration: {df['recording_timestamp'].max():.2f} seconds")
    else:
        # Convert the example data
        print("Converting sceneviewing_tobii example data...")
        print("=" * 60)

        df = convert_sceneviewing_data()

        if df is not None:
            print("\n" + "=" * 60)
            print("Sample of converted data:")
            print(df.head(10))
            print(f"\nData shape: {df.shape}")
            print(f"Duration: {df['recording_timestamp'].max():.2f} seconds")
