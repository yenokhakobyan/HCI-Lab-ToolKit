# HCI Lab ToolKit

Multi-modal physiological data collection and analysis toolkit for Human-Computer Interaction research.

## Supported Devices

| Device | Data Type | SDK/Protocol |
|--------|-----------|--------------|
| **Tobii Eye Tracker** | Gaze position, fixations, saccades, pupil diameter | Tobii Pro SDK |
| **Emotiv EEG** | Brain activity (14/32 channels), motion sensors | Cortex API |
| **Shimmer** | GSR/EDA, ECG, EMG, Accelerometer, Gyroscope | Bluetooth/Serial |
| **Behavioral** | Mouse movement, clicks, keyboard, screen capture | pynput |

## Repository Structure

```
HCI Lab ToolKit/
├── data/
│   ├── raw/                    # Unprocessed data from devices
│   │   ├── tobii/              # Eye tracking data
│   │   ├── emotiv/             # EEG data
│   │   ├── shimmer/            # Physiological sensors
│   │   └── behavioral/         # Mouse, keyboard, clicks
│   └── processed/              # Cleaned and synchronized data
│       ├── tobii/
│       ├── emotiv/
│       ├── shimmer/
│       ├── behavioral/
│       └── synchronized/       # Multi-modal aligned data
├── src/
│   ├── collectors/             # Data collection modules
│   ├── processors/             # Data cleaning and preprocessing
│   ├── analyzers/              # Analysis algorithms
│   ├── utils/                  # Helper functions
│   └── visualization/          # Plotting and visualization
├── configs/                    # Configuration files
├── docs/                       # Documentation
├── notebooks/                  # Jupyter notebooks for analysis
└── tests/                      # Unit tests
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Additional Setup

#### Tobii Eye Tracker
1. Install [Tobii Pro SDK](https://www.tobii.com/products/software/applications-and-developer-kits/tobii-pro-sdk)
2. Connect eye tracker via USB

#### Emotiv EEG
1. Install [Emotiv Launcher](https://www.emotiv.com/emotiv-launcher/)
2. Create Emotiv account and get API credentials
3. Pair headset via Bluetooth

#### Shimmer
1. Pair Shimmer device via Bluetooth
2. Note the COM port / Bluetooth address

## Quick Start

```python
# Example: Record synchronized data
from src.collectors import TobiiCollector, EmotivCollector, ShimmerCollector

# Initialize collectors
tobii = TobiiCollector()
emotiv = EmotivCollector(client_id="YOUR_ID", client_secret="YOUR_SECRET")
shimmer = ShimmerCollector(port="/dev/tty.Shimmer")

# Start recording
tobii.start()
emotiv.start()
shimmer.start()
```

## Data Formats

| Source | Raw Format | Processed Format |
|--------|------------|------------------|
| Tobii | CSV/JSON | Parquet |
| Emotiv | EDF/CSV | Parquet |
| Shimmer | CSV | Parquet |
| Behavioral | CSV | Parquet |
| Synchronized | - | HDF5/Parquet |

## License

MIT License
