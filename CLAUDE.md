# CLAUDE.md

## Project Overview

HCI Lab ToolKit — a multi-modal physiological data collection and analysis toolkit for Human-Computer Interaction research. Built at the HCI + L Laboratory, Yerevan State University, for the EcoInsight adaptive learning project.

Collects synchronized data from eye trackers (Tobii), EEG (Enobio), physiological sensors (Shimmer), keyboard/mouse, and webcam-based facial analysis. Includes real-time processing, ML-based state classification, and interactive analytics dashboards.

## Tech Stack

- **Language:** Python 3.x, JavaScript (web collector), Bash (scripts)
- **Web server:** FastAPI + uvicorn + WebSockets
- **Streaming/sync:** Lab Streaming Layer (LSL)
- **ML/DL:** PyTorch, scikit-learn, MNE-Python, neurokit2, heartpy, biosppy
- **Visualization:** Streamlit, Plotly, Matplotlib, Seaborn
- **Computer vision:** OpenCV, MediaPipe, L2CS-Net (gaze), DenseAttNet (emotion)
- **Hardware comms:** Bleak (Bluetooth/Shimmer), pySerial

## Directory Structure

```
src/
  collectors/           # Data collection modules (Tobii, AOI, web HCI)
    web_hci_collector/  # FastAPI-based multimodal web collector
  analyzers/            # Analysis algorithms
  processors/           # Data preprocessing
  visualization/        # Plotting utilities
  utils/                # Helpers and converters
  tobiiresearch/        # Tobii SDK wrapper
  analysis/             # Streamlit dashboard
configs/                # YAML configuration (config.yaml)
data/raw/               # Raw data (tobii, emotiv, shimmer, behavioral)
data/processed/         # Cleaned/synchronized data
notebooks/              # Jupyter analysis notebooks
docs/                   # Technical notes (eye tracking, EEG, Shimmer)
tests/                  # Test suite (pytest)
```

## Setup & Running

```bash
# Install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Analytics dashboard
./run_dashboard.sh
# or: streamlit run src/analysis/dashboard.py --server.port 8502

# Web HCI collector server
python src/collectors/web_hci_collector/server.py

# Streamlit demo
streamlit run src/collectors/web_hci_collector/run_demo_streamlit.py
```

## Testing

```bash
pytest tests/ --cov=src
```

Tests use pytest with coverage. Test directory is at `tests/`.

## Configuration

Central config: `configs/config.yaml` — controls enabled devices (tobii, emotiv, shimmer, behavioral), sample rates, data directories, LSL sync settings, and logging.

## Key Patterns

- **Collector architecture:** Modular collectors with `connect()`, `start_recording()`, `process_data()`, `save_data()` interface
- **Synchronization:** LSL for unified timestamps across devices; software event marking as fallback
- **Data pipeline:** Device Acquisition -> Real-time Streaming (LSL/WebSocket) -> Processing -> Feature Extraction -> Classification -> Dashboard
- **Data formats:** CSV (raw), Parquet (processed), HDF5 (synchronized multimodal), JSON (metadata), WebM (video)
- **Code style:** PEP 8, type hints in newer modules, docstrings present
- **No CI/CD:** Local development and research use only
