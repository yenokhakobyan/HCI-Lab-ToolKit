#!/usr/bin/env python3
"""
Run the Web HCI Collector Demo

This script starts the server with real-time visualization.
Uses HTTPS by default (required for webcam/WebGazer access).

Usage:
    python run_demo.py
    python run_demo.py --port 8080
    python run_demo.py --no-browser
    python run_demo.py --no-ssl  # HTTP only (webcam won't work)
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.collectors.web_hci_collector.server import WebHCICollectorServer, ServerConfig


def main():
    parser = argparse.ArgumentParser(description="Run Web HCI Collector Demo")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--no-ssl", action="store_true", help="Disable HTTPS (webcam won't work)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output-dir", default="/Users/yenokhakobyan/HCI Lab ToolKit/data/raw/web_hci", help="Output directory for data")

    args = parser.parse_args()

    config = ServerConfig(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        debug=args.debug,
        enable_emotion_detection=True,
        ssl_enabled=not args.no_ssl
    )

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           Web HCI Collector - Demo Mode                   ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  This demo collects:                                      ║
    ║    • Eye gaze (WebGazer.js)                              ║
    ║    • Face mesh landmarks (MediaPipe)                     ║
    ║    • Cognitive states (landmark-based heuristics)         ║
    ║    • Mouse movement and clicks                           ║
    ║    • Keyboard events                                     ║
    ║                                                           ║
    ║  Pages:                                                   ║
    ║    • /dashboard  - Real-time visualization               ║
    ║    • /           - Experiment page                       ║
    ║    • /calibration - Gaze calibration                     ║
    ║                                                           ║
    ║  IMPORTANT: WebGazer requires HTTPS for webcam access.   ║
    ║  The server will auto-generate SSL certificates.         ║
    ║  You may need to accept the self-signed cert in browser. ║
    ║                                                           ║
    ║  Cognitive states are estimated from facial landmarks     ║
    ║  using FACS-based heuristics. For model-based detection, ║
    ║  add a DenseAttNet model to: models/denseattnet.onnx     ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    server = WebHCICollectorServer(config)
    server.run(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
