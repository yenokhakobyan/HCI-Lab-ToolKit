#!/usr/bin/env python3
"""
Run the Web HCI Collector Server

Starts the FastAPI server for real-time HCI data collection.
Configuration via environment variables (see .env.example) or CLI args.

Usage:
    python -m src.collectors.web_hci_collector.run_server
    python run_server.py --port 8080
    python run_server.py --no-browser
    python run_server.py --ssl  # Enable HTTPS (for direct access without reverse proxy)
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.collectors.web_hci_collector.server import WebHCICollectorServer, ServerConfig


def main():
    parser = argparse.ArgumentParser(description="Run Web HCI Collector Server")
    parser.add_argument("--host", default=None, help="Host to bind to (default: $HCI_HOST or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to (default: $HCI_PORT or 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--ssl", action="store_true", help="Enable HTTPS (for direct access without reverse proxy)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output-dir", default=None, help="Output directory for data (default: $HCI_DATA_DIR)")

    args = parser.parse_args()

    # Build config from env vars (ServerConfig defaults), override with CLI args
    config = ServerConfig()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.debug:
        config.debug = True
    if args.ssl:
        config.ssl_enabled = True

    protocol = "https" if config.ssl_enabled else "http"
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║           Web HCI Collector Server                        ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  Collects:                                                ║
    ║    - Eye gaze (WebGazer.js)                              ║
    ║    - Face mesh landmarks (MediaPipe)                     ║
    ║    - Cognitive states (landmark-based heuristics)         ║
    ║    - Mouse movement and clicks                           ║
    ║    - Keyboard events                                     ║
    ║                                                           ║
    ║  Listening: {protocol}://{config.host}:{config.port:<26s}║
    ║  Data dir:  {config.output_dir[:45]:<46s}║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    server = WebHCICollectorServer(config)
    server.run(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
