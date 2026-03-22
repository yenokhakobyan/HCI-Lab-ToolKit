"""Allow running as: python -m src.collectors.web_hci_collector"""
import argparse
import os

from .server import WebHCICollectorServer, ServerConfig


def main():
    parser = argparse.ArgumentParser(description="Run Web HCI Collector Server")
    parser.add_argument("--host", default=None, help="Host to bind to (default: $HCI_HOST or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to (default: $HCI_PORT or 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--ssl", action="store_true", help="Enable HTTPS")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output-dir", default=None, help="Output directory for data")

    args = parser.parse_args()

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

    server = WebHCICollectorServer(config)
    server.run(open_browser=not args.no_browser)


main()
