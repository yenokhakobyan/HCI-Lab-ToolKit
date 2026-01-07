#!/usr/bin/env python3
"""
Run the Web HCI Collector Demo with Streamlit

This script provides a Streamlit-based interface for the Web HCI Collector,
making it easy to deploy and manage on a server.

Usage:
    streamlit run run_demo_streamlit.py
    streamlit run run_demo_streamlit.py -- --port 8000
"""

import argparse
import sys
import threading
import time
import socket
from pathlib import Path
from datetime import datetime
from contextlib import closing

import streamlit as st

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.collectors.web_hci_collector.server import WebHCICollectorServer, ServerConfig


def find_free_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is already in use."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0


def run_server_thread(server: WebHCICollectorServer):
    """Run the server in a background thread."""
    server.run(open_browser=False)


def main():
    st.set_page_config(
        page_title="Web HCI Collector",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧠 Web HCI Collector")
    st.markdown("Real-time HCI data collection with eye gaze, face mesh, and cognitive state tracking")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Server settings
    host = st.sidebar.text_input("Host", value="0.0.0.0", help="Use 0.0.0.0 to allow external connections")
    port = st.sidebar.number_input("Port", min_value=1024, max_value=65535, value=8000)

    # SSL settings
    use_ssl = st.sidebar.checkbox("Enable HTTPS", value=True, help="Required for webcam access")

    # Output directory
    output_dir = st.sidebar.text_input("Output Directory", value="data/raw/web_hci")

    # Feature toggles
    st.sidebar.subheader("Features")
    enable_emotion = st.sidebar.checkbox("Emotion Detection", value=True)
    enable_l2cs = st.sidebar.checkbox("L2CS Gaze Estimation", value=True)
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    # Screen dimensions for gaze mapping
    st.sidebar.subheader("Screen Settings")
    screen_width = st.sidebar.number_input("Screen Width", min_value=800, max_value=7680, value=1920)
    screen_height = st.sidebar.number_input("Screen Height", min_value=600, max_value=4320, value=1080)

    # Initialize session state
    if 'server_running' not in st.session_state:
        st.session_state.server_running = False
    if 'server_thread' not in st.session_state:
        st.session_state.server_thread = None
    if 'server' not in st.session_state:
        st.session_state.server = None
    if 'server_port' not in st.session_state:
        st.session_state.server_port = None

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Server Control")

        # Check if server is actually running (port is in use)
        actual_port = st.session_state.server_port or port
        port_in_use = is_port_in_use(actual_port, "127.0.0.1") if st.session_state.server_running else False

        if st.session_state.server_running and port_in_use:
            st.success(f"✅ Server is running on port {actual_port}")

            protocol = "https" if use_ssl else "http"
            base_url = f"{protocol}://{host}:{actual_port}"

            # Show URLs
            st.markdown("### 🔗 Access URLs")

            # For external access, show the actual server address
            if host == "0.0.0.0":
                display_host = "YOUR_SERVER_IP"
                st.info(f"Replace `{display_host}` with your server's actual IP address")
            else:
                display_host = host

            display_url = f"{protocol}://{display_host}:{actual_port}"

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"**Dashboard**")
                st.code(f"{display_url}/dashboard")
            with col_b:
                st.markdown(f"**Experiment**")
                st.code(f"{display_url}/")
            with col_c:
                st.markdown(f"**Calibration**")
                st.code(f"{display_url}/calibration")

            if st.button("🛑 Stop Server", type="primary"):
                st.session_state.server_running = False
                st.session_state.server = None
                st.session_state.server_thread = None
                st.session_state.server_port = None
                st.warning("Server stop requested. You may need to restart the Streamlit app to fully stop the server.")
                st.rerun()
        else:
            st.warning("⚠️ Server is not running")

            if st.button("🚀 Start Server", type="primary"):
                # Check if port is available
                if is_port_in_use(port):
                    st.error(f"Port {port} is already in use. Please choose a different port.")
                else:
                    # Create server configuration
                    config = ServerConfig(
                        host=host,
                        port=port,
                        output_dir=output_dir,
                        debug=debug_mode,
                        enable_emotion_detection=enable_emotion,
                        enable_l2cs_gaze=enable_l2cs,
                        ssl_enabled=use_ssl,
                        screen_width=screen_width,
                        screen_height=screen_height
                    )

                    try:
                        # Create and start server in background thread
                        server = WebHCICollectorServer(config)
                        server_thread = threading.Thread(
                            target=run_server_thread,
                            args=(server,),
                            daemon=True
                        )
                        server_thread.start()

                        # Wait a moment for server to start
                        time.sleep(2)

                        # Update session state
                        st.session_state.server_running = True
                        st.session_state.server = server
                        st.session_state.server_thread = server_thread
                        st.session_state.server_port = port

                        st.success("Server started successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to start server: {e}")

    with col2:
        st.subheader("📊 Status")

        # Status indicators
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            if st.session_state.server_running:
                st.metric("Server", "Running", delta="Active")
            else:
                st.metric("Server", "Stopped", delta="Inactive")

        with status_col2:
            if st.session_state.server_running and st.session_state.server:
                try:
                    session_count = len(st.session_state.server.session_manager.get_active_sessions())
                    st.metric("Sessions", session_count)
                except:
                    st.metric("Sessions", "N/A")
            else:
                st.metric("Sessions", "0")

    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About Web HCI Collector")

    with st.expander("Features", expanded=True):
        st.markdown("""
        This demo collects:
        - 👁️ **Eye gaze** (WebGazer.js)
        - 🎭 **Face mesh landmarks** (MediaPipe)
        - 🧠 **Cognitive states** (confusion, engagement, boredom, frustration)
        - 🖱️ **Mouse movement and clicks**
        - ⌨️ **Keyboard events**
        """)

    with st.expander("Pages"):
        st.markdown("""
        - **Dashboard** (`/dashboard`) - Real-time visualization
        - **Experiment** (`/`) - Main experiment page
        - **Calibration** (`/calibration`) - Gaze calibration
        """)

    with st.expander("Important Notes"):
        st.markdown("""
        - ⚠️ **HTTPS Required**: WebGazer requires HTTPS for webcam access. The server will auto-generate self-signed SSL certificates.
        - 🔐 **Certificate Warning**: You may need to accept the self-signed certificate in your browser.
        - 🧪 **Demo Mode**: Cognitive states use simulated data unless you add a DenseAttNet model to `models/denseattnet.onnx`
        """)

    # Data management section
    if st.session_state.server_running:
        st.markdown("---")
        st.subheader("📁 Data Management")

        output_path = Path(output_dir)
        if output_path.exists():
            sessions = list(output_path.iterdir())
            if sessions:
                st.write(f"Found {len(sessions)} session(s) in output directory:")
                for session_dir in sessions:
                    if session_dir.is_dir():
                        with st.expander(f"📂 {session_dir.name}"):
                            files = list(session_dir.iterdir())
                            for f in files:
                                size = f.stat().st_size / 1024
                                st.text(f"  {f.name} ({size:.1f} KB)")
            else:
                st.info("No session data yet. Start a collection session from the experiment page.")
        else:
            st.info(f"Output directory will be created at: {output_dir}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Web HCI Collector - Part of HCI Lab ToolKit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
