/**
 * HCI Iframe Mouse Tracker
 *
 * Include this script in your iframe content to enable mouse tracking
 * when embedded in the HCI Collector experiment page.
 *
 * Usage: Add this script to your page:
 * <script src="https://your-hci-server/static/js/iframe-tracker.js"></script>
 *
 * Or copy this code directly into your page.
 */

(function() {
    'use strict';

    let isTracking = false;
    let lastMouseMove = 0;

    // Listen for enable message from parent
    window.addEventListener('message', (e) => {
        if (e.data && e.data.type === 'hci_enable_mouse_tracking') {
            console.log('[HCI Tracker] Mouse tracking enabled for session:', e.data.sessionId);
            isTracking = true;
        }
    });

    // Auto-enable if parent is detected (optional - can be removed for explicit enable only)
    if (window.parent !== window) {
        // We're in an iframe, auto-enable tracking
        isTracking = true;
        console.log('[HCI Tracker] Running in iframe, mouse tracking active');
    }

    function sendToParent(eventType, data) {
        if (!isTracking || window.parent === window) return;

        window.parent.postMessage({
            type: 'hci_mouse_event',
            data: {
                event: eventType,
                ...data
            }
        }, '*');
    }

    // Mouse move
    document.addEventListener('mousemove', (e) => {
        const now = performance.now();
        if (now - lastMouseMove < 50) return; // 20 Hz max
        lastMouseMove = now;

        sendToParent('move', {
            x: e.clientX,
            y: e.clientY
        });
    }, true);

    // Click
    document.addEventListener('click', (e) => {
        sendToParent('click', {
            x: e.clientX,
            y: e.clientY,
            button: e.button,
            target: e.target?.tagName || 'unknown'
        });
    }, true);

    // Scroll
    window.addEventListener('scroll', () => {
        sendToParent('scroll', {
            scrollX: window.scrollX,
            scrollY: window.scrollY
        });
    }, true);

    // Wheel
    document.addEventListener('wheel', (e) => {
        sendToParent('wheel', {
            x: e.clientX,
            y: e.clientY,
            deltaX: e.deltaX,
            deltaY: e.deltaY
        });
    }, true);

    // Right-click
    document.addEventListener('contextmenu', (e) => {
        sendToParent('rightclick', {
            x: e.clientX,
            y: e.clientY
        });
    }, true);

    console.log('[HCI Tracker] Iframe mouse tracker loaded');
})();
