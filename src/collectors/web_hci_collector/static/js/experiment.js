/**
 * Experiment Page Controller
 *
 * Manages data collection during an experiment:
 * - WebGazer.js for eye tracking
 * - MediaPipe Face Mesh for facial landmarks (468 points)
 * - Mouse and keyboard tracking
 * - WebSocket communication with server
 */

// Session configuration
const SESSION_ID = crypto.randomUUID().slice(0, 8);
// Use wss:// for HTTPS, ws:// for HTTP
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${WS_PROTOCOL}//${window.location.host}/ws/collect/${SESSION_ID}`;

// Content URL - can be set via URL parameter: ?content=https://example.com
const urlParams = new URLSearchParams(window.location.search);
const CONTENT_URL = urlParams.get('content') || 'https://superlative-bavarois-0e36a8.netlify.app/';

// State
let ws = null;
let isCollecting = false;

// Trackers
let faceMesh = null;
let camera = null;

// DOM Elements
const connectionStatus = document.getElementById('connection-status');
const connectionText = document.getElementById('connection-text');
const gazeCursor = document.getElementById('gaze-cursor');
const preExperiment = document.getElementById('pre-experiment');
const activeExperiment = document.getElementById('active-experiment');
const floatingControls = document.getElementById('floating-controls');
const experimentContent = document.getElementById('experiment-content');
const contentFrame = document.getElementById('content-frame');

/**
 * Initialize the experiment page
 */
async function init() {
    console.log('Initializing experiment...', SESSION_ID);
    console.log('Content URL:', CONTENT_URL);

    // Connect to WebSocket
    connectWebSocket();

    // Initialize WebGazer
    await initWebGazer();

    // Initialize MediaPipe Face Mesh
    await initFaceMesh();

    // Setup input tracking
    setupMouseTracking();
    setupKeyboardTracking();
}

/**
 * Connect to the WebSocket server
 */
function connectWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('WebSocket connected');
        connectionStatus.classList.add('connected');
        connectionText.textContent = 'Connected';
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        connectionStatus.classList.remove('connected');
        connectionText.textContent = 'Disconnected';

        // Attempt reconnection
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

/**
 * Send data to the server
 */
function sendData(type, data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: type,
            timestamp: performance.now(),
            data: data
        }));
    }
}

/**
 * Initialize WebGazer.js for eye tracking
 */
async function initWebGazer() {
    try {
        await webgazer
            .setGazeListener((data, timestamp) => {
                if (data && isCollecting) {
                    // Update gaze cursor position
                    gazeCursor.style.left = `${data.x}px`;
                    gazeCursor.style.top = `${data.y}px`;

                    // Send gaze data
                    sendData('gaze', {
                        x: data.x,
                        y: data.y,
                        timestamp: timestamp
                    });
                }
            })
            .saveDataAcrossSessions(true)
            .begin();

        // Configure WebGazer display - hide all previews
        webgazer.showVideoPreview(false);
        webgazer.showPredictionPoints(false);
        webgazer.showFaceOverlay(false);
        webgazer.showFaceFeedbackBox(false);

        console.log('WebGazer initialized');

    } catch (error) {
        console.error('WebGazer initialization error:', error);
    }
}

/**
 * Initialize MediaPipe Face Mesh
 */
async function initFaceMesh() {
    try {
        faceMesh = new FaceMesh({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
            }
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,  // Includes iris landmarks (468 + 10 = 478 total)
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceMesh.onResults(onFaceMeshResults);

        // Get video element and start camera (hidden)
        const videoElement = document.getElementById('webcam-video');

        camera = new Camera(videoElement, {
            onFrame: async () => {
                if (isCollecting) {
                    await faceMesh.send({ image: videoElement });
                }
            },
            width: 640,
            height: 480
        });

        await camera.start();
        console.log('MediaPipe Face Mesh initialized');

    } catch (error) {
        console.error('Face Mesh initialization error:', error);
    }
}

/**
 * Handle Face Mesh results - sends all 468 landmarks
 */
function onFaceMeshResults(results) {
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];

        // Key landmark indices for reference:
        // 1: Nose tip
        // 33: Left eye inner corner
        // 133: Left eye outer corner
        // 263: Right eye inner corner
        // 362: Right eye outer corner
        // 61: Left mouth corner
        // 291: Right mouth corner
        // 10: Forehead center
        // 152: Chin

        // Extract key landmarks for quick access
        const noseTip = landmarks[1];
        const leftEyeInner = landmarks[33];
        const leftEyeOuter = landmarks[133];
        const rightEyeInner = landmarks[263];
        const rightEyeOuter = landmarks[362];
        const forehead = landmarks[10];
        const chin = landmarks[152];

        // Calculate eye centers
        const leftEyeCenter = {
            x: (leftEyeInner.x + leftEyeOuter.x) / 2,
            y: (leftEyeInner.y + leftEyeOuter.y) / 2,
            z: (leftEyeInner.z + leftEyeOuter.z) / 2
        };

        const rightEyeCenter = {
            x: (rightEyeInner.x + rightEyeOuter.x) / 2,
            y: (rightEyeInner.y + rightEyeOuter.y) / 2,
            z: (rightEyeInner.z + rightEyeOuter.z) / 2
        };

        const eyeCenter = {
            x: (leftEyeCenter.x + rightEyeCenter.x) / 2,
            y: (leftEyeCenter.y + rightEyeCenter.y) / 2,
            z: (leftEyeCenter.z + rightEyeCenter.z) / 2
        };

        // Calculate head pose from landmarks
        const headPose = {
            pitch: (noseTip.y - eyeCenter.y) * 100,  // Up/down
            yaw: (noseTip.x - eyeCenter.x) * 100,    // Left/right
            roll: (leftEyeCenter.y - rightEyeCenter.y) * 100  // Tilt
        };

        // Send complete face mesh data with all 468 landmarks
        sendData('face_mesh', {
            // All 468 landmarks (each with x, y, z normalized 0-1)
            landmarks: landmarks.map(l => ({ x: l.x, y: l.y, z: l.z })),
            landmark_count: landmarks.length,

            // Key points for quick access
            key_points: {
                nose_tip: { x: noseTip.x, y: noseTip.y, z: noseTip.z },
                left_eye_center: leftEyeCenter,
                right_eye_center: rightEyeCenter,
                eye_center: eyeCenter,
                forehead: { x: forehead.x, y: forehead.y, z: forehead.z },
                chin: { x: chin.x, y: chin.y, z: chin.z }
            },

            // Computed head pose
            head_pose: headPose,

            // Face bounding box (approximate)
            bounding_box: {
                min_x: Math.min(...landmarks.map(l => l.x)),
                max_x: Math.max(...landmarks.map(l => l.x)),
                min_y: Math.min(...landmarks.map(l => l.y)),
                max_y: Math.max(...landmarks.map(l => l.y))
            }
        });
    }
}

/**
 * Setup mouse tracking
 */
function setupMouseTracking() {
    // Mouse move (throttled)
    let lastMouseMove = 0;
    document.addEventListener('mousemove', (e) => {
        if (!isCollecting) return;

        const now = performance.now();
        if (now - lastMouseMove < 50) return; // 20 Hz max
        lastMouseMove = now;

        sendData('mouse', {
            event: 'move',
            x: e.clientX,
            y: e.clientY,
            screenX: e.screenX,
            screenY: e.screenY
        });
    });

    // Mouse click
    document.addEventListener('click', (e) => {
        if (!isCollecting) return;

        sendData('mouse', {
            event: 'click',
            x: e.clientX,
            y: e.clientY,
            button: e.button,
            target: e.target.tagName
        });
    });

    // Mouse scroll
    document.addEventListener('scroll', () => {
        if (!isCollecting) return;

        sendData('mouse', {
            event: 'scroll',
            scrollX: window.scrollX,
            scrollY: window.scrollY
        });
    });
}

/**
 * Setup keyboard tracking
 */
function setupKeyboardTracking() {
    document.addEventListener('keydown', (e) => {
        if (!isCollecting) return;

        // Don't log sensitive keys fully
        const safeKey = getSafeKey(e);

        sendData('keyboard', {
            event: 'keydown',
            key: safeKey,
            code: e.code,
            ctrlKey: e.ctrlKey,
            altKey: e.altKey,
            shiftKey: e.shiftKey,
            metaKey: e.metaKey
        });
    });

    document.addEventListener('keyup', (e) => {
        if (!isCollecting) return;

        const safeKey = getSafeKey(e);

        sendData('keyboard', {
            event: 'keyup',
            key: safeKey,
            code: e.code
        });
    });
}

/**
 * Get a safe (privacy-preserving) key representation
 */
function getSafeKey(e) {
    // For printable characters, just indicate 'char'
    // For special keys, keep them
    const specialKeys = [
        'Enter', 'Tab', 'Escape', 'Backspace', 'Delete',
        'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
        'Home', 'End', 'PageUp', 'PageDown',
        'Control', 'Alt', 'Shift', 'Meta',
        'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
        'F7', 'F8', 'F9', 'F10', 'F11', 'F12'
    ];

    if (specialKeys.includes(e.key)) {
        return e.key;
    }

    return 'char'; // Anonymize regular character keys
}

/**
 * Start the experiment
 */
window.startExperiment = function() {
    console.log('Starting experiment...');
    console.log('Loading content:', CONTENT_URL);
    isCollecting = true;

    // Hide pre-experiment, show active experiment
    preExperiment.classList.add('hidden');
    activeExperiment.classList.remove('hidden');
    floatingControls.classList.remove('hidden');

    // Make experiment content fullscreen - add class to body for CSS targeting
    document.body.classList.add('experiment-active');
    experimentContent.classList.add('fullscreen');

    // Load content URL into iframe
    contentFrame.src = CONTENT_URL;

    // Show gaze cursor
    gazeCursor.style.display = 'block';

    // Send session start event
    sendData('session', {
        event: 'start',
        content_url: CONTENT_URL,
        userAgent: navigator.userAgent,
        screenWidth: window.screen.width,
        screenHeight: window.screen.height,
        windowWidth: window.innerWidth,
        windowHeight: window.innerHeight
    });
};

/**
 * End the experiment
 */
window.endExperiment = function() {
    console.log('Ending experiment...');
    isCollecting = false;

    // Send session end event
    sendData('session', {
        event: 'end'
    });

    // Hide gaze cursor and floating controls
    gazeCursor.style.display = 'none';
    floatingControls.classList.add('hidden');

    // Remove fullscreen mode
    document.body.classList.remove('experiment-active');
    experimentContent.classList.remove('fullscreen');

    // Show completion message
    activeExperiment.innerHTML = `
        <div class="experiment-area">
            <div class="sample-content">
                <h2>Experiment Complete</h2>
                <p>Thank you for participating! Your data has been recorded.</p>
                <p>Session ID: ${SESSION_ID}</p>
                <div style="display: flex; gap: 16px; justify-content: center; margin-top: 30px;">
                    <button class="btn btn-secondary" onclick="window.location.reload()">
                        Start New Session
                    </button>
                    <button class="btn btn-primary" onclick="window.location.href='/dashboard'">
                        View Dashboard
                    </button>
                </div>
            </div>
        </div>
    `;

    // Stop WebGazer
    webgazer.end();
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
