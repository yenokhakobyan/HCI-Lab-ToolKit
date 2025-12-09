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

// Screen recording using MediaRecorder API
let screenStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = 0;
const VIDEO_CHUNK_INTERVAL = 1000; // Send video chunk every 1 second

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
 *
 * NOTE: Due to browser security (Same-Origin Policy), mouse events INSIDE
 * cross-origin iframes cannot be captured by the parent document.
 *
 * What we CAN track:
 * - Mouse position when cursor is outside the iframe (on parent document)
 * - Mouse entering/leaving the iframe area
 * - Scroll and wheel events on the parent document
 *
 * What we CANNOT track (cross-origin limitation):
 * - Mouse movement inside the iframe
 * - Clicks inside the iframe
 *
 * SOLUTION: Screen recording captures the actual cursor position visually.
 * The recorded video shows where the user's mouse is at all times.
 */
function setupMouseTracking() {
    let lastMouseMove = 0;
    let isOverIframe = false;

    // Track mouse position on parent document
    // This works when mouse is outside the iframe
    document.addEventListener('mousemove', (e) => {
        if (!isCollecting) return;

        const now = performance.now();
        if (now - lastMouseMove < 50) return; // 20 Hz max
        lastMouseMove = now;

        // Check if mouse is over the iframe
        const iframe = document.getElementById('content-frame');
        if (iframe) {
            const rect = iframe.getBoundingClientRect();
            const wasOverIframe = isOverIframe;
            isOverIframe = (
                e.clientX >= rect.left &&
                e.clientX <= rect.right &&
                e.clientY >= rect.top &&
                e.clientY <= rect.bottom
            );

            // Log when mouse enters/exits iframe area
            if (isOverIframe && !wasOverIframe) {
                sendData('mouse', {
                    event: 'enter_iframe',
                    x: e.clientX,
                    y: e.clientY
                });
            } else if (!isOverIframe && wasOverIframe) {
                sendData('mouse', {
                    event: 'exit_iframe',
                    x: e.clientX,
                    y: e.clientY
                });
            }
        }

        // Always send position (even if over iframe, we get the last known position)
        sendData('mouse', {
            event: 'move',
            x: e.clientX,
            y: e.clientY,
            screenX: e.screenX,
            screenY: e.screenY,
            overIframe: isOverIframe
        });
    }, true);

    // Track clicks on parent document (won't capture iframe clicks)
    document.addEventListener('click', (e) => {
        if (!isCollecting) return;

        sendData('mouse', {
            event: 'click',
            x: e.clientX,
            y: e.clientY,
            button: e.button,
            target: e.target?.tagName || 'unknown'
        });
    }, true);

    // Mouse scroll on parent window
    window.addEventListener('scroll', () => {
        if (!isCollecting) return;

        sendData('mouse', {
            event: 'scroll',
            scrollX: window.scrollX,
            scrollY: window.scrollY
        });
    }, true);

    // Wheel events
    document.addEventListener('wheel', (e) => {
        if (!isCollecting) return;

        sendData('mouse', {
            event: 'wheel',
            x: e.clientX,
            y: e.clientY,
            deltaX: e.deltaX,
            deltaY: e.deltaY
        });
    }, true);

    // Right-click
    document.addEventListener('contextmenu', (e) => {
        if (!isCollecting) return;

        sendData('mouse', {
            event: 'rightclick',
            x: e.clientX,
            y: e.clientY
        });
    }, true);

    console.log('Mouse tracking initialized. Note: Cross-origin iframe mouse events are captured via screen recording.');
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
window.startExperiment = async function() {
    console.log('Starting experiment...');
    console.log('Loading content:', CONTENT_URL);

    // Hide pre-experiment, show active experiment
    preExperiment.classList.add('hidden');
    activeExperiment.classList.remove('hidden');

    // Make experiment content fullscreen - add class to body for CSS targeting
    document.body.classList.add('experiment-active');
    experimentContent.classList.add('fullscreen');

    // Load content URL into iframe FIRST
    contentFrame.src = CONTENT_URL;

    // Show a loading message while waiting
    const loadingOverlay = document.createElement('div');
    loadingOverlay.id = 'loading-overlay';
    loadingOverlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(26, 26, 46, 0.95);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        color: white;
        font-family: system-ui, sans-serif;
    `;
    loadingOverlay.innerHTML = `
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 20px;">Loading Experiment Content...</div>
            <div style="font-size: 0.9rem; color: #888; margin-bottom: 30px;">
                Please wait for the content to load, then you'll be asked to share your screen.
            </div>
            <div style="width: 200px; height: 4px; background: #333; border-radius: 2px; overflow: hidden;">
                <div id="loading-progress" style="width: 0%; height: 100%; background: #4ecca3; transition: width 0.3s;"></div>
            </div>
        </div>
    `;
    document.body.appendChild(loadingOverlay);

    // Animate progress bar
    const progressBar = document.getElementById('loading-progress');
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + 10, 90);
        if (progressBar) progressBar.style.width = progress + '%';
    }, 200);

    // Wait for iframe to load (or timeout after 5 seconds)
    await new Promise((resolve) => {
        const timeout = setTimeout(() => {
            console.log('Iframe load timeout, proceeding anyway');
            resolve();
        }, 5000);

        contentFrame.onload = () => {
            console.log('Iframe content loaded');
            clearTimeout(timeout);
            resolve();
        };
    });

    // Complete the progress bar
    clearInterval(progressInterval);
    if (progressBar) progressBar.style.width = '100%';

    // Update loading message
    loadingOverlay.innerHTML = `
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 20px;">Content Loaded!</div>
            <div style="font-size: 0.9rem; color: #4ecca3; margin-bottom: 10px;">
                Now select the screen sharing option...
            </div>
            <div style="font-size: 0.8rem; color: #888; max-width: 400px; line-height: 1.5;">
                In the dialog that appears, select <strong>"This Tab"</strong> or <strong>"Chrome Tab"</strong>
                and choose this browser tab to record what you see during the experiment.
            </div>
        </div>
    `;

    // Small delay to let user see the message
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Now start screen recording (will prompt user for permission)
    const recordingStarted = await startScreenRecording();

    // Remove loading overlay
    loadingOverlay.remove();

    // Now enable data collection
    isCollecting = true;
    floatingControls.classList.remove('hidden');

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
        windowHeight: window.innerHeight,
        screenRecording: recordingStarted
    });

    if (!recordingStarted) {
        console.warn('Screen recording was not started - user may have denied permission');
    }
};

/**
 * End the experiment
 */
window.endExperiment = function() {
    console.log('Ending experiment...');
    isCollecting = false;

    // Stop screen recording
    stopScreenRecording();

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

/**
 * Start screen recording using MediaRecorder API
 * Prompts user to share their screen/tab
 */
async function startScreenRecording() {
    console.log('Starting screen recording...');

    try {
        // Request screen capture permission
        // User selects which screen/window/tab to share
        screenStream = await navigator.mediaDevices.getDisplayMedia({
            video: {
                cursor: 'always',
                displaySurface: 'browser', // Prefer browser tab
                frameRate: { ideal: 15, max: 30 }
            },
            audio: false
        });

        // Create MediaRecorder
        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
            ? 'video/webm;codecs=vp9'
            : MediaRecorder.isTypeSupported('video/webm;codecs=vp8')
                ? 'video/webm;codecs=vp8'
                : 'video/webm';

        mediaRecorder = new MediaRecorder(screenStream, {
            mimeType: mimeType,
            videoBitsPerSecond: 1000000 // 1 Mbps for good quality but reasonable size
        });

        recordedChunks = [];
        recordingStartTime = Date.now();

        // Handle data available event - send chunks to server
        mediaRecorder.ondataavailable = async (event) => {
            if (event.data && event.data.size > 0) {
                recordedChunks.push(event.data);

                // Convert chunk to base64 and send
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64Data = reader.result.split(',')[1];
                    const currentTime = Date.now() - recordingStartTime;

                    sendData('video_chunk', {
                        data: base64Data,
                        mimeType: mimeType,
                        chunkIndex: recordedChunks.length - 1,
                        timestamp: currentTime,
                        size: event.data.size
                    });

                    console.log(`Video chunk #${recordedChunks.length} sent (${(event.data.size / 1024).toFixed(1)} KB)`);
                };
                reader.readAsDataURL(event.data);
            }
        };

        // Handle recording stop
        mediaRecorder.onstop = () => {
            console.log('Screen recording stopped');
            // Send final video blob info
            if (recordedChunks.length > 0) {
                const totalSize = recordedChunks.reduce((acc, chunk) => acc + chunk.size, 0);
                sendData('video_complete', {
                    totalChunks: recordedChunks.length,
                    totalSize: totalSize,
                    duration: Date.now() - recordingStartTime,
                    mimeType: mimeType
                });
            }
        };

        // Handle stream ending (user stops sharing)
        screenStream.getVideoTracks()[0].onended = () => {
            console.log('Screen sharing stopped by user');
            stopScreenRecording();
        };

        // Start recording with timeslice for chunked data
        mediaRecorder.start(VIDEO_CHUNK_INTERVAL);

        console.log('Screen recording started with', mimeType);

        // Send recording start event
        sendData('video_start', {
            mimeType: mimeType,
            width: screenStream.getVideoTracks()[0].getSettings().width,
            height: screenStream.getVideoTracks()[0].getSettings().height,
            startTime: recordingStartTime
        });

        return true;

    } catch (error) {
        console.error('Screen recording error:', error);
        // User denied permission or API not supported
        return false;
    }
}

/**
 * Stop screen recording and release resources
 */
function stopScreenRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }

    if (screenStream) {
        screenStream.getTracks().forEach(track => track.stop());
        screenStream = null;
    }

    mediaRecorder = null;
    console.log('Screen recording stopped, total chunks:', recordedChunks.length);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
