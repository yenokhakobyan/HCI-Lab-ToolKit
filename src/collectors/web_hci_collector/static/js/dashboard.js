/**
 * Dashboard Controller
 *
 * Real-time visualization of HCI data collection:
 * - Gaze point visualization
 * - Face mesh (468 landmarks) visualization
 * - Cognitive state bars
 * - Event log
 * - Statistics
 */

// Configuration
// Use wss:// for HTTPS, ws:// for HTTP
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${WS_PROTOCOL}//${window.location.host}/ws/dashboard`;

// State
let ws = null;
let isCollecting = false;
let sessionId = null;
let startTime = null;
let stats = {
    gazeSamples: 0,
    faceSamples: 0,
    mouseEvents: 0,
    keyboardEvents: 0,
    totalSamples: 0
};

// Gaze visualization
let gazeHistory = [];
const GAZE_HISTORY_LENGTH = 100;

// Face mesh visualization
let latestLandmarks = null;
let faceMesh = null;
let localFaceMeshActive = false;

// Cognitive state estimation
let cognitiveStateHistory = {
    blinkHistory: [],        // Track blinks for blink rate
    eyeOpennessHistory: [],  // Track eye openness over time
    headPoseHistory: [],     // Track head movement
    mouthOpennessHistory: [], // Track mouth openness (yawning/confusion)
    browHistory: []          // Track eyebrow position
};
const HISTORY_WINDOW = 90;   // ~3 seconds at 30fps
let lastBlinkTime = 0;
let blinkCount = 0;
let cognitiveStates = {
    confusion: 0,
    engagement: 0,
    boredom: 0,
    frustration: 0
};

// Canvas contexts
let gazeCanvas = null;
let gazeCtx = null;
let faceCanvas = null;
let faceCtx = null;

// DOM Elements
const connectionStatus = document.getElementById('connection-status');
const connectionText = document.getElementById('connection-text');
const gazePoint = document.getElementById('gaze-point');
const eventLog = document.getElementById('event-log');

/**
 * Initialize the dashboard
 */
function init() {
    console.log('Initializing dashboard...');

    // Setup canvases
    setupGazeCanvas();
    setupFaceCanvas();

    // Initialize webcam preview
    initWebcam();

    // Connect to WebSocket
    connectWebSocket();

    // Start stats update loop
    setInterval(updateStats, 1000);

    // Start face mesh rendering loop
    requestAnimationFrame(renderFaceMesh);
}

/**
 * Initialize webcam preview and local face mesh for dashboard
 */
async function initWebcam() {
    const video = document.getElementById('webcam-video');
    const cameraStatus = document.getElementById('camera-status');

    try {
        // Initialize MediaPipe Face Mesh for local real-time tracking
        await initLocalFaceMesh();

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });

        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            if (cameraStatus) {
                cameraStatus.style.display = 'none';
            }
            console.log('Dashboard webcam initialized');

            // Start local face mesh processing after video is ready
            startLocalFaceMesh(video);
        };
    } catch (error) {
        console.error('Dashboard webcam error:', error);
        if (cameraStatus) {
            cameraStatus.textContent = 'Camera unavailable';
            cameraStatus.style.color = 'var(--error)';
        }
    }
}

/**
 * Initialize MediaPipe Face Mesh for local real-time tracking
 */
async function initLocalFaceMesh() {
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

        faceMesh.onResults(onLocalFaceMeshResults);

        console.log('Local MediaPipe Face Mesh initialized');
        addLogEntry('system', 'Face mesh ready');

    } catch (error) {
        console.error('Local Face Mesh initialization error:', error);
        addLogEntry('system', 'Face mesh init failed');
    }
}

/**
 * Start local face mesh processing loop
 */
function startLocalFaceMesh(videoElement) {
    if (!faceMesh) {
        console.warn('Face mesh not initialized');
        return;
    }

    localFaceMeshActive = true;

    // Use Camera utility from MediaPipe for smooth frame processing
    const camera = new Camera(videoElement, {
        onFrame: async () => {
            if (localFaceMeshActive && faceMesh) {
                await faceMesh.send({ image: videoElement });
            }
        },
        width: 640,
        height: 480
    });

    camera.start();
    console.log('Local face mesh processing started');
}

/**
 * Handle local Face Mesh results - processes all 468 landmarks in real-time
 */
function onLocalFaceMeshResults(results) {
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];

        // Key landmark indices:
        // 1: Nose tip, 33/133: Left eye corners, 263/362: Right eye corners
        // 61/291: Mouth corners, 10: Forehead center, 152: Chin

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

        // Calculate head pose
        const headPose = {
            pitch: (noseTip.y - eyeCenter.y) * 100,
            yaw: (noseTip.x - eyeCenter.x) * 100,
            roll: (leftEyeCenter.y - rightEyeCenter.y) * 100
        };

        // Update latestLandmarks for rendering (same format as experiment page)
        latestLandmarks = {
            landmarks: landmarks.map(l => ({ x: l.x, y: l.y, z: l.z })),
            landmark_count: landmarks.length,
            key_points: {
                nose_tip: { x: noseTip.x, y: noseTip.y, z: noseTip.z },
                left_eye_center: leftEyeCenter,
                right_eye_center: rightEyeCenter,
                eye_center: eyeCenter,
                forehead: { x: forehead.x, y: forehead.y, z: forehead.z },
                chin: { x: chin.x, y: chin.y, z: chin.z }
            },
            head_pose: headPose
        };

        // Estimate cognitive states from facial features
        estimateCognitiveStates(landmarks, headPose);

        // Update stats (local tracking)
        stats.faceSamples++;
        stats.totalSamples++;

        // Update UI
        document.getElementById('landmark-count').textContent = landmarks.length;
        const pitch = headPose.pitch.toFixed(1);
        const yaw = headPose.yaw.toFixed(1);
        document.getElementById('head-pose').textContent = `P:${pitch} Y:${yaw}`;
    }
}

/**
 * Estimate cognitive states from facial landmarks
 * Based on research indicators:
 * - Confusion: furrowed brows, squinting, head tilt
 * - Engagement: eyes wide open, forward head pose, stable gaze
 * - Boredom: drooping eyelids, yawning, looking away
 * - Frustration: tense jaw, furrowed brows, rapid blinking
 */
function estimateCognitiveStates(landmarks, headPose) {
    // Extract facial features for cognitive state estimation
    const features = extractFacialFeatures(landmarks);

    // Update history buffers
    updateFeatureHistory(features, headPose);

    // Calculate cognitive states based on features
    const states = calculateCognitiveStates(features, headPose);

    // Smooth the states using exponential moving average
    const smoothingFactor = 0.15;
    cognitiveStates.confusion = smoothValue(cognitiveStates.confusion, states.confusion, smoothingFactor);
    cognitiveStates.engagement = smoothValue(cognitiveStates.engagement, states.engagement, smoothingFactor);
    cognitiveStates.boredom = smoothValue(cognitiveStates.boredom, states.boredom, smoothingFactor);
    cognitiveStates.frustration = smoothValue(cognitiveStates.frustration, states.frustration, smoothingFactor);

    // Update UI
    updateCognitiveStateUI();
}

/**
 * Extract key facial features from landmarks
 */
function extractFacialFeatures(landmarks) {
    // Eye landmarks (left eye: 159-145 vertical, right eye: 386-374 vertical)
    // Upper eyelid: 159 (left), 386 (right)
    // Lower eyelid: 145 (left), 374 (right)
    const leftEyeTop = landmarks[159];
    const leftEyeBottom = landmarks[145];
    const rightEyeTop = landmarks[386];
    const rightEyeBottom = landmarks[374];

    // Eye openness (Eye Aspect Ratio)
    const leftEyeOpenness = Math.abs(leftEyeTop.y - leftEyeBottom.y);
    const rightEyeOpenness = Math.abs(rightEyeTop.y - rightEyeBottom.y);
    const avgEyeOpenness = (leftEyeOpenness + rightEyeOpenness) / 2;

    // Eyebrow landmarks (left: 70, 63, right: 300, 293)
    const leftBrowInner = landmarks[70];
    const leftBrowOuter = landmarks[63];
    const rightBrowInner = landmarks[300];
    const rightBrowOuter = landmarks[293];

    // Eyebrow position relative to eyes (higher = raised, lower = furrowed)
    const leftBrowHeight = leftEyeTop.y - ((leftBrowInner.y + leftBrowOuter.y) / 2);
    const rightBrowHeight = rightEyeTop.y - ((rightBrowInner.y + rightBrowOuter.y) / 2);
    const avgBrowHeight = (leftBrowHeight + rightBrowHeight) / 2;

    // Inner brow distance (closer = furrowed)
    const browFurrow = Math.abs(leftBrowInner.x - rightBrowInner.x);

    // Mouth landmarks
    const upperLip = landmarks[13];
    const lowerLip = landmarks[14];
    const leftMouth = landmarks[61];
    const rightMouth = landmarks[291];

    // Mouth openness (for yawning detection)
    const mouthOpenness = Math.abs(upperLip.y - lowerLip.y);
    const mouthWidth = Math.abs(leftMouth.x - rightMouth.x);

    // Detect blink (eye openness below threshold)
    const blinkThreshold = 0.015;
    const isBlinking = avgEyeOpenness < blinkThreshold;

    // Track blink
    const now = Date.now();
    if (isBlinking && (now - lastBlinkTime) > 200) { // Minimum 200ms between blinks
        lastBlinkTime = now;
        blinkCount++;
    }

    return {
        eyeOpenness: avgEyeOpenness,
        leftEyeOpenness,
        rightEyeOpenness,
        browHeight: avgBrowHeight,
        browFurrow,
        mouthOpenness,
        mouthWidth,
        isBlinking,
        blinkCount
    };
}

/**
 * Update feature history for temporal analysis
 */
function updateFeatureHistory(features, headPose) {
    // Add to history
    cognitiveStateHistory.eyeOpennessHistory.push(features.eyeOpenness);
    cognitiveStateHistory.headPoseHistory.push({ ...headPose });
    cognitiveStateHistory.mouthOpennessHistory.push(features.mouthOpenness);
    cognitiveStateHistory.browHistory.push(features.browHeight);

    // Trim history to window size
    if (cognitiveStateHistory.eyeOpennessHistory.length > HISTORY_WINDOW) {
        cognitiveStateHistory.eyeOpennessHistory.shift();
    }
    if (cognitiveStateHistory.headPoseHistory.length > HISTORY_WINDOW) {
        cognitiveStateHistory.headPoseHistory.shift();
    }
    if (cognitiveStateHistory.mouthOpennessHistory.length > HISTORY_WINDOW) {
        cognitiveStateHistory.mouthOpennessHistory.shift();
    }
    if (cognitiveStateHistory.browHistory.length > HISTORY_WINDOW) {
        cognitiveStateHistory.browHistory.shift();
    }
}

/**
 * Calculate cognitive states from features
 */
function calculateCognitiveStates(features, headPose) {
    const states = {
        confusion: 0,
        engagement: 0,
        boredom: 0,
        frustration: 0
    };

    // Get baseline values from history
    const avgEyeOpenness = average(cognitiveStateHistory.eyeOpennessHistory) || features.eyeOpenness;
    const eyeOpennessStd = standardDeviation(cognitiveStateHistory.eyeOpennessHistory) || 0.01;

    // Head movement variance (higher = less engaged or frustrated)
    const headMovement = calculateHeadMovementVariance();

    // Normalize features
    const eyeOpennessNorm = (features.eyeOpenness - avgEyeOpenness) / (eyeOpennessStd + 0.001);

    // --- ENGAGEMENT ---
    // High engagement: eyes wide open, stable head, forward looking
    const eyeWideOpen = Math.max(0, eyeOpennessNorm);
    const headStable = Math.max(0, 1 - headMovement * 10);
    const lookingForward = Math.max(0, 1 - (Math.abs(headPose.yaw) + Math.abs(headPose.pitch)) / 20);
    states.engagement = clamp((eyeWideOpen * 0.3 + headStable * 0.3 + lookingForward * 0.4), 0, 1);

    // --- BOREDOM ---
    // Boredom: droopy eyelids, yawning, looking away, slow blinks
    const eyeDroopy = Math.max(0, -eyeOpennessNorm * 0.5);
    const yawning = features.mouthOpenness > 0.05 ? Math.min(1, features.mouthOpenness * 10) : 0;
    const lookingAway = Math.min(1, (Math.abs(headPose.yaw) + Math.abs(headPose.pitch)) / 30);
    states.boredom = clamp((eyeDroopy * 0.4 + yawning * 0.3 + lookingAway * 0.3), 0, 1);

    // --- CONFUSION ---
    // Confusion: furrowed brows, squinting, head tilt
    const browFurrowed = Math.max(0, 0.1 - features.browFurrow) * 10; // Closer brows = more furrowed
    const squinting = Math.max(0, -eyeOpennessNorm * 0.3);
    const headTilt = Math.min(1, Math.abs(headPose.roll) / 15);
    states.confusion = clamp((browFurrowed * 0.4 + squinting * 0.3 + headTilt * 0.3), 0, 1);

    // --- FRUSTRATION ---
    // Frustration: rapid blinking, tense expression, head movement
    const recentBlinks = blinkCount; // Blinks in recent window
    const rapidBlinking = Math.min(1, recentBlinks / 10); // More than 10 blinks = high
    const headRestless = Math.min(1, headMovement * 15);
    const tenseBrows = browFurrowed * 0.5;
    states.frustration = clamp((rapidBlinking * 0.4 + headRestless * 0.3 + tenseBrows * 0.3), 0, 1);

    // Reset blink count periodically (every ~3 seconds)
    if (cognitiveStateHistory.eyeOpennessHistory.length >= HISTORY_WINDOW) {
        blinkCount = Math.max(0, blinkCount - 1);
    }

    return states;
}

/**
 * Calculate head movement variance from history
 */
function calculateHeadMovementVariance() {
    if (cognitiveStateHistory.headPoseHistory.length < 2) return 0;

    const pitchValues = cognitiveStateHistory.headPoseHistory.map(h => h.pitch);
    const yawValues = cognitiveStateHistory.headPoseHistory.map(h => h.yaw);

    return (standardDeviation(pitchValues) + standardDeviation(yawValues)) / 2;
}

/**
 * Update cognitive state UI bars
 */
function updateCognitiveStateUI() {
    const confusionPct = Math.round(cognitiveStates.confusion * 100);
    const engagementPct = Math.round(cognitiveStates.engagement * 100);
    const boredomPct = Math.round(cognitiveStates.boredom * 100);
    const frustrationPct = Math.round(cognitiveStates.frustration * 100);

    document.getElementById('confusion-bar').style.width = `${confusionPct}%`;
    document.getElementById('confusion-value').textContent = `${confusionPct}%`;

    document.getElementById('engagement-bar').style.width = `${engagementPct}%`;
    document.getElementById('engagement-value').textContent = `${engagementPct}%`;

    document.getElementById('boredom-bar').style.width = `${boredomPct}%`;
    document.getElementById('boredom-value').textContent = `${boredomPct}%`;

    document.getElementById('frustration-bar').style.width = `${frustrationPct}%`;
    document.getElementById('frustration-value').textContent = `${frustrationPct}%`;
}

// --- Utility Functions ---

function smoothValue(current, target, factor) {
    return current + (target - current) * factor;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function average(arr) {
    if (!arr || arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function standardDeviation(arr) {
    if (!arr || arr.length < 2) return 0;
    const avg = average(arr);
    const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
    return Math.sqrt(average(squareDiffs));
}

/**
 * Setup the gaze visualization canvas
 */
function setupGazeCanvas() {
    gazeCanvas = document.getElementById('gaze-canvas');
    const container = document.getElementById('gaze-display');

    // Set canvas size
    const resizeCanvas = () => {
        gazeCanvas.width = container.clientWidth;
        gazeCanvas.height = container.clientHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    gazeCtx = gazeCanvas.getContext('2d');
}

/**
 * Setup the face mesh visualization canvas
 */
function setupFaceCanvas() {
    faceCanvas = document.getElementById('face-canvas');
    const container = faceCanvas.parentElement;

    // Set canvas size
    const resizeCanvas = () => {
        faceCanvas.width = container.clientWidth;
        faceCanvas.height = container.clientHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    faceCtx = faceCanvas.getContext('2d');

    // Fill with dark background
    faceCtx.fillStyle = '#0f3460';
    faceCtx.fillRect(0, 0, faceCanvas.width, faceCanvas.height);
}

/**
 * Connect to the WebSocket server
 */
function connectWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('Dashboard WebSocket connected');
        connectionStatus.classList.add('connected');
        connectionText.textContent = 'Connected';
        addLogEntry('system', 'Connected to server');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleIncomingData(data);
    };

    ws.onclose = () => {
        console.log('Dashboard WebSocket disconnected');
        connectionStatus.classList.remove('connected');
        connectionText.textContent = 'Disconnected';
        addLogEntry('system', 'Disconnected from server');

        // Attempt reconnection
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('Dashboard WebSocket error:', error);
    };
}

/**
 * Handle incoming data from WebSocket
 */
function handleIncomingData(data) {
    const { type, session_id } = data;

    // Update session ID if needed
    if (session_id && sessionId !== session_id) {
        sessionId = session_id;
        document.getElementById('session-id').textContent = session_id;
        if (!startTime) startTime = Date.now();
    }

    switch (type) {
        case 'gaze':
            handleGazeData(data.data);
            break;
        case 'face_mesh':
            handleFaceMeshData(data.data);
            break;
        case 'emotion':
            handleEmotionData(data.data);
            break;
        case 'mouse':
            handleMouseData(data.data);
            break;
        case 'keyboard':
            handleKeyboardData(data.data);
            break;
        case 'session':
            handleSessionEvent(data.data);
            break;
    }
}

/**
 * Handle gaze data
 */
function handleGazeData(data) {
    stats.gazeSamples++;
    stats.totalSamples++;

    // Update gaze position display
    document.getElementById('gaze-x').textContent = Math.round(data.x);
    document.getElementById('gaze-y').textContent = Math.round(data.y);

    // Normalize to canvas coordinates
    const canvasX = (data.x / window.screen.width) * gazeCanvas.width;
    const canvasY = (data.y / window.screen.height) * gazeCanvas.height;

    // Add to history
    gazeHistory.push({ x: canvasX, y: canvasY, time: Date.now() });
    if (gazeHistory.length > GAZE_HISTORY_LENGTH) {
        gazeHistory.shift();
    }

    // Draw gaze visualization
    drawGazeVisualization();

    // Update gaze point overlay
    gazePoint.style.left = `${(data.x / window.screen.width) * 100}%`;
    gazePoint.style.top = `${(data.y / window.screen.height) * 100}%`;
}

/**
 * Draw gaze visualization on canvas
 */
function drawGazeVisualization() {
    // Clear canvas
    gazeCtx.fillStyle = 'rgba(15, 52, 96, 0.1)';
    gazeCtx.fillRect(0, 0, gazeCanvas.width, gazeCanvas.height);

    if (gazeHistory.length < 2) return;

    // Draw trail
    gazeCtx.beginPath();
    gazeCtx.moveTo(gazeHistory[0].x, gazeHistory[0].y);

    for (let i = 1; i < gazeHistory.length; i++) {
        gazeCtx.lineTo(gazeHistory[i].x, gazeHistory[i].y);
    }

    gazeCtx.strokeStyle = 'rgba(233, 69, 96, 0.5)';
    gazeCtx.lineWidth = 2;
    gazeCtx.stroke();

    // Draw current point
    const current = gazeHistory[gazeHistory.length - 1];
    gazeCtx.beginPath();
    gazeCtx.arc(current.x, current.y, 10, 0, Math.PI * 2);
    gazeCtx.fillStyle = 'rgba(233, 69, 96, 0.8)';
    gazeCtx.fill();

    // Draw heatmap effect for recent points
    for (let i = Math.max(0, gazeHistory.length - 20); i < gazeHistory.length; i++) {
        const point = gazeHistory[i];
        const alpha = (i - (gazeHistory.length - 20)) / 20 * 0.3;

        gazeCtx.beginPath();
        gazeCtx.arc(point.x, point.y, 20, 0, Math.PI * 2);
        gazeCtx.fillStyle = `rgba(233, 69, 96, ${alpha})`;
        gazeCtx.fill();
    }
}

/**
 * Handle face mesh data - stores all 468 landmarks
 */
function handleFaceMeshData(data) {
    stats.faceSamples++;
    stats.totalSamples++;

    // Store landmarks for rendering
    latestLandmarks = data;

    // Update landmark count display
    if (data.landmark_count) {
        document.getElementById('landmark-count').textContent = data.landmark_count;
    }

    // Update head pose display
    if (data.head_pose) {
        const pitch = data.head_pose.pitch.toFixed(1);
        const yaw = data.head_pose.yaw.toFixed(1);
        document.getElementById('head-pose').textContent = `P:${pitch} Y:${yaw}`;
    }
}

/**
 * Render face mesh landmarks on canvas (called in animation loop)
 */
function renderFaceMesh() {
    if (!faceCtx || !faceCanvas) {
        requestAnimationFrame(renderFaceMesh);
        return;
    }

    // Clear canvas with dark background
    faceCtx.fillStyle = '#0f3460';
    faceCtx.fillRect(0, 0, faceCanvas.width, faceCanvas.height);

    if (!latestLandmarks || !latestLandmarks.landmarks) {
        // Draw placeholder text
        faceCtx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        faceCtx.font = '14px sans-serif';
        faceCtx.textAlign = 'center';
        faceCtx.fillText('Waiting for face data...', faceCanvas.width / 2, faceCanvas.height / 2);
        requestAnimationFrame(renderFaceMesh);
        return;
    }

    const landmarks = latestLandmarks.landmarks;
    const w = faceCanvas.width;
    const h = faceCanvas.height;

    // Draw connections (face mesh tesselation - simplified)
    // Using key face outline connections
    const faceOutline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10];
    const leftEye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33];
    const rightEye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362];
    const lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61];

    // Draw connections
    faceCtx.strokeStyle = 'rgba(78, 204, 163, 0.4)';
    faceCtx.lineWidth = 1;

    const drawPath = (indices) => {
        faceCtx.beginPath();
        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            if (idx < landmarks.length) {
                const x = (1 - landmarks[idx].x) * w;  // Mirror horizontally
                const y = landmarks[idx].y * h;
                if (i === 0) {
                    faceCtx.moveTo(x, y);
                } else {
                    faceCtx.lineTo(x, y);
                }
            }
        }
        faceCtx.stroke();
    };

    drawPath(faceOutline);
    drawPath(leftEye);
    drawPath(rightEye);
    drawPath(lips);

    // Draw all 468 landmark points
    for (let i = 0; i < landmarks.length; i++) {
        const lm = landmarks[i];
        const x = (1 - lm.x) * w;  // Mirror horizontally
        const y = lm.y * h;

        // Color based on depth (z value)
        const depth = Math.max(0, Math.min(1, (lm.z + 0.1) * 5));
        const r = Math.round(78 + depth * 155);
        const g = Math.round(204 - depth * 100);
        const b = Math.round(163 - depth * 50);

        faceCtx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
        faceCtx.beginPath();
        faceCtx.arc(x, y, 1.5, 0, Math.PI * 2);
        faceCtx.fill();
    }

    // Draw key points with larger dots
    if (latestLandmarks.key_points) {
        const keyPoints = latestLandmarks.key_points;
        faceCtx.fillStyle = '#e94560';

        const drawKeyPoint = (point, label) => {
            if (!point) return;
            const x = (1 - point.x) * w;
            const y = point.y * h;

            faceCtx.beginPath();
            faceCtx.arc(x, y, 4, 0, Math.PI * 2);
            faceCtx.fill();
        };

        drawKeyPoint(keyPoints.nose_tip);
        drawKeyPoint(keyPoints.left_eye_center);
        drawKeyPoint(keyPoints.right_eye_center);
        drawKeyPoint(keyPoints.forehead);
        drawKeyPoint(keyPoints.chin);
    }

    requestAnimationFrame(renderFaceMesh);
}

/**
 * Handle emotion/cognitive state data
 */
function handleEmotionData(data) {
    // Update emotion bars
    if (data.confusion !== undefined) {
        const pct = Math.round(data.confusion * 100);
        document.getElementById('confusion-bar').style.width = `${pct}%`;
        document.getElementById('confusion-value').textContent = `${pct}%`;
    }

    if (data.engagement !== undefined) {
        const pct = Math.round(data.engagement * 100);
        document.getElementById('engagement-bar').style.width = `${pct}%`;
        document.getElementById('engagement-value').textContent = `${pct}%`;
    }

    if (data.boredom !== undefined) {
        const pct = Math.round(data.boredom * 100);
        document.getElementById('boredom-bar').style.width = `${pct}%`;
        document.getElementById('boredom-value').textContent = `${pct}%`;
    }

    if (data.frustration !== undefined) {
        const pct = Math.round(data.frustration * 100);
        document.getElementById('frustration-bar').style.width = `${pct}%`;
        document.getElementById('frustration-value').textContent = `${pct}%`;
    }
}

/**
 * Handle mouse data
 */
function handleMouseData(data) {
    stats.mouseEvents++;
    stats.totalSamples++;

    // Update mouse position display
    if (data.x !== undefined) {
        document.getElementById('mouse-x').textContent = Math.round(data.x);
        document.getElementById('mouse-y').textContent = Math.round(data.y);
    }

    // Log click events
    if (data.event === 'click') {
        addLogEntry('mouse', `Click at (${data.x}, ${data.y})`);
    }
}

/**
 * Handle keyboard data
 */
function handleKeyboardData(data) {
    stats.keyboardEvents++;
    stats.totalSamples++;

    // Log key events
    if (data.event === 'keydown') {
        addLogEntry('keyboard', `Key: ${data.key}`);
    }
}

/**
 * Handle session events
 */
function handleSessionEvent(data) {
    if (data.event === 'start') {
        startTime = Date.now();
        isCollecting = true;
        addLogEntry('system', 'Session started');
        if (data.content_url) {
            addLogEntry('system', `Content: ${data.content_url}`);
        }
    } else if (data.event === 'end') {
        isCollecting = false;
        addLogEntry('system', 'Session ended');
    }
}

/**
 * Add entry to event log
 */
function addLogEntry(type, message) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString();

    const entry = document.createElement('div');
    entry.className = 'event-item';
    entry.innerHTML = `
        <span class="event-time">${timeStr}</span>
        <span class="event-type ${type}">${type}</span>
        <span>${message}</span>
    `;

    eventLog.insertBefore(entry, eventLog.firstChild);

    // Keep log size manageable
    while (eventLog.children.length > 100) {
        eventLog.removeChild(eventLog.lastChild);
    }
}

/**
 * Update statistics display
 */
function updateStats() {
    // Update sample counts
    document.getElementById('gaze-samples').textContent = stats.gazeSamples;
    document.getElementById('face-samples').textContent = stats.faceSamples;
    document.getElementById('total-samples').textContent = stats.totalSamples;
    document.getElementById('mouse-events').textContent = stats.mouseEvents;
    document.getElementById('keyboard-events').textContent = stats.keyboardEvents;

    // Update duration
    if (startTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        document.getElementById('session-duration').textContent =
            `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    // Update gaze rate
    if (startTime) {
        const elapsed = (Date.now() - startTime) / 1000;
        const rate = elapsed > 0 ? Math.round(stats.gazeSamples / elapsed) : 0;
        document.getElementById('gaze-rate').textContent = rate;
    }
}

/**
 * Start data collection (from dashboard)
 */
window.startCollection = function() {
    // Open experiment page in new tab
    window.open('/', '_blank');
    addLogEntry('system', 'Experiment page opened');
};

/**
 * Export session data
 */
window.exportData = async function() {
    if (!sessionId) {
        alert('No active session to export');
        return;
    }

    try {
        const response = await fetch(`/api/session/${sessionId}/export?format=csv`, {
            method: 'POST'
        });
        const result = await response.json();

        if (result.success) {
            addLogEntry('system', `Data exported to ${result.filepath}`);
            alert(`Data exported to: ${result.filepath}`);
        } else {
            alert('Export failed: ' + result.error);
        }
    } catch (error) {
        console.error('Export error:', error);
        alert('Export failed');
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
