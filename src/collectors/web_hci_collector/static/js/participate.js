/**
 * Participant Page Controller
 *
 * Manages the full experiment flow:
 * 1. Welcome — instructions, webcam permission
 * 2. Calibration — inline 9-point gaze calibration
 * 3. Content — iframe experiment with full data collection
 * 4. Complete — thank-you screen
 *
 * Data collected: gaze, face mesh, mouse, keyboard, hover, answers, video
 */

// ── Session from URL ──────────────────────────────────────
const SESSION_ID = window.location.pathname.split('/participate/')[1];
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${WS_PROTOCOL}//${window.location.host}/ws/collect/${SESSION_ID}`;

// ── State ─────────────────────────────────────────────────
let ws = null;
let isCollecting = false;
let experimentConfig = {};
let currentStep = 'welcome';

// Trackers
let faceMesh = null;
let camera = null;

// Screen recording
let screenStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = 0;
const VIDEO_CHUNK_INTERVAL = 1000;

// L2CS gaze
let l2csCanvas = null;
let l2csCtx = null;
let lastL2CSFrameTime = 0;
const L2CS_FRAME_INTERVAL = 100;
const L2CS_ENABLED = true;

// Calibration
const CAL_POINTS = [
    { x: 10, y: 10 }, { x: 50, y: 10 }, { x: 90, y: 10 },
    { x: 10, y: 50 }, { x: 50, y: 50 }, { x: 90, y: 50 },
    { x: 10, y: 90 }, { x: 50, y: 90 }, { x: 90, y: 90 },
];
const CLICKS_PER_POINT = 5;
let calCurrentPoint = 0;
let calCurrentClicks = 0;
let calPointElements = [];

// DOM
const gazeCursor = document.getElementById('gaze-cursor');
const contentFrame = document.getElementById('content-frame');
const floatingControls = document.getElementById('floating-controls');

// ── Initialization ────────────────────────────────────────

async function init() {
    console.log('Participant page init, session:', SESSION_ID);

    if (!SESSION_ID) {
        document.body.innerHTML = '<div style="padding:40px;text-align:center;color:#fff;"><h2>Invalid session</h2><p>No session ID in URL.</p></div>';
        return;
    }

    // Fetch session config
    try {
        const resp = await fetch(`/api/session/${SESSION_ID}/config`);
        const data = await resp.json();
        if (data.error) {
            document.body.innerHTML = `<div style="padding:40px;text-align:center;color:#fff;"><h2>Error</h2><p>${data.error}</p></div>`;
            return;
        }
        experimentConfig = data.experiment_config || {};
    } catch (e) {
        console.error('Failed to fetch config:', e);
    }

    // Update UI
    document.getElementById('session-tag').textContent = `Session: ${SESSION_ID}`;
    if (experimentConfig.experiment_name) {
        document.getElementById('experiment-name').textContent = experimentConfig.experiment_name;
    }

    // Connect WebSocket
    connectWebSocket();

    // Pre-init WebGazer
    await initWebGazer();

    // Build calibration UI
    buildCalibrationUI();
}

// ── WebSocket ─────────────────────────────────────────────

function connectWebSocket() {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => console.log('WS connected');
    ws.onclose = () => { console.log('WS disconnected'); setTimeout(connectWebSocket, 3000); };
    ws.onerror = (e) => console.error('WS error:', e);
}

function sendData(type, data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type, timestamp: performance.now(), data }));
    }
}

// ── Step Navigation ───────────────────────────────────────

const STEPS = ['welcome', 'calibration', 'content', 'complete'];

function showStep(step) {
    STEPS.forEach(s => {
        const el = document.getElementById(`step-${s}`);
        if (el) el.classList.toggle('active', s === step);
    });
    currentStep = step;
    updateProgressBar(step);

    // Notify server
    sendData('step_transition', { step });
}

function updateProgressBar(step) {
    const idx = STEPS.indexOf(step);
    STEPS.forEach((s, i) => {
        const dot = document.getElementById(`dot-${s}`);
        const lbl = document.getElementById(`lbl-${s}`);
        if (dot) {
            dot.classList.remove('current', 'completed');
            if (i < idx) dot.classList.add('completed');
            else if (i === idx) dot.classList.add('current');
        }
        if (lbl) {
            lbl.classList.remove('current');
            if (i === idx) lbl.classList.add('current');
        }
        // Connectors
        if (i > 0) {
            const conn = document.getElementById(`conn-${i}`);
            if (conn) {
                conn.classList.toggle('completed', i <= idx);
            }
        }
    });

    // Hide progress bar during content step (fullscreen)
    const bar = document.getElementById('step-progress');
    bar.classList.toggle('hidden', step === 'content');
}

// ── Step 1: Welcome → Begin ───────────────────────────────

window.beginStudy = function () {
    const requireCal = experimentConfig.require_calibration !== false;
    if (requireCal) {
        showStep('calibration');
        startCalibration();
    } else {
        showStep('content');
        startContent();
    }
};

// ── Step 2: Calibration ───────────────────────────────────

function buildCalibrationUI() {
    const container = document.getElementById('step-calibration');
    const progress = document.getElementById('cal-progress');

    CAL_POINTS.forEach((pt, i) => {
        // Calibration dot
        const el = document.createElement('div');
        el.className = 'calibration-point';
        el.id = `cal-pt-${i}`;
        el.style.left = `${pt.x}%`;
        el.style.top = `${pt.y}%`;
        el.onclick = () => handleCalClick(i);
        container.appendChild(el);
        calPointElements.push(el);

        // Progress dot
        const pd = document.createElement('div');
        pd.className = 'cal-dot';
        pd.id = `cal-pdot-${i}`;
        progress.appendChild(pd);
    });
}

function startCalibration() {
    calCurrentPoint = 0;
    calCurrentClicks = 0;
    showCalPoint(0);
}

function showCalPoint(idx) {
    calPointElements.forEach((el, i) => {
        el.classList.remove('active');
        if (i === idx) el.classList.add('active');
    });
    CAL_POINTS.forEach((_, i) => {
        const pd = document.getElementById(`cal-pdot-${i}`);
        pd.classList.remove('cur', 'done');
        if (i < idx) pd.classList.add('done');
        if (i === idx) pd.classList.add('cur');
    });
    document.getElementById('cal-sub').textContent = `Point ${idx + 1} of ${CAL_POINTS.length} — click ${CLICKS_PER_POINT} times`;
}

function handleCalClick(idx) {
    if (idx !== calCurrentPoint) return;
    calCurrentClicks++;

    // Visual feedback
    const pt = calPointElements[idx];
    pt.style.transform = 'translate(-50%, -50%) scale(0.8)';
    setTimeout(() => { pt.style.transform = 'translate(-50%, -50%) scale(1)'; }, 100);

    const remaining = CLICKS_PER_POINT - calCurrentClicks;
    document.getElementById('cal-sub').textContent =
        remaining > 0
            ? `Point ${idx + 1} of ${CAL_POINTS.length} — click ${remaining} more`
            : 'Moving to next point...';

    if (calCurrentClicks >= CLICKS_PER_POINT) {
        pt.classList.remove('active');
        pt.classList.add('done');
        document.getElementById(`cal-pdot-${idx}`).classList.add('done');
        document.getElementById(`cal-pdot-${idx}`).classList.remove('cur');

        calCurrentClicks = 0;
        calCurrentPoint++;

        if (calCurrentPoint >= CAL_POINTS.length) {
            finishCalibration();
        } else {
            showCalPoint(calCurrentPoint);
        }
    }
}

function finishCalibration() {
    localStorage.setItem('webgazer_calibrated', 'true');
    localStorage.setItem('webgazer_calibrated_at', new Date().toISOString());
    sendData('calibration_complete', { points: CAL_POINTS.length, clicks_per_point: CLICKS_PER_POINT });

    document.getElementById('cal-instruction').textContent = 'Calibration complete!';
    document.getElementById('cal-sub').textContent = 'Starting experiment...';

    setTimeout(() => {
        showStep('content');
        startContent();
    }, 800);
}

// ── Step 3: Content & Data Collection ─────────────────────

async function startContent() {
    const contentUrl = experimentConfig.content_url || '/static/experiments/green_energy_demo.html';
    console.log('Loading content:', contentUrl);

    contentFrame.src = contentUrl;

    // Wait for iframe load
    await new Promise(resolve => {
        const timeout = setTimeout(resolve, 5000);
        contentFrame.onload = () => {
            clearTimeout(timeout);
            setupIframeTracking();
            resolve();
        };
    });

    // Init MediaPipe + camera
    await initFaceMesh();

    // Start screen recording
    await startScreenRecording();

    // Enable collection
    isCollecting = true;
    floatingControls.classList.remove('hidden');
    gazeCursor.style.display = 'block';

    sendData('session', {
        event: 'start',
        content_url: contentUrl,
        userAgent: navigator.userAgent,
        screenWidth: window.screen.width,
        screenHeight: window.screen.height,
        windowWidth: window.innerWidth,
        windowHeight: window.innerHeight,
    });

    // Listen for iframe messages
    window.addEventListener('message', handleIframeMessage);
}

function handleIframeMessage(e) {
    if (!isCollecting) return;

    // Mouse events from iframe
    if (e.data?.type === 'hci_mouse_event') {
        const iframe = contentFrame;
        const rect = iframe.getBoundingClientRect();
        const d = e.data.data;
        sendData('mouse', {
            event: d.event,
            x: rect.left + d.x,
            y: rect.top + d.y,
            iframeX: d.x, iframeY: d.y,
            button: d.button, target: d.target,
            deltaX: d.deltaX, deltaY: d.deltaY,
            scrollX: d.scrollX, scrollY: d.scrollY,
            overIframe: true,
            source: 'iframe_postmessage',
        });
    }

    // Structured answer from experiment content
    if (e.data?.type === 'hci_answer') {
        sendData('answer', e.data.data);
    }

    // Generic experiment event
    if (e.data?.type === 'hci_experiment_event') {
        const ev = e.data.data;
        sendData('experiment_event', ev);

        // Detect answer submission from legacy events
        if (ev.type === 'answer_submit' || ev.type === 'answer_select') {
            sendData('answer', {
                question_id: ev.question_id || `${ev.session || 'unknown'}_q`,
                selected_answer: ev.answer,
                response_time_ms: ev.timestamp || ev.ts,
                step: String(ev.session || ''),
                raw_event: ev,
            });
        }
    }

    // Experiment complete signal
    if (e.data?.type === 'hci_experiment_complete') {
        sendData('experiment_event', { type: 'experiment_complete', ...e.data.data });
        endExperiment();
    }
}

// ── End Experiment ────────────────────────────────────────

window.endExperiment = async function () {
    if (!isCollecting) return;
    isCollecting = false;

    await stopScreenRecordingAsync();

    sendData('session', { event: 'end', saveData: true });

    gazeCursor.style.display = 'none';
    floatingControls.classList.add('hidden');

    // Export data
    try {
        await fetch(`/api/session/${SESSION_ID}/export?format=csv`, { method: 'POST' });
    } catch (e) { console.error('Export error:', e); }

    webgazer.end();

    document.getElementById('complete-session').textContent = `Session: ${SESSION_ID}`;
    showStep('complete');
};

// ── WebGazer ──────────────────────────────────────────────

async function initWebGazer() {
    try {
        await webgazer
            .setGazeListener((data, timestamp) => {
                if (data && isCollecting) {
                    gazeCursor.style.left = `${data.x}px`;
                    gazeCursor.style.top = `${data.y}px`;
                    sendData('gaze', { x: data.x, y: data.y, timestamp });
                }
            })
            .saveDataAcrossSessions(true)
            .begin();

        webgazer.showVideoPreview(false);
        webgazer.showPredictionPoints(false);
        webgazer.showFaceOverlay(false);
        webgazer.showFaceFeedbackBox(false);
        console.log('WebGazer initialized');
    } catch (e) {
        console.error('WebGazer init error:', e);
    }
}

// ── MediaPipe Face Mesh ───────────────────────────────────

async function initFaceMesh() {
    try {
        faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
        });
        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });
        faceMesh.onResults(onFaceMeshResults);

        const videoEl = document.getElementById('webcam-video');
        camera = new Camera(videoEl, {
            onFrame: async () => {
                if (isCollecting) {
                    await faceMesh.send({ image: videoEl });
                    if (L2CS_ENABLED) sendL2CSFrame(videoEl);
                }
            },
            width: 640, height: 480,
        });
        await camera.start();

        if (L2CS_ENABLED) {
            l2csCanvas = document.createElement('canvas');
            l2csCanvas.width = 640; l2csCanvas.height = 480;
            l2csCtx = l2csCanvas.getContext('2d');
        }
        console.log('Face Mesh initialized');
    } catch (e) {
        console.error('Face Mesh init error:', e);
    }
}

function onFaceMeshResults(results) {
    if (!results.multiFaceLandmarks?.length) return;
    const landmarks = results.multiFaceLandmarks[0];

    const noseTip = landmarks[1];
    const leftEyeInner = landmarks[33], leftEyeOuter = landmarks[133];
    const rightEyeInner = landmarks[263], rightEyeOuter = landmarks[362];
    const forehead = landmarks[10], chin = landmarks[152];

    const leftEyeCenter = {
        x: (leftEyeInner.x + leftEyeOuter.x) / 2,
        y: (leftEyeInner.y + leftEyeOuter.y) / 2,
        z: (leftEyeInner.z + leftEyeOuter.z) / 2,
    };
    const rightEyeCenter = {
        x: (rightEyeInner.x + rightEyeOuter.x) / 2,
        y: (rightEyeInner.y + rightEyeOuter.y) / 2,
        z: (rightEyeInner.z + rightEyeOuter.z) / 2,
    };
    const eyeCenter = {
        x: (leftEyeCenter.x + rightEyeCenter.x) / 2,
        y: (leftEyeCenter.y + rightEyeCenter.y) / 2,
        z: (leftEyeCenter.z + rightEyeCenter.z) / 2,
    };

    sendData('face_mesh', {
        landmarks: landmarks.map(l => ({ x: l.x, y: l.y, z: l.z })),
        landmark_count: landmarks.length,
        key_points: {
            nose_tip: { x: noseTip.x, y: noseTip.y, z: noseTip.z },
            left_eye_center: leftEyeCenter,
            right_eye_center: rightEyeCenter,
            eye_center: eyeCenter,
            forehead: { x: forehead.x, y: forehead.y, z: forehead.z },
            chin: { x: chin.x, y: chin.y, z: chin.z },
        },
        head_pose: {
            pitch: (noseTip.y - eyeCenter.y) * 100,
            yaw: (noseTip.x - eyeCenter.x) * 100,
            roll: (leftEyeCenter.y - rightEyeCenter.y) * 100,
        },
        bounding_box: {
            min_x: Math.min(...landmarks.map(l => l.x)),
            max_x: Math.max(...landmarks.map(l => l.x)),
            min_y: Math.min(...landmarks.map(l => l.y)),
            max_y: Math.max(...landmarks.map(l => l.y)),
        },
    });
}

// ── L2CS Frame Capture ────────────────────────────────────

function sendL2CSFrame(videoEl) {
    const now = performance.now();
    if (now - lastL2CSFrameTime < L2CS_FRAME_INTERVAL) return;
    lastL2CSFrameTime = now;
    if (!l2csCanvas || !l2csCtx || !videoEl) return;
    try {
        l2csCtx.drawImage(videoEl, 0, 0, l2csCanvas.width, l2csCanvas.height);
        sendData('gaze_frame', {
            frame: l2csCanvas.toDataURL('image/jpeg', 0.7),
            width: l2csCanvas.width,
            height: l2csCanvas.height,
            screen_width: window.screen.width,
            screen_height: window.screen.height,
        });
    } catch (e) { /* ignore */ }
}

// ── Mouse & Keyboard Tracking ─────────────────────────────

function setupMouseTracking() {
    let lastMove = 0;
    document.addEventListener('mousemove', (e) => {
        if (!isCollecting) return;
        const now = performance.now();
        if (now - lastMove < 50) return;
        lastMove = now;
        sendData('mouse', { event: 'move', x: e.clientX, y: e.clientY, source: 'parent' });
    }, true);

    document.addEventListener('click', (e) => {
        if (!isCollecting) return;
        sendData('mouse', { event: 'click', x: e.clientX, y: e.clientY, button: e.button, target: e.target?.tagName || '', source: 'parent' });
    }, true);

    document.addEventListener('wheel', (e) => {
        if (!isCollecting) return;
        sendData('mouse', { event: 'wheel', x: e.clientX, y: e.clientY, deltaX: e.deltaX, deltaY: e.deltaY, source: 'parent' });
    }, true);

    window.addEventListener('scroll', () => {
        if (!isCollecting) return;
        sendData('mouse', { event: 'scroll', scrollX: window.scrollX, scrollY: window.scrollY, source: 'parent' });
    }, true);

    document.addEventListener('contextmenu', (e) => {
        if (!isCollecting) return;
        sendData('mouse', { event: 'rightclick', x: e.clientX, y: e.clientY, source: 'parent' });
    }, true);
}

function setupKeyboardTracking() {
    const special = [
        'Enter','Tab','Escape','Backspace','Delete',
        'ArrowUp','ArrowDown','ArrowLeft','ArrowRight',
        'Home','End','PageUp','PageDown',
        'Control','Alt','Shift','Meta',
        'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12',
    ];
    const safeKey = (e) => special.includes(e.key) ? e.key : 'char';

    document.addEventListener('keydown', (e) => {
        if (!isCollecting) return;
        sendData('keyboard', { event: 'keydown', key: safeKey(e), code: e.code, ctrlKey: e.ctrlKey, altKey: e.altKey, shiftKey: e.shiftKey, metaKey: e.metaKey });
    });
    document.addEventListener('keyup', (e) => {
        if (!isCollecting) return;
        sendData('keyboard', { event: 'keyup', key: safeKey(e), code: e.code });
    });
}

// ── Iframe Tracking (same-origin) ─────────────────────────

function setupIframeTracking() {
    const iframe = contentFrame;
    setTimeout(() => {
        try {
            const iDoc = iframe.contentDocument || iframe.contentWindow?.document;
            if (iDoc) {
                injectIframeMouseTracking(iframe, iDoc);
                injectHoverTracking(iDoc);
            }
        } catch (e) {
            console.log('Cross-origin iframe — relying on postMessage');
            try {
                iframe.contentWindow.postMessage({ type: 'hci_enable_mouse_tracking', sessionId: SESSION_ID }, '*');
            } catch (_) {}
        }
    }, 500);
}

function injectIframeMouseTracking(iframe, iDoc) {
    let lastMove = 0;
    const rect = () => iframe.getBoundingClientRect();

    iDoc.addEventListener('mousemove', (e) => {
        if (!isCollecting) return;
        const now = performance.now();
        if (now - lastMove < 50) return;
        lastMove = now;
        const r = rect();
        sendData('mouse', { event: 'move', x: r.left + e.clientX, y: r.top + e.clientY, iframeX: e.clientX, iframeY: e.clientY, overIframe: true, source: 'iframe_injected' });
    }, true);

    iDoc.addEventListener('click', (e) => {
        if (!isCollecting) return;
        const r = rect();
        sendData('mouse', { event: 'click', x: r.left + e.clientX, y: r.top + e.clientY, iframeX: e.clientX, iframeY: e.clientY, button: e.button, target: e.target?.tagName || '', overIframe: true, source: 'iframe_injected' });
    }, true);

    iDoc.addEventListener('wheel', (e) => {
        if (!isCollecting) return;
        const r = rect();
        sendData('mouse', { event: 'wheel', x: r.left + e.clientX, y: r.top + e.clientY, deltaX: e.deltaX, deltaY: e.deltaY, overIframe: true, source: 'iframe_injected' });
    }, true);

    iframe.contentWindow.addEventListener('scroll', () => {
        if (!isCollecting) return;
        sendData('mouse', { event: 'scroll', scrollX: iframe.contentWindow.scrollX, scrollY: iframe.contentWindow.scrollY, overIframe: true, source: 'iframe_injected' });
    }, true);
}

function injectHoverTracking(iDoc) {
    iDoc.querySelectorAll('[data-aoi]').forEach(el => {
        let enterTime = null;
        el.addEventListener('mouseenter', () => {
            enterTime = performance.now();
            sendData('hover', { event: 'enter', aoi: el.dataset.aoi, element: el.tagName });
        });
        el.addEventListener('mouseleave', () => {
            const dwell = enterTime ? performance.now() - enterTime : 0;
            sendData('hover', { event: 'leave', aoi: el.dataset.aoi, dwell_time_ms: dwell });
            enterTime = null;
        });
    });
}

// ── Screen Recording ──────────────────────────────────────

async function startScreenRecording() {
    try {
        screenStream = await navigator.mediaDevices.getDisplayMedia({
            video: { cursor: 'always', displaySurface: 'browser', frameRate: { ideal: 15, max: 30 } },
            audio: false,
        });

        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
            ? 'video/webm;codecs=vp9'
            : MediaRecorder.isTypeSupported('video/webm;codecs=vp8')
                ? 'video/webm;codecs=vp8'
                : 'video/webm';

        mediaRecorder = new MediaRecorder(screenStream, { mimeType, videoBitsPerSecond: 1000000 });
        recordedChunks = [];
        recordingStartTime = Date.now();

        mediaRecorder.ondataavailable = (event) => {
            if (event.data?.size > 0) {
                recordedChunks.push(event.data);
                const reader = new FileReader();
                reader.onloadend = () => {
                    sendData('video_chunk', {
                        data: reader.result.split(',')[1],
                        mimeType, chunkIndex: recordedChunks.length - 1,
                        timestamp: Date.now() - recordingStartTime,
                        size: event.data.size,
                    });
                };
                reader.readAsDataURL(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            if (recordedChunks.length) {
                sendData('video_complete', {
                    totalChunks: recordedChunks.length,
                    totalSize: recordedChunks.reduce((a, c) => a + c.size, 0),
                    duration: Date.now() - recordingStartTime,
                    mimeType,
                });
            }
        };

        screenStream.getVideoTracks()[0].onended = () => stopScreenRecording();
        mediaRecorder.start(VIDEO_CHUNK_INTERVAL);

        sendData('video_start', {
            mimeType,
            width: screenStream.getVideoTracks()[0].getSettings().width,
            height: screenStream.getVideoTracks()[0].getSettings().height,
            startTime: recordingStartTime,
        });
        return true;
    } catch (e) {
        console.error('Screen recording error:', e);
        return false;
    }
}

function stopScreenRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    if (screenStream) { screenStream.getTracks().forEach(t => t.stop()); screenStream = null; }
    mediaRecorder = null;
}

function stopScreenRecordingAsync() {
    return new Promise(resolve => {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') {
            if (screenStream) { screenStream.getTracks().forEach(t => t.stop()); screenStream = null; }
            resolve(); return;
        }
        mediaRecorder.onstop = () => {
            if (recordedChunks.length) {
                sendData('video_complete', {
                    totalChunks: recordedChunks.length,
                    totalSize: recordedChunks.reduce((a, c) => a + c.size, 0),
                    duration: Date.now() - recordingStartTime,
                });
            }
            if (screenStream) { screenStream.getTracks().forEach(t => t.stop()); screenStream = null; }
            mediaRecorder = null;
            setTimeout(resolve, 500);
        };
        mediaRecorder.stop();
    });
}

// ── Boot ──────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    setupMouseTracking();
    setupKeyboardTracking();
    init();
});
