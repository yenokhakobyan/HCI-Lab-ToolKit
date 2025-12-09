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
    clickCount: 0,
    totalSamples: 0
};

// Participant view state
let participantScreenWidth = 1920;  // Default, updated from session data
let participantScreenHeight = 1080;
let participantWindowWidth = 1920;
let participantWindowHeight = 1080;
let contentUrl = null;

// Gaze visualization
let gazeHistory = [];
const GAZE_HISTORY_LENGTH = 100;

// Mouse tracking visualization
let mouseHistory = [];
const MOUSE_HISTORY_LENGTH = 50;
let mouseTrailCanvas = null;
let mouseTrailCtx = null;
let gazeTrailCanvas = null;
let gazeTrailCtx = null;

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

// Timeline state
let timelineData = {
    engagement: [],    // { time: ms, value: 0-1 }
    gaze: [],          // { time: ms, x, y }
    clicks: [],        // { time: ms, x, y }
    keys: [],          // { time: ms, key }
    events: [],        // { time: ms, type, data }
    faceMesh: [],      // { time: ms, landmarks, keyPoints, headPose }
    cognitiveStates: [], // { time: ms, confusion, engagement, boredom, frustration }
    mouse: [],         // { time: ms, x, y }
    scroll: [],        // { time: ms, scrollX, scrollY }
    navigation: [],    // { time: ms, url, title }
    screenshots: [],   // { time: ms, dataUrl } - captured screen images (legacy)
    videoChunks: []    // { time: ms, blob } - video recording chunks
};

// Video playback state
let videoBlob = null;
let videoUrl = null;
let videoElement = null;
let videoMimeType = 'video/webm';
let videoStartTime = 0;
let videoDuration = 0;

// Track current URL for navigation change detection
let currentNavigationUrl = null;
const TIMELINE_MAX_DURATION = 30 * 60 * 1000; // 30 minutes max
const TIMELINE_SAMPLE_INTERVAL = 500; // Sample engagement every 500ms
const FACEMESH_SAMPLE_INTERVAL = 100; // Sample face mesh every 100ms
const SCREENSHOT_INTERVAL = 2000; // Capture screenshot every 2 seconds
let lastScreenshotTime = 0;
let timelineMode = 'live'; // 'live' or 'playback'
let timelinePlaybackTime = 0; // Current playback position in ms
let timelineIsPlaying = false;
let timelineDragging = false;
let lastTimelineSample = 0;
let lastFaceMeshSample = 0;

// Canvas contexts
let faceCanvas = null;
let faceCtx = null;

// Timeline canvas contexts
let engagementChartCanvas = null;
let engagementChartCtx = null;
let gazeChartCanvas = null;
let gazeChartCtx = null;
let eventsChartCanvas = null;
let eventsChartCtx = null;

// DOM Elements
const connectionStatus = document.getElementById('connection-status');
const connectionText = document.getElementById('connection-text');
const eventLog = document.getElementById('event-log');

/**
 * Initialize the dashboard
 */
function init() {
    console.log('Initializing dashboard...');

    // Setup canvases
    setupFaceCanvas();
    setupTrailCanvases();

    // Initialize timeline
    initTimeline();

    // Initialize webcam preview
    initWebcam();

    // Connect to WebSocket
    connectWebSocket();

    // Start stats update loop
    setInterval(updateStats, 1000);

    // Start face mesh rendering loop
    requestAnimationFrame(renderFaceMesh);

    // Start trail rendering loop
    requestAnimationFrame(renderTrails);

    // Start timeline update loop
    setInterval(updateTimeline, 100);
}

/**
 * Setup trail canvases for mouse and gaze visualization
 */
function setupTrailCanvases() {
    const container = document.getElementById('participant-view-container');
    if (!container) return;

    // Mouse trail canvas
    mouseTrailCanvas = document.getElementById('mouse-trail-canvas');
    if (mouseTrailCanvas) {
        const resizeMouseCanvas = () => {
            mouseTrailCanvas.width = container.clientWidth;
            mouseTrailCanvas.height = container.clientHeight;
        };
        resizeMouseCanvas();
        window.addEventListener('resize', resizeMouseCanvas);
        mouseTrailCtx = mouseTrailCanvas.getContext('2d');
    }

    // Gaze trail canvas
    gazeTrailCanvas = document.getElementById('gaze-trail-canvas');
    if (gazeTrailCanvas) {
        const resizeGazeCanvas = () => {
            gazeTrailCanvas.width = container.clientWidth;
            gazeTrailCanvas.height = container.clientHeight;
        };
        resizeGazeCanvas();
        window.addEventListener('resize', resizeGazeCanvas);
        gazeTrailCtx = gazeTrailCanvas.getContext('2d');
    }
}

/**
 * Render trails for mouse and gaze
 */
function renderTrails() {
    // Render mouse trail
    if (mouseTrailCtx && mouseTrailCanvas) {
        mouseTrailCtx.clearRect(0, 0, mouseTrailCanvas.width, mouseTrailCanvas.height);

        if (mouseHistory.length > 1) {
            // Draw connected line segments with increasing opacity and width
            for (let i = 1; i < mouseHistory.length; i++) {
                const alpha = i / mouseHistory.length;
                mouseTrailCtx.beginPath();
                mouseTrailCtx.moveTo(mouseHistory[i - 1].x, mouseHistory[i - 1].y);
                mouseTrailCtx.lineTo(mouseHistory[i].x, mouseHistory[i].y);
                mouseTrailCtx.strokeStyle = `rgba(78, 204, 163, ${alpha * 0.7})`;
                mouseTrailCtx.lineWidth = 2 + alpha * 3;
                mouseTrailCtx.lineCap = 'round';
                mouseTrailCtx.stroke();
            }

            // Draw dots at each position for better visibility
            for (let i = 0; i < mouseHistory.length; i++) {
                const alpha = (i + 1) / mouseHistory.length;
                mouseTrailCtx.beginPath();
                mouseTrailCtx.arc(mouseHistory[i].x, mouseHistory[i].y, 2 + alpha * 2, 0, Math.PI * 2);
                mouseTrailCtx.fillStyle = `rgba(78, 204, 163, ${alpha * 0.8})`;
                mouseTrailCtx.fill();
            }
        }
    }

    // Render gaze trail (heatmap style)
    if (gazeTrailCtx && gazeTrailCanvas) {
        // Fade existing trail
        gazeTrailCtx.fillStyle = 'rgba(15, 52, 96, 0.05)';
        gazeTrailCtx.fillRect(0, 0, gazeTrailCanvas.width, gazeTrailCanvas.height);

        // Draw recent gaze points as heatmap
        for (let i = Math.max(0, gazeHistory.length - 30); i < gazeHistory.length; i++) {
            const point = gazeHistory[i];
            const age = (gazeHistory.length - i) / 30;
            const alpha = (1 - age) * 0.3;

            gazeTrailCtx.beginPath();
            gazeTrailCtx.arc(point.x, point.y, 15 + (1 - age) * 10, 0, Math.PI * 2);
            gazeTrailCtx.fillStyle = `rgba(233, 69, 96, ${alpha})`;
            gazeTrailCtx.fill();
        }
    }

    requestAnimationFrame(renderTrails);
}

/**
 * Initialize the timeline component
 */
function initTimeline() {
    // Get timeline canvases
    engagementChartCanvas = document.getElementById('timeline-engagement-chart');
    gazeChartCanvas = document.getElementById('timeline-gaze-chart');
    eventsChartCanvas = document.getElementById('timeline-events-chart');

    if (engagementChartCanvas) {
        engagementChartCtx = engagementChartCanvas.getContext('2d');
        setupTimelineCanvas(engagementChartCanvas);
    }
    if (gazeChartCanvas) {
        gazeChartCtx = gazeChartCanvas.getContext('2d');
        setupTimelineCanvas(gazeChartCanvas);
    }
    if (eventsChartCanvas) {
        eventsChartCtx = eventsChartCanvas.getContext('2d');
        setupTimelineCanvas(eventsChartCanvas);
    }

    // Setup scrubber interaction
    const track = document.getElementById('timeline-track');
    if (track) {
        track.addEventListener('mousedown', onTimelineMouseDown);
        track.addEventListener('touchstart', onTimelineTouchStart);
    }

    document.addEventListener('mousemove', onTimelineMouseMove);
    document.addEventListener('mouseup', onTimelineMouseUp);
    document.addEventListener('touchmove', onTimelineTouchMove);
    document.addEventListener('touchend', onTimelineMouseUp);

    // Setup play/pause button
    const playPauseBtn = document.getElementById('timeline-play-pause');
    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', toggleTimelinePlayback);
    }

    // Setup live button
    const liveBtn = document.getElementById('timeline-live');
    if (liveBtn) {
        liveBtn.addEventListener('click', jumpToLive);
    }

    console.log('Timeline initialized');
}

/**
 * Setup timeline canvas dimensions
 */
function setupTimelineCanvas(canvas) {
    const container = canvas.parentElement;
    if (!container) return;

    const resizeCanvas = () => {
        canvas.width = container.clientWidth;
        canvas.height = canvas.clientHeight || 30;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

/**
 * Handle timeline mouse down (start dragging)
 */
function onTimelineMouseDown(e) {
    e.preventDefault();
    timelineDragging = true;
    timelineMode = 'playback';
    updateTimelinePosition(e.clientX);
    updateLiveButtonState();
}

/**
 * Handle timeline touch start
 */
function onTimelineTouchStart(e) {
    e.preventDefault();
    timelineDragging = true;
    timelineMode = 'playback';
    if (e.touches.length > 0) {
        updateTimelinePosition(e.touches[0].clientX);
    }
    updateLiveButtonState();
}

/**
 * Handle timeline mouse move (dragging)
 */
function onTimelineMouseMove(e) {
    if (!timelineDragging) return;
    updateTimelinePosition(e.clientX);
}

/**
 * Handle timeline touch move
 */
function onTimelineTouchMove(e) {
    if (!timelineDragging) return;
    if (e.touches.length > 0) {
        updateTimelinePosition(e.touches[0].clientX);
    }
}

/**
 * Handle timeline mouse up (stop dragging)
 */
function onTimelineMouseUp() {
    timelineDragging = false;
}

/**
 * Update timeline position from mouse/touch position
 */
function updateTimelinePosition(clientX) {
    const track = document.getElementById('timeline-track');
    if (!track || !startTime) return;

    const rect = track.getBoundingClientRect();
    const x = clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));

    const sessionDuration = Date.now() - startTime;
    timelinePlaybackTime = ratio * sessionDuration;

    // Update scrubber position
    updateTimelineScrubber();
}

/**
 * Update timeline scrubber visual position
 */
function updateTimelineScrubber() {
    if (!startTime) return;

    const progress = document.getElementById('timeline-progress');
    const scrubber = document.getElementById('timeline-scrubber');
    const currentTimeEl = document.getElementById('timeline-current-time');

    const sessionDuration = Date.now() - startTime;
    const currentTime = timelineMode === 'live' ? sessionDuration : timelinePlaybackTime;
    const ratio = sessionDuration > 0 ? (currentTime / sessionDuration) * 100 : 0;

    if (progress) {
        progress.style.width = `${ratio}%`;
    }
    if (scrubber) {
        scrubber.style.left = `${ratio}%`;
    }
    if (currentTimeEl) {
        currentTimeEl.textContent = formatTime(currentTime);
    }
}

/**
 * Toggle timeline playback (play/pause)
 */
function toggleTimelinePlayback() {
    const playIcon = document.getElementById('play-icon');
    const pauseIcon = document.getElementById('pause-icon');

    if (timelineMode === 'live') {
        // Switch to playback mode, paused at current position
        timelineMode = 'playback';
        timelinePlaybackTime = startTime ? Date.now() - startTime : 0;
        timelineIsPlaying = false;
    } else {
        // Toggle play/pause in playback mode
        timelineIsPlaying = !timelineIsPlaying;
    }

    // Update button icons
    if (playIcon && pauseIcon) {
        if (timelineIsPlaying) {
            playIcon.style.display = 'none';
            pauseIcon.style.display = 'block';
        } else {
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
        }
    }

    updateLiveButtonState();
}

/**
 * Jump to live mode
 */
function jumpToLive() {
    timelineMode = 'live';
    timelineIsPlaying = false;
    timelinePlaybackTime = 0;

    // Reset play button
    const playIcon = document.getElementById('play-icon');
    const pauseIcon = document.getElementById('pause-icon');
    if (playIcon && pauseIcon) {
        playIcon.style.display = 'block';
        pauseIcon.style.display = 'none';
    }

    // Clear playback click markers
    const clickContainer = document.getElementById('click-ripple-container');
    if (clickContainer) {
        const playbackMarkers = clickContainer.querySelectorAll('.playback-click-marker');
        playbackMarkers.forEach(m => m.remove());
    }

    // Hide scroll indicator
    const scrollIndicator = document.getElementById('scroll-position-indicator');
    if (scrollIndicator) {
        scrollIndicator.style.display = 'none';
    }

    // Hide navigation indicator
    const navIndicator = document.getElementById('navigation-indicator');
    if (navIndicator) {
        navIndicator.style.display = 'none';
    }

    // Hide video element and stop playback
    if (videoElement) {
        videoElement.pause();
        videoElement.style.display = 'none';
    }

    // Hide screenshot overlay and badge, restore iframe
    const screenshotOverlay = document.getElementById('screenshot-overlay');
    if (screenshotOverlay) {
        screenshotOverlay.style.display = 'none';
    }
    const screenshotBadge = document.getElementById('screenshot-badge');
    if (screenshotBadge) {
        screenshotBadge.style.display = 'none';
    }
    const frame = document.getElementById('participant-frame');
    if (frame) {
        frame.style.opacity = '1';
    }

    // Restore current live URL in iframe
    if (contentUrl) {
        if (frame) {
            frame.src = contentUrl;
        }
        const urlDisplay = document.getElementById('participant-url');
        if (urlDisplay) {
            urlDisplay.textContent = contentUrl;
        }
    }

    // Clear history arrays (will be repopulated by live data)
    gazeHistory = [];
    mouseHistory = [];

    updateLiveButtonState();
}

/**
 * Update live button state
 */
function updateLiveButtonState() {
    const liveBtn = document.getElementById('timeline-live');
    if (liveBtn) {
        if (timelineMode === 'live') {
            liveBtn.classList.add('active');
        } else {
            liveBtn.classList.remove('active');
        }
    }

    // Update playback badge visibility
    const playbackBadge = document.getElementById('playback-badge');
    if (playbackBadge) {
        playbackBadge.style.display = timelineMode === 'playback' ? 'inline-block' : 'none';
    }
}

/**
 * Update timeline (called periodically)
 */
function updateTimeline() {
    if (!startTime) return;

    const now = Date.now();
    const sessionDuration = now - startTime;

    // Sample engagement data periodically (only in live mode)
    if (timelineMode === 'live' && now - lastTimelineSample >= TIMELINE_SAMPLE_INTERVAL) {
        lastTimelineSample = now;
        recordTimelineData();
    }

    // Record face mesh to timeline
    if (timelineMode === 'live') {
        recordTimelineFaceMesh();
    }

    // Update time displays
    const totalTimeEl = document.getElementById('timeline-total-time');
    if (totalTimeEl) {
        totalTimeEl.textContent = formatTime(sessionDuration);
    }

    // Advance playback if playing
    if (timelineMode === 'playback' && timelineIsPlaying) {
        timelinePlaybackTime += 100; // Add 100ms (update interval)
        if (timelinePlaybackTime >= sessionDuration) {
            // Reached end, jump to live
            jumpToLive();
        }
    }

    // Apply timeline state in playback mode
    if (timelineMode === 'playback') {
        applyTimelineState();
    }

    // Update scrubber position
    updateTimelineScrubber();

    // Render timeline charts
    renderTimelineCharts();
}

/**
 * Record current data to timeline
 */
function recordTimelineData() {
    if (!startTime) return;

    const time = Date.now() - startTime;

    // Record engagement
    timelineData.engagement.push({
        time,
        value: cognitiveStates.engagement
    });

    // Record cognitive states
    timelineData.cognitiveStates.push({
        time,
        confusion: cognitiveStates.confusion,
        engagement: cognitiveStates.engagement,
        boredom: cognitiveStates.boredom,
        frustration: cognitiveStates.frustration
    });

    // Trim old data
    trimTimelineData();
}

/**
 * Record face mesh to timeline (called less frequently for performance)
 */
function recordTimelineFaceMesh() {
    if (!startTime || !latestLandmarks) return;

    const now = Date.now();
    if (now - lastFaceMeshSample < FACEMESH_SAMPLE_INTERVAL) return;
    lastFaceMeshSample = now;

    const time = now - startTime;

    // Store a simplified version of landmarks for playback
    timelineData.faceMesh.push({
        time,
        landmarks: latestLandmarks.landmarks,
        key_points: latestLandmarks.key_points,
        head_pose: latestLandmarks.head_pose
    });
}

/**
 * Record mouse position to timeline
 */
function recordTimelineMouse(x, y) {
    if (!startTime) return;
    timelineData.mouse.push({
        time: Date.now() - startTime,
        x, y
    });
}

/**
 * Record scroll position to timeline
 */
function recordTimelineScroll(scrollX, scrollY) {
    if (!startTime) return;
    timelineData.scroll.push({
        time: Date.now() - startTime,
        scrollX, scrollY
    });
}

/**
 * Record navigation/URL change to timeline
 */
function recordTimelineNavigation(url, title = '') {
    if (!startTime) return;
    // Only record if URL actually changed
    if (url === currentNavigationUrl) return;

    currentNavigationUrl = url;
    const time = Date.now() - startTime;
    timelineData.navigation.push({
        time,
        url,
        title: title || url
    });

    // Add marker to timeline
    addTimelineMarker(time, 'navigation');
    addLogEntry('system', `Navigation: ${url}`);
}

/**
 * Record screenshot to timeline
 * Captures the current state of participant screen from received screenshot data
 */
function recordTimelineScreenshot(dataUrl) {
    if (!startTime || !dataUrl) return;

    const now = Date.now();
    // Throttle screenshots to avoid memory issues
    if (now - lastScreenshotTime < SCREENSHOT_INTERVAL) return;
    lastScreenshotTime = now;

    const time = now - startTime;
    timelineData.screenshots.push({
        time,
        dataUrl
    });

    // Limit screenshots to prevent memory issues (keep last 100)
    if (timelineData.screenshots.length > 100) {
        timelineData.screenshots.shift();
    }
}

/**
 * Record gaze data to timeline
 */
function recordTimelineGaze(x, y) {
    if (!startTime) return;
    timelineData.gaze.push({
        time: Date.now() - startTime,
        x, y
    });
}

/**
 * Record click to timeline
 */
function recordTimelineClick(x, y) {
    if (!startTime) return;
    const time = Date.now() - startTime;
    timelineData.clicks.push({ time, x, y });

    // Add marker to timeline
    addTimelineMarker(time, 'click');
}

/**
 * Record key event to timeline
 */
function recordTimelineKey(key) {
    if (!startTime) return;
    const time = Date.now() - startTime;
    timelineData.keys.push({ time, key });

    // Add marker to timeline
    addTimelineMarker(time, 'keyboard');
}

/**
 * Record custom event to timeline
 */
function recordTimelineEvent(type, data) {
    if (!startTime) return;
    const time = Date.now() - startTime;
    timelineData.events.push({ time, type, data });

    // Add marker to timeline
    addTimelineMarker(time, type);
}

/**
 * Add a marker to the timeline track
 */
function addTimelineMarker(time, type) {
    const markersContainer = document.getElementById('timeline-markers');
    const track = document.getElementById('timeline-track');
    if (!markersContainer || !track || !startTime) return;

    const sessionDuration = Date.now() - startTime;
    const position = (time / sessionDuration) * 100;

    const marker = document.createElement('div');
    marker.className = `timeline-marker ${type}`;
    marker.style.left = `${position}%`;
    marker.title = `${type} at ${formatTime(time)}`;
    marker.dataset.time = time;

    markersContainer.appendChild(marker);

    // Update marker positions on subsequent updates
    // (handled in updateTimelineMarkers)
}

/**
 * Update all timeline markers positions
 */
function updateTimelineMarkers() {
    const markersContainer = document.getElementById('timeline-markers');
    if (!markersContainer || !startTime) return;

    const sessionDuration = Date.now() - startTime;
    const markers = markersContainer.querySelectorAll('.timeline-marker');

    markers.forEach(marker => {
        const time = parseInt(marker.dataset.time, 10);
        const position = (time / sessionDuration) * 100;
        marker.style.left = `${position}%`;
    });
}

/**
 * Trim old timeline data
 */
function trimTimelineData() {
    const maxTime = TIMELINE_MAX_DURATION;

    // Trim each data array
    timelineData.engagement = timelineData.engagement.filter(d => d.time <= maxTime);
    timelineData.gaze = timelineData.gaze.filter(d => d.time <= maxTime);
    timelineData.clicks = timelineData.clicks.filter(d => d.time <= maxTime);
    timelineData.keys = timelineData.keys.filter(d => d.time <= maxTime);
    timelineData.events = timelineData.events.filter(d => d.time <= maxTime);
    timelineData.faceMesh = timelineData.faceMesh.filter(d => d.time <= maxTime);
    timelineData.cognitiveStates = timelineData.cognitiveStates.filter(d => d.time <= maxTime);
    timelineData.mouse = timelineData.mouse.filter(d => d.time <= maxTime);
    timelineData.scroll = timelineData.scroll.filter(d => d.time <= maxTime);
    timelineData.navigation = timelineData.navigation.filter(d => d.time <= maxTime);
    timelineData.screenshots = timelineData.screenshots.filter(d => d.time <= maxTime);
}

/**
 * Render timeline mini charts
 */
function renderTimelineCharts() {
    if (!startTime) return;

    const sessionDuration = Date.now() - startTime;

    // Render engagement chart
    if (engagementChartCtx && engagementChartCanvas) {
        renderEngagementChart(sessionDuration);
    }

    // Render gaze activity chart
    if (gazeChartCtx && gazeChartCanvas) {
        renderGazeChart(sessionDuration);
    }

    // Render events chart
    if (eventsChartCtx && eventsChartCanvas) {
        renderEventsChart(sessionDuration);
    }

    // Update marker positions
    updateTimelineMarkers();
}

/**
 * Render engagement chart
 */
function renderEngagementChart(sessionDuration) {
    const ctx = engagementChartCtx;
    const canvas = engagementChartCanvas;
    const w = canvas.width;
    const h = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, w, h);

    // Draw background
    ctx.fillStyle = 'rgba(15, 52, 96, 0.3)';
    ctx.fillRect(0, 0, w, h);

    if (timelineData.engagement.length < 2) return;

    // Draw engagement line
    ctx.beginPath();
    ctx.strokeStyle = '#4ecca3';
    ctx.lineWidth = 2;

    timelineData.engagement.forEach((point, i) => {
        const x = (point.time / sessionDuration) * w;
        const y = h - (point.value * h);

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    // Draw fill
    ctx.lineTo((timelineData.engagement[timelineData.engagement.length - 1].time / sessionDuration) * w, h);
    ctx.lineTo((timelineData.engagement[0].time / sessionDuration) * w, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(78, 204, 163, 0.2)';
    ctx.fill();
}

/**
 * Render gaze activity chart (heatmap style)
 */
function renderGazeChart(sessionDuration) {
    const ctx = gazeChartCtx;
    const canvas = gazeChartCanvas;
    const w = canvas.width;
    const h = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, w, h);

    // Draw background
    ctx.fillStyle = 'rgba(15, 52, 96, 0.3)';
    ctx.fillRect(0, 0, w, h);

    // Calculate gaze density in time buckets
    const bucketCount = Math.min(100, w);
    const bucketWidth = sessionDuration / bucketCount;
    const buckets = new Array(bucketCount).fill(0);

    timelineData.gaze.forEach(point => {
        const bucketIndex = Math.floor(point.time / bucketWidth);
        if (bucketIndex >= 0 && bucketIndex < bucketCount) {
            buckets[bucketIndex]++;
        }
    });

    const maxBucket = Math.max(...buckets, 1);

    // Draw bars
    const barWidth = w / bucketCount;
    buckets.forEach((count, i) => {
        const intensity = count / maxBucket;
        const barHeight = intensity * h;

        ctx.fillStyle = `rgba(233, 69, 96, ${0.3 + intensity * 0.7})`;
        ctx.fillRect(i * barWidth, h - barHeight, barWidth - 1, barHeight);
    });
}

/**
 * Render events chart (clicks, keys, and mouse activity)
 */
function renderEventsChart(sessionDuration) {
    const ctx = eventsChartCtx;
    const canvas = eventsChartCanvas;
    const w = canvas.width;
    const h = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, w, h);

    // Draw background
    ctx.fillStyle = 'rgba(15, 52, 96, 0.3)';
    ctx.fillRect(0, 0, w, h);

    // Calculate mouse activity density in time buckets (bottom layer - subtle)
    const bucketCount = Math.min(50, w);
    const bucketWidth = sessionDuration / bucketCount;
    const mouseBuckets = new Array(bucketCount).fill(0);

    timelineData.mouse.forEach(point => {
        const bucketIndex = Math.floor(point.time / bucketWidth);
        if (bucketIndex >= 0 && bucketIndex < bucketCount) {
            mouseBuckets[bucketIndex]++;
        }
    });

    const maxMouseBucket = Math.max(...mouseBuckets, 1);

    // Draw mouse activity as subtle background bars
    const barWidth = w / bucketCount;
    mouseBuckets.forEach((count, i) => {
        const intensity = count / maxMouseBucket;
        const barHeight = intensity * h * 0.8;

        ctx.fillStyle = `rgba(78, 204, 163, ${intensity * 0.2})`;
        ctx.fillRect(i * barWidth, h - barHeight, barWidth - 1, barHeight);
    });

    // Draw click markers (middle layer)
    ctx.fillStyle = '#4ecca3';
    timelineData.clicks.forEach(click => {
        const x = (click.time / sessionDuration) * w;
        ctx.beginPath();
        ctx.arc(x, h / 3, 3, 0, Math.PI * 2);
        ctx.fill();
    });

    // Draw key markers (top layer)
    ctx.fillStyle = '#f39c12';
    timelineData.keys.forEach(key => {
        const x = (key.time / sessionDuration) * w;
        ctx.beginPath();
        ctx.arc(x, (2 * h) / 3, 2, 0, Math.PI * 2);
        ctx.fill();
    });
}

/**
 * Format time in mm:ss
 */
function formatTime(ms) {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

/**
 * Find the data point closest to (but not exceeding) the given time
 */
function findDataAtTime(dataArray, targetTime) {
    if (!dataArray || dataArray.length === 0) return null;

    // Binary search for efficiency
    let left = 0;
    let right = dataArray.length - 1;

    while (left < right) {
        const mid = Math.floor((left + right + 1) / 2);
        if (dataArray[mid].time <= targetTime) {
            left = mid;
        } else {
            right = mid - 1;
        }
    }

    return dataArray[left].time <= targetTime ? dataArray[left] : null;
}

/**
 * Get the current time for visualization (live or playback)
 */
function getCurrentVisualizationTime() {
    if (!startTime) return 0;
    if (timelineMode === 'live') {
        return Date.now() - startTime;
    }
    return timelinePlaybackTime;
}

/**
 * Apply timeline state to visualizations
 * This updates all views to match the current timeline position
 */
function applyTimelineState() {
    if (timelineMode === 'live') return; // In live mode, data flows naturally

    const time = timelinePlaybackTime;
    const container = document.getElementById('participant-view-container');
    if (!container) return;

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    // Apply face mesh data
    const faceMeshData = findDataAtTime(timelineData.faceMesh, time);
    if (faceMeshData) {
        latestLandmarks = {
            landmarks: faceMeshData.landmarks,
            key_points: faceMeshData.key_points,
            head_pose: faceMeshData.head_pose
        };

        // Update head pose display
        if (faceMeshData.head_pose) {
            const pitch = faceMeshData.head_pose.pitch.toFixed(1);
            const yaw = faceMeshData.head_pose.yaw.toFixed(1);
            document.getElementById('head-pose').textContent = `P:${pitch} Y:${yaw}`;
        }
        if (faceMeshData.landmarks) {
            document.getElementById('landmark-count').textContent = faceMeshData.landmarks.length;
        }
    }

    // Apply cognitive states
    const cogStateData = findDataAtTime(timelineData.cognitiveStates, time);
    if (cogStateData) {
        // Update the bars (but don't overwrite the actual cognitiveStates which are still being calculated live)
        document.getElementById('confusion-bar').style.width = `${Math.round(cogStateData.confusion * 100)}%`;
        document.getElementById('confusion-value').textContent = `${Math.round(cogStateData.confusion * 100)}%`;
        document.getElementById('engagement-bar').style.width = `${Math.round(cogStateData.engagement * 100)}%`;
        document.getElementById('engagement-value').textContent = `${Math.round(cogStateData.engagement * 100)}%`;
        document.getElementById('boredom-bar').style.width = `${Math.round(cogStateData.boredom * 100)}%`;
        document.getElementById('boredom-value').textContent = `${Math.round(cogStateData.boredom * 100)}%`;
        document.getElementById('frustration-bar').style.width = `${Math.round(cogStateData.frustration * 100)}%`;
        document.getElementById('frustration-value').textContent = `${Math.round(cogStateData.frustration * 100)}%`;
    }

    // Apply gaze position and update gaze history for trail
    const gazeData = findDataAtTime(timelineData.gaze, time);
    if (gazeData) {
        const normalizedX = (gazeData.x / participantWindowWidth) * containerWidth;
        const normalizedY = (gazeData.y / participantWindowHeight) * containerHeight;

        const gazeOverlay = document.getElementById('gaze-point-overlay');
        if (gazeOverlay) {
            gazeOverlay.style.left = `${normalizedX}px`;
            gazeOverlay.style.top = `${normalizedY}px`;
        }

        document.getElementById('gaze-x').textContent = Math.round(gazeData.x);
        document.getElementById('gaze-y').textContent = Math.round(gazeData.y);
    }

    // Build gaze history from timeline data for trail visualization
    updatePlaybackGazeHistory(time, containerWidth, containerHeight);

    // Apply mouse position and update mouse history for trail
    const mouseData = findDataAtTime(timelineData.mouse, time);
    if (mouseData) {
        const normalizedX = (mouseData.x / participantWindowWidth) * containerWidth;
        const normalizedY = (mouseData.y / participantWindowHeight) * containerHeight;

        const mouseCursor = document.getElementById('mouse-cursor-overlay');
        if (mouseCursor) {
            mouseCursor.style.left = `${normalizedX}px`;
            mouseCursor.style.top = `${normalizedY}px`;
        }

        document.getElementById('mouse-x').textContent = Math.round(mouseData.x);
        document.getElementById('mouse-y').textContent = Math.round(mouseData.y);
    }

    // Build mouse history from timeline data for trail visualization
    updatePlaybackMouseHistory(time, containerWidth, containerHeight);

    // Show clicks that happened recently relative to current playback time
    updatePlaybackClicks(time, containerWidth, containerHeight);

    // Apply scroll position to participant view
    updatePlaybackScroll(time);

    // Apply navigation/URL changes
    updatePlaybackNavigation(time);

    // Apply screenshot (shows captured screen instead of live iframe)
    updatePlaybackScreenshot(time);
}

/**
 * Update gaze history for playback mode (for trail visualization)
 */
function updatePlaybackGazeHistory(currentTime, containerWidth, containerHeight) {
    // Get gaze data from the last 2 seconds relative to current playback time
    const windowStart = Math.max(0, currentTime - 2000);

    gazeHistory = timelineData.gaze
        .filter(g => g.time >= windowStart && g.time <= currentTime)
        .map(g => ({
            x: (g.x / participantWindowWidth) * containerWidth,
            y: (g.y / participantWindowHeight) * containerHeight,
            time: g.time
        }))
        .slice(-GAZE_HISTORY_LENGTH);
}

/**
 * Update mouse history for playback mode (for trail visualization)
 */
function updatePlaybackMouseHistory(currentTime, containerWidth, containerHeight) {
    // Get mouse data from the last 2 seconds relative to current playback time (longer for better visibility)
    const windowStart = Math.max(0, currentTime - 2000);

    mouseHistory = timelineData.mouse
        .filter(m => m.time >= windowStart && m.time <= currentTime)
        .map(m => ({
            x: (m.x / participantWindowWidth) * containerWidth,
            y: (m.y / participantWindowHeight) * containerHeight,
            time: m.time
        }))
        .slice(-MOUSE_HISTORY_LENGTH);
}

/**
 * Apply scroll position during playback
 * This scrolls the iframe to match the participant's scroll position at the given time
 */
function updatePlaybackScroll(currentTime) {
    const scrollData = findDataAtTime(timelineData.scroll, currentTime);
    if (!scrollData) return;

    const frame = document.getElementById('participant-frame');
    if (!frame) return;

    // Always show the scroll indicator for visibility
    updateScrollIndicator(scrollData.scrollX, scrollData.scrollY);

    try {
        // Try to scroll the iframe content
        // Note: This will only work if the iframe is same-origin or allows access
        if (frame.contentWindow) {
            frame.contentWindow.scrollTo(scrollData.scrollX, scrollData.scrollY);
        }
    } catch (e) {
        // Cross-origin restriction - we can't directly scroll the iframe
        // The visual indicator is already shown above
        console.log('Cannot scroll cross-origin iframe, showing indicator only');
    }
}

/**
 * Apply navigation/URL during playback
 * Loads the appropriate page in the iframe based on timeline position
 */
function updatePlaybackNavigation(currentTime) {
    const navData = findDataAtTime(timelineData.navigation, currentTime);
    if (!navData) return;

    const frame = document.getElementById('participant-frame');
    const urlDisplay = document.getElementById('participant-url');

    // Only change iframe if URL is different from current
    if (frame && frame.src !== navData.url) {
        frame.src = navData.url;
    }

    // Update URL display
    if (urlDisplay) {
        urlDisplay.textContent = navData.url;
        urlDisplay.title = navData.url;
    }

    // Update navigation indicator
    updateNavigationIndicator(navData.url, navData.time);
}

/**
 * Update navigation indicator overlay
 */
function updateNavigationIndicator(url, time) {
    let indicator = document.getElementById('navigation-indicator');
    const container = document.getElementById('participant-view-container');
    if (!container) return;

    // Create indicator if it doesn't exist
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'navigation-indicator';
        indicator.style.cssText = `
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            background: rgba(15, 52, 96, 0.95);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.7rem;
            z-index: 102;
            border: 1px solid var(--accent);
            pointer-events: none;
            display: flex;
            align-items: center;
            gap: 8px;
        `;
        container.appendChild(indicator);
    }

    // Only show in playback mode
    if (timelineMode === 'playback') {
        indicator.style.display = 'flex';
        // Truncate long URLs
        const displayUrl = url.length > 60 ? url.substring(0, 60) + '...' : url;
        indicator.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style="flex-shrink: 0;">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
            </svg>
            <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${displayUrl}</span>
            <span style="color: var(--text-secondary); margin-left: auto; flex-shrink: 0;">@ ${formatTime(time)}</span>
        `;
    } else {
        indicator.style.display = 'none';
    }
}

/**
 * Update visual scroll position indicator (for cross-origin iframes)
 */
function updateScrollIndicator(scrollX, scrollY) {
    let indicator = document.getElementById('scroll-position-indicator');
    const container = document.getElementById('participant-view-container');
    if (!container) return;

    // Create indicator if it doesn't exist
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'scroll-position-indicator';
        indicator.style.cssText = `
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(15, 52, 96, 0.9);
            color: var(--text-primary);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-family: monospace;
            z-index: 100;
            border: 1px solid var(--accent);
            pointer-events: none;
        `;
        container.appendChild(indicator);
    }

    // Only show indicator in playback mode
    if (timelineMode === 'playback') {
        indicator.style.display = 'block';
        indicator.innerHTML = `Scroll: X:${scrollX} Y:${scrollY}`;
    } else {
        indicator.style.display = 'none';
    }
}

/**
 * Apply video/screenshot during playback
 * Shows recorded video or captured screenshot instead of live iframe content
 */
function updatePlaybackScreenshot(currentTime) {
    const container = document.getElementById('participant-view-container');
    const frame = document.getElementById('participant-frame');
    if (!container) return;

    // Ensure video element exists
    ensureVideoElement();

    let screenshotOverlay = document.getElementById('screenshot-overlay');
    let playbackBadge = document.getElementById('screenshot-badge');

    // Create screenshot overlay if it doesn't exist (fallback for when no video)
    if (!screenshotOverlay) {
        screenshotOverlay = document.createElement('img');
        screenshotOverlay.id = 'screenshot-overlay';
        screenshotOverlay.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #1a1a2e;
            z-index: 90;
            pointer-events: none;
            display: none;
        `;
        container.appendChild(screenshotOverlay);
    }

    // Create badge if it doesn't exist
    if (!playbackBadge) {
        playbackBadge = document.createElement('div');
        playbackBadge.id = 'screenshot-badge';
        playbackBadge.style.cssText = `
            position: absolute;
            bottom: 40px;
            right: 10px;
            background: rgba(78, 204, 163, 0.9);
            color: #000;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.65rem;
            font-weight: 600;
            z-index: 103;
            pointer-events: none;
            display: none;
        `;
        container.appendChild(playbackBadge);
    }

    if (timelineMode === 'playback') {
        // Check if we have video recording available
        if (videoElement && videoUrl && timelineData.videoChunks.length > 0) {
            // Use video playback
            videoElement.style.display = 'block';
            screenshotOverlay.style.display = 'none';
            if (frame) frame.style.opacity = '0';

            // Seek video to current time (convert ms to seconds)
            const videoTime = currentTime / 1000;
            if (Math.abs(videoElement.currentTime - videoTime) > 0.5) {
                videoElement.currentTime = videoTime;
            }

            // Play/pause based on timeline state
            if (timelineIsPlaying && videoElement.paused) {
                videoElement.play().catch(e => console.log('Video play error:', e));
            } else if (!timelineIsPlaying && !videoElement.paused) {
                videoElement.pause();
            }

            // Show badge
            playbackBadge.textContent = `🎬 Recording @ ${formatTime(currentTime)}`;
            playbackBadge.style.display = 'block';

        } else {
            // Fallback to screenshots if no video
            const screenshotData = findDataAtTime(timelineData.screenshots, currentTime);

            if (videoElement) videoElement.style.display = 'none';

            if (screenshotData && screenshotData.dataUrl) {
                screenshotOverlay.src = screenshotData.dataUrl;
                screenshotOverlay.style.display = 'block';
                if (frame) frame.style.opacity = '0';

                playbackBadge.textContent = `📸 Captured @ ${formatTime(screenshotData.time)}`;
                playbackBadge.style.display = 'block';
            } else {
                // No video or screenshots available
                screenshotOverlay.style.display = 'none';
                playbackBadge.style.display = 'none';
                if (frame) frame.style.opacity = '1';
            }
        }
    } else {
        // Live mode - hide video/screenshot overlays
        if (videoElement) {
            videoElement.style.display = 'none';
            videoElement.pause();
        }
        screenshotOverlay.style.display = 'none';
        playbackBadge.style.display = 'none';
        if (frame) frame.style.opacity = '1';
    }
}

/**
 * Show click markers during playback
 */
function updatePlaybackClicks(currentTime, containerWidth, containerHeight) {
    const clickContainer = document.getElementById('click-ripple-container');
    if (!clickContainer) return;

    // Clear existing playback click markers
    const existingMarkers = clickContainer.querySelectorAll('.playback-click-marker');
    existingMarkers.forEach(m => m.remove());

    // Show clicks that happened in the last 2 seconds relative to current playback time
    const windowStart = Math.max(0, currentTime - 2000);
    const recentClicks = timelineData.clicks.filter(c => c.time >= windowStart && c.time <= currentTime);

    recentClicks.forEach(click => {
        const normalizedX = (click.x / participantWindowWidth) * containerWidth;
        const normalizedY = (click.y / participantWindowHeight) * containerHeight;

        // Calculate opacity based on age (newer = more opaque)
        const age = currentTime - click.time;
        const opacity = Math.max(0.2, 1 - (age / 2000));

        // Create click marker
        const marker = document.createElement('div');
        marker.className = 'playback-click-marker';
        marker.style.cssText = `
            position: absolute;
            left: ${normalizedX}px;
            top: ${normalizedY}px;
            width: ${16 + (1 - opacity) * 20}px;
            height: ${16 + (1 - opacity) * 20}px;
            border-radius: 50%;
            border: 2px solid var(--success);
            background: rgba(78, 204, 163, ${opacity * 0.3});
            transform: translate(-50%, -50%);
            pointer-events: none;
            z-index: 99;
            opacity: ${opacity};
        `;
        clickContainer.appendChild(marker);
    });
}

/**
 * Reset timeline data for new session
 */
function resetTimelineData() {
    timelineData = {
        engagement: [],
        gaze: [],
        clicks: [],
        keys: [],
        events: [],
        faceMesh: [],
        cognitiveStates: [],
        mouse: [],
        scroll: [],
        navigation: [],
        screenshots: [],
        videoChunks: []
    };

    // Reset video playback state
    if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
        videoUrl = null;
    }
    videoBlob = null;
    videoDuration = 0;
    videoStartTime = 0;
    if (videoElement) {
        videoElement.src = '';
        videoElement.style.display = 'none';
    }

    // Reset navigation tracking
    currentNavigationUrl = null;
    lastScreenshotTime = 0;

    // Clear markers
    const markersContainer = document.getElementById('timeline-markers');
    if (markersContainer) {
        markersContainer.innerHTML = '';
    }

    // Reset sample timers
    lastFaceMeshSample = 0;

    // Reset to live mode
    jumpToLive();
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
    // Only update UI in live mode; playback mode updates via applyTimelineState
    if (timelineMode !== 'live') return;

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
        case 'navigation':
            handleNavigationEvent(data.data);
            break;
        case 'screenshot':
            handleScreenshotData(data.data);
            break;
        case 'video_start':
            handleVideoStart(data.data);
            break;
        case 'video_chunk':
            handleVideoChunk(data.data);
            break;
        case 'video_complete':
            handleVideoComplete(data.data);
            break;
        case 'session':
            handleSessionEvent(data.data);
            break;
    }
}

/**
 * Handle screenshot data from participant (legacy fallback)
 */
function handleScreenshotData(data) {
    if (data.dataUrl) {
        // Record screenshot to timeline
        recordTimelineScreenshot(data.dataUrl);

        // Log for debugging
        const captureNum = data.captureNumber || timelineData.screenshots.length;
        const method = data.captureMethod || 'unknown';
        console.log(`Screenshot #${captureNum} received (${method}), total stored: ${timelineData.screenshots.length}`);
    }
}

/**
 * Handle video recording start
 */
function handleVideoStart(data) {
    console.log('Video recording started:', data);
    videoMimeType = data.mimeType || 'video/webm';
    videoStartTime = data.startTime || Date.now();
    timelineData.videoChunks = [];

    // Clean up any existing video
    if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
        videoUrl = null;
    }
    videoBlob = null;

    addLogEntry('system', `Screen recording started (${data.width}x${data.height})`);
}

/**
 * Handle incoming video chunk
 */
function handleVideoChunk(data) {
    if (!data.data) return;

    // Convert base64 to blob
    const byteCharacters = atob(data.data);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: data.mimeType || videoMimeType });

    // Store chunk with timestamp
    timelineData.videoChunks.push({
        time: data.timestamp || 0,
        blob: blob,
        index: data.chunkIndex
    });

    // Rebuild video blob from all chunks for playback
    rebuildVideoBlob();

    console.log(`Video chunk #${data.chunkIndex + 1} received (${(data.size / 1024).toFixed(1)} KB), total: ${timelineData.videoChunks.length}`);
}

/**
 * Handle video recording complete
 */
function handleVideoComplete(data) {
    console.log('Video recording complete:', data);
    videoDuration = data.duration || 0;

    // Final rebuild of video blob
    rebuildVideoBlob();

    addLogEntry('system', `Screen recording complete (${(data.totalSize / 1024 / 1024).toFixed(2)} MB, ${timelineData.videoChunks.length} chunks)`);
}

/**
 * Rebuild video blob from all chunks
 */
function rebuildVideoBlob() {
    if (timelineData.videoChunks.length === 0) return;

    // Sort chunks by index
    const sortedChunks = [...timelineData.videoChunks].sort((a, b) => a.index - b.index);

    // Create combined blob
    const blobs = sortedChunks.map(c => c.blob);
    videoBlob = new Blob(blobs, { type: videoMimeType });

    // Create object URL for playback
    if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
    }
    videoUrl = URL.createObjectURL(videoBlob);

    // Update or create video element
    ensureVideoElement();
}

/**
 * Ensure video element exists in participant view container
 */
function ensureVideoElement() {
    const container = document.getElementById('participant-view-container');
    if (!container) return;

    if (!videoElement) {
        videoElement = document.createElement('video');
        videoElement.id = 'playback-video';
        videoElement.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #1a1a2e;
            z-index: 91;
            pointer-events: none;
            display: none;
        `;
        videoElement.muted = true;
        videoElement.playsInline = true;
        container.appendChild(videoElement);
    }

    // Update video source if we have a URL
    if (videoUrl && videoElement.src !== videoUrl) {
        videoElement.src = videoUrl;
        videoElement.load();
    }
}

/**
 * Handle navigation events (URL changes from participant)
 */
function handleNavigationEvent(data) {
    if (data.url) {
        contentUrl = data.url;

        // Record navigation to timeline
        recordTimelineNavigation(data.url, data.title || '');

        // Update URL display (only in live mode)
        if (timelineMode === 'live') {
            const urlDisplay = document.getElementById('participant-url');
            if (urlDisplay) {
                urlDisplay.textContent = data.url;
                urlDisplay.title = data.url;
            }

            // Load content in iframe
            const frame = document.getElementById('participant-frame');
            if (frame) {
                frame.src = data.url;
            }
        }
    }
}

/**
 * Handle gaze data
 */
function handleGazeData(data) {
    stats.gazeSamples++;
    stats.totalSamples++;

    // Record gaze to timeline (always record)
    recordTimelineGaze(data.x, data.y);

    // Only update live visualization if in live mode
    if (timelineMode !== 'live') return;

    // Update gaze position display
    document.getElementById('gaze-x').textContent = Math.round(data.x);
    document.getElementById('gaze-y').textContent = Math.round(data.y);

    // Get participant view container dimensions
    const container = document.getElementById('participant-view-container');
    if (!container) return;

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    // Normalize participant coordinates to container coordinates
    const normalizedX = (data.x / participantWindowWidth) * containerWidth;
    const normalizedY = (data.y / participantWindowHeight) * containerHeight;

    // Add to history for trail visualization
    gazeHistory.push({ x: normalizedX, y: normalizedY, time: Date.now() });
    if (gazeHistory.length > GAZE_HISTORY_LENGTH) {
        gazeHistory.shift();
    }

    // Update gaze point overlay on participant view
    const gazeOverlay = document.getElementById('gaze-point-overlay');
    if (gazeOverlay) {
        gazeOverlay.style.left = `${normalizedX}px`;
        gazeOverlay.style.top = `${normalizedY}px`;
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

    // Get participant view container dimensions
    const container = document.getElementById('participant-view-container');
    if (!container) return;

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    // Normalize participant coordinates to container coordinates
    const normalizedX = (data.x / participantWindowWidth) * containerWidth;
    const normalizedY = (data.y / participantWindowHeight) * containerHeight;

    // Handle mouse move events
    if (data.event === 'move') {
        // Always record mouse position to timeline (regardless of mode)
        recordTimelineMouse(data.x, data.y);

        // Only update live visualization if in live mode
        if (timelineMode === 'live') {
            // Update mouse position display
            document.getElementById('mouse-x').textContent = Math.round(data.x);
            document.getElementById('mouse-y').textContent = Math.round(data.y);

            // Update mouse cursor overlay position
            const mouseCursor = document.getElementById('mouse-cursor-overlay');
            if (mouseCursor) {
                mouseCursor.style.left = `${normalizedX}px`;
                mouseCursor.style.top = `${normalizedY}px`;
            }

            // Add to mouse history for trail
            mouseHistory.push({ x: normalizedX, y: normalizedY, time: Date.now() });
            if (mouseHistory.length > MOUSE_HISTORY_LENGTH) {
                mouseHistory.shift();
            }
        }
    }

    // Handle click events with visual feedback
    if (data.event === 'click') {
        stats.clickCount++;
        document.getElementById('click-count').textContent = stats.clickCount;

        // Create click ripple effect (only in live mode)
        if (timelineMode === 'live') {
            createClickRipple(normalizedX, normalizedY);
        }

        // Record click to timeline
        recordTimelineClick(data.x, data.y);

        // Log click events
        addLogEntry('mouse', `Click at (${Math.round(data.x)}, ${Math.round(data.y)})`);
    }

    // Handle scroll events
    if (data.event === 'scroll') {
        // Record scroll to timeline
        recordTimelineScroll(data.scrollX, data.scrollY);
        addLogEntry('mouse', `Scroll to (${data.scrollX}, ${data.scrollY})`);
    }

    // Handle wheel events (mouse wheel without page scroll)
    if (data.event === 'wheel') {
        // Record as scroll for timeline
        recordTimelineScroll(data.deltaX, data.deltaY);
    }
}

/**
 * Create a click ripple animation at the specified position
 */
function createClickRipple(x, y) {
    const container = document.getElementById('click-ripple-container');
    if (!container) return;

    // Create ripple element
    const ripple = document.createElement('div');
    ripple.className = 'click-ripple';
    ripple.style.left = `${x}px`;
    ripple.style.top = `${y}px`;
    container.appendChild(ripple);

    // Create click marker (persists briefly)
    const marker = document.createElement('div');
    marker.className = 'click-marker';
    marker.style.left = `${x}px`;
    marker.style.top = `${y}px`;
    container.appendChild(marker);

    // Remove ripple after animation
    setTimeout(() => {
        ripple.remove();
    }, 600);

    // Remove marker after fade
    setTimeout(() => {
        marker.remove();
    }, 2000);
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

        // Record key to timeline
        recordTimelineKey(data.key);
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

        // Update participant screen dimensions
        if (data.screenWidth) participantScreenWidth = data.screenWidth;
        if (data.screenHeight) participantScreenHeight = data.screenHeight;
        if (data.windowWidth) participantWindowWidth = data.windowWidth;
        if (data.windowHeight) participantWindowHeight = data.windowHeight;

        // Load participant's content URL in the iframe
        if (data.content_url) {
            contentUrl = data.content_url;
            addLogEntry('system', `Content: ${data.content_url}`);

            // Record initial navigation to timeline
            recordTimelineNavigation(data.content_url, 'Initial Page');

            // Update URL display
            const urlDisplay = document.getElementById('participant-url');
            if (urlDisplay) {
                urlDisplay.textContent = data.content_url;
                urlDisplay.title = data.content_url;
            }

            // Load content in iframe
            const frame = document.getElementById('participant-frame');
            if (frame) {
                frame.src = data.content_url;
            }
        }

        // Reset stats for new session
        stats.gazeSamples = 0;
        stats.faceSamples = 0;
        stats.mouseEvents = 0;
        stats.keyboardEvents = 0;
        stats.clickCount = 0;
        stats.totalSamples = 0;
        gazeHistory = [];
        mouseHistory = [];

        // Reset timeline data for new session
        resetTimelineData();

    } else if (data.event === 'end') {
        isCollecting = false;
        addLogEntry('system', 'Session ended');

        // Update URL display
        const urlDisplay = document.getElementById('participant-url');
        if (urlDisplay) {
            urlDisplay.textContent = 'Session ended';
        }
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
