/**
 * HCI Analytics Toolbox — Frontend
 *
 * Fetches computed metrics from /api/session/{id}/analytics and renders
 * interactive Plotly.js charts across 7 analysis sections.
 */

// ── State ──────────────────────────────────────────────────────────────
let sessionId = null;
let analyticsData = null;
let currentSection = 'overview';
let allSessions = [];
let selectedQuestion = 'all';
let selectedQuestionDetail = null;  // for per-question detail view

// ── Plotly defaults ────────────────────────────────────────────────────
const PLOT_BG = '#0f3460';
const PLOT_PAPER = '#0f3460';
const PLOT_GRID = 'rgba(255,255,255,0.08)';
const PLOT_TEXT = '#a0a0a0';
const ACCENT = '#e94560';
const SUCCESS = '#4ecca3';
const WARNING = '#ffd93d';
const COLORS = ['#e94560', '#4ecca3', '#ffd93d', '#6c5ce7', '#00cec9', '#fd79a8', '#fdcb6e'];

function plotLayout(title, extra) {
    return Object.assign({
        title: { text: title || '', font: { color: PLOT_TEXT, size: 13 } },
        paper_bgcolor: PLOT_PAPER,
        plot_bgcolor: PLOT_BG,
        font: { color: PLOT_TEXT, size: 11 },
        margin: { l: 50, r: 20, t: title ? 35 : 10, b: 40 },
        xaxis: { gridcolor: PLOT_GRID, zerolinecolor: PLOT_GRID },
        yaxis: { gridcolor: PLOT_GRID, zerolinecolor: PLOT_GRID },
    }, extra || {});
}

const PLOT_CONFIG = { responsive: true, displayModeBar: false };

// ── Init ───────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    await loadSessionList();
    // Check URL for session ID
    const pathParts = window.location.pathname.split('/');
    const urlSid = pathParts[pathParts.length - 1];
    if (urlSid && urlSid !== 'analytics') {
        document.getElementById('session-select').value = urlSid;
        await loadAnalytics(urlSid);
    }
});

async function loadSessionList() {
    try {
        const res = await fetch('/api/sessions');
        const data = await res.json();
        allSessions = data.sessions || [];
        const select = document.getElementById('session-select');
        const compareSelect = document.getElementById('compare-session-select');
        allSessions.forEach(s => {
            const label = `${s.session_id.slice(0, 8)}... (${s.status || 'unknown'})`;
            select.innerHTML += `<option value="${s.session_id}">${label}</option>`;
            compareSelect.innerHTML += `<option value="${s.session_id}">${label}</option>`;
        });
    } catch (e) {
        console.error('Failed to load sessions:', e);
    }
}

function onSessionChange(sid) {
    if (sid) loadAnalytics(sid);
}

async function loadAnalytics(sid) {
    sessionId = sid;
    setStatus('Computing analytics...');
    document.getElementById('empty-state').style.display = 'none';
    document.getElementById('loading-state').style.display = 'flex';
    hideAllSections();

    try {
        const res = await fetch(`/api/session/${sid}/analytics`);
        analyticsData = await res.json();
        if (analyticsData.error) {
            alert('Error: ' + analyticsData.error);
            document.getElementById('loading-state').style.display = 'none';
            document.getElementById('empty-state').style.display = 'block';
            return;
        }
        document.getElementById('loading-state').style.display = 'none';
        document.getElementById('status-session').textContent = `Session: ${sid.slice(0, 12)}...`;
        setStatus('Ready');
        selectedQuestion = 'all';
        selectedQuestionDetail = null;
        populateQuestionFilter();
        showSection(currentSection);
    } catch (e) {
        console.error('Failed to load analytics:', e);
        alert('Failed to compute analytics. Check console.');
        document.getElementById('loading-state').style.display = 'none';
        document.getElementById('empty-state').style.display = 'block';
    }
}

// ── Navigation ─────────────────────────────────────────────────────────

function showSection(name) {
    if (!analyticsData && name !== 'compare') return;
    currentSection = name;
    // Update nav
    document.querySelectorAll('.nav-item').forEach(el => {
        el.classList.toggle('active', el.dataset.section === name);
    });
    hideAllSections();
    const sec = document.getElementById(`section-${name}`);
    if (sec) {
        sec.classList.add('active');
        sec.style.display = 'block';
        renderCurrentSection();
    }
}

function hideAllSections() {
    document.querySelectorAll('.section').forEach(el => {
        el.classList.remove('active');
        el.style.display = 'none';
    });
}

function renderCurrentSection() {
    if (!analyticsData) return;
    switch (currentSection) {
        case 'overview': renderOverview(analyticsData.overview); break;
        case 'gaze': renderGaze(analyticsData.gaze); break;
        case 'attention': renderAttention(analyticsData.attention); break;
        case 'behavioral': renderBehavioral(analyticsData.behavioral); break;
        case 'coordination': renderCoordination(analyticsData.coordination, analyticsData.behavioral); break;
        case 'emotion': renderEmotion(analyticsData.emotion); break;
        case 'temporal': renderTemporal(analyticsData.temporal); break;
        case 'questions': renderQuestions(analyticsData.questions); break;
        case 'answers': renderAnswers(analyticsData.answers); break;
        case 'survey': renderSurvey(analyticsData.survey); break;
    }
}

// ── Overview ───────────────────────────────────────────────────────────

function renderOverview(ov) {
    if (!ov) return;
    // Summary cards
    const cards = document.getElementById('overview-cards');
    cards.innerHTML = `
        ${statCard('Duration', ov.duration_formatted, 'accent')}
        ${statCard('Total Samples', fmtNum(ov.total_samples))}
        ${statCard('Quality Score', ov.quality_score + '%', ov.quality_score > 70 ? 'success' : 'warning')}
        ${statCard('Gaze Tracking', ov.gaze_tracking_rate + '%', ov.gaze_tracking_rate > 80 ? 'success' : 'warning')}
        ${statCard('Emotion Conf.', ov.emotion_confidence + '%')}
        ${statCard('Status', ov.status)}
    `;

    // Completeness
    const comp = document.getElementById('overview-completeness');
    const streamNames = {
        gaze: 'Gaze', mouse: 'Mouse', keyboard: 'Keyboard',
        emotion: 'Emotion', face_mesh: 'Face Mesh', hover: 'Hover',
        experiment_event: 'Events', answer: 'Answers',
        calibration_click: 'Calibration', drift_sample: 'Drift',
    };
    let compHtml = '';
    for (const [key, label] of Object.entries(streamNames)) {
        const on = ov.completeness[key];
        const count = ov.stream_counts[key] || 0;
        compHtml += `
            <div class="completeness-item">
                <div class="completeness-dot ${on ? 'on' : 'off'}"></div>
                <span class="completeness-label">${label}</span>
                <span class="completeness-count">${fmtNum(count)}</span>
            </div>`;
    }
    comp.innerHTML = compHtml;
}

// ── Gaze Analysis ──────────────────────────────────────────────────────

function renderGaze(g) {
    if (!g || !g.available) {
        showEmpty('section-gaze', 'No gaze data available');
        return;
    }
    const fs = g.fixation_stats || {};
    const ss = g.saccade_stats || {};

    // Stats cards
    document.getElementById('gaze-stats-cards').innerHTML = `
        ${statCard('Fixations', fs.count || 0)}
        ${statCard('Mean Duration', Math.round(fs.duration_mean || 0) + ' ms')}
        ${statCard('Saccades', ss.count || 0)}
        ${statCard('Mean Amplitude', Math.round(ss.amplitude_mean || 0) + ' px')}
        ${statCard('Max Velocity', Math.round(ss.velocity_max || 0) + ' px/ms')}
        ${statCard('Total Fix Time', Math.round((fs.total_fixation_time || 0) / 1000) + ' s')}
    `;

    // Heatmap
    if (g.heatmap && g.heatmap.z) {
        Plotly.newPlot('chart-gaze-heatmap', [{
            z: g.heatmap.z,
            type: 'heatmap',
            colorscale: [[0, '#1a1a2e'], [0.3, '#0f3460'], [0.6, '#e94560'], [1, '#ffd93d']],
            showscale: false,
        }], plotLayout('', { yaxis: { autorange: 'reversed' } }), PLOT_CONFIG);
    }

    // Scanpath
    if (g.scanpath && g.scanpath.length > 0) {
        const sp = g.scanpath;
        Plotly.newPlot('chart-scanpath', [
            { x: sp.map(p => p.x), y: sp.map(p => p.y), mode: 'lines+markers',
              marker: { size: sp.map(p => Math.min(30, Math.max(6, p.duration / 30))),
                        color: sp.map((_, i) => i), colorscale: 'Viridis' },
              line: { color: 'rgba(255,255,255,0.3)', width: 1 },
              type: 'scatter', hovertext: sp.map(p => `#${p.id} ${Math.round(p.duration)}ms`) },
        ], plotLayout('', { yaxis: { autorange: 'reversed' } }), PLOT_CONFIG);
    }

    // Fixation duration histogram
    plotHistogramFromBins('chart-fix-duration', g.fixation_duration_histogram, 'Duration (ms)', ACCENT);

    // Saccade amplitude histogram
    plotHistogramFromBins('chart-sac-amplitude', g.saccade_amplitude_histogram, 'Amplitude (px)', SUCCESS);

    // Saccade direction polar
    if (ss.directions && ss.directions.length > 0) {
        Plotly.newPlot('chart-sac-direction', [{
            r: ss.amplitudes,
            theta: ss.directions,
            type: 'scatterpolar',
            mode: 'markers',
            marker: { color: ACCENT, size: 5, opacity: 0.6 },
        }], plotLayout('', {
            polar: {
                bgcolor: PLOT_BG,
                radialaxis: { gridcolor: PLOT_GRID, color: PLOT_TEXT },
                angularaxis: { gridcolor: PLOT_GRID, color: PLOT_TEXT },
            }
        }), PLOT_CONFIG);
    }

    // Gaze dispersion timeline
    plotTimeline('chart-gaze-dispersion', g.dispersion_timeline, 'Dispersion (px)', ACCENT);
}

// ── Attention & Cognitive Load ─────────────────────────────────────────

function renderAttention(a) {
    if (!a || !a.available) {
        showEmpty('section-attention', 'No attention data available');
        return;
    }
    plotTimeline('chart-k-coeff', a.k_coefficient_timeline, 'K Coefficient', ACCENT,
        { shapes: [{ type: 'line', y0: 0, y1: 0, x0: 0, x1: 1, xref: 'paper',
                     line: { color: 'rgba(255,255,255,0.3)', dash: 'dash', width: 1 } }] });
    plotTimeline('chart-entropy', a.gaze_entropy_timeline, 'Entropy (bits)', SUCCESS);
    plotTimeline('chart-fix-stability', a.fixation_stability_timeline, 'Std Dev (ms)', WARNING);
    plotTimeline('chart-cog-load', a.cognitive_load_timeline, 'Load Index (0-1)', ACCENT);
    plotTimeline('chart-engagement', a.engagement_timeline, 'Engagement', SUCCESS);
}

// ── Behavioral ─────────────────────────────────────────────────────────

function renderBehavioral(b) {
    if (!b || !b.available) {
        showEmpty('section-behavioral', 'No behavioral data available');
        return;
    }
    const cp = b.click_patterns || {};
    const kd = b.keystroke_dynamics || {};
    const gm = b.gaze_mouse_coordination || {};
    const idle = b.idle_periods || [];
    const ps = b.mouse_path_stats || {};
    const bp = b.behavioral_patterns || {};

    document.getElementById('behavioral-stats-cards').innerHTML = `
        ${statCard('Total Clicks', cp.total_clicks || 0)}
        ${statCard('Key Events', kd.total_events || 0)}
        ${statCard('Gaze-Mouse Corr.', gm.available ? (gm.correlation || 0).toFixed(2) : 'N/A')}
        ${statCard('Coordination', gm.available ? (gm.coordination_score || 0).toFixed(2) : 'N/A', gm.coordination_score > 0.7 ? 'success' : 'warning')}
        ${statCard('Idle Periods', idle.length)}
    `;

    // Mouse path & pattern cards
    document.getElementById('behavioral-pattern-cards').innerHTML = `
        ${statCard('Path Distance', ps.total_distance_px ? fmtNum(ps.total_distance_px) + ' px' : 'N/A')}
        ${statCard('Dir. Changes', ps.direction_changes || 0)}
        ${statCard('Avg Speed', ps.avg_speed_px_ms ? ps.avg_speed_px_ms.toFixed(2) + ' px/ms' : 'N/A')}
        ${statCard('Hesitations', bp.hesitation_count || 0, bp.hesitation_count > 10 ? 'warning' : '')}
        ${statCard('Jitter Events', bp.jitter_events || 0, bp.jitter_events > 5 ? 'warning' : '')}
        ${statCard('Backtracks', bp.backtrack_count || 0)}
    `;

    // Mouse trajectory
    const traj = b.mouse_trajectory;
    if (traj && traj.available && traj.points && traj.points.length > 0) {
        const pts = traj.points;
        Plotly.newPlot('chart-mouse-trajectory', [{
            x: pts.map(p => p.x), y: pts.map(p => p.y),
            mode: 'lines+markers',
            type: 'scatter',
            marker: {
                size: 3,
                color: pts.map(p => p.time / 1000),
                colorscale: [[0, '#4ecca3'], [0.5, '#ffd93d'], [1, '#e94560']],
                colorbar: { title: 'Time (s)', titlefont: { color: PLOT_TEXT }, tickfont: { color: PLOT_TEXT } },
            },
            line: { color: 'rgba(255,255,255,0.15)', width: 1 },
        }], plotLayout('', { yaxis: { autorange: 'reversed' }, xaxis: { title: 'X (px)' }, yaxis2: { title: 'Y (px)' } }), PLOT_CONFIG);
    }

    plotTimeline('chart-mouse-velocity', b.mouse_velocity_timeline, 'Velocity (px/ms)', ACCENT);

    // Jitter timeline
    const jt = bp.jitter_timeline || [];
    if (jt.length > 0) {
        plotTimeline('chart-jitter-timeline', jt, 'Direction Changes / Window', WARNING);
    } else {
        clearChart('chart-jitter-timeline');
    }

    plotTimeline('chart-typing-speed', kd.typing_speed_timeline, 'Chars/min', SUCCESS);
    plotTimeline('chart-gaze-mouse-dist', gm.available ? gm.distance_timeline : [], 'Distance (px)', WARNING);

    // Click scatter
    if (cp.click_x && cp.click_x.length > 0) {
        Plotly.newPlot('chart-click-scatter', [{
            x: cp.click_x, y: cp.click_y,
            mode: 'markers',
            marker: { color: ACCENT, size: 8, opacity: 0.7 },
            type: 'scatter',
        }], plotLayout('', { yaxis: { autorange: 'reversed' } }), PLOT_CONFIG);
    }
}

// ── Cross-Stream Coordination ─────────────────────────────────────────

function renderCoordination(coord, behavioral) {
    if (!coord || !coord.available) {
        showEmpty('section-coordination', 'No cross-stream coordination data available. Both gaze and mouse+emotion data required.');
        return;
    }

    // Lag analysis cards
    const gm = (behavioral || {}).gaze_mouse_coordination || {};
    const lag = gm.lag_analysis || {};
    const lagCardsEl = document.getElementById('coordination-lag-cards');
    if (lag.available) {
        lagCardsEl.innerHTML = `
            ${statCard('Best Lag', lag.best_lag + ' samples')}
            ${statCard('Lag Correlation', (lag.best_correlation || 0).toFixed(3))}
            ${statCard('Who Leads?', lag.gaze_leads_mouse ? 'Gaze' : lag.best_lag < 0 ? 'Mouse' : 'Sync', lag.gaze_leads_mouse ? 'accent' : 'success')}
            ${statCard('Coordination', gm.available ? (gm.coordination_score || 0).toFixed(2) : 'N/A', gm.coordination_score > 0.7 ? 'success' : 'warning')}
        `;

        // Lag cross-correlation bar chart
        const corrs = lag.correlations || [];
        if (corrs.length > 0) {
            Plotly.newPlot('chart-lag-correlation', [{
                x: corrs.map(c => c.lag),
                y: corrs.map(c => c.correlation),
                type: 'bar',
                marker: {
                    color: corrs.map(c => c.lag === lag.best_lag ? ACCENT : 'rgba(78,204,163,0.6)'),
                },
            }], plotLayout('', {
                xaxis: { title: 'Lag (samples)', dtick: 2 },
                yaxis: { title: 'Pearson r', range: [-1, 1] },
                shapes: [{ type: 'line', y0: 0, y1: 0, x0: corrs[0].lag, x1: corrs[corrs.length-1].lag,
                           line: { color: 'rgba(255,255,255,0.3)', dash: 'dash', width: 1 } }],
            }), PLOT_CONFIG);
        }
    } else {
        lagCardsEl.innerHTML = statCard('Lag Analysis', 'Insufficient data', 'warning');
        clearChart('chart-lag-correlation');
    }

    // Coordination score timeline
    const ct = gm.coordination_timeline || [];
    if (ct.length > 0) {
        plotTimeline('chart-coord-timeline', ct, 'Score (0-1)', SUCCESS);
    } else {
        clearChart('chart-coord-timeline');
    }

    // Emotion × Gaze correlation heatmap
    renderCorrelationMatrix('chart-emotion-gaze-matrix', coord.emotion_gaze);

    // Emotion × Mouse correlation heatmap
    renderCorrelationMatrix('chart-emotion-mouse-matrix', coord.emotion_mouse);

    // Combined state timeline
    const combined = coord.combined_timeline;
    if (combined && combined.available && combined.times) {
        const traces = [];
        const t = combined.times.map(v => v / 1000);

        if (combined.coordination) {
            traces.push({
                x: t, y: combined.coordination,
                name: 'Coordination', type: 'scatter', mode: 'lines',
                line: { color: SUCCESS, width: 2 },
            });
        }
        if (combined.cognitive_load) {
            traces.push({
                x: t, y: combined.cognitive_load,
                name: 'Cognitive Load', type: 'scatter', mode: 'lines',
                line: { color: WARNING, width: 2 },
            });
        }
        if (traces.length > 0) {
            Plotly.newPlot('chart-combined-timeline', traces, plotLayout('', {
                xaxis: { title: 'Time (s)', gridcolor: PLOT_GRID },
                yaxis: { title: 'Score (0-1)', range: [0, 1], gridcolor: PLOT_GRID },
                showlegend: true,
                legend: { font: { color: PLOT_TEXT, size: 10 }, bgcolor: 'transparent' },
            }), PLOT_CONFIG);
        }
    } else {
        clearChart('chart-combined-timeline');
    }
}

function renderCorrelationMatrix(divId, matrixData) {
    if (!matrixData || !matrixData.available) {
        clearChart(divId);
        return;
    }
    const labelsX = matrixData.labels_x || [];
    const labelsY = matrixData.labels_y || [];
    const matrix = matrixData.matrix || {};

    // Build 2D array
    const z = labelsY.map(em => labelsX.map(gm => (matrix[em] || {})[gm] || 0));

    // Annotate text
    const annotations = [];
    labelsY.forEach((em, i) => {
        labelsX.forEach((gm, j) => {
            annotations.push({
                x: gm, y: capitalize(em),
                text: z[i][j].toFixed(2),
                font: { color: Math.abs(z[i][j]) > 0.3 ? '#fff' : PLOT_TEXT, size: 13 },
                showarrow: false,
            });
        });
    });

    Plotly.newPlot(divId, [{
        z: z,
        x: labelsX.map(capitalize),
        y: labelsY.map(capitalize),
        type: 'heatmap',
        colorscale: [[0, '#0f3460'], [0.25, '#1a4a7a'], [0.5, '#2a2a4e'], [0.75, '#e94560'], [1, '#ff6b6b']],
        zmin: -1, zmax: 1,
        showscale: true,
        colorbar: { title: 'r', titlefont: { color: PLOT_TEXT }, tickfont: { color: PLOT_TEXT } },
    }], plotLayout('', {
        annotations: annotations,
        xaxis: { side: 'bottom' },
        margin: { l: 100, r: 60, t: 10, b: 60 },
    }), PLOT_CONFIG);
}

// ── Emotion & Facial ───────────────────────────────────────────────────

function renderEmotion(em) {
    if (!em || !em.available) {
        showEmpty('section-emotion', 'No emotion data available');
        return;
    }
    const summary = em.summary || {};
    const states = Object.keys(summary);

    // Summary cards
    let cardsHtml = '';
    const stateColors = { confusion: 'warning', engagement: 'success', boredom: '', frustration: 'accent' };
    states.forEach(s => {
        const v = summary[s];
        cardsHtml += statCard(capitalize(s), (v.mean * 100).toFixed(0) + '%', stateColors[s] || '');
    });
    document.getElementById('emotion-summary-cards').innerHTML = cardsHtml;

    // Emotion timeline (multi-line)
    if (em.timeline) {
        const traces = [];
        const traceColors = { confusion: WARNING, engagement: SUCCESS, boredom: '#6c5ce7', frustration: ACCENT };
        for (const [state, data] of Object.entries(em.timeline)) {
            if (data && data.length > 0) {
                traces.push({
                    x: data.map(d => d.time / 1000),
                    y: data.map(d => d.value),
                    name: capitalize(state),
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: traceColors[state] || COLORS[traces.length], width: 2 },
                    fill: 'tozeroy',
                    opacity: 0.7,
                });
            }
        }
        if (traces.length > 0) {
            Plotly.newPlot('chart-emotion-timeline', traces, plotLayout('', {
                xaxis: { title: 'Time (s)', gridcolor: PLOT_GRID },
                yaxis: { title: 'Score', range: [0, 1], gridcolor: PLOT_GRID },
                showlegend: true,
                legend: { font: { color: PLOT_TEXT, size: 10 }, bgcolor: 'transparent' },
            }), PLOT_CONFIG);
        }
    }

    // Distribution donut
    if (em.distribution) {
        const labels = Object.keys(em.distribution).map(capitalize);
        const values = Object.values(em.distribution);
        Plotly.newPlot('chart-emotion-dist', [{
            labels, values,
            type: 'pie',
            hole: 0.5,
            marker: { colors: [WARNING, SUCCESS, '#6c5ce7', ACCENT] },
            textfont: { color: '#fff', size: 12 },
        }], plotLayout('', { showlegend: true, legend: { font: { color: PLOT_TEXT } } }), PLOT_CONFIG);
    }

    // Head pose
    const hp = em.head_pose_stability;
    if (hp && hp.available && hp.timeline) {
        const t = hp.timeline;
        Plotly.newPlot('chart-head-pose', [
            { x: t.map(d => d.time / 1000), y: t.map(d => d.yaw), name: 'Yaw', type: 'scatter', mode: 'lines', line: { color: ACCENT, width: 1.5 } },
            { x: t.map(d => d.time / 1000), y: t.map(d => d.pitch), name: 'Pitch', type: 'scatter', mode: 'lines', line: { color: SUCCESS, width: 1.5 } },
            { x: t.map(d => d.time / 1000), y: t.map(d => d.roll), name: 'Roll', type: 'scatter', mode: 'lines', line: { color: WARNING, width: 1.5 } },
        ], plotLayout('', {
            xaxis: { title: 'Time (s)', gridcolor: PLOT_GRID },
            yaxis: { title: 'Degrees', gridcolor: PLOT_GRID },
            showlegend: true, legend: { font: { color: PLOT_TEXT, size: 10 }, bgcolor: 'transparent' },
        }), PLOT_CONFIG);
    }

    // Eye openness
    const ff = em.facial_features;
    if (ff && ff.available && ff.eye_openness_timeline) {
        plotTimeline('chart-eye-openness', ff.eye_openness_timeline, 'Openness', SUCCESS, {}, 'eye_openness');
    }
}

// ── Temporal Patterns ──────────────────────────────────────────────────

function renderTemporal(tp) {
    if (!tp) return;

    // Activity rate (multi-stream)
    const ar = tp.activity_rate || {};
    const traces = [];
    const streamColors = { gaze: ACCENT, mouse: SUCCESS, keyboard: WARNING };
    for (const [stream, data] of Object.entries(ar)) {
        if (data && data.length > 0) {
            traces.push({
                x: data.map(d => d.time / 1000),
                y: data.map(d => d.value),
                name: capitalize(stream),
                type: 'scatter',
                mode: 'lines',
                line: { color: streamColors[stream] || COLORS[traces.length], width: 2 },
            });
        }
    }
    if (traces.length > 0) {
        Plotly.newPlot('chart-activity-rate', traces, plotLayout('', {
            xaxis: { title: 'Time (s)', gridcolor: PLOT_GRID },
            yaxis: { title: 'Events/sec', gridcolor: PLOT_GRID },
            showlegend: true,
            legend: { font: { color: PLOT_TEXT, size: 10 }, bgcolor: 'transparent' },
        }), PLOT_CONFIG);
    }
}

// ── Per-Question Analysis ──────────────────────────────────────────────

function renderQuestions(qData) {
    if (!qData || !qData.available) {
        showEmpty('section-questions', 'No per-question data available. Run a session with the updated experiment to collect answer events.');
        return;
    }
    const questions = qData.questions || [];
    const comp = qData.comparison || {};

    // Question pills
    const pillsEl = document.getElementById('question-pills');
    let pillsHtml = `<span class="q-pill ${!selectedQuestionDetail ? 'active' : ''}" onclick="selectQuestionDetail(null)">All</span>`;
    questions.forEach(q => {
        const isActive = selectedQuestionDetail === q.question_id;
        pillsHtml += `<span class="q-pill ${isActive ? 'active' : ''}" onclick="selectQuestionDetail('${q.question_id}')">${q.question_id.toUpperCase()}</span>`;
    });
    pillsEl.innerHTML = pillsHtml;

    // Summary table
    const tableEl = document.getElementById('question-summary-table');
    let html = '<table class="compare-table"><thead><tr><th>Q#</th><th>Title</th><th>Time</th><th>Ans</th><th>Fix</th><th>Load</th><th>Coord</th><th>Hesit.</th><th>Clicks</th><th>Emotion</th></tr></thead><tbody>';
    questions.forEach(q => {
        const domEm = q.dominant_emotion || 'N/A';
        html += `<tr>
            <td>${q.question_id.toUpperCase()}</td>
            <td style="text-align:left;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${q.title}">${q.title}</td>
            <td>${q.duration_formatted}</td>
            <td><strong>${q.selected_answer || '-'}</strong></td>
            <td>${q.fixation_count}</td>
            <td>${(q.cognitive_load || 0).toFixed(2)}</td>
            <td>${(q.coordination_score || 0).toFixed(2)}</td>
            <td>${q.hesitation_count || 0}</td>
            <td>${q.click_count}</td>
            <td>${capitalize(domEm)}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    tableEl.innerHTML = html;

    // Comparison bar charts
    const qLabels = comp.question_ids || [];

    if (comp.duration_ms && comp.duration_ms.length > 0) {
        Plotly.newPlot('chart-q-duration', [{
            x: qLabels.map(q => q.toUpperCase()), y: comp.duration_ms.map(d => d / 1000),
            type: 'bar', marker: { color: ACCENT, opacity: 0.85 },
        }], plotLayout('', { xaxis: { title: 'Question' }, yaxis: { title: 'Seconds' } }), PLOT_CONFIG);
    }

    if (comp.cognitive_load && comp.cognitive_load.length > 0) {
        Plotly.newPlot('chart-q-cogload', [{
            x: qLabels.map(q => q.toUpperCase()), y: comp.cognitive_load,
            type: 'bar', marker: { color: WARNING, opacity: 0.85 },
        }], plotLayout('', { xaxis: { title: 'Question' }, yaxis: { title: 'Load Index (0-1)' } }), PLOT_CONFIG);
    }

    if (comp.fixation_count && comp.fixation_count.length > 0) {
        Plotly.newPlot('chart-q-fixations', [{
            x: qLabels.map(q => q.toUpperCase()), y: comp.fixation_count,
            type: 'bar', marker: { color: SUCCESS, opacity: 0.85 },
        }], plotLayout('', { xaxis: { title: 'Question' }, yaxis: { title: 'Count' } }), PLOT_CONFIG);
    }

    // Emotion stacked bar
    const emotionCols = ['confusion', 'engagement', 'boredom', 'frustration'];
    const emotionColors = { confusion: WARNING, engagement: SUCCESS, boredom: '#6c5ce7', frustration: ACCENT };
    const emTraces = emotionCols.map(col => ({
        x: qLabels.map(q => q.toUpperCase()),
        y: comp[`mean_${col}`] || [],
        name: capitalize(col),
        type: 'bar',
        marker: { color: emotionColors[col] },
    }));
    if (emTraces.some(t => t.y.length > 0)) {
        Plotly.newPlot('chart-q-emotion', emTraces, plotLayout('', {
            barmode: 'group', xaxis: { title: 'Question' }, yaxis: { title: 'Mean Score' },
            showlegend: true, legend: { font: { color: PLOT_TEXT, size: 10 }, bgcolor: 'transparent' },
        }), PLOT_CONFIG);
    }

    // Detail area for selected question
    renderQuestionDetailArea(questions);
}

function selectQuestionDetail(qid) {
    selectedQuestionDetail = qid;
    if (analyticsData && analyticsData.questions) renderQuestions(analyticsData.questions);
}

function renderQuestionDetailArea(questions) {
    const area = document.getElementById('question-detail-area');
    if (!selectedQuestionDetail) {
        area.innerHTML = '<div style="color:var(--text-secondary);text-align:center;padding:30px;font-size:0.9em;">Click a question pill above to see detailed metrics</div>';
        return;
    }
    const q = questions.find(q => q.question_id === selectedQuestionDetail);
    if (!q) { area.innerHTML = ''; return; }

    area.innerHTML = `
        <h3 class="section-subtitle">${q.question_id.toUpperCase()}: ${q.title}</h3>
        <div class="card-grid">
            ${statCard('Duration', q.duration_formatted, 'accent')}
            ${statCard('Answer', q.selected_answer || '-')}
            ${statCard('Fixations', q.fixation_count)}
            ${statCard('Fix. Duration', Math.round(q.fixation_duration_mean || 0) + ' ms')}
            ${statCard('Saccades', q.saccade_count)}
            ${statCard('Cog. Load', (q.cognitive_load || 0).toFixed(2), q.cognitive_load > 0.5 ? 'warning' : 'success')}
            ${statCard('K Coeff', (q.k_coefficient || 0).toFixed(2))}
            ${statCard('Clicks', q.click_count)}
            ${statCard('Emotion', capitalize(q.dominant_emotion || 'N/A'))}
            ${statCard('Calculator', q.calc_used ? 'Yes' : 'No')}
            ${statCard('Gaze Samples', q.gaze_samples || 0)}
            ${statCard('Mouse Events', q.mouse_events || 0)}
        </div>
        ${q.heatmap ? '<div class="chart-row single"><div class="chart-box"><div class="chart-title">Gaze Heatmap</div><div id="chart-q-detail-heatmap" class="plotly-chart"></div></div></div>' : ''}
        ${Object.keys(q.aoi_dwell || {}).length > 0 ? '<div class="chart-row single"><div class="chart-box"><div class="chart-title">AOI Dwell Time</div><div id="chart-q-detail-aoi" class="plotly-chart"></div></div></div>' : ''}
    `;

    // Render heatmap if available
    if (q.heatmap && q.heatmap.z) {
        setTimeout(() => {
            Plotly.newPlot('chart-q-detail-heatmap', [{
                z: q.heatmap.z, type: 'heatmap',
                colorscale: [[0, '#1a1a2e'], [0.3, '#0f3460'], [0.6, '#e94560'], [1, '#ffd93d']],
                showscale: false,
            }], plotLayout('', { yaxis: { autorange: 'reversed' } }), PLOT_CONFIG);
        }, 50);
    }

    // AOI dwell chart
    const aoiKeys = Object.keys(q.aoi_dwell || {});
    if (aoiKeys.length > 0) {
        setTimeout(() => {
            Plotly.newPlot('chart-q-detail-aoi', [{
                x: aoiKeys.map(capitalize),
                y: aoiKeys.map(k => q.aoi_dwell[k] / 1000),
                type: 'bar', marker: { color: SUCCESS, opacity: 0.85 },
            }], plotLayout('', { xaxis: { title: 'AOI Region' }, yaxis: { title: 'Dwell Time (s)' } }), PLOT_CONFIG);
        }, 50);
    }
}

// ── Answers ────────────────────────────────────────────────────────────

function renderAnswers(aData) {
    if (!aData || !aData.available) {
        showEmpty('section-answers', 'No answer data available. Run a session with the updated experiment to collect answers.');
        return;
    }

    // Summary cards
    document.getElementById('answer-summary-cards').innerHTML = `
        ${statCard('Answered', `${aData.total_answered}/${aData.total_questions}`, 'accent')}
        ${statCard('Avg Response', aData.avg_response_time_formatted)}
        ${statCard('Calculator Used', `${aData.calc_usage_count} (${aData.calc_usage_rate}%)`)}
        ${statCard('Skipped', aData.summary.skipped || 0, aData.summary.skipped > 0 ? 'warning' : '')}
    `;

    // Answer table
    const answers = aData.answers || [];
    let html = '<table class="answer-table"><thead><tr><th>Q#</th><th>Title</th><th>Answer</th><th>Response Time</th><th>Calculator</th></tr></thead><tbody>';
    answers.forEach(a => {
        html += `<tr>
            <td>${a.question_id.toUpperCase()}</td>
            <td style="text-align:left;">${a.title}</td>
            <td><strong>${a.selected_answer || '-'}</strong></td>
            <td>${(a.response_time_ms / 1000).toFixed(1)}s</td>
            <td class="${a.calc_used ? 'calc-yes' : 'calc-no'}">${a.calc_used ? '✓ Yes' : '—'}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('answer-table-container').innerHTML = html;

    // Response time bar chart
    if (answers.length > 0) {
        Plotly.newPlot('chart-response-time', [{
            x: answers.map(a => a.question_id.toUpperCase()),
            y: answers.map(a => a.response_time_ms / 1000),
            type: 'bar',
            marker: {
                color: answers.map(a => a.calc_used ? SUCCESS : ACCENT),
                opacity: 0.85,
            },
            text: answers.map(a => a.selected_answer),
            textposition: 'outside',
            textfont: { color: PLOT_TEXT, size: 12 },
        }], plotLayout('', {
            xaxis: { title: 'Question' },
            yaxis: { title: 'Response Time (s)' },
        }), PLOT_CONFIG);
    }

    // Answer timeline (scatter with timestamps)
    if (answers.length > 0) {
        Plotly.newPlot('chart-answer-timeline', [{
            x: answers.map(a => a.timestamp / 1000),
            y: answers.map(a => a.question_id.toUpperCase()),
            mode: 'markers+text',
            type: 'scatter',
            marker: { size: 14, color: ACCENT, symbol: 'diamond' },
            text: answers.map(a => a.selected_answer),
            textposition: 'top center',
            textfont: { color: PLOT_TEXT, size: 11 },
        }], plotLayout('', {
            xaxis: { title: 'Session Time (s)' },
            yaxis: { title: '' },
        }), PLOT_CONFIG);
    }
}

// ── Survey ─────────────────────────────────────────────────────────────

function renderSurvey(sData) {
    if (!sData || !sData.available) {
        showEmpty('section-survey', 'No survey data available. Run a session and complete the post-study survey.');
        return;
    }

    // Demographics cards
    const demo = sData.demographics || {};
    const demoLabels = sData.labels || {};
    let demoHtml = '';
    const demoMap = { q16: 'Gender', q17: 'Age Group', q18: 'Student Level', q19: 'Field of Study' };
    for (const [key, label] of Object.entries(demoMap)) {
        const val = demo[key] || 'N/A';
        demoHtml += statCard(label, capitalize(val.replace(/_/g, ' ')));
    }
    document.getElementById('survey-demographics-cards').innerHTML = demoHtml;

    // Helper to render horizontal bar chart for a survey section
    function renderSurveySection(chartId, sectionKey) {
        const sec = (sData.sections || {})[sectionKey];
        if (!sec) { clearChart(chartId); return; }
        const responses = sec.responses || {};
        const labels = sec.labels || {};
        const keys = Object.keys(responses).filter(k => typeof responses[k] === 'number');
        if (keys.length === 0) { clearChart(chartId); return; }

        // Sort by question number
        keys.sort((a, b) => parseInt(a.replace('q', '')) - parseInt(b.replace('q', '')));

        Plotly.newPlot(chartId, [{
            y: keys.map(k => {
                const lbl = labels[k] || k;
                return lbl.length > 45 ? lbl.slice(0, 42) + '...' : lbl;
            }),
            x: keys.map(k => responses[k]),
            type: 'bar',
            orientation: 'h',
            marker: { color: ACCENT, opacity: 0.85 },
            text: keys.map(k => responses[k]),
            textposition: 'outside',
            textfont: { color: PLOT_TEXT },
        }], plotLayout('', {
            xaxis: { title: 'Rating (1-5)', range: [0, 5.5], dtick: 1 },
            yaxis: { automargin: true },
            margin: { l: 250, r: 40, t: 10, b: 40 },
        }), PLOT_CONFIG);
    }

    renderSurveySection('chart-survey-difficulty', 'difficulty_effort');
    renderSurveySection('chart-survey-info', 'information_use');
    renderSurveySection('chart-survey-decision', 'decision_style');
    renderSurveySection('chart-survey-experience', 'experience');
    renderSurveySection('chart-survey-about', 'about_you');

    // Radar chart of section averages
    const avgs = sData.section_averages || {};
    const radarLabels = [];
    const radarValues = [];
    for (const [, info] of Object.entries(avgs)) {
        radarLabels.push(info.label);
        radarValues.push(info.mean);
    }
    if (radarLabels.length > 0) {
        Plotly.newPlot('chart-survey-radar', [{
            type: 'scatterpolar',
            r: [...radarValues, radarValues[0]], // close the shape
            theta: [...radarLabels, radarLabels[0]],
            fill: 'toself',
            fillcolor: ACCENT + '33',
            line: { color: ACCENT, width: 2 },
            marker: { size: 8, color: ACCENT },
        }], plotLayout('', {
            polar: {
                bgcolor: PLOT_BG,
                radialaxis: { range: [0, 5], gridcolor: PLOT_GRID, color: PLOT_TEXT, dtick: 1 },
                angularaxis: { gridcolor: PLOT_GRID, color: PLOT_TEXT },
            },
        }), PLOT_CONFIG);
    }
}

// ── Question Filter ───────────────────────────────────────────────────

function onQuestionFilter(qid) {
    selectedQuestion = qid;
    // Update question filter dropdown label
    document.getElementById('question-filter').value = qid;
    // Re-render current section
    renderCurrentSection();
}

function populateQuestionFilter() {
    const select = document.getElementById('question-filter');
    if (!select) return;
    // Reset options
    select.innerHTML = '<option value="all">All Questions</option>';
    if (analyticsData && analyticsData.questions && analyticsData.questions.available) {
        const questions = analyticsData.questions.questions || [];
        questions.forEach(q => {
            select.innerHTML += `<option value="${q.question_id}">${q.question_id.toUpperCase()} - ${q.title}</option>`;
        });
    }
}

// ── Comparison ─────────────────────────────────────────────────────────

async function runComparison() {
    const select = document.getElementById('compare-session-select');
    const selected = Array.from(select.selectedOptions).map(o => o.value);
    if (selected.length < 2) {
        alert('Select at least 2 sessions to compare.');
        return;
    }
    setStatus('Comparing sessions...');
    try {
        const res = await fetch(`/api/sessions/compare?ids=${selected.join(',')}`);
        const data = await res.json();
        renderComparison(data);
        setStatus('Ready');
    } catch (e) {
        console.error('Comparison failed:', e);
        setStatus('Comparison failed');
    }
}

function renderComparison(data) {
    const sessions = data.sessions || {};
    const sids = Object.keys(sessions);
    if (sids.length === 0) return;

    const metrics = [
        ['Duration', 'duration'],
        ['Quality Score', 'quality_score'],
        ['Total Samples', 'total_samples'],
        ['Fixation Count', 'fixation_count'],
        ['Mean Fix. Duration', 'fixation_duration_mean'],
        ['Saccade Count', 'saccade_count'],
        ['Mean Sac. Amplitude', 'saccade_amplitude_mean'],
        ['Total Clicks', 'total_clicks'],
        ['Total Keystrokes', 'total_keystrokes'],
    ];

    let html = '<table class="compare-table"><thead><tr><th>Metric</th>';
    sids.forEach(sid => { html += `<th>${sid.slice(0, 8)}...</th>`; });
    html += '</tr></thead><tbody>';

    metrics.forEach(([label, key]) => {
        html += `<tr><td>${label}</td>`;
        sids.forEach(sid => {
            const v = sessions[sid][key];
            html += `<td>${v !== undefined && v !== null ? (typeof v === 'number' ? fmtNum(Math.round(v)) : v) : 'N/A'}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    document.getElementById('compare-table-container').innerHTML = html;
}

// ── Chart Helpers ──────────────────────────────────────────────────────

function plotTimeline(divId, data, yLabel, color, extraLayout, valueKey) {
    if (!data || data.length === 0) {
        clearChart(divId);
        return;
    }
    const vk = valueKey || 'value';
    Plotly.newPlot(divId, [{
        x: data.map(d => d.time / 1000),
        y: data.map(d => d[vk]),
        type: 'scatter',
        mode: 'lines',
        line: { color: color || ACCENT, width: 2 },
        fill: 'tozeroy',
        fillcolor: (color || ACCENT) + '22',
    }], plotLayout('', Object.assign({
        xaxis: { title: 'Time (s)', gridcolor: PLOT_GRID },
        yaxis: { title: yLabel || '', gridcolor: PLOT_GRID },
    }, extraLayout || {})), PLOT_CONFIG);
}

function plotHistogramFromBins(divId, bins, xLabel, color) {
    if (!bins || bins.length === 0) {
        clearChart(divId);
        return;
    }
    Plotly.newPlot(divId, [{
        x: bins.map(b => (b.bin_start + b.bin_end) / 2),
        y: bins.map(b => b.count),
        type: 'bar',
        marker: { color: color || ACCENT, opacity: 0.8 },
        width: bins.length > 0 ? (bins[0].bin_end - bins[0].bin_start) * 0.9 : undefined,
    }], plotLayout('', {
        xaxis: { title: xLabel || '', gridcolor: PLOT_GRID },
        yaxis: { title: 'Count', gridcolor: PLOT_GRID },
        bargap: 0.05,
    }), PLOT_CONFIG);
}

function clearChart(divId) {
    const el = document.getElementById(divId);
    if (el) el.innerHTML = '<div style="color:var(--text-secondary);text-align:center;padding:60px 0;font-size:0.9em;">No data</div>';
}

// ── Utility Helpers ────────────────────────────────────────────────────

function statCard(label, value, colorClass) {
    return `<div class="stat-card"><div class="label">${label}</div><div class="value ${colorClass || ''}">${value}</div></div>`;
}

function fmtNum(n) {
    if (n === undefined || n === null) return '0';
    return n.toLocaleString();
}

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}

function showEmpty(sectionId, msg) {
    const sec = document.getElementById(sectionId);
    if (sec) {
        const existing = sec.querySelector('.empty-state');
        if (!existing) {
            sec.insertAdjacentHTML('beforeend', `<div class="empty-state"><h2>${msg}</h2></div>`);
        }
    }
}

function setStatus(text) {
    const el = document.getElementById('status-text');
    if (el) el.textContent = text;
}

function exportReport() {
    if (!analyticsData) {
        alert('No analytics data to export. Load a session first.');
        return;
    }
    const blob = new Blob([JSON.stringify(analyticsData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analytics_${sessionId || 'report'}.json`;
    a.click();
    URL.revokeObjectURL(url);
}
