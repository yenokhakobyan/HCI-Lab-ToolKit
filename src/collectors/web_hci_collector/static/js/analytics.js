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
        case 'emotion': renderEmotion(analyticsData.emotion); break;
        case 'temporal': renderTemporal(analyticsData.temporal); break;
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

    document.getElementById('behavioral-stats-cards').innerHTML = `
        ${statCard('Total Clicks', cp.total_clicks || 0)}
        ${statCard('Key Events', kd.total_events || 0)}
        ${statCard('Gaze-Mouse Corr.', gm.available ? (gm.correlation || 0).toFixed(2) : 'N/A')}
        ${statCard('Mean G-M Dist.', gm.available ? Math.round(gm.mean_distance || 0) + ' px' : 'N/A')}
        ${statCard('Idle Periods', idle.length)}
    `;

    plotTimeline('chart-mouse-velocity', b.mouse_velocity_timeline, 'Velocity (px/ms)', ACCENT);
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
