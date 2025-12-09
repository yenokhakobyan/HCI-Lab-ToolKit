# HCI Session Analysis Dashboard

A comprehensive Streamlit-based dashboard for analyzing web HCI (Human-Computer Interaction) session data including eye tracking, mouse movement, facial expressions, and behavioral patterns.

## Quick Start

```bash
# Run the dashboard
./run_dashboard.sh

# Or directly with Streamlit
streamlit run src/analysis/dashboard.py
```

The dashboard will open at `http://localhost:8502`

## Data Requirements

The dashboard expects session data in `data/raw/web_hci/{session_id}/` with:

| File | Description |
|------|-------------|
| `gaze_*.csv` | Eye tracking data (timestamp, x, y) |
| `mouse_*.csv` | Mouse events (timestamp, x, y, event) |
| `face_mesh_*.csv` | Facial landmarks (468 points per frame) |
| `metadata_*.json` | Session metadata |
| `timeline_*.json` | Event timeline |
| `recording.webm` | Screen recording video |

---

## Dashboard Tabs

### 1. Video & Timeline
- **Video playback** of the recorded session
- **Synchronized timeline** showing all data streams:
  - Gaze velocity
  - Mouse activity
  - Face detection
  - Click events

### 2. Eye Tracking
- **Gaze Heatmap**: Density visualization of where the user looked
- **Gaze Scanpath**: Sequential path of eye movements
- **Time filtering**: Analyze specific portions of the session
- **Statistics**: Mean position, sampling rate

### 3. Mouse Analysis
- **Movement Trajectory**: Path of mouse movement
- **Click Positions**: Map of all click locations
- **Statistics**: Total events, clicks, movements, distance traveled

### 4. Fixations & Saccades
Analyzes eye movement patterns using the I-DT (Dispersion-Threshold) algorithm.

#### Fixations
Points where the eye remains relatively stable (reading, processing).

| Parameter | Default | Description |
|-----------|---------|-------------|
| Velocity Threshold | 50 px/ms | Max velocity for fixation |
| Min Duration | 100 ms | Minimum fixation time |
| Max Dispersion | 150 px | Maximum spread within fixation |

**Tip**: For webcam eye tracking, increase dispersion to 200-300px.

#### Saccades
Rapid eye movements between fixations.

| Metric | Description |
|--------|-------------|
| Amplitude | Distance traveled (pixels) |
| Duration | Time of movement (ms) |
| Velocity | Speed (px/ms) |
| Direction | Angle in degrees (0=right, 90=down) |

#### Visualizations
- **Scanpath**: Numbered fixations connected by saccade lines
- **Duration histogram**: Distribution of fixation durations
- **Amplitude histogram**: Distribution of saccade distances
- **Direction polar plot**: Which directions eyes moved most

### 5. AOI Analysis (Areas of Interest)
Define rectangular regions to analyze attention distribution.

#### How to Use
1. Set number of AOIs (1-10)
2. Define each AOI: name, x, y, width, height
3. Click "Analyze AOIs"

#### Metrics Per AOI
| Metric | Description |
|--------|-------------|
| Gaze Points | Number of gaze samples in AOI |
| Gaze Duration | Total time looking at AOI (ms) |
| Gaze Entries | How many times gaze entered AOI |
| First Gaze Time | Time to first look (TTFF) |
| Mouse Clicks | Clicks within AOI |

#### Transition Matrix
Shows how attention moved between AOIs (useful for understanding navigation patterns).

### 6. Attention Metrics
Comprehensive attention analysis.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Total Fixations | Number of fixations | More = more detailed scanning |
| Mean Fixation Duration | Average fixation length | Longer = deeper processing |
| Fixation Rate | Fixations per second | Higher = more active scanning |
| Gaze Dispersion | Spread of gaze positions | Higher = wider exploration |
| Saccade Count | Number of saccades | More = more visual search |
| Mean Saccade Amplitude | Average saccade distance | Larger = bigger jumps |

#### TTFF (Time to First Fixation)
How long until user first looked at each AOI - critical for UX research.

### 7. Gaze-Mouse Coordination
Analyzes relationship between eye and hand movements.

| Metric | Description |
|--------|-------------|
| X/Y Correlation | How well gaze and mouse positions correlate (-1 to 1) |
| Coordination Score | Overall coordination (0-1, higher = better) |
| Mean Distance | Average pixels between gaze and mouse |
| Temporal Relationship | Who leads - gaze or mouse |

#### Lag Analysis
Cross-correlation at different time offsets to determine if:
- **Gaze leads mouse**: User looks then moves mouse (typical)
- **Mouse leads gaze**: User moves mouse then looks (unusual)
- **Synchronized**: Both move together

### 8. Behavior & Emotion

#### Behavioral Patterns

| Pattern | Description | Indicates |
|---------|-------------|-----------|
| Reading Pattern | F-pattern, Z-pattern, Vertical, Mixed | How user scans content |
| Hesitations | Mouse pauses without clicking (>300ms) | Uncertainty, decision points |
| Backtracking | Returning to previous gaze positions | Re-reading, confusion |
| Rapid Scanning | High-velocity eye movements | Visual search |
| Focused Attention | Long fixations (>300ms) | Areas of high interest |
| Mouse Jitter | Small rapid mouse movements | Nervousness, frustration |

#### Emotion Analysis (Face Mesh Based)
Uses MediaPipe face mesh landmarks to detect emotions via facial feature analysis.

**Detected Emotions:**
- Neutral, Happy, Surprised, Confused, Focused, Frustrated, Tired

**Metrics:**
| Feature | Detection Method |
|---------|-----------------|
| Eye Aspect Ratio (EAR) | Eye openness (blinks, tiredness) |
| Mouth Aspect Ratio (MAR) | Mouth openness (surprise, yawn) |
| Eyebrow Position | Raised = surprise, furrowed = confusion |
| Smile Ratio | Mouth corner positions |

**Valence-Arousal Model (Circumplex):**
- **Valence**: Positive (happy) to Negative (frustrated)
- **Arousal**: High energy (excited) to Low energy (tired)

| Quadrant | Emotions |
|----------|----------|
| +Valence, +Arousal | Excited, Happy |
| -Valence, +Arousal | Angry, Stressed |
| +Valence, -Arousal | Calm, Relaxed |
| -Valence, -Arousal | Sad, Bored |

**Blink Analysis:**
- Normal rate: 15-20 blinks/min
- Low (<10): Focused/staring
- High (>20): Fatigue/stress

### 10. Compare & Export

#### Multi-Session Comparison
1. Select multiple sessions
2. Click "Compare Sessions"
3. View:
   - Metrics comparison table
   - Duration/activity bar charts
   - Aggregate heatmap across all sessions

#### Export Options
- **JSON Report**: Complete analysis with recommendations
- **CSV Export**: Raw data for each data type

---

## Troubleshooting

### No Fixations/Saccades Detected
- **Increase Max Dispersion** to 200-400px (webcam tracking is noisy)
- **Decrease Min Duration** to 50-80ms
- Check if gaze data exists in the CSV

### No Behavioral Patterns Showing
- Ensure session has both gaze AND mouse data
- Check if data timestamps are valid (not all zeros)

### Emotion Analysis Empty
- Requires face_mesh CSV with landmark data
- Webcam must have been enabled during session
- Face must be visible and detected

### Slow Performance
- Large sessions (>10k data points) may take time
- Emotion analysis parses landmarks for each frame
- Use time filtering to analyze portions

---

## Key Metrics Reference

### For UX Research
| Metric | Good For |
|--------|----------|
| TTFF (Time to First Fixation) | Measuring visibility of elements |
| Dwell Time | Interest/engagement with content |
| AOI Transitions | Understanding navigation flow |
| Hesitations | Identifying confusing UI elements |
| Click positions | Verifying clickable areas |

### For Cognitive Assessment
| Metric | Indicates |
|--------|-----------|
| Fixation Duration | Processing depth |
| Saccade Amplitude | Visual search strategy |
| Blink Rate | Mental fatigue |
| Cognitive Load Score | Overall mental effort |

### For Emotional Response
| Metric | Indicates |
|--------|-----------|
| Valence | Positive/negative reaction |
| Arousal | Engagement level |
| Frustration Events | Pain points in UX |
| Engagement Timeline | Interest over time |

---

## Algorithm Details

### Fixation Detection (I-DT Algorithm)
```
1. Start at first gaze point
2. Expand window while dispersion < threshold
3. If duration >= min_duration → fixation found
4. Move to next point after fixation
5. Repeat
```

### Emotion Classification
Based on Facial Action Coding System (FACS) principles:
- Eye Aspect Ratio (EAR) = vertical / horizontal eye distance
- Mouth Aspect Ratio (MAR) = vertical / horizontal mouth distance
- Combined with eyebrow position and smile detection

### Cognitive Load Estimation
Composite score from:
- Fixation duration trend (increasing = higher load)
- Saccade velocity (decreasing = higher load)
- Blink rate (lower = higher load)

---

## Tips for Best Results

1. **Calibrate eye tracking** before sessions for better accuracy
2. **Good lighting** improves face mesh detection
3. **Consistent head position** reduces noise
4. **Multiple sessions** allow for comparative analysis
5. **Define AOIs** based on your research questions
6. **Adjust thresholds** based on your tracking hardware quality
