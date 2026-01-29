# Eye Tracking in Human-Computer Interaction: Technical Reference Guide

**HCI + L Laboratory | Yerevan State University**  
**Technical Notes for Research Team**

---

## Table of Contents

1. [Introduction to Eye Tracking](#1-introduction-to-eye-tracking)
2. [Eye Tracking Hardware](#2-eye-tracking-hardware)
3. [Fundamental Concepts](#3-fundamental-concepts)
4. [Core Eye Movement Types](#4-core-eye-movement-types)
5. [Key Metrics and Measures](#5-key-metrics-and-measures)
6. [Areas of Interest (AOI) Analysis](#6-areas-of-interest-aoi-analysis)
7. [Scanpath Analysis](#7-scanpath-analysis)
8. [Cognitive Load and Pupillometry](#8-cognitive-load-and-pupillometry)
9. [Theoretical Models and Frameworks](#9-theoretical-models-and-frameworks)
10. [Data Processing and Analysis](#10-data-processing-and-analysis)
11. [Experimental Design Considerations](#11-experimental-design-considerations)
12. [Software Tools and Platforms](#12-software-tools-and-platforms)
13. [Quality Assurance and Validation](#13-quality-assurance-and-validation)
14. [Applications in HCI Research](#14-applications-in-hci-research)
15. [Visual Attention Models](#15-visual-attention-models)
16. [Noteworthy Papers and Landmark Studies](#16-noteworthy-papers-and-landmark-studies)
17. [Important Datasets](#17-important-datasets)
18. [Noteworthy Approaches and Methods](#18-noteworthy-approaches-and-methods)
19. [Notable Projects and Research Initiatives](#19-notable-projects-and-research-initiatives)
20. [Emerging Trends and Future Directions](#20-emerging-trends-and-future-directions)
21. [References and Further Reading](#21-references-and-further-reading)

---

## 1. Introduction to Eye Tracking

### 1.1 What is Eye Tracking?

Eye tracking is the process of measuring either the point of gaze (where a person is looking) or the motion of an eye relative to the head. In HCI research, eye tracking provides objective, quantitative data about visual attention, cognitive processing, and user behavior during interaction with digital interfaces.

### 1.2 Why Eye Tracking Matters for HCI

Eye tracking bridges the gap between what users say and what they actually do. It reveals:

- **Attention allocation**: Where users look and for how long
- **Information processing**: How users read, scan, and comprehend content
- **Cognitive states**: Mental workload, confusion, engagement
- **Usability issues**: Navigation problems, missed elements, inefficient search patterns
- **Decision-making processes**: How visual information influences choices

### 1.3 The Eye-Mind Hypothesis

The foundational assumption in eye tracking research comes from Just and Carpenter (1980):

> "There is no appreciable lag between what is being fixated and what is being processed... the eye remains fixated on a word as long as the word is being processed."

**Key Assumptions:**

1. **Immediacy Assumption**: Cognitive processing occurs immediately upon encountering information
2. **Eye-Mind Assumption**: The eye remains fixated on information while it is being cognitively processed

**Limitations to Consider:**
- Covert attention (processing without direct fixation)
- Parafoveal processing (processing information outside direct fixation)
- Mind wandering during fixations

---

## 2. Eye Tracking Hardware

### 2.1 Types of Eye Trackers

| Type | Description | Use Cases | Examples |
|------|-------------|-----------|----------|
| **Remote/Screen-based** | Mounted below or integrated into display | Desktop studies, controlled experiments | Tobii Pro Spectrum, EyeLink 1000 Plus |
| **Head-mounted** | Worn on head like glasses | Mobile studies, real-world environments | Tobii Pro Glasses 3, Pupil Labs |
| **Tower-mounted** | High-precision with head stabilization | Reading research, precise measurements | EyeLink 1000 Plus with chin rest |
| **Webcam-based** | Uses standard webcams | Large-scale studies, remote research | WebGazer.js, GazeRecorder |

### 2.2 EyeLink 1000 Plus (Our Primary System)

**Specifications:**
- Sampling rate: Up to 2000 Hz (monocular), 1000 Hz (binocular)
- Accuracy: 0.25° - 0.5° average
- Precision (RMS): 0.01° typical
- Head movement tolerance: 22 × 18 × 20 cm (remote mode)
- Pupil tracking: Centroid and ellipse fitting

**Key Features:**
- Supports both head-fixed and head-free tracking
- Integration capabilities with EEG, fMRI, MEG
- Real-time gaze position output
- Built-in 9-point and 13-point calibration

**Tracking Modes:**
1. **Pupil-CR (Corneal Reflection)**: Most accurate, uses infrared reflection
2. **Pupil-only**: For situations where CR is not available

### 2.3 Tobii Eye Trackers

**Tobii Pro Spectrum:**
- Sampling rates: 60, 120, 150, 300, 600, 1200 Hz
- Screen size support: Up to 24"
- Binocular tracking with individual eye data

**Tobii Pro Fusion:**
- Portable design
- 120 Hz or 250 Hz sampling
- Suitable for laptops and external monitors

**Key Tobii Technology:**
- **Bright Pupil**: Infrared light creates bright pupil effect
- **Dark Pupil**: Alternative method for certain conditions
- **3D eye model**: Compensates for head movement

### 2.4 Webcam-Based Solutions

**WebGazer.js:**
- Open-source JavaScript library
- Runs in browser without special hardware
- Uses machine learning for gaze prediction
- Lower accuracy (~100-200 pixels) but highly scalable

**Considerations for Webcam Tracking:**
- Requires good lighting conditions
- Lower precision than dedicated hardware
- Suitable for large-scale, less precision-critical studies
- Important for our Study 3 (scalability assessment)

### 2.5 Sampling Rate Selection Guide

| Research Type | Recommended Rate | Rationale |
|--------------|------------------|-----------|
| Reading research | 1000+ Hz | Capture brief fixations |
| Usability testing | 60-120 Hz | Sufficient for AOI analysis |
| Saccade dynamics | 500+ Hz | Accurate saccade measurement |
| Smooth pursuit | 250+ Hz | Track continuous movements |
| General HCI | 120-300 Hz | Balance of detail and data size |

---

## 3. Fundamental Concepts

### 3.1 Visual Anatomy Relevant to Eye Tracking

**Fovea:**
- Central 2° of visual field
- Highest visual acuity
- Where we "really" see

**Parafovea:**
- Extends 2-5° from fixation point
- Reduced but useful acuity
- Important for reading (preview benefit)

**Periphery:**
- Beyond 5° from fixation
- Low acuity but sensitive to motion
- Guides saccade targeting

### 3.2 Coordinate Systems

**Screen Coordinates:**
- Origin typically at top-left corner
- X increases rightward
- Y increases downward
- Units: pixels

**Gaze Coordinates:**
- Can be normalized (0-1) or in pixels
- May include depth (Z) for 3D tracking

**Visual Angle:**
- Measured in degrees
- 1° ≈ 30-35 pixels at typical viewing distance (60 cm)
- More comparable across setups than pixels

**Conversion Formula:**
```
Visual Angle (degrees) = 2 × arctan(size / (2 × distance))
```

### 3.3 Calibration and Validation

**Calibration Process:**
1. Present calibration targets (typically 5, 9, or 13 points)
2. Participant fixates each point
3. System maps pupil/CR positions to screen coordinates
4. Creates mathematical model for gaze estimation

**Validation:**
- Present additional points after calibration
- Measure error between actual and estimated gaze
- Accept if average error < 1° (or study-specific threshold)

**Calibration Quality Indicators:**
- **Average error**: Mean deviation across all points
- **Maximum error**: Worst-case deviation
- **Spatial distribution**: Errors may vary across screen regions

**Best Practices:**
- Recalibrate if participant moves significantly
- Validate before critical trial blocks
- Document calibration quality in data files
- Consider drift correction for long sessions

---

## 4. Core Eye Movement Types

### 4.1 Fixations

**Definition:** Periods when the eye is relatively stationary, allowing visual information intake.

**Characteristics:**
- Duration: Typically 150-600 ms (mean ~250 ms for reading)
- Not perfectly still (includes tremor, drift, microsaccades)
- Primary source of visual information acquisition

**What Fixations Indicate:**
- Attention allocation
- Information processing time
- Cognitive effort (longer = more processing)
- Interest/relevance of content

### 4.2 Saccades

**Definition:** Rapid, ballistic eye movements between fixations.

**Characteristics:**
- Duration: 20-200 ms
- Velocity: Up to 500°/second
- Amplitude: Typically 2-15° in reading; larger in scene viewing
- Suppressed vision during movement (saccadic suppression)

**Types of Saccades:**
- **Progressive**: Forward movement (e.g., left-to-right in English reading)
- **Regressive**: Backward movement (indicates re-reading or confusion)
- **Return sweeps**: Large saccades to beginning of next line

**Saccade Parameters:**
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| Amplitude | Distance traveled | 2-15° |
| Duration | Time of movement | 20-200 ms |
| Peak velocity | Maximum speed | 300-500°/s |
| Latency | Time to initiate | 150-250 ms |

### 4.3 Smooth Pursuit

**Definition:** Slow, continuous eye movements tracking moving objects.

**Characteristics:**
- Velocity: Up to ~30°/second (matches target)
- Requires moving target to initiate
- Cannot be made voluntarily without target

**Relevance in HCI:**
- Tracking moving UI elements
- Animation perception
- Video content viewing

### 4.4 Vergence Movements

**Definition:** Coordinated movements of both eyes in opposite directions.

**Types:**
- **Convergence**: Eyes move inward (near objects)
- **Divergence**: Eyes move outward (far objects)

**Relevance:**
- Depth perception studies
- VR/AR research
- 3D display evaluation

### 4.5 Microsaccades

**Definition:** Tiny, involuntary saccades during fixations.

**Characteristics:**
- Amplitude: < 1°
- Frequency: 1-2 per second
- Function: Prevent retinal adaptation, maintain vision

**Research Applications:**
- Attention indicators
- Cognitive load markers
- May need high sampling rate (500+ Hz) to detect

---

## 5. Key Metrics and Measures

### 5.1 Fixation-Based Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Fixation Count** | Number of fixations on AOI/stimulus | Higher = more attention/interest OR confusion |
| **Fixation Duration** | Length of individual fixation (ms) | Longer = deeper processing |
| **Total Fixation Duration** | Sum of all fixation durations on AOI | Overall attention/processing time |
| **Mean Fixation Duration** | Average fixation length | Processing efficiency indicator |
| **First Fixation Duration** | Duration of first fixation on AOI | Initial processing/recognition time |
| **Gaze Duration** | Sum of fixations before leaving AOI | First-pass processing time |
| **Total Dwell Time** | All time spent looking at AOI | Cumulative attention measure |
| **Time to First Fixation** | Time until first fixation on AOI | Noticeability/salience indicator |

### 5.2 Saccade-Based Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Saccade Count** | Number of saccades made | Search/scanning activity |
| **Saccade Amplitude** | Distance of saccade (degrees) | Search strategy indicator |
| **Saccade Duration** | Time of saccade movement | Related to amplitude |
| **Saccade Velocity** | Speed of eye movement | Motor control indicator |
| **Regression Rate** | Proportion of backward saccades | Comprehension difficulty |
| **Saccade/Fixation Ratio** | Relationship between movements | Scanning efficiency |

### 5.3 Spatial Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Spatial Density** | Concentration of fixations | Focused vs. distributed attention |
| **Convex Hull Area** | Area encompassing all fixations | Extent of visual exploration |
| **Fixation Dispersion** | Spread of fixation locations | Search pattern diffusion |
| **Nearest Neighbor Index** | Clustering of fixations | Systematic vs. random scanning |

### 5.4 Temporal Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Entry Time** | When AOI was first fixated | Priority in visual search |
| **Dwell Time** | Total time in AOI | Engagement level |
| **Revisits** | Number of returns to AOI | Re-processing need |
| **Time to First Fixation** | Latency to initial fixation | Salience/noticeability |

### 5.5 Reading-Specific Metrics

| Metric | Definition | Use |
|--------|------------|-----|
| **First Fixation Duration** | Duration of first fixation on word | Early lexical processing |
| **Single Fixation Duration** | When word receives exactly one fixation | Word recognition time |
| **Gaze Duration** | Sum of first-pass fixations | Lexical access time |
| **Go-Past Time** | Time from first fixation until leaving rightward | Integration difficulty |
| **Total Reading Time** | All fixations including regressions | Complete processing time |
| **Skipping Rate** | Proportion of words not fixated | Predictability/frequency effect |
| **Regression Path Duration** | Includes regressive fixations | Comprehension difficulty |

### 5.6 Pupillometric Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Pupil Diameter** | Size of pupil (mm or pixels) | Cognitive load, arousal |
| **Pupil Dilation** | Change from baseline | Task difficulty, emotional response |
| **PCPD** | Peak change in pupil diameter | Maximum cognitive effort |
| **LHIPA** | Low/High Index of Pupillary Activity | Cognitive processing indicator |
| **IPA** | Index of Pupillary Activity | Overall cognitive effort |

---

## 6. Areas of Interest (AOI) Analysis

### 6.1 What are AOIs?

Areas of Interest are researcher-defined regions on a stimulus that are analyzed separately. They allow comparison of attention across different interface elements.

### 6.2 Types of AOIs

**Static AOIs:**
- Fixed regions that don't change
- Defined before experiment
- Suitable for static stimuli

**Dynamic AOIs:**
- Move or change over time
- Track moving elements
- More complex to implement

**Semantic AOIs:**
- Based on content meaning
- May overlap or have irregular shapes
- Examples: navigation menu, content area, advertisements

### 6.3 AOI Design Principles

**Size Considerations:**
- Minimum size: ~1° visual angle (accounts for accuracy limits)
- Add margin around elements (typically 0.5-1°)
- Larger AOIs = more reliable data but less precision

**Shape Options:**
- Rectangles: Simplest, fastest processing
- Ellipses: Better for faces, circular elements
- Polygons: Precise boundaries for irregular shapes
- Freeform: Maximum precision, most labor-intensive

**Best Practices:**
1. Define AOIs based on research questions, not convenience
2. Document AOI boundaries and rationale
3. Consider eye tracker accuracy when sizing
4. Avoid overlapping AOIs when possible
5. Include "white space" or non-AOI regions in analysis

### 6.4 AOI Metrics

**Hit-based:**
- Fixation count in AOI
- First fixation in AOI
- Number of visits/entries

**Time-based:**
- Total dwell time
- Average visit duration
- Time to first fixation

**Sequence-based:**
- Entry order
- Transition probabilities between AOIs
- Scanpath within AOIs

### 6.5 AOI Analysis Considerations

**Challenges:**
- AOI size affects metrics (larger AOIs = more fixations)
- Edge fixations may be miscategorized
- Observer bias in AOI definition

**Solutions:**
- Normalize by AOI size when comparing
- Use multiple coders for AOI definition
- Report AOI dimensions and rationale
- Consider data-driven AOI generation

---

## 7. Scanpath Analysis

### 7.1 What is a Scanpath?

A scanpath is the sequence of fixations and saccades made while viewing a stimulus. It represents the spatial and temporal pattern of visual attention.

### 7.2 Scanpath Representations

**String Representation:**
- Encode AOIs as letters (A, B, C, etc.)
- Sequence becomes string: "ABCBDA"
- Enables string comparison algorithms

**Coordinate Sequence:**
- Raw (x, y) fixation coordinates
- Preserves spatial detail
- Used for geometric analysis

**Vector Representation:**
- Direction and amplitude of saccades
- Captures movement patterns
- Useful for pattern recognition

### 7.3 Scanpath Comparison Methods

| Method | Description | Output |
|--------|-------------|--------|
| **Levenshtein Distance** | Edit distance between AOI strings | Similarity score |
| **Needleman-Wunsch** | Global sequence alignment | Aligned sequences |
| **Smith-Waterman** | Local sequence alignment | Best matching subsequence |
| **ScanMatch** | Combines spatial and temporal info | Similarity matrix |
| **MultiMatch** | Multi-dimensional comparison | 5 dimension scores |

### 7.4 MultiMatch Dimensions

1. **Vector**: Saccade direction similarity
2. **Length**: Saccade amplitude similarity  
3. **Position**: Fixation location similarity
4. **Duration**: Fixation duration similarity
5. **Shape**: Overall path shape similarity

### 7.5 Scanpath Visualization

**Static Visualizations:**
- Gaze plots (fixations as circles, saccades as lines)
- Numbered sequences showing order
- Heat maps (aggregated attention)

**Dynamic Visualizations:**
- Gaze replays (animated playback)
- Attention spotlights
- Cumulative visualizations

### 7.6 Common Scanpath Patterns

| Pattern | Description | Indication |
|---------|-------------|------------|
| **F-Pattern** | Horizontal movements, decreasing depth | Web page scanning |
| **Z-Pattern** | Diagonal scanning | Print-like reading |
| **Exhaustive** | Systematic coverage | Thorough search |
| **Focused** | Concentrated on specific area | Clear target |
| **Random** | No clear pattern | Confusion or exploration |

---

## 8. Cognitive Load and Pupillometry

### 8.1 Pupil Dilation and Cognition

The pupil dilates in response to cognitive effort, independent of lighting conditions. This relationship, known as Task-Evoked Pupillary Response (TEPR), was established by Hess and Polt (1964) and Kahneman (1973).

### 8.2 Factors Affecting Pupil Size

**Cognitive Factors:**
- Mental workload (increases dilation)
- Memory load (increases dilation)
- Decision difficulty (increases dilation)
- Emotional arousal (increases dilation)

**Non-Cognitive Factors (Confounds):**
- Luminance (must control)
- Accommodation (focusing distance)
- Age (pupil range decreases)
- Medications/substances
- Fatigue

### 8.3 Pupillometry Methods

**Baseline Correction:**
```
Corrected Pupil = Raw Pupil - Baseline Pupil
```
or
```
Percent Change = ((Task - Baseline) / Baseline) × 100
```

**Index of Cognitive Activity (ICA):**
- Developed by Marshall (2002)
- Based on rapid, small pupil dilations
- Less affected by luminance
- Proprietary algorithm

**Low/High Index of Pupillary Activity (LHIPA):**
- Analyzes wavelet decomposition
- Separates cognitive from light responses
- More robust measure

### 8.4 Experimental Design for Pupillometry

**Requirements:**
1. Control luminance (constant across conditions)
2. Establish baseline period (typically 1-2 seconds)
3. Account for pupillary light reflex latency (~250 ms)
4. Sample at adequate rate (minimum 60 Hz, preferably higher)

**Analysis Considerations:**
- Filter out blinks and artifacts
- Interpolate missing data carefully
- Consider individual differences in baseline
- Report absolute and relative measures

### 8.5 Interpreting Pupil Data

| Pupil Response | Possible Interpretation |
|----------------|------------------------|
| Sustained dilation | Ongoing cognitive effort |
| Peak then decrease | Task completed |
| No change | Low cognitive demand |
| Large variability | Fluctuating attention/effort |
| Constriction | Reduced arousal or task disengagement |

---

## 9. Theoretical Models and Frameworks

### 9.1 Just and Carpenter's Model of Reading (1980)

**Core Principles:**
- Eye-Mind Assumption: Fixation location = processing location
- Immediacy Assumption: Words processed upon encounter

**Implications:**
- Fixation duration reflects processing difficulty
- Longer fixations on low-frequency, unpredictable words
- Regressions indicate comprehension failure

### 9.2 E-Z Reader Model (Reichle et al., 2003)

**Key Features:**
- Two-stage word identification:
  1. **L1**: Familiarity check (initiates saccade programming)
  2. **L2**: Full lexical access (completes identification)
- Serial attention shift (one word at a time)
- Explains skipping and refixations

**Predictions:**
- Word frequency affects fixation duration
- Predictability affects skipping
- Spillover effects from previous words

### 9.3 SWIFT Model (Engbert et al., 2005)

**Key Features:**
- Parallel processing of multiple words
- Spatially distributed attention gradient
- Random timing for saccade initiation
- Accounts for word skipping through parallel processing

**Differs from E-Z Reader:**
- Parallel vs. serial attention
- Processing gradient vs. spotlight

### 9.4 Saliency Models

**Itti-Koch Model (1998):**
- Bottom-up visual saliency
- Combines color, intensity, orientation features
- Creates saliency map predicting fixation locations

**Components:**
1. Feature extraction at multiple scales
2. Center-surround differences
3. Normalization and combination
4. Winner-take-all selection

**Limitations for HCI:**
- Ignores top-down task demands
- Doesn't account for semantic content
- Better for free-viewing than task-oriented behavior

### 9.5 SEEV Model (Wickens et al., 2003)

**SEEV = Salience, Effort, Expectancy, Value**

Predicts attention allocation based on:
- **Salience**: Bottom-up visual properties
- **Effort**: Cost of eye movements
- **Expectancy**: Probability of relevant information
- **Value**: Importance of information

**Application:**
- Interface design optimization
- Predicting attention distribution
- Workload assessment

### 9.6 Cognitive Load Theory and Eye Tracking

**Types of Cognitive Load (Sweller, 1988):**
1. **Intrinsic**: Inherent task complexity
2. **Extraneous**: Poor design/presentation
3. **Germane**: Schema construction (beneficial)

**Eye Tracking Indicators:**
| Load Type | Eye Movement Indicator |
|-----------|----------------------|
| High intrinsic | Longer fixations, more revisits |
| High extraneous | More saccades, scattered pattern |
| Germane | Systematic scanning, integration saccades |

### 9.7 Multimedia Learning and Eye Tracking

**Mayer's Cognitive Theory of Multimedia Learning (CTML):**
- Dual channels (visual/verbal)
- Limited capacity
- Active processing

**Eye Tracking Insights:**
- Integration patterns between text and images
- Split attention indicators
- Redundancy detection
- Coherence violations

---

## 10. Data Processing and Analysis

### 10.1 Event Detection

**Fixation Detection Algorithms:**

| Algorithm | Description | Parameters |
|-----------|-------------|------------|
| **Velocity-based (I-VT)** | Velocity below threshold = fixation | Threshold (typically 30°/s) |
| **Dispersion-based (I-DT)** | Points within spatial area = fixation | Dispersion threshold, duration |
| **Hidden Markov Model** | Probabilistic state transitions | Training data required |
| **Machine Learning** | Learned classification | Labeled training data |

**I-VT Algorithm:**
```
For each sample:
    Calculate velocity from position change
    If velocity < threshold:
        Mark as fixation
    Else:
        Mark as saccade
```

**I-DT Algorithm:**
```
For sliding window of minimum duration:
    If dispersion (max - min for x and y) < threshold:
        Expand window until dispersion exceeds threshold
        Mark window as fixation
```

### 10.2 Data Cleaning

**Blink Detection and Handling:**
- Pupil loss indicates blink
- Typical blink duration: 100-400 ms
- Remove or interpolate blink periods

**Artifact Removal:**
- Track loss (look away, obstruction)
- Physiologically impossible values
- Equipment noise/glitches

**Interpolation Methods:**
- Linear interpolation (simple, fast)
- Cubic spline (smoother)
- Do not interpolate gaps > 75-100 ms

### 10.3 Data Quality Metrics

| Metric | Acceptable Range | Calculation |
|--------|------------------|-------------|
| **Tracking Ratio** | > 90% | Valid samples / Total samples |
| **Calibration Error** | < 1° | Mean deviation from targets |
| **Precision (RMS)** | < 0.1° | Root mean square of successive samples |
| **Data Loss** | < 10% | Missing or invalid data |

### 10.4 Statistical Analysis Approaches

**For Continuous Metrics:**
- ANOVA (for comparing conditions)
- Mixed-effects models (handles repeated measures)
- t-tests (pairwise comparisons)

**For Count Data:**
- Poisson regression
- Negative binomial (for overdispersion)
- Chi-square tests

**For Temporal Data:**
- Survival analysis (time to first fixation)
- Growth curve models
- Time series analysis

**For Scanpath Data:**
- Sequence analysis
- Markov chain models
- String edit distance comparisons

### 10.5 Mixed-Effects Models

**Why Mixed-Effects?**
- Accounts for participant variability
- Handles missing data
- Appropriate for nested designs

**Basic Model Structure:**
```
Metric ~ Fixed_Effects + (Random_Effects | Participant) + (Random_Effects | Item)
```

**Example in R:**
```r
library(lme4)
model <- lmer(fixation_duration ~ condition * AOI + 
              (1 + condition | participant) + 
              (1 | item), 
              data = eye_data)
```

### 10.6 Visualization Techniques

**Heat Maps:**
- Aggregate fixation data across participants
- Color intensity = attention concentration
- Good for identifying "hot spots"

**Gaze Plots:**
- Individual scanpaths
- Circles = fixations (size = duration)
- Lines = saccades
- Numbers = sequence order

**Bee Swarm Plots:**
- Animated fixation replay
- Show temporal progression
- Multiple participants simultaneously

**Attention Maps:**
- Gaussian blur applied to fixations
- Creates smooth attention landscape
- Useful for comparing conditions

---

## 11. Experimental Design Considerations

### 11.1 Participant Requirements

**Inclusion Criteria:**
- Normal or corrected-to-normal vision
- No strabismus (eye misalignment)
- Ability to follow instructions
- Age-appropriate for study

**Exclusion Considerations:**
- Contact lenses (may affect tracking quality)
- Heavy eye makeup (can interfere with pupil detection)
- Certain medications (affect pupil response)
- Conditions affecting eye movements (e.g., nystagmus)

**Sample Size:**
- Depends on effect size and variability
- Typically 20-50 participants for lab studies
- Power analysis recommended

### 11.2 Environmental Setup

**Lighting:**
- Consistent across sessions
- Avoid direct light on screen or eyes
- Control natural light sources
- Document lighting conditions

**Display Settings:**
- Fixed brightness and contrast
- Known resolution and dimensions
- Consistent viewing distance (typically 60-70 cm)
- Minimize reflections

**Participant Positioning:**
- Use chin rest for high-precision studies
- Mark floor position for consistency
- Ensure comfortable posture
- Allow breaks to prevent fatigue

### 11.3 Calibration Protocol

**Before Each Session:**
1. Adjust equipment for participant
2. Run calibration procedure
3. Validate calibration accuracy
4. Re-calibrate if error > threshold

**During Session:**
- Monitor tracking quality
- Drift correct between blocks if needed
- Re-calibrate if quality degrades

**Documentation:**
- Record calibration quality
- Note any issues or re-calibrations
- Save calibration data

### 11.4 Stimulus Design

**General Principles:**
- Control for low-level visual features when comparing content
- Ensure text is readable (appropriate size, contrast)
- Consider screen real estate
- Account for calibration accuracy in element spacing

**For Reading Studies:**
- Single or double line displays for precise measures
- Controlled word properties (frequency, length, predictability)
- Boundary paradigm for parafoveal studies

**For Interface Studies:**
- Representative of actual use
- Controlled variations between conditions
- Clear AOI boundaries
- Appropriate task instructions

### 11.5 Task Design

**Considerations:**
- Natural vs. artificial tasks (ecological validity trade-off)
- Free viewing vs. directed tasks
- Task difficulty and duration
- Instruction clarity

**Types of Tasks:**
| Task Type | Description | Metrics Emphasized |
|-----------|-------------|-------------------|
| Free viewing | Explore without specific goal | Saliency, interest |
| Visual search | Find specific target | Search efficiency |
| Reading | Comprehend text | Processing time |
| Comparison | Judge between options | Decision process |
| Problem solving | Complete complex task | Strategy, workload |

---

## 12. Software Tools and Platforms

### 12.1 Data Collection Software

**EyeLink Software Suite:**
- **Experiment Builder**: Visual experiment design
- **Data Viewer**: Analysis and visualization
- **EyeLink API**: Custom programming (Python, MATLAB, C)

**Tobii Pro Software:**
- **Tobii Pro Lab**: Complete research platform
- **Tobii Pro SDK**: Development kit
- **Extensions**: MATLAB, Python, E-Prime integration

**Open Source Options:**
- **PyGaze**: Python eye tracking toolbox
- **OpenSesame**: With eye tracking plugins
- **PsychoPy**: With Tobii/EyeLink support

### 12.2 Analysis Software

| Software | Type | Strengths |
|----------|------|-----------|
| **MATLAB** | General | Flexible, extensive toolboxes |
| **R** | Statistical | Mixed models, visualization |
| **Python** | General | Machine learning, automation |
| **OGAMA** | Specialized | Free, complete pipeline |
| **GazePoint Analysis** | Specialized | User-friendly |

### 12.3 Useful R Packages

```r
# Eye tracking specific
library(eyetrackingR)  # Analysis and visualization
library(saccades)       # Saccade detection
library(popEye)         # Reading research

# General analysis
library(lme4)           # Mixed-effects models
library(ggplot2)        # Visualization
library(tidyverse)      # Data manipulation
```

### 12.4 Useful Python Libraries

```python
# Eye tracking specific
import pygaze           # Experiment control
import remodnav         # Eye movement classification

# General analysis
import pandas           # Data manipulation
import numpy            # Numerical computing
import scipy            # Statistical analysis
import matplotlib       # Visualization
import seaborn          # Statistical visualization
```

### 12.5 Online/Cloud Solutions

**For Large-Scale Studies:**
- **Labvanced**: Online experiments with webcam tracking
- **Gorilla**: Experiment platform with eye tracking
- **JATOS**: Just Another Tool for Online Studies

---

## 13. Quality Assurance and Validation

### 13.1 Data Quality Checks

**Pre-Analysis Checks:**
1. Tracking ratio acceptable (>90%)
2. Calibration accuracy documented
3. No systematic data loss
4. Reasonable value ranges

**Automated Flagging:**
- Fixations too short (<50 ms) or too long (>2000 ms)
- Saccades with impossible velocities
- Pupil size outliers
- Excessive track loss periods

### 13.2 Validation Approaches

**Internal Validation:**
- Split-half reliability
- Test-retest reliability
- Inter-rater reliability for AOI coding

**External Validation:**
- Comparison with known effects (e.g., word frequency)
- Correlation with other measures
- Replication of established findings

### 13.3 Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Poor tracking | Glasses, makeup, lighting | Adjust setup, exclude if persistent |
| Drift | Head movement, equipment | Drift correction, chin rest |
| Missing data | Blinks, look-aways | Interpolation, exclude trials |
| Noisy data | Low sampling rate, poor calibration | Filter, re-calibrate |
| Ceiling/floor effects | Task too easy/hard | Adjust difficulty |

### 13.4 Reporting Standards

**Method Section Should Include:**
- Eye tracker model and specifications
- Sampling rate used
- Calibration procedure and acceptance criteria
- Event detection algorithm and parameters
- Data cleaning procedures
- Analysis approach and software

**Results Should Report:**
- Data loss and exclusions
- Tracking quality metrics
- Effect sizes, not just p-values
- Visualizations where appropriate

---

## 14. Applications in HCI Research

### 14.1 Usability Evaluation

**What Eye Tracking Reveals:**
- Where users look vs. where they should look
- Confusing or missed interface elements
- Inefficient navigation patterns
- Reading behavior on interfaces

**Key Metrics:**
- Time to first fixation on target
- Fixation count before finding target
- Revisits to incorrect elements
- Scanpath efficiency

### 14.2 Information Visualization

**Research Questions:**
- How do users read data visualizations?
- What makes visualizations effective?
- How does complexity affect comprehension?

**Relevant to EcoInsight Project:**
- Infographic comprehension patterns
- Text-image integration
- Cognitive load from visual complexity

### 14.3 Reading and Text Processing

**Applications:**
- Font legibility comparison
- Layout optimization
- Reading comprehension assessment
- Language learning research

**Key Phenomena:**
- Word frequency effect
- Predictability effect
- Parafoveal preview benefit
- Regressions and re-reading

### 14.4 Attention and Advertising

**Research Questions:**
- What captures attention?
- How long do users engage?
- What is remembered?

**Metrics:**
- First fixation location
- Dwell time on ads
- Attention to brand vs. content

### 14.5 Accessibility Research

**Applications:**
- Screen reader optimization
- Interface design for low vision
- Cognitive accessibility
- Age-related differences

### 14.6 Gaming and Entertainment

**Research Areas:**
- Player attention patterns
- Tutorial effectiveness
- Difficulty assessment
- Immersion indicators

### 14.7 Educational Technology

**Directly Relevant to HCI+L:**
- Learning from multimedia
- Adaptive content delivery
- Attention in online learning
- Individual differences in learning

---

## 15. Visual Attention Models

### 15.1 Overview of Visual Attention Modeling

Visual attention models aim to predict where humans look in images, videos, or interfaces. These models are crucial for HCI research as they help understand user behavior, evaluate designs, and create adaptive systems.

**Two Main Categories:**
1. **Bottom-up (stimulus-driven)**: Based on low-level visual features
2. **Top-down (goal-driven)**: Based on task demands, knowledge, and intentions

### 15.2 Classical Bottom-Up Models

#### Feature Integration Theory (FIT) - Treisman & Gelade (1980)

**Core Concept:** Visual attention operates in two stages:
1. **Pre-attentive stage**: Parallel processing of basic features (color, orientation, size)
2. **Attentive stage**: Serial binding of features into objects

**Key Predictions:**
- Feature search is parallel (pop-out effect)
- Conjunction search is serial (requires attention)
- Illusory conjunctions occur without attention

**Relevance to HCI:** Explains why certain UI elements "pop out" and are easily detected.

#### Itti-Koch-Niebur Model (1998)

**Architecture:**
```
Input Image
    ↓
Feature Extraction (Color, Intensity, Orientation)
    ↓
Center-Surround Differences (Multiple Scales)
    ↓
Normalization
    ↓
Feature Maps Combination
    ↓
Saliency Map
    ↓
Winner-Take-All + Inhibition of Return
    ↓
Attention Shifts
```

**Features Computed:**
| Feature | Channels | Description |
|---------|----------|-------------|
| **Intensity** | 1 | Luminance contrast |
| **Color** | 2 | Red-green, blue-yellow opponency |
| **Orientation** | 4 | 0°, 45°, 90°, 135° Gabor filters |

**Key Mechanisms:**
- **Center-surround**: Detects local contrast at multiple scales
- **Normalization**: Promotes sparse, unique features
- **Winner-take-all (WTA)**: Selects most salient location
- **Inhibition of return (IOR)**: Prevents re-attending same location

**Strengths:** Biologically plausible, interpretable, fast
**Limitations:** Ignores semantics, top-down factors, context

#### Graph-Based Visual Saliency (GBVS) - Harel et al. (2006)

**Improvements over Itti-Koch:**
- Uses Markov chains on graph of image locations
- Equilibrium distribution defines saliency
- Better captures "spread" of salient regions

**Algorithm:**
1. Compute feature maps (similar to Itti-Koch)
2. Form fully-connected graph over locations
3. Edge weights based on feature dissimilarity
4. Compute equilibrium distribution via random walk
5. Normalize and combine across features

**Performance:** Consistently outperforms Itti-Koch on benchmarks

#### Attention based on Information Maximization (AIM) - Bruce & Tsotsos (2006)

**Core Idea:** Salient locations are those that maximize information (Shannon entropy)

**Principle:** 
- Learn statistics of natural images using ICA
- Salient = surprising/unexpected given learned statistics
- Self-information: I(x) = -log P(x)

**Strengths:** Principled information-theoretic foundation
**Limitations:** Computationally expensive, sensitive to image statistics

#### Saliency Using Natural Statistics (SUN) - Zhang et al. (2008)

**Bayesian Framework:**
```
P(saliency | features) ∝ P(features | saliency) × P(saliency)
```

**Key Innovation:** 
- Models bottom-up saliency as self-information
- Incorporates top-down priors (e.g., target features)
- Unifies bottom-up and top-down in single framework

### 15.3 Top-Down and Combined Models

#### Guided Search - Wolfe (1994, 2007)

**Core Concept:** Visual search guided by both bottom-up salience and top-down target features

**Activation Map:**
```
Activation = Σ(wi × bottom-up_i) + Σ(wj × top-down_j)
```

**Top-Down Guidance:**
- Feature-based: Attend to target color, size, orientation
- Categorical: Attend to semantic categories
- Scene-based: Use scene context to guide search

**Relevance to HCI:** Explains how users search for specific interface elements

#### SEEV Model - Wickens et al. (2003)

**SEEV = Salience, Effort, Expectancy, Value**

**Attention Allocation:**
```
P(attend to AOI) = f(Salience - Effort + Expectancy × Value)
```

| Factor | Description | Example in HCI |
|--------|-------------|----------------|
| **Salience** | Visual prominence | Bright button, animation |
| **Effort** | Cost of eye movement | Distance from current fixation |
| **Expectancy** | Probability of relevant info | Frequently updated area |
| **Value** | Importance of information | Critical system status |

**Applications:**
- Aviation cockpit design
- Dashboard layout optimization
- Interface attention prediction

#### Contextual Guidance Model - Torralba et al. (2006)

**Innovation:** Incorporates global scene context

**Components:**
1. **Local saliency**: Bottom-up feature contrast
2. **Global context**: Scene gist guides search
3. **Top-down task**: Target template matching

**Equation:**
```
P(location | target, image) ∝ P(local features | target) × P(location | global context)
```

**Key Finding:** Context dramatically improves search prediction (e.g., people on sidewalks, not sky)

### 15.4 Deep Learning Saliency Models

#### DeepGaze I - Kümmerer et al. (2015)

**Architecture:**
- Pre-trained AlexNet features
- Simple readout network on top
- No fine-tuning of CNN layers

**Key Insight:** Features learned for object recognition transfer well to saliency prediction

#### DeepGaze II - Kümmerer et al. (2017)

**Improvements:**
- VGG-19 features (deeper network)
- Multi-scale feature integration
- Center bias prior

**Performance:** State-of-the-art on MIT300 benchmark for several years

#### DeepGaze III - Kümmerer et al. (2022)

**Innovations:**
- Probabilistic predictions (uncertainty estimation)
- Better handling of individual differences
- Improved scanpath prediction

#### SalGAN - Pan et al. (2017)

**Architecture:**
```
Generator (Encoder-Decoder)
    ↓
Saliency Map Prediction
    ↓
Discriminator
    ↓
Adversarial Loss + Content Loss
```

**Key Innovation:** Adversarial training produces sharper, more realistic saliency maps

#### SAM (Saliency Attentive Model) - Cornia et al. (2018)

**Components:**
- Dilated ResNet encoder
- Attentive ConvLSTM for sequential refinement
- Learned prior (center bias)

**Key Feature:** Iteratively refines predictions using attention mechanism

#### EML-NET - Jia & Bruce (2020)

**Multi-Level Approach:**
- Extracts features at multiple CNN layers
- Combines low-level and high-level information
- Efficient encoder-decoder architecture

#### TranSalNet - Lou et al. (2022)

**Transformer-Based:**
- Vision Transformer (ViT) backbone
- Self-attention captures global dependencies
- Competitive with CNN-based methods

#### SimpleNet - Reddy et al. (2020)

**Philosophy:** Simple architectures can be effective

**Architecture:**
- Standard encoder-decoder
- Skip connections
- Minimal complexity

**Benefit:** Fast inference, easy to deploy

### 15.5 Video Saliency Models

#### Dynamic Saliency Models

| Model | Approach | Key Feature |
|-------|----------|-------------|
| **SUSiNet** | Two-stream CNN | Separate appearance/motion |
| **ACLNet** | Attention + ConvLSTM | Temporal attention |
| **TASED-Net** | 3D convolutions | Spatiotemporal features |
| **ViNet** | Video transformer | Long-range temporal |
| **STSANet** | Spatial-temporal self-attention | Efficient attention |

**Additional Features for Video:**
- **Optical flow**: Motion direction and magnitude
- **Temporal contrast**: Changes over time
- **Flicker**: Rapid luminance changes
- **Looming**: Expanding objects (approach detection)

### 15.6 Task-Specific Attention Models

#### Visual Question Answering (VQA) Attention

**Models:** 
- Stacked Attention Networks (SAN)
- Bottom-Up Top-Down Attention
- MCAN (Multi-modal Co-Attention)

**Mechanism:** Question guides visual attention to relevant image regions

#### Reading Attention Models

| Model | Focus | Application |
|-------|-------|-------------|
| **E-Z Reader** | Serial word processing | Reading simulation |
| **SWIFT** | Parallel processing | Eye movement prediction |
| **OB1-Reader** | Open bigram coding | Word recognition |
| **Über-Reader** | Comprehensive | Multiple phenomena |

#### Interface-Specific Models

**UIBert / Screen Recognition:**
- Trained on UI screenshots
- Understands UI semantics
- Predicts interaction targets

**Pix2Struct:**
- Converts screenshots to structured representations
- Understands visual-textual UI elements

### 15.7 Attention in Transformers

#### Self-Attention Mechanism

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Components:**
- **Q (Query)**: What am I looking for?
- **K (Key)**: What do I contain?
- **V (Value)**: What information do I provide?

#### Vision Transformer (ViT) Attention

**Attention Patterns:**
- Early layers: Local, texture-like attention
- Later layers: Global, semantic attention
- [CLS] token: Aggregates global information

**Visualization Methods:**
- Attention rollout
- Attention flow
- Gradient-weighted attention

#### CLIP Attention for Saliency

**Zero-Shot Saliency:**
- Use text prompts to guide attention
- "A photo of a salient object"
- Cross-modal attention reveals relevant regions

### 15.8 Scanpath Prediction Models

#### Probabilistic Models

| Model | Approach | Output |
|-------|----------|--------|
| **SceneWalk** | Activation + IOR | Scanpath sequence |
| **CLE** | Conditional likelihood | Fixation probability |
| **IHMM** | Infinite HMM | State transitions |

#### Deep Learning Scanpath Models

| Model | Architecture | Innovation |
|-------|--------------|------------|
| **PathGAN** | GAN | Adversarial scanpath generation |
| **IRL-Gaze** | Inverse RL | Learns reward function |
| **ScanpathNet** | Seq2Seq | Encoder-decoder for sequences |
| **Gazeformer** | Transformer | Attention-based prediction |
| **HAT** | Hierarchical transformer | Multi-scale prediction |

#### Scanpath Comparison Metrics

| Metric | What it Measures |
|--------|------------------|
| **String Edit Distance** | Sequence similarity (AOI-based) |
| **ScanMatch** | Spatial-temporal alignment |
| **MultiMatch** | 5 dimensions (shape, direction, length, position, duration) |
| **RecurrenceQuantification** | Determinism, recurrence patterns |

### 15.9 Evaluation Metrics for Attention Models

#### Location-Based Metrics

| Metric | Formula/Description | Range |
|--------|---------------------|-------|
| **AUC-Judd** | ROC using fixations as positives | 0-1 |
| **AUC-Borji** | Shuffled fixations as negatives | 0-1 |
| **sAUC** | Shuffled AUC (center-bias corrected) | 0-1 |
| **NSS** | Mean saliency at fixation locations | -∞ to +∞ |

#### Distribution-Based Metrics

| Metric | Formula/Description | Range |
|--------|---------------------|-------|
| **CC** | Pearson correlation | -1 to 1 |
| **SIM** | Histogram intersection | 0-1 |
| **KL Divergence** | KL(Human || Model) | 0 to +∞ |
| **EMD** | Earth mover's distance | 0 to +∞ |

#### Information-Theoretic Metrics

| Metric | Description |
|--------|-------------|
| **IG** | Information gain over baseline |
| **LL** | Log-likelihood of fixations |
| **AUC-shuffled** | AUC with other-image fixations |

### 15.10 Implementing Attention Models

#### Using Pre-trained Models (Python)

```python
# DeepGaze II example
import torch
from deepgaze_pytorch import DeepGazeIIE

model = DeepGazeIIE(pretrained=True)
model.eval()

# Predict saliency
with torch.no_grad():
    saliency = model(image, centerbias)
```

#### Simple Itti-Koch Implementation

```python
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def itti_koch_saliency(image, scales=[1, 2, 3, 4]):
    """Simplified Itti-Koch saliency computation"""
    
    # Convert to different color spaces
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    saliency_maps = []
    
    for channel in [l, a, b]:
        channel_saliency = np.zeros_like(channel, dtype=float)
        
        for scale in scales:
            # Create center and surround
            sigma_c = scale
            sigma_s = scale * 2
            
            center = gaussian_filter(channel.astype(float), sigma_c)
            surround = gaussian_filter(channel.astype(float), sigma_s)
            
            # Center-surround difference
            cs_diff = np.abs(center - surround)
            channel_saliency += cs_diff
        
        saliency_maps.append(channel_saliency)
    
    # Combine channels
    combined = np.mean(saliency_maps, axis=0)
    
    # Normalize
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    
    return combined
```

### 15.11 Applications in HCI Research

#### Usability Evaluation

**Use Cases:**
- Predict attention before user testing
- Identify overlooked UI elements
- Compare design alternatives
- Validate visual hierarchy

**Example Workflow:**
1. Generate saliency map for interface design
2. Overlay with intended attention areas
3. Identify mismatches (salient but unimportant areas)
4. Iterate on design

#### Adaptive Interfaces

**Gaze-Contingent Displays:**
- Foveated rendering (high detail at gaze point)
- Attention-aware information presentation
- Dynamic content adaptation

**Relevance to EcoInsight:**
- Predict where users will look on infographics
- Adapt content based on attention patterns
- Personalize based on individual differences

#### Accessibility

**Applications:**
- Predict visibility issues for low vision users
- Optimize contrast for attention guidance
- Design for diverse visual abilities

### 15.12 Comparison of Major Models

| Model | Type | Strengths | Weaknesses | Best For |
|-------|------|-----------|------------|----------|
| **Itti-Koch** | Classical | Fast, interpretable | No semantics | Real-time |
| **GBVS** | Classical | Better spread | Still bottom-up | Quick analysis |
| **DeepGaze II** | Deep learning | High accuracy | Black box | Benchmarks |
| **SalGAN** | Deep learning | Sharp maps | Training data needed | Research |
| **SEEV** | Cognitive | Top-down factors | Requires AOI weights | Applied HCI |
| **Guided Search** | Cognitive | Task-specific | Complex setup | Search studies |

### 15.13 Resources and Tools

#### Pre-trained Models

| Resource | Models Available | Link |
|----------|------------------|------|
| **MIT Saliency** | Multiple classical | saliency.mit.edu |
| **DeepGaze PyTorch** | DeepGaze I, II, III | github.com/matthias-k/DeepGaze |
| **OpenSALICON** | SALICON models | github.com/CLT29/OpenSALICON |
| **TorchSaliency** | Multiple models | github.com |

#### Evaluation Tools

| Tool | Purpose |
|------|---------|
| **MIT/Tuebingen Saliency Benchmark** | Model evaluation |
| **SMAC** | Saliency Model Analysis |
| **pysaliency** | Python evaluation tools |

---

## 16. Noteworthy Papers and Landmark Studies

### 16.1 Foundational Papers in Eye Tracking

| Paper | Authors (Year) | Key Contribution |
|-------|---------------|------------------|
| **A Theory of Reading** | Just & Carpenter (1980) | Established Eye-Mind and Immediacy hypotheses |
| **Eye Movements in Reading** | Rayner (1998) | Comprehensive 20-year review of reading research |
| **The E-Z Reader Model** | Reichle et al. (2003) | Serial attention shift model for reading |
| **SWIFT Model** | Engbert et al. (2005) | Parallel processing model for reading |
| **A Model of Saliency-Based Visual Attention** | Itti, Koch & Niebur (1998) | Computational saliency model |
| **Graph-Based Visual Saliency** | Harel, Koch & Perona (2006) | GBVS algorithm for saliency prediction |

### 16.2 Eye Tracking in HCI - Key Papers

**Usability and Interface Design:**

| Paper | Citation | Summary |
|-------|----------|---------|
| **Eye Tracking in HCI and Usability Research** | Jacob & Karn (2003) | Foundational overview of eye tracking applications in HCI |
| **What Can You Do with Eye Movements?** | Goldberg & Wichansky (2003) | Practical guidelines for usability practitioners |
| **Eye Tracking the User Experience** | Bergstrom & Schall (2014) | Comprehensive book on UX eye tracking methods |
| **How People Look at Pictures** | Buswell (1935) | One of the earliest systematic eye tracking studies |

**Attention and Visual Search:**

| Paper | Citation | Summary |
|-------|----------|---------|
| **Guided Search 2.0** | Wolfe (1994) | Influential model of visual search |
| **A Feature-Integration Theory of Attention** | Treisman & Gelade (1980) | Foundation for understanding visual attention |
| **Eye Movements During Visual Search** | Zelinsky (2008) | Review of search behavior |
| **Optimal Eye Movement Strategies in Visual Search** | Najemnik & Geisler (2005) | Ideal observer analysis of search |

### 16.3 Multimedia Learning and Eye Tracking

| Paper | Authors (Year) | Key Finding |
|-------|---------------|-------------|
| **Eye Tracking as a Tool to Study and Enhance Multimedia Learning** | Van Gog & Scheiter (2010) | Comprehensive review linking eye tracking to multimedia learning theory |
| **Attention Guidance in Multimedia Learning** | De Koning et al. (2010) | How visual cues guide attention in animations |
| **Effects of Visual Cues on Learning** | Jarodzka et al. (2013) | Expert gaze displays improve novice learning |
| **Integrating Information from Text and Pictures** | Scheiter & Eitel (2015) | Text-picture integration processes |
| **Eye Movements and Cognitive Load** | Zu et al. (2020) | Meta-analysis of cognitive load indicators |

### 16.4 Pupillometry Landmark Papers

| Paper | Authors (Year) | Contribution |
|-------|---------------|--------------|
| **Pupil Size in Relation to Mental Activity** | Hess & Polt (1964) | Established pupil-cognition link |
| **Attention and Effort** | Kahneman (1973) | Book establishing pupillometry for cognitive load |
| **The Index of Cognitive Activity** | Marshall (2002) | ICA algorithm development |
| **Pupillometry: A Window to the Preconscious?** | Laeng et al. (2012) | Comprehensive review |
| **Cognitive Pupillometry** | Beatty & Lucero-Wagoner (2000) | Methodological guidelines |

### 16.5 Deep Learning and Eye Tracking

| Paper | Authors (Year) | Innovation |
|-------|---------------|------------|
| **DeepGaze I** | Kümmerer et al. (2015) | Deep learning for saliency prediction |
| **DeepGaze II** | Kümmerer et al. (2017) | Improved deep saliency model |
| **SalGAN** | Pan et al. (2017) | GAN-based saliency prediction |
| **SAM-ResNet** | Cornia et al. (2018) | Attentive convolutional LSTM for saliency |
| **EML-NET** | Jia & Bruce (2020) | Multi-level feature aggregation |
| **Learning to Predict Gaze** | Kummerer et al. (2022) | State-of-the-art deep gaze models |

### 16.6 Reading Research Milestones

| Paper | Focus | Key Insight |
|-------|-------|-------------|
| **Rayner (2009)** | 35 years of research | Comprehensive update on eye movement research |
| **Kliegl et al. (2004)** | Individual differences | Mixed-effects modeling for reading |
| **Inhoff & Radach (1998)** | Parafoveal processing | Definition of preview benefit |
| **McConkie & Rayner (1975)** | Perceptual span | Moving window paradigm |
| **Staub & Rayner (2007)** | Eye movements in reading | Methodological review |

---

## 17. Important Datasets

### 17.1 Eye Tracking Datasets for Saliency and Attention

| Dataset | Description | Size | Access |
|---------|-------------|------|--------|
| **MIT Saliency Benchmark** | Natural images with fixations | 300 images, 39 observers | saliency.mit.edu |
| **SALICON** | Saliency in Context - mouse tracking as proxy | 20,000 images | salicon.net |
| **CAT2000** | 2000 images across 20 categories | 2000 images, 24 observers | saliency.mit.edu |
| **OSIE** | Objects and Semantic Images | 700 images with object/semantic labels | github |
| **FIGRIM** | Fine-grained image memorability | 2222 images | figrim.mit.edu |
| **TORONTO** | Classic saliency dataset | 120 color images | bruce-lab.ca |
| **NUSEF** | Emotional images | 758 images | github |

### 17.2 Eye Tracking Datasets for Reading

| Dataset | Description | Language | Access |
|---------|-------------|----------|--------|
| **Dundee Corpus** | Newspaper reading | English | Request from authors |
| **GECO** | Ghent Eye-tracking Corpus | English/Dutch | github |
| **Provo Corpus** | Predictability norms with eye tracking | English | osf.io |
| **ZuCo** | EEG + Eye tracking during reading | English | osf.io |
| **MECO** | Multilingual Eye-tracking Corpus | 13 languages | github |
| **Beijing Sentence Corpus** | Chinese reading | Chinese | Request from authors |
| **Potsdam Sentence Corpus** | German sentence reading | German | uni-potsdam.de |
| **Russian Sentence Corpus** | Russian reading | Russian | Request from authors |

### 17.3 Eye Tracking Datasets for UI/UX and Web

| Dataset | Description | Size | Access |
|---------|-------------|------|--------|
| **FiWI** | Fixations in Webpage Images | 149 webpages | github |
| **UMASS WebTrack** | Web page viewing | 361 pages | Request from authors |
| **GazeBase** | Biometric eye movements | 12,334 recordings | figshare |
| **WebGaze** | Webcam gaze dataset | 2.4M images | webgazer.cs.brown.edu |
| **UI Salience** | Mobile UI saliency | 2,800 screenshots | github |
| **UEYES** | User interface eye tracking | 1,980 images | github |
| **Rico** | Mobile app UI (with attention) | 72,219 screenshots | interactionmining.org |

### 17.4 Gaze Estimation Datasets

| Dataset | Description | Participants | Access |
|---------|-------------|--------------|--------|
| **GazeCapture** | Large-scale gaze from mobile devices | 1,450 | gazecapture.csail.mit.edu |
| **MPIIGaze** | Appearance-based gaze estimation | 15 | mpi-inf.mpg.de |
| **ETH-XGaze** | Extreme head pose gaze | 110 | ait.ethz.ch |
| **RT-GENE** | Gaze with head pose | 15 | github |
| **OpenEDS** | Eye region segmentation | 152 | research.facebook.com |
| **NVGaze** | Gaze in VR environments | 35 | research.nvidia.com |
| **Columbia Gaze** | Lab-collected gaze | 56 | cs.columbia.edu |

### 17.5 Specialized Datasets

**Scene Viewing:**
| Dataset | Focus | Details |
|---------|-------|---------|
| **MIT300** | Benchmark for saliency | 300 held-out images |
| **PASCAL-S** | Semantic segmentation + saliency | 850 images |
| **ECSSD** | Extended complex scene saliency | 1000 images |
| **DUT-OMRON** | Complex backgrounds | 5,168 images |

**Video and Dynamic Stimuli:**
| Dataset | Focus | Details |
|---------|-------|---------|
| **DHF1K** | Video saliency | 1,000 videos |
| **Hollywood-2** | Movie clips | 1,707 clips |
| **UCF Sports** | Action recognition | 150 videos |
| **DIEM** | Dynamic images and eye movements | 85 videos |
| **LEDOV** | Large-scale eye tracking video | 538 videos |

**Driving:**
| Dataset | Focus | Details |
|---------|-------|---------|
| **DR(eye)VE** | Driving attention | 74 videos |
| **BDD-A** | Berkeley driving attention | 1,232 videos |
| **MAAD** | Multimodal anomaly driving | 8+ hours |

---

## 18. Noteworthy Approaches and Methods

### 18.1 Saliency Prediction Approaches

**Classical Approaches:**
| Approach | Key Idea | Paper |
|----------|----------|-------|
| **Itti-Koch** | Center-surround feature competition | Itti et al. (1998) |
| **GBVS** | Graph-based visual saliency | Harel et al. (2006) |
| **AIM** | Attention based on information maximization | Bruce & Tsotsos (2006) |
| **SUN** | Saliency using natural statistics | Zhang et al. (2008) |
| **AWS** | Adaptive whitening saliency | Garcia-Diaz et al. (2012) |
| **Boolean Maps** | Boolean map based saliency | Zhang & Sclaroff (2013) |

**Deep Learning Approaches:**
| Model | Architecture | Innovation |
|-------|--------------|------------|
| **DeepGaze I/II/III** | VGG + readout | Transfer from ImageNet features |
| **SalGAN** | GAN-based | Adversarial training for saliency |
| **SAM** | Attentive ConvLSTM | Learned attention mechanisms |
| **SALICON Net** | Fine-tuned VGG | Large-scale mouse tracking training |
| **TranSalNet** | Transformer | Self-attention for saliency |
| **SimpleNet** | Efficient CNN | Lightweight saliency prediction |

### 18.2 Scanpath Prediction Approaches

| Approach | Method | Paper |
|----------|--------|-------|
| **SceneWalk** | Probabilistic + inhibition of return | Engbert et al. (2015) |
| **IRL-Gaze** | Inverse reinforcement learning | Chen et al. (2020) |
| **PathGAN** | GAN-based scanpath | Assens et al. (2018) |
| **ScanPath-VQA** | Visual question answering | Chen et al. (2021) |
| **Gazeformer** | Transformer-based | Mondal et al. (2023) |
| **HAT** | Human attention transformer | Yang et al. (2022) |

### 18.3 Gaze Estimation Approaches

**Model-Based:**
| Method | Principle | Accuracy |
|--------|-----------|----------|
| **PCCR** | Pupil center-corneal reflection | < 1° (with hardware) |
| **3D Eye Model** | Geometric eye modeling | ~1-2° |
| **Polynomial Mapping** | Regression from features | ~2-3° |

**Appearance-Based (Deep Learning):**
| Method | Architecture | Performance |
|--------|--------------|-------------|
| **iTracker** | Multi-branch CNN | 2.5 cm (mobile) |
| **GazeNet** | VGG-based | 4.8° |
| **RT-GENE** | VGG + ResNet | 7.7° |
| **ETH-XGaze baseline** | ResNet-50 | 4.5° |
| **Full-Face Gaze** | Full face input | 4.2° |
| **Dilated-Net** | Dilated convolutions | 3.2° |
| **FAZE** | Few-shot adaptation | 3.1° |
| **GazeTR** | Transformer | 4.0° |

### 18.4 Fixation Detection Algorithms

| Algorithm | Type | Key Parameters |
|-----------|------|----------------|
| **I-VT** | Velocity threshold | Velocity threshold (30°/s typical) |
| **I-DT** | Dispersion threshold | Dispersion, minimum duration |
| **I-HMM** | Hidden Markov Model | State probabilities |
| **I-KF** | Kalman filter | State space model |
| **Engbert-Kliegl** | Adaptive velocity | Adaptive threshold based on noise |
| **NH2010** | Machine learning | Trained classifier |
| **REMoDNaV** | Robust eye movement detection | Multiple movement types |

### 18.5 AOI Analysis Approaches

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Grid-based** | Divide stimulus into regular grid | Exploratory analysis |
| **Semantic** | Based on content meaning | Interface evaluation |
| **Data-driven** | Clustering fixations | Discovery of attention patterns |
| **Voronoi tessellation** | Partitions based on elements | Balanced AOI coverage |
| **Dynamic AOI** | Time-varying regions | Video, interactive content |
| **Probabilistic AOI** | Soft boundaries | Uncertainty handling |

### 18.6 Sequence Analysis Methods

| Method | Description | Output |
|--------|-------------|--------|
| **Levenshtein distance** | String edit distance | Single similarity score |
| **Needleman-Wunsch** | Global alignment | Aligned sequences |
| **Smith-Waterman** | Local alignment | Best matching subsequence |
| **ScanMatch** | Spatial-temporal comparison | Similarity matrix |
| **MultiMatch** | Multi-dimensional | 5 dimension scores |
| **SubsMatch** | Common subsequences | Shared patterns |
| **eMine** | Sequential pattern mining | Frequent patterns |
| **T-pattern** | Temporal patterns | Recurring sequences |

### 18.7 Statistical Approaches for Eye Tracking Data

| Approach | Use Case | Reference |
|----------|----------|-----------|
| **Linear Mixed-Effects Models** | Repeated measures, nested data | Baayen et al. (2008) |
| **Generalized Additive Models** | Non-linear relationships | Wood (2017) |
| **Survival Analysis** | Time-to-event (first fixation) | Singer & Willett (2003) |
| **Growth Curve Analysis** | Temporal dynamics | Mirman (2014) |
| **Bayesian Models** | Uncertainty quantification | Nicenboim & Vasishth (2016) |
| **Permutation Tests** | Non-parametric comparison | Maris & Oostenveld (2007) |

---

## 19. Notable Projects and Research Initiatives

### 19.1 Major Research Projects

**Eye Tracking for Everyone (MIT/Google):**
- Large-scale gaze dataset collection
- GazeCapture dataset (1.5M images)
- Enabled mobile gaze estimation research
- Paper: Krafka et al. (2016)

**COGAIN (Communication by Gaze Interaction):**
- EU-funded network of excellence
- Focus on assistive technology
- Standards for eye tracking research
- Website: cogain.org

**OpenEyes Project:**
- Open-source eye tracking
- Low-cost hardware solutions
- Community-driven development
- GitHub: opengazer

**Pupil Labs:**
- Open-source eye tracking glasses
- Affordable research-grade hardware
- Active developer community
- Website: pupil-labs.com

### 19.2 Benchmark Initiatives

**MIT Saliency Benchmark:**
- Standard evaluation for saliency models
- Multiple metrics (AUC, NSS, CC, SIM, KL)
- Leaderboard for model comparison
- Website: saliency.mit.edu

**LSUN Challenge:**
- Large-scale understanding
- Includes saliency prediction track
- Deep learning benchmark

**SALICON Challenge:**
- Saliency in context
- Large-scale evaluation
- Mouse tracking as gaze proxy

### 19.3 Open-Source Software Projects

| Project | Purpose | Language/Platform |
|---------|---------|-------------------|
| **PyGaze** | Experiment programming | Python |
| **WebGazer.js** | Webcam eye tracking | JavaScript |
| **OpenGaze** | Gaze estimation | Python/C++ |
| **GazeML** | ML-based gaze | TensorFlow |
| **Pupil** | Complete eye tracking | Python |
| **OGAMA** | Analysis platform | C#/.NET |
| **PyTrack** | Analysis library | Python |
| **I2MC** | Fixation detection | MATLAB/Python |
| **REMoDNaV** | Movement classification | Python |
| **gazehound** | Analysis tools | Python |
| **eyekit** | Reading analysis | Python |
| **popEye** | Reading research | R |

### 19.4 Industry Research Labs

| Organization | Focus Areas | Notable Contributions |
|--------------|-------------|----------------------|
| **Tobii** | Hardware, UX research | Pro Lab, consumer tracking |
| **SR Research** | High-precision tracking | EyeLink systems |
| **Google AI** | Gaze prediction, accessibility | GazeCapture, Screen Recognition |
| **Meta/Facebook** | VR/AR gaze | Eye tracking in Quest headsets |
| **Apple** | Privacy-preserving gaze | On-device ML for accessibility |
| **Microsoft** | HCI, accessibility | Gaze interaction research |
| **NVIDIA** | Foveated rendering | Gaze-contingent graphics |

### 19.5 Academic Research Centers

| Center | Institution | Focus |
|--------|-------------|-------|
| **MIT AgeLab** | MIT | Driving, aging |
| **Stanford Vision Lab** | Stanford | Attention, perception |
| **Computational Cognition Lab** | UC Berkeley | Cognitive modeling |
| **Eye Tracking Lab** | UCSD | Reading, comprehension |
| **Visual Cognition Lab** | Cambridge | Attention, memory |
| **Potsdam Eye Movement Lab** | U Potsdam | Reading research |
| **Collaboration Technology Lab** | Northwestern | HCI, collaboration |
| **Graphics Lab** | ETH Zurich | Gaze estimation, VR |

### 19.6 Relevant Conferences and Venues

**Primary Venues:**
| Conference | Focus | Frequency |
|------------|-------|-----------|
| **ETRA** | Eye Tracking Research & Applications | Annual |
| **CHI** | Human Factors in Computing | Annual |
| **UIST** | User Interface Software & Technology | Annual |
| **CSCW** | Computer-Supported Cooperative Work | Annual |

**Related Venues:**
| Conference | Relevance |
|------------|-----------|
| **VSS** | Vision Sciences Society - Perception research |
| **CVPR/ICCV/ECCV** | Computer vision - Gaze estimation, saliency |
| **NeurIPS/ICML** | Machine learning - Deep learning approaches |
| **CogSci** | Cognitive Science - Cognitive modeling |
| **ICLS/CSCL** | Learning Sciences - Educational applications |
| **APS** | Psychological Science - Cognitive psychology |

### 19.7 Journals Publishing Eye Tracking Research

| Journal | Focus | Impact |
|---------|-------|--------|
| **Journal of Eye Movement Research** | Dedicated eye tracking | Specialized |
| **Behavior Research Methods** | Methodology | High |
| **Attention, Perception, & Psychophysics** | Perception | High |
| **Vision Research** | Vision science | High |
| **ACM TOCHI** | Human-Computer Interaction | High |
| **IJHCS** | Human-Computer Studies | High |
| **Computers in Human Behavior** | Technology + behavior | High |
| **Journal of Vision** | Vision research | High |
| **Reading and Writing** | Reading research | Specialized |
| **Learning and Instruction** | Educational | High |

---

## 20. Emerging Trends and Future Directions

### 20.1 Current Research Frontiers

**AI and Deep Learning:**
- End-to-end gaze estimation from raw images
- Transformer architectures for scanpath prediction
- Self-supervised learning reducing labeled data needs
- Real-time mobile gaze estimation

**Extended Reality (XR):**
- Eye tracking in VR/AR headsets
- Foveated rendering optimization
- Social presence and avatar eye contact
- Gaze-based interaction in mixed reality

**Multimodal Integration:**
- EEG + eye tracking (joint cognitive measures)
- Facial expression + gaze
- Physiological signals (GSR, HR) + eye tracking
- Audio attention + visual attention

**Privacy-Preserving Approaches:**
- On-device processing
- Federated learning for gaze models
- Differential privacy in gaze data
- Iris anonymization

### 20.2 Technological Developments

**Hardware Advances:**
- Integrated eye tracking in consumer devices
- Lower-cost research-grade systems
- Improved webcam-based tracking
- Miniaturized sensors for wearables

**Software and Methods:**
- Automated analysis pipelines
- Cloud-based processing
- Standardized data formats
- Open science practices

### 20.3 Application Domains Expanding

| Domain | Application |
|--------|-------------|
| **Healthcare** | Autism screening, neurological assessment |
| **Education** | Adaptive learning, reading intervention |
| **Marketing** | Neuromarketing, shelf attention |
| **Gaming** | Player experience, accessibility |
| **Automotive** | Driver monitoring, ADAS |
| **Aviation** | Pilot training, cockpit design |
| **Accessibility** | Gaze-based input, assistive tech |
| **Security** | Biometric authentication |

---

## 21. References and Further Reading

### 15.1 Foundational Papers

- Just, M. A., & Carpenter, P. A. (1980). A theory of reading: From eye fixations to comprehension. *Psychological Review, 87*(4), 329-354.

- Rayner, K. (1998). Eye movements in reading and information processing: 20 years of research. *Psychological Bulletin, 124*(3), 372-422.

- Rayner, K. (2009). Eye movements and attention in reading, scene perception, and visual search. *Quarterly Journal of Experimental Psychology, 62*(8), 1457-1506.

- Holmqvist, K., et al. (2011). *Eye Tracking: A Comprehensive Guide to Methods and Measures*. Oxford University Press.

### 15.2 Cognitive Models

- Reichle, E. D., Rayner, K., & Pollatsek, A. (2003). The E-Z Reader model of eye-movement control in reading: Comparisons to other models. *Behavioral and Brain Sciences, 26*(4), 445-476.

- Engbert, R., Nuthmann, A., Richter, E. M., & Kliegl, R. (2005). SWIFT: A dynamical model of saccade generation during reading. *Psychological Review, 112*(4), 777-813.

- Itti, L., & Koch, C. (2001). Computational modelling of visual attention. *Nature Reviews Neuroscience, 2*(3), 194-203.

### 15.3 Pupillometry

- Kahneman, D. (1973). *Attention and Effort*. Prentice-Hall.

- Beatty, J., & Lucero-Wagoner, B. (2000). The pupillary system. In *Handbook of Psychophysiology* (pp. 142-162).

### 15.4 HCI and Eye Tracking

- Jacob, R. J., & Karn, K. S. (2003). Eye tracking in human-computer interaction and usability research. In *The Mind's Eye* (pp. 573-605). Elsevier.

- Duchowski, A. T. (2017). *Eye Tracking Methodology: Theory and Practice* (3rd ed.). Springer.

### 15.5 Multimedia Learning

- Mayer, R. E. (2014). *The Cambridge Handbook of Multimedia Learning* (2nd ed.). Cambridge University Press.

- Van Gog, T., & Scheiter, K. (2010). Eye tracking as a tool to study and enhance multimedia learning. *Learning and Instruction, 20*(2), 95-99.

### 15.6 Methods and Statistics

- Barr, D. J. (2008). Analyzing 'visual world' eyetracking data using multilevel logistic regression. *Journal of Memory and Language, 59*(4), 457-474.

- Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008). Mixed-effects modeling with crossed random effects for subjects and items. *Journal of Memory and Language, 59*(4), 390-412.

---

## Appendix A: Quick Reference Card

### Event Detection Parameters (Starting Points)

| Parameter | Recommended Value |
|-----------|------------------|
| Velocity threshold (I-VT) | 30°/s |
| Dispersion threshold (I-DT) | 1° |
| Minimum fixation duration | 50-100 ms |
| Maximum fixation duration | 1500-2000 ms |
| Blink threshold | 75-100 ms |

### Typical Fixation Durations by Task

| Task | Mean Duration |
|------|---------------|
| Silent reading | 225-250 ms |
| Oral reading | 275-325 ms |
| Scene viewing | 260-330 ms |
| Visual search | 180-275 ms |
| Music reading | 375-400 ms |

### Unit Conversions

| Conversion | Formula |
|------------|---------|
| Pixels to degrees | degrees = arctan(pixels × pixel_size / distance) × (180/π) |
| Degrees to pixels | pixels = tan(degrees × π/180) × distance / pixel_size |
| 1° at 60cm | ~35 pixels (typical monitor) |

---

## Appendix B: Checklist for Eye Tracking Studies

### Pre-Experiment
- [ ] Ethics approval obtained
- [ ] Equipment tested and calibrated
- [ ] Stimuli prepared and validated
- [ ] Participant screening criteria defined
- [ ] Consent forms prepared
- [ ] Data storage plan established

### Experiment Session
- [ ] Room lighting controlled
- [ ] Participant positioned correctly
- [ ] Calibration successful (error < 1°)
- [ ] Practice trials completed
- [ ] Tracking quality monitored throughout
- [ ] Drift corrections applied as needed
- [ ] Post-experiment questions administered

### Data Processing
- [ ] Raw data backed up
- [ ] Data quality assessed
- [ ] Exclusion criteria applied consistently
- [ ] Event detection parameters documented
- [ ] AOIs defined and validated
- [ ] Artifacts removed/corrected

### Analysis and Reporting
- [ ] Appropriate statistical methods used
- [ ] Effect sizes reported
- [ ] Visualizations included
- [ ] Methods fully documented
- [ ] Limitations acknowledged

---

## Appendix C: Saliency Model Evaluation Metrics

### Standard Metrics for Saliency Benchmarks

| Metric | Full Name | Range | Interpretation |
|--------|-----------|-------|----------------|
| **AUC-Judd** | Area Under ROC Curve (Judd) | 0-1 | Higher = better prediction |
| **AUC-Borji** | Area Under ROC Curve (Borji) | 0-1 | Shuffled baseline comparison |
| **sAUC** | Shuffled AUC | 0-1 | Controls for center bias |
| **NSS** | Normalized Scanpath Saliency | -∞ to +∞ | Mean saliency at fixations |
| **CC** | Pearson Correlation | -1 to 1 | Linear correlation with ground truth |
| **SIM** | Similarity | 0-1 | Histogram intersection |
| **KL** | KL Divergence | 0 to +∞ | Lower = better (distribution match) |
| **IG** | Information Gain | bits | Information above baseline |

### Metric Selection Guide

| Research Goal | Recommended Metrics |
|---------------|---------------------|
| Model comparison | AUC-Judd, NSS, CC |
| Center bias analysis | sAUC, AUC-Borji |
| Distribution matching | SIM, KL |
| Benchmark submission | All standard metrics |

---

## Appendix D: Common Eye Tracking File Formats

### Data Formats by System

| System | Native Format | Export Options |
|--------|---------------|----------------|
| **EyeLink** | .edf | .asc, .csv |
| **Tobii** | .gazedata | .tsv, .csv, .xlsx |
| **SMI** | .idf | .txt, .csv |
| **Gazepoint** | .csv | .csv |
| **WebGazer** | JSON | JSON, CSV |

### Standard Data Fields

| Field | Description | Units |
|-------|-------------|-------|
| timestamp | Time of sample | ms |
| x_left, y_left | Left eye position | pixels or degrees |
| x_right, y_right | Right eye position | pixels or degrees |
| pupil_left | Left pupil size | mm or arbitrary |
| pupil_right | Right pupil size | mm or arbitrary |
| validity | Data quality flag | 0-4 or boolean |

---

## Appendix E: Code Snippets

### Python: Basic Fixation Detection (I-VT)

```python
import numpy as np

def detect_fixations_ivt(x, y, timestamps, velocity_threshold=30, 
                          min_duration=100, sampling_rate=1000):
    """
    I-VT fixation detection algorithm
    
    Parameters:
    - x, y: gaze coordinates (degrees)
    - timestamps: time in ms
    - velocity_threshold: degrees per second
    - min_duration: minimum fixation duration in ms
    - sampling_rate: Hz
    
    Returns:
    - List of fixations with start, end, x, y, duration
    """
    # Calculate velocities
    dt = np.diff(timestamps) / 1000  # Convert to seconds
    dx = np.diff(x)
    dy = np.diff(y)
    velocity = np.sqrt(dx**2 + dy**2) / dt
    
    # Classify samples
    is_fixation = velocity < velocity_threshold
    is_fixation = np.append(is_fixation, is_fixation[-1])  # Pad
    
    # Find fixation boundaries
    fixations = []
    in_fixation = False
    start_idx = 0
    
    for i, fix in enumerate(is_fixation):
        if fix and not in_fixation:
            start_idx = i
            in_fixation = True
        elif not fix and in_fixation:
            duration = timestamps[i-1] - timestamps[start_idx]
            if duration >= min_duration:
                fixations.append({
                    'start': timestamps[start_idx],
                    'end': timestamps[i-1],
                    'x': np.mean(x[start_idx:i]),
                    'y': np.mean(y[start_idx:i]),
                    'duration': duration
                })
            in_fixation = False
    
    return fixations
```

### R: Mixed-Effects Model for Eye Tracking

```r
library(lme4)
library(lmerTest)

# Example: Analyzing fixation duration
model <- lmer(
  log(fixation_duration) ~ condition * aoi_type + 
    (1 + condition | participant) + 
    (1 | item),
  data = eye_data,
  control = lmerControl(optimizer = "bobyqa")
)

# Summary with p-values
summary(model)

# Effect sizes
library(effectsize)
eta_squared(model)
```

### Python: Heat Map Generation

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def create_heatmap(fixations, image_size, sigma=30):
    """
    Create attention heat map from fixations
    
    Parameters:
    - fixations: list of dicts with 'x', 'y', 'duration'
    - image_size: (width, height)
    - sigma: Gaussian blur sigma
    
    Returns:
    - 2D numpy array (heat map)
    """
    heatmap = np.zeros((image_size[1], image_size[0]))
    
    for fix in fixations:
        x, y = int(fix['x']), int(fix['y'])
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            # Weight by duration
            heatmap[y, x] += fix['duration']
    
    # Apply Gaussian blur
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap
```

---

## Appendix F: Glossary of Terms

| Term | Definition |
|------|------------|
| **AOI** | Area of Interest - defined region for analysis |
| **Calibration** | Process of mapping eye position to screen coordinates |
| **Corneal reflection** | Light reflection from cornea used for tracking |
| **Drift** | Gradual change in calibration accuracy over time |
| **Dwell time** | Total time spent looking at an area |
| **Fixation** | Period when eyes are relatively stationary |
| **Fovea** | Central retinal area with highest acuity |
| **Gaze** | Direction of looking (point of fixation) |
| **Heat map** | Visualization showing attention distribution |
| **Microsaccade** | Small involuntary saccade during fixation |
| **Parafovea** | Area surrounding fovea (2-5° eccentricity) |
| **Pupil-CR** | Tracking method using pupil and corneal reflection |
| **Regression** | Backward saccade (e.g., in reading) |
| **Saccade** | Rapid eye movement between fixations |
| **Saliency** | Visual prominence that attracts attention |
| **Scanpath** | Sequence of fixations and saccades |
| **Smooth pursuit** | Slow tracking of moving objects |
| **Vergence** | Eye movements in opposite directions (depth) |
| **Visual angle** | Size measurement in degrees of arc |

---

## Appendix G: Key Dataset URLs

| Dataset | URL |
|---------|-----|
| MIT Saliency Benchmark | https://saliency.mit.edu |
| SALICON | http://salicon.net |
| GazeCapture | https://gazecapture.csail.mit.edu |
| MPIIGaze | https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild |
| ETH-XGaze | https://ait.ethz.ch/xgaze |
| GECO Corpus | https://github.com/eeemma/GECO |
| ZuCo | https://osf.io/q3zws/ |
| WebGazer.js | https://webgazer.cs.brown.edu |

---

*Document prepared for HCI + L Laboratory research team*  
*Last updated: January 2026*
