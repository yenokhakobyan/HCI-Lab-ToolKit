# EEG in Human-Computer Interaction Research: A Comprehensive Guide

**HCI + L Laboratory Technical Documentation**  
**Version 1.0 | January 2026**

---

## Table of Contents

1. [Introduction to EEG](#1-introduction-to-eeg)
2. [Hardware: Enobio 20 and Alternatives](#2-hardware-enobio-20-and-alternatives)
3. [EEG Signal Fundamentals](#3-eeg-signal-fundamentals)
4. [Frequency Bands and Their Significance](#4-frequency-bands-and-their-significance)
5. [Key Metrics for HCI Research](#5-key-metrics-for-hci-research)
6. [Cognitive Load Measurement](#6-cognitive-load-measurement)
7. [Emotional State Detection](#7-emotional-state-detection)
8. [Event-Related Potentials (ERPs)](#8-event-related-potentials-erps)
9. [Data Processing Pipeline](#9-data-processing-pipeline)
10. [Analysis Approaches and Models](#10-analysis-approaches-and-models)
11. [Integration with Eye-Tracking](#11-integration-with-eye-tracking)
12. [Experimental Design Considerations](#12-experimental-design-considerations)
13. [Common Challenges and Solutions](#13-common-challenges-and-solutions)
14. [Software Tools and Libraries](#14-software-tools-and-libraries)
15. [References and Further Reading](#15-references-and-further-reading)

---

## 1. Introduction to EEG

### What is EEG?

Electroencephalography (EEG) is a non-invasive neuroimaging technique that measures electrical activity in the brain through electrodes placed on the scalp. The electrical signals (measured in microvolts, μV) reflect the synchronized activity of millions of neurons and provide real-time insight into brain function.

### Why EEG for HCI Research?

EEG offers several advantages for studying human-computer interaction:

- **High temporal resolution** (~milliseconds) — captures rapid cognitive processes
- **Real-time monitoring** — enables adaptive systems
- **Portable options available** — can be used outside traditional lab settings
- **Non-invasive** — comfortable for participants
- **Cost-effective** compared to fMRI or MEG
- **Direct neural measurement** — not dependent on behavioral responses

### Relevance to Our Project

For the EcoInsight adaptive learning system, EEG provides:
- Objective measures of cognitive load during infographic processing
- Emotional engagement indicators
- Attention and focus metrics
- Individual differences in working memory utilization
- Real-time data for adaptive content delivery

---

## 2. Hardware: Enobio 20 and Alternatives

### Enobio 20 (Primary System)

The Enobio 20 by Neuroelectrics is our primary EEG system, selected for its balance of research-grade quality and practical usability.

#### Technical Specifications

| Specification | Value |
|--------------|-------|
| Channels | 20 EEG channels |
| Electrode System | 10-20 International System |
| Sampling Rate | Up to 500 Hz |
| Resolution | 24-bit ADC |
| Bandwidth | 0-125 Hz |
| Wireless | Bluetooth connectivity |
| Battery Life | ~8 hours continuous use |
| Electrode Type | Dry or gel-based options |

#### Electrode Placement (10-20 System)

Standard positions available on Enobio 20:
```
Frontal:    Fp1, Fp2, F7, F3, Fz, F4, F8
Central:    C3, Cz, C4
Temporal:   T3 (T7), T4 (T8), T5 (P7), T6 (P8)
Parietal:   P3, Pz, P4
Occipital:  O1, O2
```

#### Key Features for HCI Research

- **Portability**: Wireless design allows natural interaction with interfaces
- **Dry electrodes option**: Reduces setup time (though gel electrodes provide better signal quality)
- **NIC2 Software**: Proprietary software for data acquisition and visualization
- **LSL Integration**: Lab Streaming Layer support for multi-modal synchronization
- **Accelerometer**: Built-in motion detection for artifact identification

#### Setup Best Practices

1. **Impedance check**: Aim for <20 kΩ (gel) or <200 kΩ (dry)
2. **Cap positioning**: Cz at vertex (measured 50% nasion-to-inion and 50% preauricular points)
3. **Reference electrode**: Typically mastoid (behind ear) or earlobe
4. **Ground electrode**: Usually at AFz position
5. **Acclimation period**: Allow 5-10 minutes for signal stabilization

### Alternative Systems

#### Research-Grade Alternatives

| System | Channels | Sampling Rate | Key Features |
|--------|----------|---------------|--------------|
| BioSemi ActiveTwo | 32-256 | 2048 Hz | Gold standard for research |
| g.tec g.USBamp | 16-256 | 38.4 kHz | High precision, BCI focus |
| Brain Products actiCHamp | 32-160 | 100 kHz | Mobile research system |
| ANT Neuro eego | 32-256 | 2048 Hz | MRI compatible |

#### Consumer/Portable Alternatives

| System | Channels | Use Case |
|--------|----------|----------|
| Emotiv EPOC X | 14 | Rapid prototyping, pilot studies |
| Muse 2 | 4 | Meditation, basic attention |
| OpenBCI | 8-16 | Open-source development |
| Neurosity Crown | 8 | Focus tracking, BCI |

### Hardware Selection Criteria

For our research needs, consider:
- **Signal quality**: Minimum 250 Hz sampling for ERP studies
- **Channel count**: 20+ for source localization; fewer for specific ROI studies
- **Mobility**: Wireless for natural HCI tasks
- **Synchronization**: Must support external triggers and multi-modal integration
- **Software compatibility**: MATLAB, Python, LSL support

---

## 3. EEG Signal Fundamentals

### Signal Characteristics

#### Amplitude
- Typical range: 10-100 μV
- Very small compared to other biosignals (ECG ~1mV, EMG ~10mV)
- Requires high amplification (×10,000 to ×100,000)

#### Frequency Content
- Meaningful signal: 0.5-100 Hz (practical focus: 1-50 Hz)
- Different frequency bands reflect different cognitive states

### Signal Generation

EEG signals arise from:
- **Post-synaptic potentials** in cortical pyramidal neurons
- Requires **synchronous activity** of thousands of neurons
- Surface electrodes detect activity primarily from **cortex** (not deep structures)
- Signal attenuated by skull, scalp, and distance

### Spatial Resolution Limitations

- EEG has poor spatial resolution (~1-2 cm)
- **Volume conduction**: Signals spread through tissue
- **Inverse problem**: Multiple source configurations can produce same scalp pattern
- Source localization requires assumptions and is approximate

### Temporal Dynamics

- Excellent temporal resolution (<1 ms with high sampling rates)
- Can track rapid cognitive processes
- Enables real-time applications
- Time-locked analysis (ERPs) possible with precise triggers

---

## 4. Frequency Bands and Their Significance

### Standard Frequency Bands

| Band | Frequency | Amplitude | Associated States |
|------|-----------|-----------|-------------------|
| **Delta** (δ) | 0.5-4 Hz | 20-200 μV | Deep sleep, unconsciousness, pathology |
| **Theta** (θ) | 4-8 Hz | 20-100 μV | Drowsiness, memory encoding, meditation |
| **Alpha** (α) | 8-13 Hz | 30-50 μV | Relaxed wakefulness, eyes closed, inhibition |
| **Beta** (β) | 13-30 Hz | 5-30 μV | Active thinking, focus, alertness |
| **Gamma** (γ) | 30-100 Hz | <5 μV | Higher cognition, perception binding |

### Sub-band Classifications

#### Alpha Sub-bands
- **Lower alpha (8-10 Hz)**: General alertness, attentional demands
- **Upper alpha (10-13 Hz)**: Semantic memory, cognitive processing

#### Beta Sub-bands
- **Low beta (13-15 Hz)**: Relaxed but alert
- **Mid beta (15-20 Hz)**: Active thinking
- **High beta (20-30 Hz)**: Anxiety, intense concentration

### Functional Interpretations for HCI

#### Frontal Theta
- **Increases with**: Working memory load, mental effort, error processing
- **Location**: Fz, F3, F4
- **HCI relevance**: Cognitive load indicator, task difficulty assessment

#### Parietal Alpha
- **Decreases with**: Visual/cognitive processing (event-related desynchronization)
- **Increases with**: Relaxation, disengagement
- **Location**: Pz, P3, P4, O1, O2
- **HCI relevance**: Attention, engagement, visual processing load

#### Frontal Alpha Asymmetry
- **Left > Right**: Approach motivation, positive affect
- **Right > Left**: Withdrawal motivation, negative affect
- **Location**: F3 vs F4
- **HCI relevance**: Emotional response to interface elements

#### Beta Activity
- **Increases with**: Focused attention, problem-solving
- **Location**: Widespread, often frontal
- **HCI relevance**: Concentration, active engagement

#### Theta/Beta Ratio
- **Higher ratio**: Lower arousal, potential disengagement
- **Lower ratio**: Higher arousal, focused attention
- **HCI relevance**: Attention/inattention classification, ADHD research

---

## 5. Key Metrics for HCI Research

### Power Spectral Metrics

#### Absolute Band Power
- Total power within frequency band (μV²)
- Computed via FFT or Welch's method
- **Formula**: Sum of squared amplitudes in frequency range
```
Power_band = Σ|X(f)|² for f in band range
```

#### Relative Band Power
- Proportion of band power to total power
- Normalizes for individual differences
- **Formula**: 
```
Relative_band = Power_band / Total_power
```

#### Power Spectral Density (PSD)
- Power per unit frequency (μV²/Hz)
- Standard representation for spectral analysis
- Enables comparison across studies

### Connectivity Metrics

#### Coherence
- Frequency-domain correlation between channels
- Measures functional connectivity
- Range: 0 (independent) to 1 (perfectly coupled)
- **Formula**:
```
Coh_xy(f) = |S_xy(f)|² / (S_xx(f) × S_yy(f))
```

#### Phase-Locking Value (PLV)
- Consistency of phase difference between channels
- Measures synchronization independent of amplitude
- Range: 0 to 1

#### Granger Causality
- Directional influence between brain regions
- Time-series based causal inference
- Useful for understanding information flow

### Asymmetry Metrics

#### Frontal Alpha Asymmetry (FAA)
```
FAA = ln(Right Alpha Power) - ln(Left Alpha Power)
     = ln(F4) - ln(F3)
```
- Positive values: Approach motivation
- Negative values: Avoidance motivation

#### Regional Asymmetry Index
- Generalizable to other bands/regions
- Used for emotional and motivational assessment

### Engagement and Attention Indices

#### Engagement Index (Pope et al.)
```
Engagement = Beta / (Alpha + Theta)
```
- Higher values indicate greater engagement
- Validated in vigilance and attention tasks

#### Attention Ratio
```
Attention = Beta / Theta
```
- Used in neurofeedback applications
- Indicator of focused vs. diffuse attention

#### Task Load Index
```
TLI = Frontal Theta / Parietal Alpha
```
- Increases with cognitive workload
- Combines two well-validated markers

### Complexity Metrics

#### Sample Entropy
- Measures signal irregularity/complexity
- Lower entropy: More regular, predictable signal
- Higher entropy: More complex, information-rich

#### Fractal Dimension
- Characterizes signal self-similarity
- Related to cognitive state complexity

#### Lempel-Ziv Complexity
- Algorithmic complexity measure
- Related to conscious awareness level

---

## 6. Cognitive Load Measurement

### Theoretical Background

Cognitive Load Theory (Sweller, 1988) distinguishes:
- **Intrinsic load**: Inherent complexity of material
- **Extraneous load**: Load from poor design/presentation
- **Germane load**: Productive effort toward learning

EEG provides objective, real-time cognitive load indicators complementing subjective measures (NASA-TLX).

### Primary Neural Markers

#### Frontal Theta Power (4-7 Hz)
- **Finding**: Robust increase with working memory demand
- **Location**: Fz, F3, F4 (frontal midline)
- **Mechanism**: Hippocampal-cortical communication
- **Evidence**: Antonenko et al. (2010), Gevins & Smith (2003)
- **Effect size**: Typically r = 0.5-0.7 correlation with load

#### Parietal Alpha Suppression (8-12 Hz)
- **Finding**: Decreased alpha with increased processing
- **Location**: Pz, P3, P4, O1, O2
- **Mechanism**: Cortical activation (desynchronization)
- **Evidence**: Klimesch (1999), Mills et al. (2017)
- **Inverse correlation**: r = -0.6 to -0.8 with task demands

#### Theta/Alpha Ratio
- **Calculation**: Frontal Theta / Parietal Alpha
- **Interpretation**: Higher ratio = higher cognitive load
- **Advantage**: Single metric combining two markers

### Secondary Markers

#### P300 Amplitude Reduction
- Secondary task paradigm
- Reduced P300 = resources diverted to primary task
- Gold standard for dual-task methodology

#### Frontal Beta Increase
- Associated with active problem-solving
- Less consistent than theta/alpha markers

#### Heart Rate Variability (with Shimmer GSR)
- Decreased HRV = increased cognitive demand
- Requires multi-modal integration

### Cognitive Load Index (CLI) Calculation

A composite measure used in our research:
```python
def compute_CLI(theta_frontal, alpha_parietal):
    """
    Compute Cognitive Load Index
    
    Parameters:
    - theta_frontal: Theta power at Fz, F3, F4 (averaged)
    - alpha_parietal: Alpha power at Pz, P3, P4 (averaged)
    
    Returns:
    - CLI: Cognitive Load Index (higher = more load)
    """
    # Log transform for normality
    log_theta = np.log(theta_frontal)
    log_alpha = np.log(alpha_parietal)
    
    # Standardize to baseline
    theta_z = (log_theta - baseline_theta_mean) / baseline_theta_std
    alpha_z = (log_alpha - baseline_alpha_mean) / baseline_alpha_std
    
    # Combine markers (theta increases, alpha decreases with load)
    CLI = theta_z - alpha_z
    
    return CLI
```

### Validation Against NASA-TLX

Expected correlations:
- Mental demand: r ≈ 0.6-0.7
- Temporal demand: r ≈ 0.4-0.5
- Effort: r ≈ 0.5-0.6
- Overall workload: r ≈ 0.6-0.7

### Real-Time Load Detection

For adaptive systems (like EcoInsight):
1. Establish baseline (resting state or easy task)
2. Compute sliding window (2-5 seconds)
3. Calculate deviation from baseline
4. Threshold for "overload" detection
5. Trigger adaptation (simplify content, slow pace)

---

## 7. Emotional State Detection

### Dimensional Model of Emotion

Russell's Circumplex Model (1980):
- **Valence**: Positive ↔ Negative (pleasure dimension)
- **Arousal**: High ↔ Low (activation dimension)

### EEG Correlates of Valence

#### Frontal Alpha Asymmetry (FAA)
- Primary marker for emotional valence
- **Positive valence**: Greater left frontal activity (F3 < F4 alpha)
- **Negative valence**: Greater right frontal activity (F3 > F4 alpha)
- **Calculation**:
```python
FAA = np.log(alpha_F4) - np.log(alpha_F3)
# Positive FAA → Positive emotion/approach
# Negative FAA → Negative emotion/withdrawal
```

#### Evidence and Reliability
- Meta-analysis effect size: d ≈ 0.4-0.6
- Better for approach/withdrawal motivation than hedonic valence
- Individual differences require baseline correction

### EEG Correlates of Arousal

#### Beta Power
- Increased beta = higher arousal
- Location: Widespread, especially frontal
- Range: 13-30 Hz

#### Alpha Power (Inverse)
- Decreased alpha = higher arousal
- Reflects cortical activation

#### Arousal Index
```python
Arousal_Index = (Beta_power) / (Alpha_power + Theta_power)
```

### Affective State Classification

#### Common Emotion Categories
Using EEG, can typically distinguish:
- Positive vs. Negative (valence)
- High vs. Low arousal
- Engagement vs. Boredom
- Frustration vs. Flow

#### Machine Learning Approaches

**Features typically used:**
- Band powers (absolute and relative)
- Asymmetry indices
- Frontal theta
- Parietal alpha
- Higher-order statistics

**Common classifiers:**
- SVM (Support Vector Machine): ~70-80% accuracy for valence
- Random Forest: Good for feature importance
- Deep learning (CNN, LSTM): Can learn spatial/temporal patterns

**Expected Performance:**
| Classification | Accuracy Range |
|---------------|----------------|
| Valence (2-class) | 65-80% |
| Arousal (2-class) | 70-85% |
| 4-quadrant | 50-65% |

### Emotion in HCI Context

For our research applications:
- **Engagement** during learning (frontal theta, beta)
- **Confusion/frustration** detection (frontal asymmetry, beta increase)
- **Interest/boredom** (alpha levels, engagement index)
- **Aesthetic response** to infographic design (N200, late positive potential)

### DEAP Dataset Reference

Standard benchmark for emotion recognition:
- 32 participants
- 40 music video stimuli
- 32-channel EEG + physiological signals
- Valence, arousal, dominance, liking ratings
- Useful for algorithm development and comparison

---

## 8. Event-Related Potentials (ERPs)

### What are ERPs?

ERPs are voltage fluctuations time-locked to specific events (stimuli or responses). They reveal the temporal dynamics of cognitive processing with millisecond precision.

### ERP Methodology

1. **Epoch extraction**: Time-lock to event (e.g., -200 to 800 ms)
2. **Baseline correction**: Subtract pre-stimulus mean
3. **Averaging**: Across trials to reduce noise
4. **Analysis**: Measure amplitude/latency of components

**Signal-to-Noise Improvement:**
```
SNR improvement = √(number of trials)
```
Typically need 30-100+ trials per condition.

### Key ERP Components for HCI

#### Early Components (Sensory)

**P1 (80-120 ms)**
- Visual attention, low-level feature processing
- Location: Occipital (O1, O2)
- Enhanced by spatial attention

**N1 (150-200 ms)**
- Early attention, discrimination processing
- Location: Occipital-parietal
- Sensitive to attended vs. unattended stimuli

#### Mid-Latency Components

**N2 (200-350 ms)**
- Conflict detection, cognitive control
- Location: Frontal-central
- **N200 in HCI**: Novelty detection, unexpected interface changes

**P3a (250-300 ms)**
- Involuntary attention shift, novelty
- Location: Frontal-central
- Triggered by unexpected events

**P3b / P300 (300-600 ms)**
- Context updating, working memory
- Location: Parietal (Pz)
- **Critical for HCI**:
  - Larger P300 = More attention/processing resources
  - Reduced P300 = Cognitive load from other tasks
  - Latency increases with processing difficulty

#### Late Components

**N400 (300-500 ms)**
- Semantic processing, meaning integration
- Location: Centro-parietal
- Larger for unexpected/incongruent content
- **HCI relevance**: Semantic mismatch in labels, icons, navigation

**Late Positive Potential (LPP, 400-800 ms)**
- Emotional significance, sustained attention
- Location: Centro-parietal
- Larger for emotionally arousing stimuli
- **HCI relevance**: Emotional engagement with content

### Error-Related Components

**Error-Related Negativity (ERN, 0-100 ms post-error)**
- Error detection, performance monitoring
- Location: Frontal-central (FCz)
- Peaks immediately after error response
- **HCI relevance**: Usability problems, interaction errors

**Error Positivity (Pe, 200-400 ms post-error)**
- Error awareness, significance
- Location: Centro-parietal
- Follows ERN
- Related to conscious error recognition

### ERP Analysis for Infographic Research

For the EcoInsight project:

**Visual Processing Analysis:**
- P1/N1 components: Early attention to visual elements
- Comparing text vs. image regions
- Effect of visual complexity

**Comprehension Analysis:**
- N400: Semantic integration of verbal/visual information
- P600: Later integration, revision of interpretation

**Attention and Memory:**
- P300: Resource allocation, importance detection
- LPP: Emotional/motivational engagement

**Example paradigm:**
```
Fixation (500ms) → Infographic segment (variable) → Probe question
                         ↑
                   ERP time-locked to segment onset
```

---

## 9. Data Processing Pipeline

### Overview

```
Raw EEG → Preprocessing → Feature Extraction → Analysis → Interpretation
```

### Step 1: Data Import and Quality Check

```python
# Typical data structure
sampling_rate = 500  # Hz
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'P7', 'P3', 'Pz', 'P4', 'P8', 
            'O1', 'O2']

# Quality checks
- Verify channel count and labels
- Check sampling rate
- Identify missing data
- Review event markers
```

### Step 2: Preprocessing

#### Filtering
```python
# Bandpass filter (standard for most analyses)
low_cutoff = 0.5  # Hz (removes slow drift)
high_cutoff = 50  # Hz (removes high-freq noise, line noise)

# Notch filter for line noise
notch_freq = 50  # Hz in Europe (60 Hz in Americas)
```

#### Re-referencing
```python
# Options:
# 1. Average reference (recommended for spectral analysis)
# 2. Linked mastoids (common in clinical)
# 3. Cz reference (for some ERP studies)
# 4. REST (standardized reference)
```

#### Artifact Removal

**Types of artifacts:**
| Artifact | Characteristics | Solution |
|----------|-----------------|----------|
| Eye blinks | High amplitude, frontal | ICA, EOG regression |
| Eye movements | Frontal propagation | ICA, EOG channels |
| Muscle (EMG) | High frequency, temporal | ICA, filtering |
| Cardiac (ECG) | Rhythmic, widespread | ICA |
| Movement | Large amplitude | Epoch rejection |
| Line noise | 50/60 Hz | Notch filter |

**ICA (Independent Component Analysis):**
```python
# Standard approach
from mne.preprocessing import ICA

ica = ICA(n_components=15, random_state=42)
ica.fit(raw_data)

# Identify artifact components (manual or automatic)
# Remove components
ica.exclude = [0, 2]  # Example: components 0 and 2 are artifacts
clean_data = ica.apply(raw_data.copy())
```

#### Epoch Rejection
```python
# Amplitude threshold
reject_criteria = dict(eeg=150e-6)  # 150 μV

# Flat channel detection
flat_criteria = dict(eeg=1e-6)  # 1 μV
```

### Step 3: Segmentation

**For Spectral Analysis:**
```python
# Continuous data or long epochs
epoch_length = 2.0  # seconds
overlap = 0.5  # 50% overlap
```

**For ERP Analysis:**
```python
# Event-locked epochs
tmin = -0.2  # 200 ms before event
tmax = 0.8   # 800 ms after event
baseline = (-0.2, 0)  # Baseline period for correction
```

### Step 4: Feature Extraction

#### Spectral Features
```python
from scipy.signal import welch

# Power spectral density
freqs, psd = welch(data, fs=sampling_rate, nperseg=1024)

# Band power
def band_power(psd, freqs, band):
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx], freqs[idx])

theta_power = band_power(psd, freqs, [4, 8])
alpha_power = band_power(psd, freqs, [8, 13])
beta_power = band_power(psd, freqs, [13, 30])
```

#### Time-Domain Features
```python
# For ERP analysis
amplitude = epochs.data.mean(axis=0)  # Average across trials
peak_amplitude = amplitude.max()
peak_latency = times[amplitude.argmax()]
mean_amplitude = amplitude[time_window].mean()
```

#### Connectivity Features
```python
from mne.connectivity import spectral_connectivity

# Coherence between channels
con = spectral_connectivity(epochs, method='coh', 
                            fmin=4, fmax=30)
```

### Step 5: Statistical Analysis

**Common approaches:**
- t-tests for condition comparisons
- ANOVA for multiple conditions
- Permutation tests (non-parametric)
- Cluster-based correction for multiple comparisons
- Linear mixed-effects models for repeated measures

---

## 10. Analysis Approaches and Models

### Traditional Statistical Approaches

#### Band Power Analysis
```python
# Between-condition comparison
from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(condition1_alpha, condition2_alpha)
```

#### Time-Frequency Analysis
- **Morlet wavelets**: Good time-frequency tradeoff
- **Short-time Fourier Transform (STFT)**: Consistent time resolution
- **Hilbert transform**: Instantaneous amplitude/phase

```python
from mne.time_frequency import tfr_morlet

freqs = np.arange(4, 40, 1)
n_cycles = freqs / 2  # Use adaptive cycles

power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                   return_itc=False)
```

### Machine Learning Approaches

#### Feature-Based Classification

**Common feature sets:**
```python
features = {
    'band_powers': ['delta', 'theta', 'alpha', 'beta', 'gamma'],
    'asymmetry': ['frontal_alpha_asym', 'parietal_alpha_asym'],
    'ratios': ['theta_alpha_ratio', 'theta_beta_ratio'],
    'connectivity': ['coherence_frontal', 'plv_central'],
    'complexity': ['sample_entropy', 'fractal_dimension']
}
```

**Classification pipeline:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf'))
])

cv_scores = cross_val_score(pipeline, X, y, cv=5)
```

#### Deep Learning Approaches

**EEGNet (Lawhern et al., 2018)**
- Compact CNN architecture for EEG
- Works well with limited data
- Interpretable features

```python
# Architecture overview
# Temporal convolution → Depthwise convolution → Separable convolution
# Input: (channels × time_samples)
# Output: (classes)
```

**LSTM for Sequential Processing**
```python
# For time-series classification
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(time_steps, features)),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])
```

**Transformer-based Models**
- Attention mechanisms for EEG
- Can model long-range dependencies
- Growing area of research

### Source Localization

**LORETA (Low Resolution Brain Electromagnetic Tomography)**
- Estimates cortical sources
- Assumes smooth distribution
- Available in open-source toolboxes

**Beamforming**
- Spatial filtering approach
- Better for localized sources
- Requires forward model

**Limitations:**
- Ill-posed inverse problem
- Results are estimates, not ground truth
- Spatial resolution still limited (~10mm)

### Cognitive Load Classification Model

For our EcoInsight project:
```python
class CognitiveLoadClassifier:
    """
    Multi-level cognitive load classifier using EEG features
    """
    
    def __init__(self):
        self.feature_extractor = EEGFeatureExtractor()
        self.classifier = RandomForestClassifier(n_estimators=100)
        
    def extract_features(self, epoch):
        features = {}
        
        # Frontal theta
        features['theta_frontal'] = self.feature_extractor.band_power(
            epoch, channels=['Fz', 'F3', 'F4'], band=[4, 8])
        
        # Parietal alpha
        features['alpha_parietal'] = self.feature_extractor.band_power(
            epoch, channels=['Pz', 'P3', 'P4'], band=[8, 13])
        
        # Engagement index
        features['engagement'] = self.feature_extractor.engagement_index(
            epoch, channels=['Fz', 'Cz'])
        
        # Theta/alpha ratio
        features['theta_alpha_ratio'] = (features['theta_frontal'] / 
                                         features['alpha_parietal'])
        
        return features
    
    def predict_load(self, epoch):
        features = self.extract_features(epoch)
        X = np.array(list(features.values())).reshape(1, -1)
        return self.classifier.predict(X)  # Low, Medium, High
```

### Multimodal Integration

Combining EEG with other data sources:
```python
# Feature-level fusion
combined_features = np.concatenate([
    eeg_features,      # Band powers, asymmetry, etc.
    eye_features,      # Fixation duration, saccades, pupil
    gsr_features       # Skin conductance, HRV
], axis=1)

# Decision-level fusion
eeg_prediction = eeg_classifier.predict(eeg_features)
eye_prediction = eye_classifier.predict(eye_features)
final_prediction = voting_classifier.predict([eeg_prediction, eye_prediction])
```

---

## 11. Integration with Eye-Tracking

### Why Combine EEG and Eye-Tracking?

| EEG Provides | Eye-Tracking Provides |
|--------------|----------------------|
| Neural processing | Visual attention |
| Cognitive load | Gaze location |
| Emotional state | Reading behavior |
| Memory encoding | Scan patterns |
| ms temporal precision | Spatial precision |

### Synchronization Methods

#### Hardware Synchronization
```python
# TTL triggers
# EEG system and eye-tracker receive same trigger
# Sub-millisecond precision
```

#### Software Synchronization (Lab Streaming Layer)
```python
from pylsl import StreamInfo, StreamOutlet, StreamInlet

# Create unified timestamp
# Post-hoc alignment based on LSL timestamps
# Typical precision: 1-5 ms
```

### Combined Analysis Approaches

#### Fixation-Related Potentials (FRPs)
Time-lock EEG to fixation onset instead of stimulus onset.

```python
# Extract epoch at each fixation
for fixation in eye_data.fixations:
    epoch = eeg_data[fixation.onset - 200 : fixation.onset + 500]
    
# Average across fixations on specific AOIs
text_frps = average(fixations_on_text)
image_frps = average(fixations_on_images)
```

**Key FRP components:**
- **Lambda response** (~100 ms): Visual processing
- **N1-like** (~200 ms): Discrimination
- **P300-like** (~300-400 ms): Semantic processing

#### Gaze-Contingent EEG Analysis
Analyze EEG based on where participant is looking.

```python
# Segment EEG by gaze location
looking_at_text_eeg = eeg_data[gaze_in_text_aoi]
looking_at_image_eeg = eeg_data[gaze_in_image_aoi]

# Compare alpha suppression in different conditions
alpha_during_text = band_power(looking_at_text_eeg, [8, 13])
alpha_during_image = band_power(looking_at_image_eeg, [8, 13])
```

#### Joint Feature Extraction

```python
features = {
    # EEG features
    'frontal_theta': compute_theta(eeg, ['Fz']),
    'parietal_alpha': compute_alpha(eeg, ['Pz']),
    'engagement_index': compute_engagement(eeg),
    
    # Eye-tracking features
    'total_fixation_time': eye.total_fixation_time,
    'fixation_count': len(eye.fixations),
    'mean_saccade_amplitude': eye.saccades.amplitude.mean(),
    'text_to_image_transitions': count_transitions(eye, text_aoi, image_aoi),
    
    # Combined features
    'theta_per_fixation': frontal_theta / fixation_count,
    'alpha_during_first_pass': alpha_during_time(first_pass_period)
}
```

### Practical Considerations

**Synchronization challenges:**
- Different sampling rates (EEG: 500 Hz, Eye: 120-1000 Hz)
- Different event definitions
- Temporal jitter in event detection

**Solutions:**
- Upsample/downsample to common rate
- Use LSL for unified timestamps
- Regular sync pulses for drift correction
- Validate synchronization with known events

---

## 12. Experimental Design Considerations

### Participant Considerations

#### Inclusion/Exclusion Criteria
```
Include:
- Age 18-30 (for our study)
- Normal or corrected-to-normal vision
- Native speakers (for language-specific tasks)

Exclude:
- Neurological/psychiatric conditions
- Psychotropic medications
- Recent head injury
- Excessive caffeine/alcohol
```

#### Sample Size Estimation

**For ERP studies:**
- Within-subject: n = 20-30 typical
- Effect size d = 0.5: n ≈ 34 (paired t-test, α = 0.05, power = 0.8)

**For spectral analysis:**
- Generally requires fewer trials
- n = 15-25 often sufficient for robust effects

**For machine learning:**
- More is generally better
- Consider: number of features, algorithm complexity
- Cross-validation essential

### Experimental Controls

#### Baseline Measurement
```
Options:
1. Eyes-closed rest (2-3 minutes)
2. Eyes-open rest (2-3 minutes)
3. Simple fixation task
4. Easy version of experimental task
```

#### Counterbalancing
```python
# Latin square for condition order
conditions = ['A', 'B', 'C', 'D']
orders = [
    ['A', 'B', 'C', 'D'],
    ['B', 'C', 'D', 'A'],
    ['C', 'D', 'A', 'B'],
    ['D', 'A', 'B', 'C']
]
# Assign participants to order based on subject number
```

#### Practice Trials
- Always include familiarization
- Exclude practice data from analysis
- Verify understanding before main experiment

### Timing Considerations

**Trial structure:**
```
Fixation (500-1000 ms, jittered)
    ↓
Stimulus (variable or fixed)
    ↓
Response period
    ↓
Inter-trial interval (1000-2000 ms, jittered)
```

**Jittering:**
- Prevents anticipation
- Reduces temporal smearing in averaging
- Helps separate overlapping neural responses

**Session length:**
- EEG setup: 15-30 minutes
- Active recording: 60-90 minutes maximum
- Include breaks every 15-20 minutes
- Watch for fatigue effects

### Data Quality During Experiment

**Online monitoring:**
- Watch for excessive artifacts
- Check impedances periodically
- Monitor participant alertness

**Quality criteria:**
```python
# Post-hoc quality check
reject_participant_if:
    - artifact_rate > 30%
    - impedance_failures > 50%
    - behavioral_accuracy < chance_level
```

### EcoInsight-Specific Design

For our infographic learning study:
```
Structure per trial:
1. Fixation cross (500 ms)
2. Instructional video segment (30-60 s)
3. Response screen (self-paced)
4. Confidence rating (optional)
5. ITI (2000 ms)

Measures:
- Continuous EEG throughout video
- Eye-tracking on infographic
- Behavioral: accuracy, RT
- Subjective: confidence, difficulty

Conditions:
- Information format (visual vs. verbal emphasis)
- Complexity level (low, medium, high)
- Adaptive vs. non-adaptive presentation
```

---

## 13. Common Challenges and Solutions

### Signal Quality Issues

#### High Impedance
| Problem | Cause | Solution |
|---------|-------|----------|
| Poor signal | Dry scalp | Apply conductive gel |
| Noise | Hair obstruction | Part hair, ensure contact |
| Drift | Movement | Re-seat electrode |

#### Artifacts

**Eye blinks:**
```python
# Detection
blink_threshold = 100e-6  # 100 μV at Fp1/Fp2

# Solution 1: Rejection
epochs.drop_bad(reject=dict(eeg=100e-6))

# Solution 2: ICA correction
ica.exclude = eog_inds  # Identified EOG components
```

**Muscle artifacts:**
- Filter aggressively (< 30 Hz) if focusing on low frequencies
- ICA can help but may remove signal
- Instruct participants to relax jaw, shoulders

**Movement:**
- Use chin rest or stabilization when possible
- Mark/exclude movement periods
- Consider accelerometer data (Enobio has built-in)

### Individual Differences

#### Baseline Variability
**Problem:** Large individual differences in absolute power

**Solution:** 
```python
# Normalization approaches
# 1. Percent change from baseline
change = (task - baseline) / baseline * 100

# 2. Z-score normalization
z_score = (value - individual_mean) / individual_std

# 3. Log transformation
log_power = np.log10(power)
```

#### Alpha Reactivity
**Problem:** Some individuals show weak alpha

**Solution:**
- Use relative power
- Consider individual alpha frequency (IAF)
- May need to exclude non-responders

### Statistical Issues

#### Multiple Comparisons
**Problem:** Many channels × time points × frequency bins

**Solutions:**
```python
# 1. Cluster-based permutation testing
from mne.stats import permutation_cluster_test

# 2. FDR correction
from statsmodels.stats.multitest import fdrcorrection

# 3. A priori ROI selection
# Only analyze theoretically motivated channels/windows
```

#### Non-Normal Data
**Problem:** Power values often skewed

**Solutions:**
```python
# Log transformation
log_power = np.log10(power)

# Non-parametric tests
from scipy.stats import wilcoxon, kruskal
```

### Real-Time Processing Challenges

**Latency requirements:**
- Neurofeedback: < 100 ms
- Adaptive systems: < 500 ms acceptable
- Our application: 1-2 second windows feasible

**Online artifact handling:**
```python
class OnlineArtifactDetector:
    def __init__(self, threshold=100e-6):
        self.threshold = threshold
        
    def check_epoch(self, epoch):
        if np.max(np.abs(epoch)) > self.threshold:
            return False  # Reject
        return True  # Accept
```

**Computational efficiency:**
- Pre-compute filter coefficients
- Use efficient FFT implementations
- Consider edge computing (Enobio can process locally)

### Cross-Cultural Considerations

For our Armenian/American comparison:
- Ensure equivalent instruction comprehension
- Use culturally neutral stimuli where possible
- Control for language differences in verbalization
- Be aware of potential differences in EEG baseline (less documented)

---

## 14. Software Tools and Libraries

### Python Ecosystem

#### MNE-Python (Primary Recommendation)
```bash
pip install mne
```
- Comprehensive EEG/MEG analysis
- Well-documented
- Active community
- Supports: preprocessing, epoching, spectral analysis, connectivity, source localization

```python
import mne

# Load data
raw = mne.io.read_raw_edf('data.edf', preload=True)

# Filter
raw.filter(l_freq=0.5, h_freq=50)

# Epoch
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8)

# Spectral analysis
spectrum = epochs.compute_psd(method='welch')
```

#### Additional Python Libraries
```bash
pip install scipy numpy pandas scikit-learn
pip install autoreject  # Automated artifact rejection
pip install pyprep      # PREP pipeline implementation
pip install antropy     # Entropy measures
```

### MATLAB Tools

#### EEGLAB
- Most widely used
- Extensive plugin ecosystem
- GUI and scripting
- Good for beginners

#### FieldTrip
- Powerful for advanced analysis
- MEG-focused but excellent for EEG
- Time-frequency, connectivity, source analysis

#### ERPLAB
- EEGLAB extension for ERPs
- Specialized ERP workflows
- Bin-based averaging

### Neuroelectrics Software (Enobio-Specific)

#### NIC2 (Neuroelectrics Instrument Controller)
- Data acquisition
- Real-time visualization
- Basic signal quality monitoring
- Export to standard formats (EDF, .easy)

#### Matnic (MATLAB interface)
```matlab
% Stream data from Enobio to MATLAB
% Real-time access for custom applications
```

### Lab Streaming Layer (LSL)
```bash
pip install pylsl
```
Essential for multi-modal synchronization:
```python
from pylsl import StreamInlet, resolve_stream

# Find EEG stream
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

# Receive data
sample, timestamp = inlet.pull_sample()
```

### Recommended Workflow

**For our lab:**
1. **Acquisition:** NIC2 + LSL
2. **Preprocessing:** MNE-Python
3. **Analysis:** MNE + custom Python scripts
4. **Statistics:** Python (scipy, statsmodels) or R
5. **Machine Learning:** scikit-learn, PyTorch
6. **Visualization:** matplotlib, seaborn, MNE plotting

### Code Repository Structure
```
eeg_analysis/
├── data/
│   ├── raw/
│   ├── preprocessed/
│   └── features/
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_spectral_analysis.ipynb
│   └── 03_classification.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── utils.py
├── config/
│   └── experiment_params.yaml
└── requirements.txt
```

---

## 15. References and Further Reading

### Foundational Texts

**General EEG:**
- Luck, S. J. (2014). *An Introduction to the Event-Related Potential Technique* (2nd ed.). MIT Press.
- Nunez, P. L., & Srinivasan, R. (2006). *Electric Fields of the Brain* (2nd ed.). Oxford University Press.

**Cognitive Neuroscience:**
- Gazzaniga, M. S., Ivry, R. B., & Mangun, G. R. (2018). *Cognitive Neuroscience: The Biology of the Mind* (5th ed.). W.W. Norton.

### Key Papers

**Cognitive Load:**
- Antonenko, P., Paas, F., Grabner, R., & van Gog, T. (2010). Using electroencephalography to measure cognitive load. *Educational Psychology Review, 22*(4), 425-438.
- Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance. *Brain Research Reviews, 29*(2-3), 169-195.
- Gevins, A., & Smith, M. E. (2003). Neurophysiological measures of cognitive workload during human-computer interaction. *Theoretical Issues in Ergonomics Science, 4*(1-2), 113-131.

**Emotion:**
- Davidson, R. J. (2004). What does the prefrontal cortex "do" in affect: Perspectives on frontal EEG asymmetry research. *Biological Psychology, 67*(1-2), 219-234.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161-1178.

**Machine Learning with EEG:**
- Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering, 15*(5), 056013.
- Lotte, F., et al. (2018). A review of classification algorithms for EEG-based brain-computer interfaces. *Journal of Neural Engineering, 15*(3), 031005.

**EEG in HCI:**
- Mills, C., et al. (2017). Put your thinking cap on: Detecting cognitive load using EEG during learning. *Proceedings of LAK*, 80-89.
- Cowley, B., et al. (2016). The psychophysiology primer: A guide to methods and a broad review with a focus on human-computer interaction. *Foundations and Trends in HCI, 9*(3-4), 151-308.

### Online Resources

**Tutorials:**
- MNE-Python tutorials: https://mne.tools/stable/auto_tutorials/
- Mike X Cohen's lectures: https://www.youtube.com/c/mikexcohen1
- EEGLAB wiki: https://eeglab.org/tutorials/

**Datasets:**
- DEAP (emotion): http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- PhysioNet (various): https://physionet.org/
- OpenNeuro (shared datasets): https://openneuro.org/

**Communities:**
- MNE-Python forum: https://mne.discourse.group/
- EEGLAB mailing list
- NeuroStars: https://neurostars.org/

---

## Appendix A: Quick Reference Card

### Frequency Bands
| Band | Range | ↑ Indicates | ↓ Indicates |
|------|-------|-------------|-------------|
| Theta (4-8 Hz) | Memory, load | Working memory ↑ | Disengagement |
| Alpha (8-13 Hz) | Attention | Relaxation | Processing ↑ |
| Beta (13-30 Hz) | Focus | Active thinking | Drowsiness |

### Key Metrics Formulas
```
Engagement Index = Beta / (Alpha + Theta)
Cognitive Load Index = Frontal Theta / Parietal Alpha
Frontal Asymmetry = ln(F4_alpha) - ln(F3_alpha)
```

### Electrode Locations (10-20)
```
        Fp1  Fp2
      F7  F3 Fz F4  F8
         C3 Cz C4
      P7  P3 Pz P4  P8
           O1 O2
```

### Typical Effect Sizes
| Measure | Effect Size (d) |
|---------|-----------------|
| Cognitive load (theta) | 0.5-0.8 |
| Alpha desynchronization | 0.6-1.0 |
| Frontal asymmetry (emotion) | 0.3-0.5 |
| P300 amplitude | 0.5-0.7 |

---

*Document prepared for the HCI + L Laboratory, Yerevan State University*  
*For questions, contact the research team at the laboratory*
