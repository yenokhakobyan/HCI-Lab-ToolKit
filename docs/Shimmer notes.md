# Shimmer Sensors in Human-Computer Interaction: Technical Reference Guide

**HCI + L Laboratory | Yerevan State University**  
**Technical Notes for Research Team**

---

## Table of Contents

1. [Introduction to Shimmer Sensors](#1-introduction-to-shimmer-sensors)
2. [Shimmer Hardware Overview](#2-shimmer-hardware-overview)
3. [Physiological Signals and Measurements](#3-physiological-signals-and-measurements)
4. [Galvanic Skin Response (GSR/EDA)](#4-galvanic-skin-response-gsreda)
5. [Photoplethysmography (PPG) and Heart Rate](#5-photoplethysmography-ppg-and-heart-rate)
6. [Heart Rate Variability (HRV)](#6-heart-rate-variability-hrv)
7. [Electrocardiography (ECG)](#7-electrocardiography-ecg)
8. [Electromyography (EMG)](#8-electromyography-emg)
9. [Motion and Inertial Sensing](#9-motion-and-inertial-sensing)
10. [Data Collection and Synchronization](#10-data-collection-and-synchronization)
11. [Signal Processing and Analysis](#11-signal-processing-and-analysis)
12. [Experimental Design Considerations](#12-experimental-design-considerations)
13. [Software Tools and Platforms](#13-software-tools-and-platforms)
14. [Theoretical Models and Frameworks](#14-theoretical-models-and-frameworks)
15. [Applications in HCI Research](#15-applications-in-hci-research)
16. [Multimodal Integration](#16-multimodal-integration)
17. [Noteworthy Papers and Studies](#17-noteworthy-papers-and-studies)
18. [Important Datasets](#18-important-datasets)
19. [Analysis Approaches and Methods](#19-analysis-approaches-and-methods)
20. [Notable Projects and Research Initiatives](#20-notable-projects-and-research-initiatives)
21. [Emerging Trends and Future Directions](#21-emerging-trends-and-future-directions)
22. [References and Further Reading](#22-references-and-further-reading)

---

## 1. Introduction to Shimmer Sensors

### 1.1 What is Shimmer?

Shimmer is a wearable sensor platform designed for research-grade physiological and kinematic data collection. Developed by Shimmer Research (Dublin, Ireland), these wireless sensors enable real-time monitoring of various biosignals in naturalistic settings, making them ideal for HCI research where ecological validity is important.

### 1.2 Why Shimmer for HCI Research?

**Advantages:**
- **Wearable and wireless**: Unobtrusive data collection
- **Multi-signal capability**: Simultaneous capture of multiple physiological signals
- **Research-grade quality**: Validated against clinical standards
- **Real-time streaming**: Low-latency data transmission
- **Synchronization**: Hardware synchronization across multiple units
- **SDK availability**: Integration with various programming environments
- **Ecological validity**: Data collection outside laboratory settings

**Key Applications in HCI:**
- Measuring cognitive load during interface interaction
- Detecting emotional responses to content
- Assessing stress and arousal in user studies
- Evaluating user engagement and attention
- Studying affective computing applications
- Validating physiological indicators of UX

### 1.3 Physiological Computing Framework

Shimmer sensors support the physiological computing paradigm:

```
User Interaction → Physiological Response → Signal Acquisition → 
Feature Extraction → State Classification → Adaptive Response
```

**Measured Constructs:**
| Construct | Primary Signals | Secondary Indicators |
|-----------|-----------------|---------------------|
| **Arousal** | GSR, HR | Pupil dilation, respiration |
| **Valence** | HRV, facial EMG | Skin temperature |
| **Cognitive Load** | HRV, GSR | EEG (with other systems) |
| **Stress** | GSR, HR, HRV | Cortisol (invasive) |
| **Engagement** | HR, GSR | Motion, posture |
| **Fatigue** | HRV, EMG | Blink rate, posture |

---

## 2. Shimmer Hardware Overview

### 2.1 Shimmer3 Platform

The Shimmer3 is the current generation platform, offering modular sensor configurations.

**Base Unit Specifications:**
| Specification | Value |
|--------------|-------|
| Dimensions | 51mm × 34mm × 14mm |
| Weight | 23.6g (with battery) |
| Processor | MSP430 (24 MHz) |
| Memory | 8MB onboard flash |
| Battery | 450mAh Li-ion |
| Battery Life | 8+ hours (typical use) |
| Wireless | Bluetooth 2.1 + EDR, 802.15.4 |
| Sampling Rate | Up to 1024 Hz (sensor dependent) |

### 2.2 Available Sensor Modules

| Module | Signals | Sampling Rate | Resolution |
|--------|---------|---------------|------------|
| **GSR+ Unit** | GSR, PPG, skin temp | 512 Hz max | 12-bit |
| **ECG Unit** | ECG (single lead) | 512 Hz | 24-bit |
| **EMG Unit** | EMG | 1024 Hz | 24-bit |
| **ExG Unit** | ECG, EMG, EEG, EOG | 1024 Hz | 24-bit |
| **IMU** | Accel, Gyro, Mag | 512 Hz | 16-bit |
| **Bridge Amplifier** | Strain, force | 1024 Hz | 24-bit |
| **Pressure** | Barometric | 50 Hz | 24-bit |

### 2.3 Consensys GSR Development Kit

**Our Lab Configuration - Components:**
1. Shimmer3 GSR+ Unit
2. Optical pulse probe (PPG)
3. GSR electrodes (finger straps)
4. Consensys software license
5. Charging dock
6. SD card for local storage

**GSR+ Unit Capabilities:**
- Galvanic Skin Response (GSR/EDA)
- Photoplethysmography (PPG)
- Continuous Heart Rate
- Heart Rate Variability (derived)
- Skin Temperature
- 3-axis accelerometer
- 3-axis gyroscope
- 3-axis magnetometer
- Integrated altimeter

### 2.4 Connectivity Options

**Bluetooth Streaming:**
- Range: ~10 meters (line of sight)
- Latency: ~50-100 ms
- Multiple device support: Up to 7 units

**SD Card Logging:**
- Standalone operation
- Higher sampling rates possible
- Post-session data transfer
- Backup for wireless dropouts

**Wireless Protocols:**
| Protocol | Use Case | Bandwidth |
|----------|----------|-----------|
| Bluetooth Classic | Single device streaming | 3 Mbps |
| Bluetooth LE | Multi-device, low power | 1 Mbps |
| 802.15.4 | Sensor networks | 250 kbps |

### 2.5 Electrode and Sensor Placement

**GSR Electrodes:**
```
Recommended: Palmar surface of distal phalanges
- Index and middle fingers (non-dominant hand)
- Alternatively: Thenar/hypothenar eminences

Preparation:
1. Clean skin with alcohol wipe
2. Allow to dry completely
3. Apply electrode gel if using dry electrodes
4. Secure with Velcro straps
5. Verify skin contact impedance < 50kΩ
```

**PPG Sensor:**
```
Placement options:
- Earlobe (most common for Shimmer)
- Fingertip
- Wrist (with appropriate clip)

Considerations:
- Avoid movement artifacts
- Ensure good skin contact
- Avoid ambient light interference
```

---

## 3. Physiological Signals and Measurements

### 3.1 Overview of Biosignals

| Signal | Origin | Frequency Range | Amplitude | Information |
|--------|--------|-----------------|-----------|-------------|
| **GSR/EDA** | Eccrine sweat glands | 0-5 Hz | 1-20 µS | Sympathetic arousal |
| **PPG** | Blood volume changes | 0.5-4 Hz | Variable | Heart rate, HRV |
| **ECG** | Cardiac electrical | 0.05-100 Hz | 0.5-2 mV | Heart rhythm, HRV |
| **EMG** | Muscle electrical | 10-500 Hz | 0.01-10 mV | Muscle activity |
| **Skin Temp** | Thermoreceptors | DC-0.1 Hz | 25-40°C | Thermoregulation |
| **Acceleration** | Body movement | 0-20 Hz | ±16 g | Motion, posture |

### 3.2 Autonomic Nervous System Basis

**Sympathetic Nervous System (SNS):**
- "Fight or flight" response
- Increases HR, GSR
- Decreases HRV
- Associated with arousal, stress, cognitive load

**Parasympathetic Nervous System (PNS):**
- "Rest and digest" response
- Decreases HR
- Increases HRV
- Associated with relaxation, recovery

**SNS/PNS Balance in HCI:**
| State | SNS Activity | PNS Activity | Observable |
|-------|--------------|--------------|------------|
| Relaxed | Low | High | Low GSR, High HRV |
| Engaged | Moderate | Moderate | Moderate GSR |
| Stressed | High | Low | High GSR, Low HRV, High HR |
| Cognitive Load | High | Low | GSR peaks, HRV decrease |
| Boredom | Low | Variable | Low GSR, variable HR |

### 3.3 Signal Quality Considerations

**Factors Affecting Signal Quality:**
| Factor | Affected Signals | Mitigation |
|--------|------------------|------------|
| Movement | All, especially PPG | Motion filtering, artifact rejection |
| Electrode contact | GSR, ECG, EMG | Proper preparation, impedance check |
| Ambient temperature | GSR, skin temp | Climate control, acclimatization |
| Caffeine/substances | HR, HRV, GSR | Participant screening |
| Time of day | HRV, GSR baseline | Consistent scheduling |
| Hydration | GSR | Participant instructions |

---

## 4. Galvanic Skin Response (GSR/EDA)

### 4.1 Fundamentals

**Terminology:**
- **GSR**: Galvanic Skin Response (older term)
- **EDA**: Electrodermal Activity (preferred term)
- **SCL**: Skin Conductance Level (tonic component)
- **SCR**: Skin Conductance Response (phasic component)
- **EDR**: Electrodermal Response

**Physiological Basis:**
- Eccrine sweat glands controlled by sympathetic nervous system
- Sweat contains electrolytes → changes skin conductance
- Palmar and plantar surfaces have highest density
- Response latency: 1-5 seconds after stimulus

### 4.2 Components of EDA

```
EDA Signal
    │
    ├── Tonic Component (SCL)
    │   └── Slow-varying baseline (0.01-0.05 Hz)
    │       - Overall arousal level
    │       - Individual differences (2-20 µS)
    │       - Gradual drift over time
    │
    └── Phasic Component (SCR)
        └── Rapid fluctuations (0.05-5 Hz)
            - Event-related responses
            - Amplitude: 0.01-1 µS
            - Rise time: 1-3 seconds
            - Recovery time: 2-10 seconds
```

### 4.3 SCR Parameters

| Parameter | Definition | Typical Values |
|-----------|------------|----------------|
| **Amplitude** | Peak height from onset | 0.01-5 µS |
| **Latency** | Time from stimulus to onset | 1-5 s |
| **Rise Time** | Onset to peak | 1-3 s |
| **Half Recovery** | Peak to 50% recovery | 2-10 s |
| **SCR Frequency** | Number of SCRs per minute | 1-20/min |

### 4.4 EDA Metrics for HCI

**Tonic Metrics:**
| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| **Mean SCL** | Average conductance | Overall arousal |
| **SCL Range** | Max - Min | Arousal variability |
| **SCL Slope** | Linear trend | Arousal trajectory |

**Phasic Metrics:**
| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| **SCR Count** | Number of responses | Response frequency |
| **SCR Amplitude** | Mean/max peak height | Response intensity |
| **SCR Sum** | Total amplitude | Cumulative arousal |
| **NS-SCR Rate** | Non-specific SCRs/min | Spontaneous arousal |

### 4.5 EDA Decomposition Methods

**Continuous Decomposition Analysis (CDA) - Benedek & Kaernbach (2010):**
- Deconvolution-based approach
- Separates tonic and phasic components
- Provides driver signal (sudomotor nerve activity)

**cvxEDA - Greco et al. (2016):**
- Convex optimization approach
- Sparse recovery of phasic component
- Robust to noise and artifacts

**Ledalab:**
- MATLAB toolbox for EDA analysis
- Multiple decomposition methods
- Widely used in research

### 4.6 EDA Processing Pipeline

```python
# Typical EDA processing steps
1. Load raw data (µS)
2. Low-pass filter (cutoff: 1-5 Hz)
3. Downsample if needed (target: 4-10 Hz)
4. Artifact detection and removal
5. Decomposition (tonic/phasic)
6. SCR detection
7. Feature extraction
8. Normalization (z-scores or range)
9. Statistical analysis
```

### 4.7 EDA and Psychological States

| State | SCL Pattern | SCR Pattern |
|-------|-------------|-------------|
| **Relaxation** | Low, stable | Few, small |
| **Attention** | Moderate | Stimulus-locked |
| **Stress** | High, rising | Frequent, large |
| **Cognitive effort** | Elevated | Task-related peaks |
| **Emotional arousal** | Variable | Large, clustered |
| **Habituation** | Decreasing | Diminishing amplitude |

---

## 5. Photoplethysmography (PPG) and Heart Rate

### 5.1 PPG Fundamentals

**Principle:**
- Light absorption by blood varies with pulse
- LED illuminates tissue
- Photodetector measures reflected/transmitted light
- Pulsatile component reflects blood volume changes

**Shimmer PPG Specifications:**
| Parameter | Value |
|-----------|-------|
| Wavelength | Green (525nm) or IR (940nm) |
| Sampling Rate | Up to 512 Hz |
| Resolution | 12-bit |
| Placement | Earlobe, fingertip |

### 5.2 PPG Signal Components

```
PPG Waveform
    │
    ├── AC Component (pulsatile)
    │   └── Cardiac-synchronous oscillations
    │       - Systolic peak
    │       - Diastolic peak/notch
    │       - Heart rate extraction
    │
    └── DC Component (baseline)
        └── Slow variations
            - Respiration effects
            - Vasomotion
            - Movement artifacts
```

### 5.3 Heart Rate Extraction from PPG

**Peak Detection Algorithm:**
```python
def extract_heart_rate(ppg_signal, fs):
    """
    Extract heart rate from PPG signal
    
    Parameters:
    - ppg_signal: raw PPG data
    - fs: sampling frequency (Hz)
    
    Returns:
    - heart_rate: BPM
    - ibi: inter-beat intervals (ms)
    """
    # 1. Bandpass filter (0.5-4 Hz)
    filtered = bandpass_filter(ppg_signal, 0.5, 4, fs)
    
    # 2. Find peaks
    peaks = find_peaks(filtered, distance=fs*0.5)  # min 0.5s between beats
    
    # 3. Calculate inter-beat intervals
    ibi = np.diff(peaks) / fs * 1000  # in ms
    
    # 4. Calculate heart rate
    heart_rate = 60000 / np.mean(ibi)  # BPM
    
    return heart_rate, ibi
```

### 5.4 Heart Rate Metrics

| Metric | Calculation | Normal Range | Interpretation |
|--------|-------------|--------------|----------------|
| **Mean HR** | 60000/mean(IBI) | 60-100 BPM | Average cardiac activity |
| **Max HR** | Maximum observed | Variable | Peak arousal |
| **Min HR** | Minimum observed | Variable | Rest state |
| **HR Range** | Max - Min | 10-30 BPM | Variability |
| **HR Reactivity** | Task HR - Baseline HR | Variable | Task response |

### 5.5 PPG Quality Assessment

**Quality Indicators:**
| Indicator | Good Quality | Poor Quality |
|-----------|--------------|--------------|
| **Signal amplitude** | Consistent | Varying greatly |
| **Peak clarity** | Sharp, distinct | Flat, noisy |
| **Baseline** | Stable | Drifting |
| **IBI consistency** | Regular (± 10%) | Irregular |

**Common Artifacts:**
- Movement (large baseline shifts)
- Loose sensor (signal dropout)
- Ambient light (periodic interference)
- Vasoconstriction (reduced amplitude)

---

## 6. Heart Rate Variability (HRV)

### 6.1 What is HRV?

Heart Rate Variability refers to the variation in time intervals between consecutive heartbeats. It reflects the dynamic interplay between sympathetic and parasympathetic nervous system activity.

**Why HRV Matters for HCI:**
- Non-invasive autonomic nervous system probe
- Indicator of cognitive load and stress
- Correlates with emotional states
- Predictive of mental workload
- Real-time monitoring capability

### 6.2 HRV Time-Domain Metrics

| Metric | Full Name | Calculation | Interpretation |
|--------|-----------|-------------|----------------|
| **SDNN** | SD of NN intervals | std(IBI) | Overall HRV |
| **RMSSD** | Root mean square of successive differences | sqrt(mean(diff(IBI)²)) | Parasympathetic activity |
| **pNN50** | % of successive differences > 50ms | count(|diff| > 50) / N | Parasympathetic activity |
| **pNN20** | % of successive differences > 20ms | count(|diff| > 20) / N | More sensitive than pNN50 |
| **SDSD** | SD of successive differences | std(diff(IBI)) | Short-term variability |
| **Mean HR** | Mean heart rate | 60000/mean(IBI) | Average activity |

**Minimum Recording Duration:**
| Metric | Minimum Duration | Recommended |
|--------|------------------|-------------|
| RMSSD | 1 minute | 5 minutes |
| SDNN | 5 minutes | 24 hours for comparison |
| pNN50 | 2 minutes | 5 minutes |

### 6.3 HRV Frequency-Domain Metrics

**Power Spectral Density Analysis:**

| Band | Frequency Range | Physiological Origin | Interpretation |
|------|-----------------|---------------------|----------------|
| **ULF** | < 0.003 Hz | Circadian rhythms | Long-term (24h) |
| **VLF** | 0.003-0.04 Hz | Thermoregulation, hormones | Longer-term regulation |
| **LF** | 0.04-0.15 Hz | Both SNS and PNS | Mixed activity |
| **HF** | 0.15-0.4 Hz | Parasympathetic (vagal) | Respiratory sinus arrhythmia |

**Derived Metrics:**
| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| **LF Power** | Integral 0.04-0.15 Hz | Sympathetic + Parasympathetic |
| **HF Power** | Integral 0.15-0.4 Hz | Parasympathetic (vagal) |
| **LF/HF Ratio** | LF Power / HF Power | Sympathovagal balance |
| **Total Power** | Integral 0-0.4 Hz | Overall HRV |
| **Normalized LF** | LF / (LF + HF) × 100 | Relative sympathetic |
| **Normalized HF** | HF / (LF + HF) × 100 | Relative parasympathetic |

### 6.4 HRV Nonlinear Metrics

| Metric | Method | Interpretation |
|--------|--------|----------------|
| **SD1** | Poincaré plot | Short-term variability |
| **SD2** | Poincaré plot | Long-term variability |
| **SD1/SD2** | Ratio | Balance of variabilities |
| **SampEn** | Sample entropy | Signal complexity |
| **ApEn** | Approximate entropy | Regularity |
| **DFA α1** | Detrended fluctuation | Short-term correlations |
| **DFA α2** | Detrended fluctuation | Long-term correlations |

### 6.5 HRV and Mental States

| Mental State | RMSSD | LF/HF | HF Power |
|--------------|-------|-------|----------|
| **Relaxation** | High | Low | High |
| **Mental stress** | Low | High | Low |
| **Cognitive load** | Decreased | Increased | Decreased |
| **Emotional arousal** | Variable | Increased | Decreased |
| **Fatigue** | Decreased | Variable | Decreased |
| **Flow state** | Moderate | Balanced | Moderate |

### 6.6 HRV Analysis Pipeline

```python
import numpy as np
from scipy import signal

def analyze_hrv(ibi_ms, fs_resample=4):
    """
    Comprehensive HRV analysis
    
    Parameters:
    - ibi_ms: inter-beat intervals in milliseconds
    - fs_resample: resampling frequency for spectral analysis
    
    Returns:
    - dict: HRV metrics
    """
    metrics = {}
    
    # Time-domain metrics
    metrics['mean_hr'] = 60000 / np.mean(ibi_ms)
    metrics['sdnn'] = np.std(ibi_ms)
    metrics['rmssd'] = np.sqrt(np.mean(np.diff(ibi_ms)**2))
    
    successive_diff = np.abs(np.diff(ibi_ms))
    metrics['pnn50'] = np.sum(successive_diff > 50) / len(successive_diff) * 100
    metrics['pnn20'] = np.sum(successive_diff > 20) / len(successive_diff) * 100
    
    # Frequency-domain metrics
    # Resample to uniform intervals
    time = np.cumsum(ibi_ms) / 1000  # seconds
    time_uniform = np.arange(time[0], time[-1], 1/fs_resample)
    ibi_resampled = np.interp(time_uniform, time, ibi_ms)
    
    # Compute PSD using Welch's method
    freqs, psd = signal.welch(ibi_resampled, fs=fs_resample, nperseg=256)
    
    # Band powers
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    
    metrics['vlf_power'] = np.trapz(psd[vlf_mask], freqs[vlf_mask])
    metrics['lf_power'] = np.trapz(psd[lf_mask], freqs[lf_mask])
    metrics['hf_power'] = np.trapz(psd[hf_mask], freqs[hf_mask])
    metrics['lf_hf_ratio'] = metrics['lf_power'] / metrics['hf_power']
    
    return metrics
```

### 6.7 HRV Considerations for Short Recordings

**Ultra-Short HRV (< 5 minutes):**
- Use time-domain metrics (RMSSD preferred)
- Avoid frequency-domain below 2 minutes
- Report recording duration
- Consider task-specific validation

**Recommendations:**
| Duration | Reliable Metrics |
|----------|------------------|
| 30 seconds | Mean HR only |
| 1 minute | RMSSD, Mean HR |
| 2 minutes | RMSSD, pNN50 |
| 5 minutes | All time-domain, frequency-domain |
| 24 hours | All metrics, circadian patterns |

---

## 7. Electrocardiography (ECG)

### 7.1 ECG Fundamentals

**What ECG Measures:**
- Electrical activity of the heart
- More accurate than PPG for HRV
- Provides morphological information
- Gold standard for cardiac timing

**Shimmer ECG Specifications:**
| Parameter | Value |
|-----------|-------|
| Resolution | 24-bit |
| Sampling Rate | Up to 512 Hz |
| Input Range | ±40 mV |
| CMRR | > 80 dB |
| Input Impedance | > 10 GΩ |

### 7.2 ECG Waveform Components

```
        R
        /\
       /  \
      /    \
     /      \
----P--Q    S----T--------
    
P wave: Atrial depolarization
QRS complex: Ventricular depolarization
T wave: Ventricular repolarization
```

**Key Intervals:**
| Interval | Normal Range | Significance |
|----------|--------------|--------------|
| **RR** | 600-1000 ms | Heart rate, HRV |
| **PR** | 120-200 ms | AV conduction |
| **QRS** | 80-120 ms | Ventricular depolarization |
| **QT** | 350-450 ms | Total ventricular activity |

### 7.3 ECG Electrode Placement

**Single Lead (Shimmer Configuration):**
```
Lead II Configuration:
- RA (Right Arm): Right clavicle
- LA (Left Arm): Left clavicle  
- RL (Right Leg/Ground): Right lower rib

Alternative placements:
- Modified chest leads for reduced motion artifacts
- Ensure good skin contact
- Use conductive gel
```

### 7.4 R-Peak Detection Algorithms

**Pan-Tompkins Algorithm (1985):**
```python
def pan_tompkins(ecg, fs):
    """
    Pan-Tompkins QRS detection algorithm
    """
    # 1. Bandpass filter (5-15 Hz)
    filtered = bandpass_filter(ecg, 5, 15, fs)
    
    # 2. Derivative
    derivative = np.diff(filtered)
    
    # 3. Squaring
    squared = derivative ** 2
    
    # 4. Moving window integration
    window = int(0.150 * fs)  # 150 ms window
    integrated = np.convolve(squared, np.ones(window)/window, mode='same')
    
    # 5. Adaptive thresholding
    threshold = np.mean(integrated) * 0.6
    peaks = find_peaks(integrated, height=threshold, distance=fs*0.3)
    
    return peaks
```

**Other Algorithms:**
| Algorithm | Approach | Performance |
|-----------|----------|-------------|
| **Hamilton-Tompkins** | Modified Pan-Tompkins | Better for noise |
| **Wavelet-based** | Multi-scale analysis | Robust to artifacts |
| **Template matching** | Cross-correlation | Good for known morphology |
| **Machine learning** | CNN, LSTM | State-of-the-art |

### 7.5 ECG vs. PPG for HRV

| Aspect | ECG | PPG |
|--------|-----|-----|
| **Accuracy** | Gold standard | Good approximation |
| **R-peak precision** | ± 1-2 ms | ± 10-20 ms |
| **Setup complexity** | Moderate (electrodes) | Simple (clip) |
| **Motion artifacts** | Moderate | High |
| **Comfort** | Less comfortable | More comfortable |
| **HRV validity** | Full validity | Valid for most metrics |
| **Cost** | Higher | Lower |

---

## 8. Electromyography (EMG)

### 8.1 EMG Fundamentals

**What EMG Measures:**
- Electrical activity of muscles
- Motor unit action potentials
- Muscle activation patterns
- Force/tension indicators

**Applications in HCI:**
- Facial EMG for emotion detection
- Forearm EMG for gesture recognition
- Muscle fatigue assessment
- Stress-related muscle tension

### 8.2 Shimmer EMG Specifications

| Parameter | Value |
|-----------|-------|
| Resolution | 24-bit |
| Sampling Rate | Up to 1024 Hz |
| Input Range | ±10 mV |
| Bandwidth | 10-500 Hz |
| CMRR | > 80 dB |

### 8.3 EMG Signal Characteristics

**Raw EMG:**
- Appears as noise-like signal
- Amplitude: 0.01-10 mV (surface)
- Frequency content: 10-500 Hz
- Stochastic in nature

**Processing Steps:**
```
Raw EMG → Bandpass Filter → Rectification → Smoothing → Feature Extraction
```

### 8.4 Facial EMG for Emotion

**Key Muscle Sites:**

| Muscle | Location | Emotional Association |
|--------|----------|----------------------|
| **Corrugator supercilii** | Above eyebrow | Negative emotions, frowning |
| **Zygomaticus major** | Cheek to mouth | Positive emotions, smiling |
| **Orbicularis oculi** | Around eye | Genuine smiling (Duchenne) |
| **Frontalis** | Forehead | Surprise, concentration |
| **Masseter** | Jaw | Tension, stress |

**Electrode Placement (Facial):**
```
Corrugator: Two electrodes above eyebrow, 
            ~2cm apart along muscle

Zygomaticus: Two electrodes from cheekbone 
             to corner of mouth

Ground: Forehead, mastoid, or earlobe
```

### 8.5 EMG Metrics

| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| **Mean Amplitude** | Mean of rectified signal | Overall activation |
| **RMS** | Root mean square | Power-related activation |
| **Integrated EMG (iEMG)** | Area under rectified curve | Total activation |
| **Peak Amplitude** | Maximum value | Peak activation |
| **Mean Frequency** | Spectral centroid | Fatigue indicator |
| **Median Frequency** | Median of power spectrum | Fatigue indicator |

### 8.6 EMG Processing

```python
def process_emg(emg_raw, fs):
    """
    Basic EMG processing pipeline
    """
    # 1. Bandpass filter (20-450 Hz)
    filtered = bandpass_filter(emg_raw, 20, 450, fs)
    
    # 2. Notch filter for power line (50/60 Hz)
    notched = notch_filter(filtered, 50, fs)  # or 60 Hz
    
    # 3. Full-wave rectification
    rectified = np.abs(notched)
    
    # 4. Smoothing (moving average or low-pass)
    window = int(0.050 * fs)  # 50 ms window
    smoothed = moving_average(rectified, window)
    
    # 5. Compute RMS in windows
    rms = compute_rms(notched, window_size=int(0.100 * fs))
    
    return smoothed, rms
```

---

## 9. Motion and Inertial Sensing

### 9.1 IMU Components

**Shimmer3 IMU Specifications:**

| Sensor | Range | Resolution | Sampling |
|--------|-------|------------|----------|
| **Accelerometer** | ±2/4/8/16 g | 16-bit | 512 Hz |
| **Gyroscope** | ±250/500/1000/2000 °/s | 16-bit | 512 Hz |
| **Magnetometer** | ±1.3/1.9/2.5/4.0/4.7/5.6/8.1 Ga | 13-bit | 100 Hz |
| **Altimeter** | 300-1100 hPa | 24-bit | 50 Hz |

### 9.2 Motion Metrics

| Metric | Calculation | Application |
|--------|-------------|-------------|
| **Activity counts** | Integrated acceleration | Activity level |
| **Movement intensity** | RMS of acceleration | Vigor of movement |
| **Orientation** | Sensor fusion | Posture |
| **Step count** | Peak detection | Locomotion |
| **Sedentary time** | Low activity periods | Behavior |

### 9.3 Posture and Movement in HCI

**Relevant Behaviors:**
- Leaning forward (engagement)
- Fidgeting (stress, boredom)
- Head movements (attention)
- Gesture recognition
- Activity classification

### 9.4 Motion Artifact Detection

Motion data helps identify artifacts in physiological signals:

```python
def detect_motion_artifact(accel, threshold=0.5):
    """
    Detect periods of excessive motion
    
    Parameters:
    - accel: acceleration magnitude
    - threshold: g threshold for artifact
    
    Returns:
    - artifact_mask: boolean array
    """
    # Compute magnitude
    magnitude = np.sqrt(np.sum(accel**2, axis=1))
    
    # Remove gravity
    magnitude_detrended = magnitude - np.median(magnitude)
    
    # Detect high motion periods
    artifact_mask = np.abs(magnitude_detrended) > threshold
    
    # Expand artifact windows (±500ms)
    kernel = np.ones(int(0.5 * fs))
    artifact_mask = np.convolve(artifact_mask, kernel, mode='same') > 0
    
    return artifact_mask
```

---

## 10. Data Collection and Synchronization

### 10.1 Consensys Software

**Features:**
- Multi-device management
- Real-time visualization
- Event marking
- Data export (CSV, MATLAB)
- Calibration tools

**Workflow:**
1. Connect Shimmer devices
2. Configure sensors and sampling rates
3. Start synchronized recording
4. Mark events during session
5. Stop and export data

### 10.2 Synchronization Methods

**Internal Synchronization:**
- Hardware sync between Shimmer units
- < 1 ms accuracy
- Uses 802.15.4 radio

**External Synchronization:**
- TTL trigger input
- Event markers via software
- Timestamp alignment post-hoc

**With Eye Tracker:**
```python
# Example: Synchronize Shimmer with EyeLink
# Method 1: TTL triggers
# - Send TTL pulse at trial start
# - Record in both systems

# Method 2: Timestamp alignment
# - Record system time at start
# - Align using common events
# - Interpolate to common time base

def synchronize_data(shimmer_data, shimmer_times, 
                     eyetracker_data, eyetracker_times,
                     common_events):
    """
    Align data from multiple sources
    """
    # Find common event times
    shimmer_event_times = find_events(shimmer_data, common_events)
    et_event_times = find_events(eyetracker_data, common_events)
    
    # Compute time offset
    offset = np.mean(shimmer_event_times - et_event_times)
    
    # Apply correction
    shimmer_times_aligned = shimmer_times - offset
    
    # Resample to common timeline
    common_timeline = eyetracker_times
    shimmer_resampled = resample(shimmer_data, shimmer_times_aligned, 
                                  common_timeline)
    
    return shimmer_resampled, eyetracker_data, common_timeline
```

### 10.3 Data Quality Monitoring

**Real-Time Checks:**
- Signal amplitude within range
- Electrode impedance acceptable
- No prolonged dropouts
- Motion within limits

**Post-Hoc Quality Assessment:**
| Check | Criterion | Action |
|-------|-----------|--------|
| **Missing data** | < 5% | Interpolate or exclude |
| **Artifacts** | < 10% | Mark and exclude |
| **Baseline drift** | Minimal | Detrend |
| **Noise level** | SNR > 10 dB | Filter or exclude |

### 10.4 Data Export Formats

**Consensys Export Options:**
| Format | Use Case |
|--------|----------|
| CSV | General analysis, Python/R |
| MATLAB (.mat) | MATLAB analysis |
| EDF | Clinical compatibility |
| Custom binary | Efficient storage |

**Recommended Data Structure:**
```
session_001/
├── raw/
│   ├── shimmer_gsr.csv
│   ├── shimmer_ppg.csv
│   └── shimmer_accel.csv
├── processed/
│   ├── eda_features.csv
│   ├── hrv_features.csv
│   └── motion_features.csv
├── events/
│   └── markers.csv
└── metadata.json
```

---

## 11. Signal Processing and Analysis

### 11.1 Preprocessing Pipeline

```
Raw Signal
    │
    ├── 1. Import and validate
    │
    ├── 2. Artifact detection
    │   ├── Motion artifacts
    │   ├── Electrode artifacts
    │   └── Signal dropouts
    │
    ├── 3. Artifact handling
    │   ├── Interpolation (short gaps)
    │   ├── Exclusion (long gaps)
    │   └── Filtering (noise)
    │
    ├── 4. Filtering
    │   ├── High-pass (baseline removal)
    │   ├── Low-pass (noise removal)
    │   └── Notch (power line)
    │
    ├── 5. Segmentation
    │   ├── Trial-based
    │   ├── Event-locked
    │   └── Sliding window
    │
    └── 6. Feature extraction
```

### 11.2 Common Filters

| Filter Type | Cutoff | Application |
|-------------|--------|-------------|
| **High-pass** | 0.05 Hz | EDA baseline drift |
| **Low-pass** | 1-5 Hz | EDA smoothing |
| **Low-pass** | 35 Hz | PPG/ECG cleanup |
| **Bandpass** | 0.5-4 Hz | PPG cardiac component |
| **Bandpass** | 20-450 Hz | EMG |
| **Notch** | 50/60 Hz | Power line removal |

### 11.3 Python Libraries for Analysis

```python
# Core libraries
import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats

# Specialized biosignal libraries
import neurokit2 as nk  # Comprehensive biosignal processing
import heartpy as hp   # Heart rate analysis
import pyhrv           # HRV analysis
import cvxEDA          # EDA decomposition
import biosppy         # Biosignal processing

# Example: NeuroKit2 workflow
import neurokit2 as nk

# Process ECG
ecg_signals, info = nk.ecg_process(ecg_data, sampling_rate=512)
hrv_indices = nk.hrv(ecg_signals, sampling_rate=512)

# Process EDA
eda_signals, info = nk.eda_process(eda_data, sampling_rate=128)
scr_peaks = info["SCR_Peaks"]

# Process PPG
ppg_signals, info = nk.ppg_process(ppg_data, sampling_rate=128)
```

### 11.4 Feature Extraction Windows

| Signal | Window Size | Overlap | Rationale |
|--------|-------------|---------|-----------|
| **EDA (SCL)** | 30-60 s | 50% | Tonic changes are slow |
| **EDA (SCR)** | Event-locked | N/A | Phasic responses |
| **HR** | 10-30 s | 50% | Beat-to-beat averaging |
| **HRV** | 60-300 s | 50% | Frequency resolution |
| **EMG** | 100-500 ms | 50% | Rapid changes |
| **Motion** | 1-10 s | 50% | Activity patterns |

### 11.5 Normalization Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| **Z-score** | (x - μ) / σ | Between-subject comparison |
| **Min-Max** | (x - min) / (max - min) | Bounded features |
| **Baseline correction** | x - baseline | Within-subject changes |
| **% Change** | (x - baseline) / baseline × 100 | Reactivity |
| **Log transform** | log(x) | Skewed distributions (EDA) |
| **Range correction** | x / (max - min) | Individual differences |

---

## 12. Experimental Design Considerations

### 12.1 Participant Preparation

**Pre-Session Instructions:**
- Avoid caffeine 2 hours before
- Avoid alcohol 24 hours before
- Get adequate sleep
- Avoid vigorous exercise
- Remove hand jewelry/watches

**Session Preparation:**
- Wash hands (but not immediately before)
- Acclimatize to room temperature (10-15 min)
- Seated comfortably
- Minimize movement during recording

### 12.2 Baseline Recording

**Purpose:**
- Establish individual reference
- Enable change scores
- Account for individual differences

**Protocol:**
```
1. Eyes-open rest: 2-5 minutes
2. Eyes-closed rest: 2-5 minutes
3. Deep breathing (optional): 1 minute
4. Recovery: 1-2 minutes

Total baseline: 5-10 minutes
```

**Baseline Metrics to Record:**
- Mean SCL
- Mean HR
- RMSSD/HRV baseline
- Activity level

### 12.3 Controlling Confounds

| Confound | Control Method |
|----------|----------------|
| **Time of day** | Consistent scheduling |
| **Room temperature** | Climate control (20-24°C) |
| **Humidity** | Moderate levels (40-60%) |
| **Caffeine/alcohol** | Participant instructions |
| **Physical activity** | Rest before session |
| **Emotional state** | Mood questionnaire |
| **Medications** | Screening, documentation |
| **Menstrual cycle** | Documentation, statistical control |

### 12.4 Event Marking

**Types of Events:**
- Trial/condition onset/offset
- Stimulus presentations
- User responses
- Artifact periods
- Notes/observations

**Marking Methods:**
```python
# Software event marking
consensys.mark_event("trial_start", timestamp)

# Hardware TTL trigger
send_ttl_pulse(channel=1)

# Synchronized with eye tracker
pylink.send_message(f"TRIAL_START {trial_num}")
shimmer.mark_event("TRIAL_START")
```

### 12.5 Session Structure

**Recommended Flow:**
```
1. Welcome and consent (5 min)
2. Equipment setup (10-15 min)
3. Calibration check (5 min)
4. Baseline recording (5-10 min)
5. Practice trials (5 min)
6. Main task blocks with breaks
   - Block 1 (10-15 min)
   - Rest (2-3 min)
   - Block 2 (10-15 min)
   - Rest (2-3 min)
   - Block 3 (10-15 min)
7. Recovery baseline (3-5 min)
8. Post-task questionnaires (5-10 min)
9. Debriefing (5 min)

Total: 60-90 minutes
```

---

## 13. Software Tools and Platforms

### 13.1 Shimmer Software Ecosystem

| Software | Purpose | Platform |
|----------|---------|----------|
| **Consensys** | Data collection, visualization | Windows, Mac |
| **Consensys PRO** | Advanced features, multi-Shimmer | Windows |
| **ShimmerCapture** | Basic capture (legacy) | Windows |
| **Shimmer MATLAB SDK** | MATLAB integration | Cross-platform |
| **Shimmer Android API** | Mobile development | Android |
| **Shimmer C# API** | Windows development | Windows |

### 13.2 Analysis Software

**Commercial:**
| Software | Strengths | Cost |
|----------|-----------|------|
| **Ledalab** | EDA decomposition | Free (MATLAB) |
| **Kubios HRV** | HRV analysis | Free/Premium |
| **AcqKnowledge** | Comprehensive biosignal | $$$ |
| **LabChart** | Physiology research | $$$ |

**Open Source:**
| Library | Language | Focus |
|---------|----------|-------|
| **NeuroKit2** | Python | Comprehensive biosignals |
| **HeartPy** | Python | HR/HRV analysis |
| **pyHRV** | Python | HRV analysis |
| **BioSPPy** | Python | Biosignal processing |
| **cvxEDA** | Python/MATLAB | EDA decomposition |
| **HRVAS** | MATLAB | HRV analysis |

### 13.3 Python Environment Setup

```python
# Recommended conda environment
conda create -n biosignals python=3.9
conda activate biosignals

# Core packages
pip install numpy pandas scipy matplotlib seaborn

# Biosignal-specific
pip install neurokit2
pip install heartpy
pip install pyhrv
pip install biosppy
pip install mne  # For EEG/multimodal

# Machine learning
pip install scikit-learn
pip install tensorflow  # or pytorch
```

### 13.4 Real-Time Processing

**For Adaptive Systems:**
```python
import asyncio
from shimmer_sdk import ShimmerDevice

async def realtime_processing():
    device = ShimmerDevice()
    await device.connect()
    
    buffer = CircularBuffer(size=1000)
    
    async for sample in device.stream():
        buffer.add(sample)
        
        if buffer.is_full():
            # Extract features
            features = extract_features(buffer.data)
            
            # Classify state
            state = classifier.predict(features)
            
            # Trigger adaptation
            if state == 'high_stress':
                trigger_intervention()
            
            buffer.slide(100)  # Overlap
```

---

## 14. Theoretical Models and Frameworks

### 14.1 Arousal Theory

**Yerkes-Dodson Law:**
- Performance vs. arousal follows inverted U
- Optimal arousal varies by task complexity
- Measurable via GSR, HR

```
Performance
    │      ___
    │     /   \
    │    /     \
    │   /       \
    │  /         \
    │ /           \
    └──────────────── Arousal
      Low   Med   High
```

### 14.2 Autonomic Space Model (Berntson et al., 1991)

**Key Concept:** SNS and PNS are not simply reciprocal

**Modes of Control:**
| Mode | SNS | PNS | Example |
|------|-----|-----|---------|
| **Reciprocal SNS** | ↑ | ↓ | Fight-or-flight |
| **Reciprocal PNS** | ↓ | ↑ | Relaxation |
| **Coactivation** | ↑ | ↑ | Attention, orientation |
| **Coinhibition** | ↓ | ↓ | Disengagement |
| **Uncoupled SNS** | ↑ | – | Thermoregulation |
| **Uncoupled PNS** | – | ↑ | Respiratory influence |

### 14.3 Polyvagal Theory (Porges, 2007)

**Three Neural Circuits:**
1. **Social engagement** (ventral vagal): Safety, connection
2. **Mobilization** (sympathetic): Fight-or-flight
3. **Immobilization** (dorsal vagal): Freeze, shutdown

**HRV Implications:**
- High HF-HRV = social engagement system active
- Low HRV = stress response or shutdown
- Respiratory sinus arrhythmia (RSA) as vagal index

### 14.4 Cognitive Load Theory Integration

**Physiological Indicators of Cognitive Load:**
| Load Level | GSR | HR | HRV | Pupil |
|------------|-----|-----|-----|-------|
| **Low** | Baseline | Normal | High | Small |
| **Moderate** | Slightly ↑ | Slightly ↑ | Moderate | Medium |
| **High** | Elevated | Elevated | Low | Large |
| **Overload** | Very high | High, variable | Very low | Very large |

### 14.5 Russell's Circumplex Model

**Emotion Mapping:**
```
            High Arousal
                 │
     Stressed    │    Excited
     Angry       │    Happy
                 │
Low Valence ─────┼───── High Valence
                 │
     Sad         │    Calm
     Depressed   │    Relaxed
                 │
            Low Arousal
```

**Physiological Correlates:**
| Quadrant | Arousal (GSR, HR) | Valence Indicator |
|----------|-------------------|-------------------|
| Excited | High | Zygomaticus EMG ↑ |
| Stressed | High | Corrugator EMG ↑ |
| Calm | Low | Zygomaticus EMG ↑ |
| Sad | Low | Corrugator EMG ↑ |

### 14.6 Somatic Marker Hypothesis (Damasio)

**Relevance to HCI:**
- Bodily states influence decision-making
- SCR changes precede conscious decisions (Iowa Gambling Task)
- Physiological signals may predict choices

---

## 15. Applications in HCI Research

### 15.1 Affective Computing

**Emotion Recognition:**
- Feature extraction from multiple signals
- Machine learning classification
- Real-time state detection

**Typical Features:**
| Signal | Features |
|--------|----------|
| GSR | SCL, SCR rate, amplitude, rise time |
| HR/HRV | Mean HR, RMSSD, LF/HF, pNN50 |
| EMG | RMS, mean amplitude, frequency |
| Temperature | Mean, rate of change |
| Motion | Activity level, posture |

### 15.2 Cognitive Load Assessment

**Applications:**
- Interface complexity evaluation
- Adaptive systems
- Learning optimization
- Workload management

**Measurement Approach:**
```python
def assess_cognitive_load(hrv_features, eda_features, baseline):
    """
    Multi-metric cognitive load assessment
    """
    # HRV indicators
    hrv_change = (baseline['rmssd'] - hrv_features['rmssd']) / baseline['rmssd']
    lf_hf_change = hrv_features['lf_hf'] - baseline['lf_hf']
    
    # EDA indicators
    scl_change = (eda_features['scl'] - baseline['scl']) / baseline['scl']
    scr_rate = eda_features['scr_rate']
    
    # Composite score (example weighting)
    load_score = (
        0.3 * normalize(hrv_change) +
        0.2 * normalize(lf_hf_change) +
        0.3 * normalize(scl_change) +
        0.2 * normalize(scr_rate)
    )
    
    return load_score
```

### 15.3 Stress Detection

**Indicators:**
- Increased GSR (SCL and SCR rate)
- Increased HR
- Decreased HRV (especially HF)
- Increased muscle tension (EMG)
- Decreased skin temperature

**Classification Approaches:**
| Method | Features | Accuracy (typical) |
|--------|----------|-------------------|
| Threshold-based | Single signal | 60-70% |
| Linear models | Multiple signals | 70-80% |
| SVM | Engineered features | 75-85% |
| Random Forest | Multiple features | 80-90% |
| Deep Learning | Raw/features | 85-95% |

### 15.4 User Experience Evaluation

**What Physiology Adds:**
- Objective complement to self-report
- Continuous measurement (not just post-task)
- Captures unconscious responses
- Temporal precision

**Example Metrics by UX Dimension:**
| UX Dimension | Physiological Indicators |
|--------------|-------------------------|
| **Frustration** | High GSR, low HRV, high corrugator EMG |
| **Engagement** | Moderate GSR, increased HR |
| **Flow** | Stable GSR, balanced HRV |
| **Boredom** | Low GSR, stable low HR |
| **Satisfaction** | Moderate physiology, zygomaticus EMG |

### 15.5 Adaptive Interfaces

**Closed-Loop Adaptation:**
```
User State Detection
        │
        ▼
State Classification
        │
        ▼
Adaptation Decision
        │
        ▼
Interface Modification
        │
        ▼
User Response (continuous)
```

**Adaptation Examples:**
| Detected State | Adaptation |
|----------------|------------|
| High cognitive load | Simplify interface, reduce options |
| Stress | Calm colors, slower pace |
| Disengagement | Add stimulation, vary content |
| Fatigue | Suggest break, reduce demands |

### 15.6 Relevance to EcoInsight Project

**Integration with Eye Tracking:**
- Correlate fixation patterns with arousal
- Link cognitive load (HRV) to visual complexity
- Detect confusion through combined signals
- Validate infographic effectiveness

**Research Questions:**
- Does arousal (GSR) correlate with infographic engagement?
- Do HRV patterns predict learning outcomes?
- Can we detect cognitive overload in real-time?
- How do physiological responses vary by culture?

---

## 16. Multimodal Integration

### 16.1 Why Multimodal?

**Benefits:**
- Increased reliability (redundancy)
- Richer information
- Better classification accuracy
- Disambiguation of ambiguous signals

**Challenges:**
- Synchronization
- Different sampling rates
- Feature fusion strategies
- Increased complexity

### 16.2 Shimmer + Eye Tracking Integration

**Synchronized Metrics:**
| Eye Tracking | Shimmer | Combined Insight |
|--------------|---------|------------------|
| Fixation duration | GSR | Arousal during attention |
| Pupil dilation | HR/HRV | Convergent cognitive load |
| Saccade patterns | Motion | Movement-attention relationship |
| Blink rate | HRV | Fatigue assessment |

### 16.3 Fusion Strategies

**Early Fusion (Feature-level):**
```python
# Concatenate features from all modalities
features = np.concatenate([
    eye_features,      # Fixation, saccade metrics
    gsr_features,      # SCL, SCR metrics
    hrv_features,      # Time and frequency domain
    motion_features    # Activity metrics
])
prediction = model.predict(features)
```

**Late Fusion (Decision-level):**
```python
# Separate classifiers, combined decision
pred_eye = eye_classifier.predict(eye_features)
pred_physio = physio_classifier.predict(physio_features)

# Weighted combination
final_pred = weighted_vote([pred_eye, pred_physio], 
                           weights=[0.4, 0.6])
```

**Hybrid Fusion:**
- Multiple fusion points
- Attention mechanisms
- Neural network approaches

### 16.4 Temporal Alignment

**Challenges:**
- Different sampling rates (eye: 1000 Hz, GSR: 128 Hz)
- Different response latencies (pupil: 200ms, GSR: 1-5s)
- Clock drift between systems

**Solutions:**
```python
def align_multimodal(eye_data, physio_data, eye_fs, physio_fs):
    """
    Align eye tracking and physiological data
    """
    # Resample to common rate
    common_fs = min(eye_fs, physio_fs)
    
    eye_resampled = resample(eye_data, eye_fs, common_fs)
    physio_resampled = resample(physio_data, physio_fs, common_fs)
    
    # Account for physiological response latency
    # GSR lags by ~1-5 seconds
    gsr_delay = int(2 * common_fs)  # 2 second delay
    
    # Align with delay compensation
    aligned_eye = eye_resampled[gsr_delay:]
    aligned_physio = physio_resampled[:-gsr_delay]
    
    return aligned_eye, aligned_physio
```

### 16.5 Multimodal Datasets

| Dataset | Modalities | Application |
|---------|------------|-------------|
| **DEAP** | EEG, GSR, PPG, video | Emotion |
| **MAHNOB-HCI** | EEG, GSR, ECG, eye, video | Emotion, attention |
| **WESAD** | ECG, EDA, EMG, respiration, motion | Stress |
| **AMIGOS** | EEG, GSR, ECG | Emotion |
| **K-EmoCon** | EEG, GSR, PPG, motion | Emotion |

---

## 17. Noteworthy Papers and Studies

### 17.1 Foundational Papers

| Paper | Authors (Year) | Contribution |
|-------|---------------|--------------|
| **Electrodermal Activity** | Boucsein (2012) | Comprehensive EDA textbook |
| **Heart Rate Variability Standards** | Task Force (1996) | HRV measurement standards |
| **Affective Computing** | Picard (1997) | Founded the field |
| **Autonomic Space** | Berntson et al. (1991) | SNS/PNS independence model |
| **Polyvagal Theory** | Porges (2007) | Vagal tone and social behavior |

### 17.2 HCI-Specific Papers

| Paper | Focus | Key Finding |
|-------|-------|-------------|
| **Scheirer et al. (2002)** | Frustration detection | GSR correlates with frustration |
| **Mandryk & Atkins (2007)** | Game experience | Physiology predicts engagement |
| **Hernandez et al. (2014)** | Workplace stress | Call center stress detection |
| **Setz et al. (2010)** | Stress discrimination | GSR distinguishes stress types |
| **Cowley et al. (2016)** | Psychophysiology primer | Comprehensive HCI guide |

### 17.3 Cognitive Load Papers

| Paper | Authors | Contribution |
|-------|---------|--------------|
| **Physiological measures of cognitive load** | Paas et al. (2003) | Review of measures |
| **HRV and cognitive load** | Mukherjee et al. (2011) | HRV as cognitive load indicator |
| **Multimodal cognitive load** | Haapalainen et al. (2010) | Sensor comparison |
| **Real-time cognitive load** | Hussain et al. (2011) | Adaptive systems |

### 17.4 Emotion Recognition Papers

| Paper | Authors | Approach |
|-------|---------|----------|
| **DEAP dataset** | Koelstra et al. (2012) | EEG + peripheral signals |
| **Emotion recognition survey** | Jerritta et al. (2011) | Comprehensive review |
| **Deep learning for emotion** | Soleymani et al. (2016) | CNN for physiological signals |
| **Wearable emotion recognition** | Schmidt et al. (2018) | WESAD dataset |

### 17.5 Methodological Papers

| Paper | Topic | Recommendation |
|-------|-------|----------------|
| **Benedek & Kaernbach (2010)** | EDA decomposition | Continuous decomposition analysis |
| **Greco et al. (2016)** | cvxEDA | Convex optimization for EDA |
| **Tarvainen et al. (2014)** | HRV analysis | Kubios software methodology |
| **Makowski et al. (2021)** | NeuroKit2 | Python signal processing |

---

## 18. Important Datasets

### 18.1 Emotion and Affect Datasets

| Dataset | Signals | Participants | Stimuli | Access |
|---------|---------|--------------|---------|--------|
| **DEAP** | EEG, GSR, PPG, EMG, EOG, temp | 32 | Music videos | eecs.qmul.ac.uk |
| **MAHNOB-HCI** | EEG, ECG, GSR, respiration, eye, video | 27 | Videos, images | mahnob-db.eu |
| **WESAD** | ECG, EDA, EMG, respiration, temp, motion | 15 | Stress protocol | ubicomp.eti.uni-siegen.de |
| **AMIGOS** | EEG, ECG, GSR | 40 | Videos | amigos-dataset.eu |
| **ASCERTAIN** | EEG, ECG, GSR | 58 | Videos | mhug.disi.unitn.it |
| **DREAMER** | EEG, ECG | 23 | Videos | zenodo.org |
| **K-EmoCon** | EEG, PPG, EDA, motion | 32 | Conversations | github |

### 18.2 Cognitive Load Datasets

| Dataset | Signals | Task | Size |
|---------|---------|------|------|
| **MAUS** | ECG, GSR | Mental arithmetic | 50 subjects |
| **CogLoad** | EEG, ECG, EDA | N-back task | 24 subjects |
| **STEW** | EEG, GSR | SIMKAP workload | 48 subjects |

### 18.3 Stress Datasets

| Dataset | Signals | Protocol | Access |
|---------|---------|----------|--------|
| **WESAD** | Multiple | TSST-like | Public |
| **SWELL-KW** | ECG, GSR, facial | Knowledge work | Request |
| **DriveDB** | ECG, EMG, GSR, respiration | Driving | physionet.org |
| **AffectiveROAD** | ECG, EDA | Driving | github |

### 18.4 Multimodal Interaction Datasets

| Dataset | Focus | Modalities |
|---------|-------|------------|
| **RECOLA** | Interaction | Audio, video, ECG, EDA |
| **SEMAINE** | Conversation | Audio, video, physiology |
| **CreativeIT** | Creativity | Motion, physiology |

---

## 19. Analysis Approaches and Methods

### 19.1 Classical Signal Processing

| Technique | Application | Implementation |
|-----------|-------------|----------------|
| **Butterworth filter** | Noise removal | scipy.signal.butter |
| **Savitzky-Golay** | Smoothing | scipy.signal.savgol_filter |
| **Welch's method** | PSD estimation | scipy.signal.welch |
| **Peak detection** | SCR, R-peaks | scipy.signal.find_peaks |
| **Detrending** | Baseline removal | scipy.signal.detrend |
| **Resampling** | Rate conversion | scipy.signal.resample |

### 19.2 EDA-Specific Methods

| Method | Description | Tool |
|--------|-------------|------|
| **CDA** | Continuous decomposition | Ledalab |
| **DDA** | Discrete decomposition | Ledalab |
| **cvxEDA** | Convex optimization | cvxEDA package |
| **SparsEDA** | Sparse deconvolution | Custom |
| **NeuroKit** | Multiple methods | NeuroKit2 |

### 19.3 HRV Analysis Methods

| Domain | Methods | Tools |
|--------|---------|-------|
| **Time** | SDNN, RMSSD, pNN50 | Kubios, pyHRV |
| **Frequency** | Welch, Lomb-Scargle, AR | Kubios, pyHRV |
| **Nonlinear** | Poincaré, entropy, DFA | Kubios, pyHRV |
| **Time-frequency** | Wavelet, STFT | Custom |

### 19.4 Machine Learning Approaches

**Traditional ML:**
| Algorithm | Strengths | Use Case |
|-----------|-----------|----------|
| **SVM** | Small datasets, clear margins | Binary classification |
| **Random Forest** | Feature importance, robust | Multi-class |
| **XGBoost** | High performance | Competitions |
| **LDA** | Interpretable | Dimensionality reduction |

**Deep Learning:**
| Architecture | Application | Input |
|--------------|-------------|-------|
| **1D CNN** | Temporal patterns | Raw signals |
| **LSTM** | Sequential data | Time series |
| **Transformer** | Long dependencies | Segments |
| **Autoencoder** | Feature learning | Raw signals |

### 19.5 Feature Selection

**Methods:**
```python
from sklearn.feature_selection import (
    SelectKBest, f_classif,
    RFE, RFECV,
    mutual_info_classif
)

# Univariate selection
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)

# Recursive feature elimination
from sklearn.ensemble import RandomForestClassifier
rfe = RFE(RandomForestClassifier(), n_features_to_select=15)
X_rfe = rfe.fit_transform(X, y)

# Feature importance from model
model = RandomForestClassifier()
model.fit(X, y)
importance = model.feature_importances_
```

### 19.6 Cross-Validation Strategies

**For Physiological Data:**
| Strategy | When to Use |
|----------|-------------|
| **Leave-One-Subject-Out (LOSO)** | Subject-independent models |
| **Leave-One-Session-Out** | Session generalization |
| **Stratified K-Fold** | Balanced classes |
| **Time-Series Split** | Temporal dependencies |
| **Nested CV** | Hyperparameter tuning |

```python
from sklearn.model_selection import LeaveOneGroupOut

# Leave-one-subject-out
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups=subjects):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

---

## 20. Notable Projects and Research Initiatives

### 20.1 Major Research Projects

**MIT Media Lab - Affective Computing Group:**
- Founded by Rosalind Picard
- Pioneering wearable sensing
- Empatica E4 development
- Autism research applications

**HUMAINE Network:**
- EU Network of Excellence
- Emotion-oriented computing
- Standards development
- Database creation

**USC Institute for Creative Technologies:**
- Virtual humans
- Multimodal sensing
- PTSD treatment
- Social skills training

### 20.2 Commercial Applications

| Company | Product | Application |
|---------|---------|-------------|
| **Empatica** | E4, Embrace | Research, epilepsy detection |
| **Shimmer** | Shimmer3, Verisense | Research, clinical |
| **iMotions** | Platform | Multimodal integration |
| **Affectiva** | Emotion AI | Automotive, media |
| **Noldus** | Observer XT | Behavior research |

### 20.3 Open-Source Projects

| Project | Focus | Link |
|---------|-------|------|
| **NeuroKit2** | Biosignal processing | github.com/neuropsychology/NeuroKit |
| **HeartPy** | Heart rate analysis | github.com/paulvangentcom/heartrate_analysis_python |
| **BioSPPy** | Biosignal processing | github.com/PIA-Group/BioSPPy |
| **pyHRV** | HRV analysis | github.com/PGomes92/pyhrv |
| **cvxEDA** | EDA decomposition | github.com/lciti/cvxEDA |
| **Ledalab** | EDA analysis | ledalab.de |

### 20.4 Relevant Conferences

| Conference | Focus | Relevance |
|------------|-------|-----------|
| **ACII** | Affective Computing | Core venue |
| **CHI** | Human-Computer Interaction | Applications |
| **UbiComp** | Ubiquitous Computing | Wearables |
| **EMBC** | Biomedical Engineering | Methods |
| **PhyCS** | Physiological Computing | Specialized |
| **PerCom** | Pervasive Computing | Context |

### 20.5 Journals

| Journal | Focus |
|---------|-------|
| **IEEE Trans. Affective Computing** | Affective computing |
| **Psychophysiology** | Psychophysiology methods |
| **Biological Psychology** | Biological basis |
| **Frontiers in Neuroscience** | Open access |
| **JMIR** | Health applications |
| **Sensors** | Sensing technologies |

---

## 21. Emerging Trends and Future Directions

### 21.1 Technological Advances

**Miniaturization:**
- Smaller, more comfortable sensors
- Integration into everyday objects
- Smart textiles and e-skin

**AI/ML Integration:**
- On-device processing
- Real-time state estimation
- Personalized models
- Transfer learning

**Novel Sensing:**
- Radar-based vital signs
- Camera-based PPG (rPPG)
- Acoustic sensing
- Molecular sensing (stress hormones)

### 21.2 Application Trends

**Mental Health:**
- Continuous monitoring
- Early intervention
- Therapy augmentation
- Crisis detection

**Workplace Wellness:**
- Stress management
- Productivity optimization
- Break recommendation
- Team dynamics

**Education:**
- Adaptive learning
- Engagement monitoring
- Attention management
- Assessment

### 21.3 Ethical Considerations

**Privacy:**
- Sensitive health data
- Continuous monitoring concerns
- Data ownership
- Consent challenges

**Bias:**
- Demographic differences
- Training data representation
- Algorithmic fairness

**Autonomy:**
- Manipulation potential
- Informed consent
- Right to disconnect

### 21.4 Future Research Directions

| Direction | Description |
|-----------|-------------|
| **Personalization** | Individual-specific models |
| **Context-awareness** | Environmental adaptation |
| **Longitudinal studies** | Long-term patterns |
| **Multimodal fusion** | Better integration methods |
| **Explainability** | Interpretable ML |
| **Real-world validation** | Ecological validity |

---

## 22. References and Further Reading

### 22.1 Textbooks

- Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.
- Cacioppo, J. T., Tassinary, L. G., & Berntson, G. G. (2016). *Handbook of Psychophysiology* (4th ed.). Cambridge.
- Picard, R. W. (1997). *Affective Computing*. MIT Press.
- Andreassi, J. L. (2007). *Psychophysiology* (5th ed.). Psychology Press.

### 22.2 Key Papers

- Task Force of ESC and NASPE. (1996). Heart rate variability: Standards of measurement. *Circulation*, 93(5), 1043-1065.
- Benedek, M., & Kaernbach, C. (2010). A continuous measure of phasic electrodermal activity. *Journal of Neuroscience Methods*, 190(1), 80-91.
- Greco, A., Valenza, G., et al. (2016). cvxEDA: A convex optimization approach to electrodermal activity processing. *IEEE TBME*, 63(4), 797-804.
- Cowley, B., et al. (2016). The psychophysiology primer: A guide to methods and a broad review. *Foundations and Trends in HCI*, 9(3-4), 151-308.

### 22.3 Review Papers

- Sharma, N., & Gedeon, T. (2012). Objective measures, sensors and computational techniques for stress recognition and classification: A survey. *Computer Methods and Programs in Biomedicine*, 108(3), 1287-1301.
- Jerritta, S., et al. (2011). Physiological signals based human emotion recognition: A review. *IEEE ICSPC*, 410-415.
- Makowski, D., et al. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*, 53, 1689-1696.

### 22.4 Online Resources

- Shimmer Documentation: shimmersensing.com/support
- NeuroKit2: neuropsychology.github.io/NeuroKit
- PhysioNet: physionet.org
- Kubios HRV: kubios.com
- Ledalab: ledalab.de

---

## Appendix A: Quick Reference - Signal Parameters

### Shimmer3 GSR+ Recommended Settings

| Signal | Sampling Rate | Filter | Notes |
|--------|---------------|--------|-------|
| GSR | 128 Hz | LP 1 Hz | Sufficient for EDA |
| PPG | 256 Hz | BP 0.5-4 Hz | HR/HRV extraction |
| Accelerometer | 128 Hz | LP 20 Hz | Motion/posture |
| Gyroscope | 128 Hz | LP 20 Hz | Rotation |
| Temperature | 1 Hz | None | Slow changes |

### Typical Metric Ranges

| Metric | Normal Range | Stress/Load |
|--------|--------------|-------------|
| SCL | 2-20 µS | Increased |
| SCR rate | 0-5 /min | Increased |
| Heart Rate | 60-100 BPM | Increased |
| RMSSD | 20-100 ms | Decreased |
| LF/HF | 0.5-2.0 | Increased |

---

## Appendix B: Checklist for Shimmer Studies

### Pre-Session
- [ ] Charge Shimmer devices
- [ ] Test Bluetooth connection
- [ ] Verify sensor calibration
- [ ] Prepare electrodes/sensors
- [ ] Set up Consensys software
- [ ] Configure sampling rates
- [ ] Prepare participant materials

### Session Setup
- [ ] Participant consent obtained
- [ ] Pre-study questionnaires completed
- [ ] Skin preparation (cleaning)
- [ ] Electrode/sensor placement
- [ ] Signal quality check
- [ ] Impedance verification (if applicable)
- [ ] Baseline recording

### During Session
- [ ] Monitor signal quality
- [ ] Mark events appropriately
- [ ] Note any artifacts/issues
- [ ] Watch for electrode displacement
- [ ] Regular quality checks

### Post-Session
- [ ] Stop and save recording
- [ ] Export data in required format
- [ ] Remove and clean sensors
- [ ] Post-task questionnaires
- [ ] Participant debriefing
- [ ] Back up data
- [ ] Document any issues

---

## Appendix C: Code Templates

### Basic Data Loading

```python
import pandas as pd
import numpy as np

def load_shimmer_data(filepath):
    """Load Shimmer CSV export"""
    data = pd.read_csv(filepath, skiprows=3)  # Skip header rows
    
    # Rename columns for clarity
    column_mapping = {
        'Shimmer_XXXX_Timestamp_Unix_CAL': 'timestamp',
        'Shimmer_XXXX_GSR_Skin_Conductance_CAL': 'gsr',
        'Shimmer_XXXX_PPG_A13_CAL': 'ppg',
        'Shimmer_XXXX_Temperature_BMP280_CAL': 'temperature',
        'Shimmer_XXXX_Accel_LN_X_CAL': 'accel_x',
        'Shimmer_XXXX_Accel_LN_Y_CAL': 'accel_y',
        'Shimmer_XXXX_Accel_LN_Z_CAL': 'accel_z',
    }
    data.rename(columns=column_mapping, inplace=True)
    
    return data
```

### Complete Processing Pipeline

```python
import neurokit2 as nk

def process_shimmer_session(gsr_data, ppg_data, fs_gsr=128, fs_ppg=256):
    """
    Complete processing pipeline for Shimmer data
    """
    results = {}
    
    # Process EDA
    eda_signals, eda_info = nk.eda_process(gsr_data, sampling_rate=fs_gsr)
    results['eda'] = {
        'scl_mean': eda_signals['EDA_Tonic'].mean(),
        'scr_count': len(eda_info['SCR_Peaks']),
        'scr_amplitude_mean': eda_signals['EDA_Phasic'].max()
    }
    
    # Process PPG for HR/HRV
    ppg_signals, ppg_info = nk.ppg_process(ppg_data, sampling_rate=fs_ppg)
    
    # Extract HRV if enough beats
    if len(ppg_info['PPG_Peaks']) > 10:
        hrv = nk.hrv(ppg_signals, sampling_rate=fs_ppg)
        results['hrv'] = {
            'mean_hr': hrv['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv else None,
            'rmssd': hrv['HRV_RMSSD'].values[0] if 'HRV_RMSSD' in hrv else None,
            'lf_hf': hrv['HRV_LFHF'].values[0] if 'HRV_LFHF' in hrv else None
        }
    
    return results
```

---

## Appendix D: Glossary

| Term | Definition |
|------|------------|
| **EDA** | Electrodermal Activity - electrical properties of skin |
| **GSR** | Galvanic Skin Response (older term for EDA) |
| **SCL** | Skin Conductance Level - tonic EDA component |
| **SCR** | Skin Conductance Response - phasic EDA component |
| **PPG** | Photoplethysmography - optical blood volume measurement |
| **HRV** | Heart Rate Variability - variation in heartbeat intervals |
| **IBI** | Inter-Beat Interval - time between heartbeats |
| **RMSSD** | Root mean square of successive differences |
| **LF/HF** | Low frequency to high frequency HRV ratio |
| **SNS** | Sympathetic Nervous System |
| **PNS** | Parasympathetic Nervous System |
| **EMG** | Electromyography - muscle electrical activity |
| **ECG** | Electrocardiography - heart electrical activity |
| **IMU** | Inertial Measurement Unit - motion sensors |

---

*Document prepared for HCI + L Laboratory research team*  
*Last updated: January 2026*
