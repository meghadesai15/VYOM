# VYOM
VYOM is an Edge AI-powered drone safety system for real-time person and vehicle detection, identity verification, behavior classification, and threat monitoring using YOLOv8n, MobileNetV2, LSTM, and TensorFlow Lite on Raspberry Pi 4.

# VYOM — Edge AI Threat-Detection Pipeline

**An autonomous, edge-AI-powered drone safety and threat-monitoring system.**

VYOM implements the RakshaDrone architecture to detect and track people and vehicles, identify an enrolled/protected user, classify the behavior of other people, and trigger appropriate safety responses using a real-time, two-loop AI pipeline.

The system combines object detection, multi-object tracking, face-based identity verification, video-based behavior classification, a per-track Finite State Machine (FSM), and an action dispatcher.

> **Core pipeline:** Detect & Track → Identity Verification → Behavior Classification → Threat Assessment → Safety Response

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Threat Classification and Decision Logic](#threat-classification-and-decision-logic)
- [Quick Start: Simulation Mode](#quick-start-simulation-mode)
- [Project Structure](#project-structure)
- [Running on Real Hardware](#running-on-real-hardware)
- [Model Training and Export](#model-training-and-export)
- [Configuration](#configuration)
- [Design Decisions](#design-decisions)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)
- [Applications](#applications)
- [Disclaimer](#disclaimer)

## Overview

VYOM is designed as an edge-AI surveillance and safety pipeline that processes camera frames locally and maintains independent behavioral states for tracked individuals.

Unlike a simple object detector, VYOM combines spatial tracking, identity verification, temporal behavior analysis, and configurable decision logic to determine when an observation warrants attention.

The architecture is designed for deployment on resource-constrained hardware, particularly the **Raspberry Pi 4**, with computationally intensive training and model preparation performed on a separate development machine.

### Project Objectives

- Detect and track people and relevant vehicles in camera footage.
- Exempt an enrolled/protected user from stranger-focused behavior analysis.
- Classify observed human behavior using a MobileNetV2 and LSTM-based pipeline.
- Reduce noisy predictions using temporal smoothing and configurable escalation rules.
- Trigger follow, alert, evidence-recording, and emergency-response actions.
- Support offline simulation for testing the complete pipeline without a camera or trained models.

## Key Features

- **Edge AI:** Designed for local inference on resource-constrained hardware.
- **Real-Time Detection and Tracking:** YOLOv8n-based detection with a simplified two-stage IoU tracker.
- **Identity Verification:** MobileFaceNet-based face embeddings for matching enrolled users.
- **Temporal Behavior Recognition:** MobileNetV2 feature extraction followed by LSTM-based sequence classification.
- **Two-Loop Architecture:** Separates frame-by-frame perception from slower, batched behavior inference.
- **Per-Track Threat Assessment:** Maintains an independent FSM for each tracked person.
- **K-of-N Decision Logic:** Requires repeated concerning predictions before escalating a track.
- **Vehicle-Aware Response:** Searches for nearby tracked cars or motorcycles when an aggressive state is reached.
- **Simulation Mode:** Supports end-to-end testing using synthetic frames and simulated model backends.

## System Architecture

VYOM uses two cooperating processing loops.

### 1. Fast Loop — Perception and Identity

The fast loop processes each incoming frame.

1. Capture a frame from the camera or simulation backend.
2. Detect people and vehicles.
3. Associate detections with existing tracks.
4. Attempt face-based re-identification for eligible lost tracks.
5. Run identity verification for newly created tracks.
6. Exempt the enrolled user from stranger-focused behavior classification.
7. Buffer observations for active stranger tracks.
8. Trigger the follow action for the enrolled user.

### 2. Slow Loop — Behavior Analysis

The slow loop periodically processes buffered observations in batches.

1. Select sparse temporal samples from each active stranger's observation window.
2. Extract frame-level features using MobileNetV2.
3. Organize features into per-track temporal sequences.
4. Run LSTM-based sequence classification.
5. Convert predicted classes into behavior tiers.
6. Pass classification results to the decision engine.

Batching the backbone inference across active tracks helps avoid redundant model calls.

### 3. Decision Engine and Action Dispatcher

The decision engine maintains an independent FSM for every track. Temporal decision rules help reduce reactions to isolated predictions.

The action dispatcher responds to state transitions by invoking the configured follow, alert, evidence-recording, emergency, or vehicle-pursuit handlers.

**High-level pipeline**

```text
Camera / Synthetic Frames
          |
          v
   Object Detection
       YOLOv8n
          |
          v
   Multi-Object Tracker
          |
          v
   Identity Verification
      MobileFaceNet
          |
     +----+----+
     |         |
 Enrolled    Stranger
   User         |
     |          v
   Follow   Observation Buffer
                |
                v
        Sparse Frame Sampling
                |
                v
        MobileNetV2 Features
                |
                v
          LSTM Classifier
                |
                v
       Per-Track Decision FSM
                |
                v
        Action Dispatcher
                |
       +--------+---------+
       |        |         |
      Alert  Emergency  Vehicle
                         Pursuit
```

## Threat Classification and Decision Logic

VYOM separates model predictions from operational decisions.

The behavior classifier predicts a behavior class. The decision engine maps the predicted class to a tier and updates the corresponding track's state using temporal rules.

### Temporal Escalation

The configured decision policy includes:

- **Escalation:** At least 3 aggressive-tier predictions among the latest 4 tier observations trigger the aggressive state.
- **Suspicious transition:** A suspicious-or-higher observation can move a watching track into the suspicious state.
- **De-escalation:** Two consecutive normal-tier observations reduce the state by one tier.
- **Transition-based actions:** The engine emits events when a track changes state, rather than treating every frame as an independent emergency.

These values are configurable and should be validated against representative footage before deployment.

### Action Mapping

| Track state or event | Configured response |
|---|---|
| Enrolled user detected | Follow the enrolled user |
| Suspicious state | Hold position, record the event, and log an alert |
| Aggressive state | Capture available face evidence, log an emergency, and search for a nearby vehicle |
| Nearby vehicle found | Invoke the configured vehicle-pursuit handler |
| No nearby vehicle found | Hold position and record the event |

The current action integrations use logging/stub implementations. Actual drone movement, remote notifications, and emergency-service integrations require separate implementation and validation.

## Quick Start: Simulation Mode

Simulation mode is the easiest way to explore the pipeline without a camera, pretrained models, or model-training dependencies.

### Prerequisites

- Python 3.10 or another Python version supported by the project's dependencies.
- `pip`
- NumPy

### 1. Install the minimum dependency

```bash
pip install numpy
```

### 2. Run the stranger-escalation scenario

From the repository root:

```bash
python -m vyom.main --simulate --scenario stranger_approach --frames 400
```

This scenario is intended to demonstrate the pipeline using synthetic frames and simulated model backends, including enrolled-user following and stranger threat escalation.

### 3. Try additional scenarios

**Vehicle threat scenario**

```bash
python -m vyom.main --simulate --scenario vehicle_threat --frames 400
```

Demonstrates the emergency-triggered nearby-vehicle response path.

**Vehicle-only scenario**

```bash
python -m vyom.main --simulate --scenario vehicle --frames 400
```

Demonstrates that a parked vehicle without an escalating stranger should not independently trigger the stranger-based threat response.

**Run until manually stopped**

```bash
python -m vyom.main --simulate --scenario stranger_approach
```

Stop the process with `Ctrl+C`.

> Simulation validates software flow and configured decision logic. It does not establish real-world detection accuracy, behavior-recognition performance, or drone safety.

## Project Structure

```text
VYOM/
├── vyom/
│   ├── config.py
│   ├── types.py
│   ├── main.py
│   │
│   ├── perception/
│   │   ├── detector.py
│   │   └── tracker.py
│   │
│   ├── identity/
│   │   ├── face_id.py
│   │   └── enroll.py
│   │
│   ├── behavior/
│   │   └── classifier.py
│   │
│   ├── decision/
│   │   ├── fsm.py
│   │   └── engine.py
│   │
│   ├── actions/
│   │   └── dispatcher.py
│   │
│   ├── pipeline/
│   │   ├── fast_loop.py
│   │   └── slow_loop.py
│   │
│   └── train/
│       ├── dataset.py
│       ├── train_behavior_model.py
│       ├── export_tflite.py
│       └── benchmark_pi.py
│
├── models/                  # Model files; normally excluded from Git
├── runs/                    # Training outputs; normally excluded from Git
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure above describes the intended repository layout. Include only files and directories that are actually present in your checked-out version.*

## Running on Real Hardware

Real deployment requires compatible model files, a camera, and the appropriate runtime dependencies.

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

Keep runtime, training, and export dependencies separated where practical because they are normally installed on different machines.

### Step 2: Train the behavior model

Run this command from the repository root:

```bash
python -m vyom.train.train_behavior_model /path/to/dataset_root
```

The expected checkpoint path is:

```text
runs/behavior_v1/best_model.pt
```

### Step 3: Export the behavior model

```bash
python -m vyom.train.export_tflite \
  runs/behavior_v1/best_model.pt \
  --calib-dir /path/to/dataset_root
```

The intended exported artifacts are:

```text
models/mobilenetv2_int8.tflite
models/lstm_head_int8.tflite
```

Verify the export script's actual output paths and supported quantization options before running deployment.

### Step 4: Prepare the pretrained perception models

Obtain compatible model files for:

- YOLOv8n object detection
- MobileFaceNet face embeddings
- A compatible face detector

Place them at the paths configured in `vyom/config.py` under `MODEL_PATHS`.

Verify input dimensions, output tensor layouts, preprocessing, and quantization compatibility for each model.

### Step 5: Enroll the protected user

```bash
python -m vyom.identity.enroll /path/to/reference_photo.jpg
```

Use a suitable reference image and verify the resulting enrollment artifact and configured storage location.

### Step 6: Benchmark on the Raspberry Pi 4

```bash
python -m vyom.train.benchmark_pi --duration 60
```

Run the benchmark on the actual target device. Desktop performance does not reliably represent Raspberry Pi CPU latency, memory pressure, or thermal throttling.

### Step 7: Run the pipeline

```bash
python -m vyom.main
```

Real mode requires a compatible camera at index `0`, the configured model files, and all required runtime dependencies.

## Model Training and Export

The behavior-recognition pipeline combines spatial feature extraction with temporal sequence modeling.

| Component | Responsibility |
|---|---|
| MobileNetV2 | Extracts features from sampled video frames |
| Feature representation | Produces frame-level feature vectors |
| LSTM head | Learns temporal relationships across frame sequences |
| Classification head | Predicts one of the configured behavior classes |
| TFLite runtime | Enables lightweight inference on supported edge devices |

Training and inference preprocessing must remain consistent, especially in frame sampling, resizing, normalization, channel order, sequence length, and class-index mapping.

Evaluate the model on a held-out test set and report per-class precision, recall, F1-score, confusion matrix, and overall accuracy. Class imbalance and false-positive rates are particularly important for safety-related applications.

## Configuration

The main configuration file is:

```text
vyom/config.py
```

It centralizes the pipeline's tunable parameters, including:

- Model paths and image dimensions
- Fast-loop and slow-loop settings
- Behavior-window duration and sampling
- Identity-verification settings
- FSM thresholds and temporal decision rules
- Action-dispatcher configuration

Keep configuration values consistent across training, evaluation, export, and deployment. Avoid changing the number or ordering of behavior classes without retraining or otherwise validating the corresponding model and its class mapping.

## Design Decisions

**1. Identity checks are not run on every frame.**

Identity verification is associated with newly created tracks, with separate re-identification logic for eligible lost tracks. This reduces repeated face-embedding computation.

**2. Behavior features are computed in batches.**

The feature-extraction backbone processes active stranger tracks together instead of performing a separate backbone call for each track.

**3. Temporal sampling reduces redundant computation.**

The slow loop samples frames from an observation window rather than processing every buffered frame through the complete sequence model.

**4. Every tracked person has an independent decision state.**

An individual's behavior history does not directly become a single global scene state.

**5. K-of-N logic reduces sensitivity to isolated predictions.**

Repeated observations are considered before aggressive escalation, while configurable de-escalation rules help the state recover when normal behavior is observed.

**6. Vehicles follow a separate processing path.**

Cars and motorcycles are tracked for spatial association but are not passed through the human behavior classifier. Vehicle-related response logic is activated through the configured association between an aggressive person track and a nearby live vehicle track.

## Current Limitations

- **Simplified tracking:** The current tracker uses two-stage IoU association and does not implement full ByteTrack with Kalman prediction.
- **Model-dependent accuracy:** Behavior recognition depends on the quality, balance, and diversity of the training dataset.
- **Identity verification:** Face-matching accuracy must be tested under changing lighting, viewing angles, occlusion, and outdoor conditions.
- **Edge performance:** Sustained throughput depends on Raspberry Pi hardware, memory, model quantization, camera resolution, and thermal conditions.
- **Stubbed integrations:** Logging-based cloud and drone-controller handlers do not provide production cloud alerts or actual flight control.
- **Vehicle association:** Pixel-distance proximity is a simplified heuristic and does not establish real-world physical distance.
- **Simulation constraints:** Synthetic scenarios validate control flow but do not reproduce the complexity of real-world video.
- **Safety validation:** The pipeline is a prototype and requires extensive testing before any operational safety use.

## Future Improvements

- Replace IoU-only tracking with Kalman-based tracking and improved occlusion handling.
- Improve behavior recognition using more diverse datasets and per-class evaluation.
- Add calibrated confidence thresholds and explicit false-alarm monitoring.
- Implement secure alert delivery and evidence-management integrations.
- Use camera calibration or geometric estimation to improve vehicle-proximity assessment.
- Add Raspberry Pi performance profiling, thermal monitoring, and memory optimization.
- Validate the complete system in controlled field trials with human oversight.
- Evaluate privacy-preserving identity storage, access controls, and evidence-retention policies.

## Applications

VYOM's intended research and development applications include:

- Women and public safety monitoring
- Campus and transit-hub surveillance
- Police and emergency-response situational awareness
- Fire and rescue operations
- NDRF/SDRF and disaster-management support
- Emergency medical services (EMS)
- Edge-AI research in autonomous UAV perception and monitoring

These are potential applications, not claims of validated operational effectiveness.

## Disclaimer

VYOM is a research and development prototype. Model predictions are probabilistic and may produce false positives or false negatives. An aggressive classification must not, by itself, be treated as proof of malicious intent or as sufficient justification for coercive action.

Any real-world deployment should include human oversight, privacy safeguards, controlled flight testing, fail-safe behavior, and compliance with applicable UAV and surveillance regulations. Automated pursuit or physical intervention should not be enabled without rigorous safety validation.

---

**VYOM — Edge AI for Safety-Aware Aerial Monitoring**

*Detect. Track. Analyze. Respond.*
