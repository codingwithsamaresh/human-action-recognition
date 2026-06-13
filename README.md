# Human Action Recognition

A production-style Human Action Recognition (HAR) system for real-time workplace safety monitoring.

## Features

- Video Action Recognition
- CNN + LSTM Baseline
- Temporal Attention Mechanism
- SlowFast Network
- TimeSformer Architecture
- YOLO Human Detection
- Multi-Person Tracking
- Real-Time Webcam Inference
- Alert Generation System
- Streamlit Dashboard
- Docker Deployment

---

## Project Structure

```text
human-action-recognition/
├── configs/
├── data/
├── notebooks/
├── src/
├── dashboard/
├── deployment/
├── tests/
├── weights/
├── outputs/
└── logs/
```

---

## Installation

Create virtual environment:

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Development Roadmap

### Phase 1

- Dataset Preparation
- Frame Extraction
- Sequence Generation
- CNN-LSTM Baseline

### Phase 2

- Temporal Attention

### Phase 3

- YOLO Human Detection

### Phase 4

- Multi-Person Tracking

### Phase 5

- SlowFast Network

### Phase 6

- TimeSformer

### Phase 7

- Dashboard + Deployment

---

Human Action Recognition system using a CNN-LSTM architecture.

* Dataset: UCF101 (25 selected action classes)
* Backbone: MobileNetV3-Small
* Temporal Encoder: LSTM
* Sequence Length: 8 frames
* Training Device: NVIDIA Tesla T4
* Validation Accuracy: 79.53%
* Framework: PyTorch

The system performs action classification from short video clips by combining spatial feature extraction with temporal sequence modeling.


## Author

Samaresh Koley