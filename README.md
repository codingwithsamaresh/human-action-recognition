# Human Action Recognition using CNN-LSTM

An end-to-end deep learning system for **Human Action Recognition (HAR)** using the **UCF101** dataset and a **CNN-LSTM** architecture.

The system extracts spatial features from video frames using a lightweight CNN backbone and models temporal dependencies across frame sequences using an LSTM. The trained model is evaluated using classification metrics, ROC-AUC analysis, confusion matrices, and inference benchmarking.

---

## Project Overview

Human Action Recognition is the task of automatically identifying human activities from video sequences.

In this project, a video is represented as a sequence of sampled frames:

Video
→ Frame Extraction
→ Frame Sequences
→ CNN Feature Extraction
→ LSTM Temporal Modeling
→ Action Classification

The current implementation uses a **MobileNetV3-Small + LSTM** baseline and is evaluated on **101 action classes from UCF101**.

---

## Key Results

The final trained CNN-LSTM model was evaluated on the UCF101 test set containing:

- **101 action classes**
- **29,649 test sequences**

### Test Performance

| Metric | Result |
|---|---:|
| Test Loss | **0.5555** |
| Top-1 Accuracy | **85.26%** |
| Top-5 Accuracy | **97.30%** |
| Precision | **82.68%** |
| Recall | **83.74%** |
| F1 Score | **82.79%** |

The reported precision, recall, and F1 values are based on the classification-report evaluation pipeline, which also generates per-class and macro/weighted statistics. 

---

## Model Benchmark

The trained model was benchmarked on a CUDA-enabled GPU.

| Property | Result |
|---|---:|
| Device | CUDA |
| Parameters | **2,333,317** |
| Model Size | **8.9 MB** |
| Inference Latency | **8.17 ms** |
| Throughput | **122.4 FPS** |

The benchmark demonstrates that the current CNN-LSTM baseline is lightweight enough to support high-throughput inference.

---

## Dataset

### UCF101

The project uses the **UCF101 Human Action Recognition Dataset**.

The dataset contains 101 human-action categories covering activities such as sports, household activities, musical activities, and human interactions.

Examples include:

- ApplyEyeMakeup
- ApplyLipstick
- Archery
- Basketball
- Billiards
- Boxing
- GolfSwing
- HorseRiding
- JumpingJack
- PlayingGuitar
- PlayingPiano
- PushUps
- Rowing
- SkateBoarding
- SoccerPenalty
- Surfing
- TennisSwing
- WalkingWithDog
- YoYo

The complete dataset contains 101 classes.

---

# Dataset Processing Pipeline

The raw UCF101 videos are converted into frame sequences before training.

```text
UCF101 Videos
      │
      ▼
Frame Extraction
      │
      ▼
Processed Frames
      │
      ▼
Temporal Sequence Generation
      │
      ▼
Train / Validation / Test Splits
      │
      ▼
CNN-LSTM Model