# TrackNetV5 SDK Documentation

**[中文版 (Chinese Version)](README_CN.md)**

This repository is the official Software Development Kit (SDK) for **TrackNetV5**, providing a standardized engineering implementation of the tennis ball tracking algorithm. Developed and maintained by **Shanghai Code Zero Sports Technology Co., Ltd.**

The core architecture and algorithmic logic of TrackNetV5 are based on our latest research:

* **Title**: *TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking*
* **Paper**: [arXiv:2512.02789](https://arxiv.org/abs/2512.02789)

## Core Specifications

* **Architecture Support**: Specifically designed for the TrackNetV5 architecture; but also implement V2/V4 versions in pytorch.
* **Integrated Features**: Encapsulates three-frame sliding window inference, Gaussian heatmap centroid extraction, trajectory enhancement visualization, and an industrial-grade training pipeline.
* **Confidentiality Notice**: Model weights and training datasets are proprietary assets of the company and are currently not open to the public.

---

## 1. Environment Configuration

This project is optimized for specific computing environments. To ensure system stability, please use the following recommended versions:

### Core Dependencies

| Component | Recommended Version |
| --- | --- |
| **Python** | 3.8+ |
| **CUDA** | 12.6 |
| **PyTorch** | 2.9.0+cu126 |
| **Torchvision** | 0.24.0+cu126 |

### Installation

```bash
# 1. Install basic scientific computing and image processing libraries
pip install -r requirements.txt

# 2. Install specific PyTorch ecosystem versions
# It is recommended to download the corresponding .whl files from the official PyTorch website

```

---

## 2. Data Preparation

The SDK utilizes **Gaussian Heatmaps** as the supervision signal.

### Dataset Standards

Please strictly follow the directory structure and labeling specifications of the following repository for custom datasets:

* **Reference**: `WASB-TrainingOK` Dataset Specification.

### Preprocessing Script

Use `tools/preprocess_data_gauss.py` to convert raw video frames and `Label.csv` into the spatio-temporal context tensors required by the V5 architecture.

```bash
python tools/preprocess_data_gauss.py \
    --input_dir <path_to_raw_data> \
    --output_dir <path_to_output> \
    --mode context \
    --train_rate 0.8 \
    --height 1080 --width 1920

```

* **Key Parameters**:
* `--mode`: Must be set to `context` (generates associative indices for the three-frame sliding window inference).
* `--size` & `--variance`: Controls the radius and variance of the generated Gaussian spots.



---

## 3. Training Guide

Training tasks are dispatched via `train.py`, which utilizes a **Factory Pattern** for dynamic component construction.

### Start Training Queue

```bash
python train.py

```

### Operational Steps

1. **Auto-Scan**: The system lists all `.py` configuration files in the `./configs/` directory.
2. **Selection**: Enter the configuration index (supports space-separated multi-task queues, e.g., `1 3 5`).
3. **Execution**: The `Runner` orchestrator automatically handles instantiation, Learning Rate Warmup, Gradient Clipping (`GradClip`), and Hook plugin mounting.

---

## 4. Inference Pipeline

The inference module supports batch video processing and structured data export.

### Execution Command

```bash
python main.py <input_dir> <weights_path> --arch v2/v4/v5 --threshold 0.5 --device cuda:0

```

### Output Description

Results are organized under `input_dir/{arch}/` by video filename:

* `_summary_report_{arch}.csv`: Summary of detection rates and frame statistics for all processed videos.
* `*_data.csv`: Frame-by-frame coordinate mapping (including detection status and `cx`, `cy` centroids).
* `*_trajectory.mp4`: **Enhanced trajectory video** (featuring a "comet tail" effect).
* `*_comparison.mp4`: Synchronized side-by-side comparison of the original video and predicted heatmaps.

---

## 5. Architecture Deep-Dive & Resources

The engineering design patterns, TrackNetV5 model details, and underlying inference logic are documented in our exclusive **Obsidian Visual Knowledge Base**.

> [!IMPORTANT]
> **Access**: The Obsidian repository is a private resource. For in-depth development, architectural study, or technical exchange, please contact the author via **Email** to request authorization.

---
## License
This SDK is proprietary software. All rights reserved by Shanghai Code Zero Sports Technology Co., Ltd. The source code is provided for technical exchange and academic study only.

© 2025 Shanghai Code Zero Sports Technology Co., Ltd.

---
