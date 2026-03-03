# YOLO Benchmark Framework

A comprehensive benchmarking framework for evaluating YOLO11 object detection models on Raspberry Pi devices. This project systematically measures performance and resource consumption across different model configurations, inference frameworks, and hardware utilization scenarios.

## 🎯 Overview

This framework enables automated benchmarking of YOLO11 models with various configurations:
- **Model Variants**: nano (n), small (s), medium (m), large (l), and extra-large (x)
- **Model Formats**: ONNX and TensorFlow Lite (TFLite)
- **Precision Levels**: float32, float16, and int8 (TFLite only)
- **Implementation Languages**: Python and C++
- **CPU Utilization**: Full cores or half cores

The system measures both **performance metrics** (preprocessing, inference, and postprocessing times) and **consumption metrics** (CPU usage, temperature, RAM usage, and current consumption).

## 📁 Project Structure

```
yolo-benchmark/
├── data/
│   ├── datasets/
│   │   └── coco128/              # COCO128 dataset for testing
│   │       ├── images/
│   │       └── labels/
│   ├── models/
│   │   ├── onnx/                 # ONNX model files
│   │   │   ├── yolo11n_float16.onnx
│   │   │   ├── yolo11n_float32.onnx
│   │   │   └── ...
│   │   └── tflite/               # TensorFlow Lite model files
│   │       ├── yolo11n_float16.tflite
│   │       ├── yolo11n_float32.tflite
│   │       ├── yolo11n_int8.tflite
│   │       └── ...
│   └── output/                   # Benchmark results by device
│       ├── Raspberry_Pi_3_Model_B_Plus_Rev_1.4/
│       └── Raspberry_Pi_5_Model_B_Rev_1.0/
├── scripts/
│   └── auto_benchmarking.sh      # Automated benchmarking script
├── src/
│   ├── InferenceBenckmark/       # Main benchmarking orchestrator
│   │   ├── main.py
│   │   ├── interface/            # MQTT consumer for IPC
│   │   ├── monitor/              # System metrics monitoring
│   │   │   ├── performancemetrics.py
│   │   │   └── consumptionmetrics.py
│   │   └── report/               # Report generation
│   │       └── table.py
│   └── InferenceEngine/          # Inference implementations
│       ├── python/               # Python YOLO inference engine
│       │   ├── main.py
│       │   ├── ai/
│       │   │   ├── architectures/    # YOLO architecture implementations
│       │   │   ├── inferencers/      # Runtime backends (LiteRT, ONNX)
│       │   │   └── processors/       # Detector interface
│       │   ├── detection/            # Post-processing (NMS, etc.)
│       │   ├── image/                # Image preprocessing & plotting
│       │   ├── interface/            # MQTT producer for IPC
│       │   └── model/
│       └── cpp/                  # C++ YOLO inference engine
│           ├── main.cpp
│           ├── CMakeLists.txt
│           ├── ai/
│           ├── detection/
│           ├── image/
│           ├── interface/
│           ├── model/
│           └── utils/
└── .gitignore
```

## 🏗️ Architecture

The framework consists of two main components that communicate via MQTT:

### 1. Inference Engine
- **Purpose**: Executes YOLO inference on images
- **Implementations**: Separate Python and C++ versions
- **Responsibilities**:
  - Load and configure YOLO models (ONNX or TFLite)
  - Process images from the dataset
  - Perform object detection
  - Measure timing for each pipeline stage
  - Send performance data via MQTT

### 2. Inference Benchmark
- **Purpose**: Orchestrates benchmarking and monitors system resources
- **Responsibilities**:
  - Launch inference engines as child processes
  - Monitor CPU usage, temperature, RAM usage
  - Measure current consumption (using Raspberry Pi 5 internal sensor or external measurements)
  - Collect performance metrics via MQTT
  - Generate comprehensive CSV reports

### Communication Flow
```
┌─────────────────────┐           MQTT Topics           ┌──────────────────────┐
│  Inference Engine   │◄────────────────────────────────►│ Inference Benchmark  │
│  (Python/C++)       │  • inferenceEngine/status       │                      │
│                     │  • inferenceEngine/data         │                      │
│  - Image Processing │                                 │  - Resource Monitor  │
│  - Model Inference  │                                 │  - Metrics Collector │
│  - Time Measurement │                                 │  - Report Generator  │
└─────────────────────┘                                 └──────────────────────┘
```

## 🚀 Getting Started

### Prerequisites

#### Hardware
- Raspberry Pi (tested on Pi 3B+ and Pi 5)
- Sufficient storage for models and datasets (~5GB recommended)

#### Software Dependencies

**Python Dependencies:**
- Python 3.7+
- OpenCV (`opencv-python`)
- ONNX Runtime (`onnxruntime`)
- TensorFlow Lite Runtime (`tflite-runtime`)
- Paho MQTT (`paho-mqtt`)
- psutil (`psutil`)
- tqdm (`tqdm`)
- numpy

**C++ Dependencies:**
- CMake 3.10+
- OpenCV
- TensorFlow Lite
- ONNX Runtime
- Paho MQTT C++ (`paho-mqtt-cpp`)
- indicators (progress bar library)
- C++17 compatible compiler

**System Requirements:**
- MQTT broker (e.g., Mosquitto) running on localhost:1883

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd yolo-benchmark
   ```

2. **Install MQTT Broker:**
   ```bash
   sudo apt-get update
   sudo apt-get install mosquitto mosquitto-clients
   sudo systemctl enable mosquitto
   sudo systemctl start mosquitto
   ```

3. **Install Python dependencies:**
   ```bash
   pip3 install opencv-python onnxruntime tflite-runtime paho-mqtt psutil tqdm numpy
   ```

4. **Build C++ inference engine:**
   ```bash
   cd src/InferenceEngine/cpp
   mkdir build && cd build
   cmake ..
   make
   ```

5. **Prepare the dataset:**
   - Download COCO128 dataset
   - Extract to `data/datasets/coco128/`

6. **Add YOLO models:**
   - Place ONNX models in `data/models/onnx/`
   - Place TFLite models in `data/models/tflite/`

### Model Preparation

Models should follow this naming convention:
- `{architecture}_{precision}.{format}`
- Examples: `yolo11n_float32.onnx`, `yolo11m_int8.tflite`

Supported architectures: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`

## 📊 Usage

### Manual Benchmarking

Run a single benchmark experiment:

```bash
python3 src/InferenceBenckmark/main.py \
    --images_folder data/datasets/coco128/images/train2017/ \
    --model_path data/models/onnx/yolo11n_float32.onnx \
    --language python
```

**Available options:**
- `--images_folder`: Path to folder containing test images
- `--model_path`: Path to YOLO model file (.onnx or .tflite)
- `--language`: Implementation language (`python` or `cpp`)
- `--half_cores`: Optional flag to use only half of available CPU cores

### Automated Benchmarking

Run comprehensive benchmarks across all configurations:

```bash
cd /home/pi/yolo-benchmark
./scripts/auto_benchmarking.sh
```

This script will:
- Test all 5 YOLO11 variants (n, s, m, l, x)
- Test all precision levels (float32, float16, int8)
- Test all formats (ONNX, TFLite)
- Test both Python and C++ implementations
- Test both full and half core configurations
- Generate 60 benchmark reports per device (5 models × 3 precisions × 2 formats × 2 languages × 2 core configs)

**Note**: Complete execution takes several hours. Each experiment includes a 10-second cooldown period between runs.

## 📈 Output and Reports

### Output Structure

Results are saved in device-specific folders:
```
data/output/{DEVICE_NAME}/{MODEL}_{PRECISION}_{FORMAT}_{LANGUAGE}_{CORES}/{TIMESTAMP}/
├── performance.csv       # Timing metrics for each image
├── consumption.csv       # Resource usage metrics
└── detections/          # (Optional) Annotated images
```

### Performance Metrics CSV

Contains per-image timing data:
| Sample | Preprocessing time (ms) | Inference time (ms) | Post processing time (ms) |
|--------|------------------------|---------------------|---------------------------|
| 0      | 45.2                   | 234.5               | 12.3                      |
| 1      | 44.8                   | 236.1               | 11.9                      |
| ...    | ...                    | ...                 | ...                       |

### Consumption Metrics CSV

Contains resource utilization data:
| Sample | CPU usage (%) | CPU temperature (°C) | RAM usage (MB) | Current consumption (mA) |
|--------|---------------|----------------------|----------------|--------------------------|
| 0      | 85.3          | 58.2                 | 1245.6         | 1850                     |
| 1      | 87.1          | 59.1                 | 1247.3         | 1920                     |
| ...    | ...           | ...                  | ...            | ...                      |

### Current Consumption Measurement

- **Raspberry Pi 5**: Uses internal current sensor for automatic measurement
- **Raspberry Pi 3B+ and earlier**: Requires manual input of minimum and maximum current readings (measured with external ammeter)

## 🔧 Technical Details

### Inference Backends

#### Python Implementation
- **ONNX Runtime**: For `.onnx` models
- **TensorFlow Lite Runtime**: For `.tflite` models
- Memory-efficient implementation suitable for resource-constrained devices

#### C++ Implementation
- **ONNX Runtime**: For `.onnx` models
- **TensorFlow Lite**: For `.tflite` models
- Optimized for performance with native compilation

### YOLO Architecture Support

Currently supports **Ultralytics YOLO** format:
- YOLO11 (all variants: n, s, m, l, x)
- Compatible with YOLOv8 architecture
- Configurable thresholds:
  - Score threshold: 0.25
  - Confidence threshold: 0.5
  - IoU threshold: 0.5

### CPU Core Management

The framework can limit CPU core usage for power consumption studies:
- **Full cores**: Uses all available CPU cores
- **Half cores**: Limits execution to half of available cores
- Useful for analyzing performance/power trade-offs

## 🧪 Benchmarking Methodology

1. **Initialization**: Load model and configure inference parameters
2. **Warm-up**: First inference may be slower (not measured separately)
3. **Execution**: Process all images in the dataset sequentially
4. **Monitoring**: Sample system metrics every 100ms during execution
5. **Collection**: Gather timing data via MQTT for each image
6. **Reporting**: Generate CSV files with aggregated results
7. **Cooldown**: 10-second pause before next experiment

### Measured Stages

- **Preprocessing**: Image resizing, normalization, format conversion
- **Inference**: Model execution time
- **Postprocessing**: Non-Maximum Suppression (NMS), bounding box formatting

## 🎓 Research Applications

This framework is designed for doctoral research and enables:
- **Performance Analysis**: Compare inference times across model variants
- **Power Efficiency Studies**: Analyze energy consumption patterns
- **Optimization Research**: Evaluate quantization effects (float32 vs float16 vs int8)
- **Edge Computing**: Assess feasibility of running YOLO models on resource-constrained devices
- **Language Comparison**: Python vs C++ implementation trade-offs
- **Hardware Evaluation**: Compare performance across Raspberry Pi generations

**Last Updated**: March 2026
