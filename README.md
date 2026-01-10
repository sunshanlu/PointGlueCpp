# PointGlueCpp

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.10+-064F8C.svg)](https://cmake.org/)
[![Doxygen](https://img.shields.io/badge/docs-Doxygen-blue.svg)](docs/)
[![Sphinx](https://img.shields.io/badge/docs-Sphinx-green.svg)](docs/)

A high-performance C++ implementation of SuperPoint and SuperGlue for feature detection and matching.

English | [简体中文](README_CN.md)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Examples](#examples)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

PointGlueCpp is a modern C++20 library that provides efficient implementations of SuperPoint and SuperGlue neural networks for feature detection and matching tasks. The library is designed with performance, flexibility, and ease of use in mind, supporting multiple inference backends including ONNX Runtime and Intel OpenVINO.

### Key Characteristics

- **High Performance**: Optimized C++20 implementation with multi-threading support via TBB
- **Flexible Inference**: Support for ONNX Runtime and OpenVINO backends
- **Production Ready**: Comprehensive error handling, logging, and testing
- **Well Documented**: Complete Doxygen and Sphinx documentation
- **Easy Integration**: Clean API design with smart pointers and RAII

## ✨ Features

### SuperPoint Module
- Feature point detection with configurable confidence thresholds
- 256-dimensional descriptor extraction
- Fast-NMS (Non-Maximum Suppression) for duplicate removal
- Support for custom input dimensions
- YAML-based configuration

### SuperGlue Module
- Attention-based feature matching
- Match confidence scoring
- Visualization utilities for match results
- Support for variable numbers of keypoints
- YAML-based configuration

### Inference Engines
- **ONNX Runtime**: CPU and CUDA support
- **OpenVINO**: CPU, GPU, and VPU acceleration
- Seamless switching between engines
- Warm-up optimization for stable inference latency

### Utilities
- **TensorView**: Zero-copy tensor wrapper with ownership management
- **ViewerTools**: OpenCV-based visualization for keypoints and matches
- **DebugTools**: Tensor data export and error checking utilities

## 🏗️ Architecture

```
PointGlueCpp/
├── include/pointgluecpp/      # Public API headers
│   ├── SuperPoint/            # SuperPoint implementation
│   ├── SuperGlue/             # SuperGlue implementation
│   └── utils/                 # Utility classes
├── src/                       # Implementation files
├── examples/                  # Example programs
├── test/                      # Unit tests
├── res/                       # Resources (models, configs, images)
└── 3rdparty/npycpp/           # NumPy file I/O library
```

### Module Dependencies

```
SuperPoint
├── SuperPointEngine (Abstract Base)
│   ├── ONNXRuntimeEngine
│   └── OpenVINOEngine
└── Utils (TensorView, ViewerTools)

SuperGlue
├── SuperGlueEngine (Abstract Base)
│   ├── ONNXRuntimeEngine
│   └── OpenVINOEngine
└── Utils (TensorView, ViewerTools)
```

## 📦 Requirements

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Compiler**: GCC 9+ / Clang 10+ with C++20 support
- **CMake**: Version 3.10 or higher

### Dependencies
- **OpenCV**: 4.x (for image processing)
- **spdlog**: 1.x (for logging)
- **TBB**: 2021.x (Intel Threading Building Blocks)
- **ONNX Runtime**: 1.10+ (optional, for ONNX Runtime backend)
- **OpenVINO**: 2022.0+ (optional, for OpenVINO backend)
- **GTest**: 1.11+ (for testing)

### Python Dependencies (for documentation)
- Python 3.8+
- Sphinx 4.x
- Breathe 4.x

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/sunshanlu/PointGlueCpp.git
cd PointGlueCpp
git submodule update --init --recursive
```

### Step 2: Install Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install cmake build-essential
sudo apt-get install libopencv-dev libspdlog-dev libtbb-dev
sudo apt-get install libgtest-dev
```

#### Install ONNX Runtime

Using vcpkg:

```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install onnxruntime
```

Or build from source: https://github.com/microsoft/onnxruntime

#### Install OpenVINO

```bash
sudo apt-get install openvino-dev
```

Or download from: https://docs.openvino.ai/

### Step 3: Build Project

```bash
mkdir build && cd build

# Basic configuration
cmake .. -DCMAKE_BUILD_TYPE=Release

# Optional: Select inference engines
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_ONNXRUNTIME=ON \
         -DUSE_OPENVINO=ON

# Optional: Build examples and tests
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_EXAMPLES=ON \
         -DBUILD_TESTING=ON

# Build
cmake --build . --config Release

# Install (optional)
cmake --install .
```

### Step 4: Prepare Models

Download SuperPoint and SuperGlue ONNX models and place them in `res/weights/`.

Configure YAML files in `res/options/`:

**SuperPoint Options** (`res/options/SuperPointOptions.yaml`):
```yaml
model_path: "/path/to/superpoint.onnx"
width: 320
height: 320
border: 8
conf_threshold: 0.015
nms_dist: 4
```

**SuperGlue Options** (`res/options/SuperGlueOptions.yaml`):
```yaml
model_path: "/path/to/superglue.onnx"
# Additional configuration...
```

## 🎮 Quick Start

### Basic Usage

```cpp
#include "pointgluecpp/SuperPoint/SuperPoint.h"
#include "pointgluecpp/SuperGlue/SuperGlue.h"

// Load images
cv::Mat image0 = cv::imread("image0.png", cv::IMREAD_GRAYSCALE);
cv::Mat image1 = cv::imread("image1.png", cv::IMREAD_GRAYSCALE);

// Initialize SuperPoint
auto sp_options = sp::SuperPoint::Options::CreateFromYaml(
    "res/options/SuperPointOptions.yaml"
);
sp::SuperPoint super_point(std::move(sp_options));

// Extract features from both images
auto [kpts0, desc0] = super_point.RunSession(image0);
auto [kpts1, desc1] = super_point.RunSession(image1);

// Initialize SuperGlue
auto sg_options = sg::SuperGlue::Options::CreateFromYaml(
    "res/options/SuperGlueOptions.yaml"
);
sg::SuperGlue super_glue(std::move(sg_options));

// Match features
sp::SuperPoint::SuperPointRet ret0{kpts0, desc0};
sp::SuperPoint::SuperPointRet ret1{kpts1, desc1};
auto matches = super_glue.RunSession(ret0, ret1, image0.size(), image1.size());

// Visualize results
sg::SuperGlue::ShowMatches(image0, image1, matches, kpts0, kpts1);
```

### Run Examples

```bash
cd build/Release/bin

# SuperPoint example
./superpoint_example

# SuperGlue example
./superglue_example

# Complete pipeline example
./pointglue_example
```

### Run Tests

```bash
cd build
ctest --output-on-failure

# Or use custom target
make test_units

# Memory leak detection (requires Valgrind)
make test_memory
```

## 📚 Documentation

### API Documentation

Generate Doxygen XML files:

```bash
cd docs
doxygen Doxyfile
```

Build Sphinx documentation website:

```bash
cd docs/source
sphinx-build -b html ../build/html
```

Open `docs/build/html/index.html` in your browser to view the documentation.

### Documentation Structure

- **index.rst**: Project overview and navigation
- **api.rst**: Complete API reference with Doxygen integration
- **quickstart.rst**: Installation and usage guide
- **changelog.rst**: Version history and release notes

## 💡 Examples

The `examples/` directory contains three example programs:

1. **superpoint.cc**: Demonstrates SuperPoint feature detection
2. **superglue.cc**: Demonstrates SuperGlue feature matching
3. **pointglue.cc**: Complete pipeline combining both modules

Each example includes detailed comments and can be used as a starting point for your own applications.

## 📊 Performance

### Running Benchmarks

The project includes performance benchmarks using Google Benchmark framework to measure inference latency.

#### Install Google Benchmark

```bash
# Using vcpkg
./vcpkg install benchmark

# Or build from source
git clone https://github.com/google/benchmark.git
cd benchmark
cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
make -j
sudo make install
```

#### Build Benchmarks

```bash
cd build
cmake .. -DBUILD_BENCHMARKS=ON
cmake --build . --config Release
```

#### Run Benchmarks

```bash
# Run all benchmarks
cd build/Release/bin
./benchmark_superpoint
./benchmark_superglue
./benchmark_pointglue

# Or use custom target
cd build
make run_benchmarks

# Run with specific options
./benchmark_superpoint --benchmark_repetitions=10 --benchmark_out=results.json
```

#### Benchmark Options

- `--benchmark_repetitions=N`: Number of repetitions (default: 5)
- `--benchmark_out=FILE`: Output results to JSON file
- `--benchmark_filter=REGEX`: Filter benchmarks by regex
- `--benchmark_display_aggregates_only`: Show only aggregate statistics

#### Benchmark Results

Benchmarks measure the following scenarios:

**SuperPoint Benchmarks** (`benchmark_superpoint`):
- Inference on 640x480 images
- Inference on 320x320 images

**SuperGlue Benchmarks** (`benchmark_superglue`):
- Matching with few keypoints (< 100)
- Matching with medium keypoints (100-500)
- Matching with many keypoints (> 500)

**Full Pipeline Benchmarks** (`benchmark_pointglue`):
- Complete pipeline on 640x480 images
- Complete pipeline on 320x320 images

*Note: Actual performance depends on hardware, model configuration, and input size. Run benchmarks on your target system for accurate measurements.*

### Optimization Tips

1. **Use OpenVINO** for Intel CPUs: Typically 20-30% faster than ONNX Runtime
2. **Enable CUDA** for NVIDIA GPUs: 5-10x speedup
3. **Adjust input size**: Smaller images are faster but may reduce accuracy
4. **Batch processing**: Process multiple images in parallel when possible
5. **Warm-up**: Always perform warm-up runs before measuring performance

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow existing code conventions
- Use C++20 features where appropriate
- Add Doxygen comments for all public APIs
- Write unit tests for new features
- Ensure all tests pass before submitting

### Development Workflow

```bash
# Install development dependencies
sudo apt-get install clang-format clang-tidy

# Format code
find . -name "*.cc" -o -name "*.h" | xargs clang-format -i

# Run static analysis
clang-tidy src/*.cc include/**/*.h -- -std=c++20

# Run tests
cd build
make test_units
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

PointGlueCpp is built upon the following excellent research and open-source projects:

- **SuperPoint**: [SuperPoint: Self-Supervised Interest Point Detection and Description](https://github.com/magicleap/SuperPoint) by Magic Leap
- **SuperGlue**: [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://github.com/magicleap/SuperGlue) by Magic Leap
- **ONNX Runtime**: [ONNX Runtime](https://github.com/microsoft/onnxruntime) by Microsoft
- **OpenVINO**: [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino) by Intel
- **OpenCV**: [Open Source Computer Vision Library](https://opencv.org/)
- **spdlog**: [Fast C++ logging library](https://github.com/gabime/spdlog)

## 📧 Contact

- **Author**: ssl
- **GitHub**: [sunshanlu/PointGlueCpp](https://github.com/sunshanlu/PointGlueCpp)
- **Issues**: [GitHub Issues](https://github.com/sunshanlu/PointGlueCpp/issues)

## 🔗 Links

- [Documentation](docs/)
- [Examples](examples/)
- [API Reference](docs/build/html/api.html)
- [Changelog](docs/build/html/changelog.html)

---

**Note**: This project is under active development. API changes may occur between versions. Please refer to the changelog for detailed release notes.