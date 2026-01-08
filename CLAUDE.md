# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Configure build (must choose at least one inference engine)
mkdir build && cd build
cmake -DUSE_ONNXRUNTIME=ON -DUSE_OPENVINO=OFF -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release ..
# OR
cmake -DUSE_ONNXRUNTIME=OFF -DUSE_OPENVINO=ON -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release ..
# OR (both enabled)
cmake -DUSE_ONNXRUNTIME=ON -DUSE_OPENVINO=ON -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release ..

# Build
make -j$(nproc)

# Run examples (from build directory)
./examples/superpoint_example
./examples/superglue_example
./examples/pointglue_example

# Install (optional)
make install
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_ONNXRUNTIME` | ON | Enable ONNX Runtime inference engine |
| `USE_OPENVINO` | ON | Enable OpenVINO inference engine |
| `BUILD_EXAMPLES` | ON | Build example executables |
| `BUILD_TESTING` | ON | Build test suite (currently no tests implemented) |
| `CMAKE_BUILD_TYPE` | Release | Debug mode adds `-DDEBUG` and defines `DEBUG_OUTPUT_DIR` |

**Note**: At least one of `USE_ONNXRUNTIME` or `USE_OPENVINO` must be enabled.

## Architecture Overview

PointGlueCpp is a C++20 implementation of SuperPoint and SuperGlue neural networks for feature detection and matching. The architecture is designed around a pluggable inference engine pattern, allowing the same high-level algorithms to work with different ML inference backends.

### Core Components

1. **SuperPoint** (`sp::SuperPoint`): Keypoint detection and descriptor extraction
   - Detects keypoints with confidence scores
   - Computes 256-dimensional descriptors
   - Uses Fast-NMS for keypoint selection
   - Configurable image size and thresholds

2. **SuperGlue** (`sg::SuperGlue`): Feature matching between image pairs
   - Attention-based feature matching
   - Outputs matches with confidence scores
   - Works with keypoints/descriptors from SuperPoint

3. **Inference Engines**: Pluggable backend abstraction
   - **ONNX Runtime** (`ONNXRuntimeEngine`): General-purpose cross-platform inference
   - **OpenVINO** (`OpenVINOEngine`): Intel-optimized inference
   - Both engines implement `SuperPointEngine` and `SuperGlueEngine` interfaces
   - Selected at compile-time via preprocessor definitions (`-DONNXRUNTIMEENGINE` or `-DOPENVINOENGINE`)

4. **Utilities**:
   - `TensorView`: Zero-copy tensor view template for efficient multi-dimensional array access
   - `ViewerTools`: OpenCV-based visualization for keypoints and matches
   - `DebugTools`: Debug output utilities (only in Debug builds)

### Build Structure

The project builds shared libraries:
- `libpoint.so` - SuperPoint implementation
- `libglue.so` - SuperGlue implementation
- `libutils.so` - Visualization and debug tools

Plus third-party:
- `libnpycpp.so` - NumPy file format support (submodule in `3rdparty/npycpp`)

### Error Handling Convention

Functions return `const char*` for error reporting:
- `nullptr` indicates success
- Non-null pointer is an error message string
- Error messages include file:line information
- Uses `SPDLOG_ERROR` for logging

### Memory Management

- Smart pointers (`std::unique_ptr`) for engine ownership
- `TensorView` supports both owned and non-owned data (zero-copy views possible)
- RAII patterns throughout

### Configuration

Models are configured via YAML files (expected in `res/` directory):
- `SuperPointOptions.yaml`: Model path, input dimensions, thresholds
- `SuperGlueOptions.yaml`: Model path, input/output specifications

The examples use `-DRESOURCES_FILES_DIR` to point to the `res/` directory at compile time.

### Dependencies

**Required:**
- OpenCV (computer vision operations)
- spdlog (logging)
- TBB (Threading Building Blocks for parallelization)
- C++20 compiler

**Required (at least one):**
- ONNX Runtime OR OpenVINO

### Include Structure

Public headers are in `include/pointgluecpp/`:
- `SuperPoint/` - SuperPoint-related headers
- `SuperGlue/` - SuperGlue-related headers
- `utils/` - Utility headers

### Testing

The `test/` directory exists but contains no tests. CTest is configured via `BUILD_TESTING` option, but test implementations need to be added.

### Performance Optimizations

- TBB parallelization for multi-threaded processing
- C++17 parallel algorithms
- Zero-copy tensor views where possible
- Efficient OpenCV integration
