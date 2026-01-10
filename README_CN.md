# PointGlueCpp

[![许可证](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.10+-064F8C.svg)](https://cmake.org/)
[![Doxygen](https://img.shields.io/badge/docs-Doxygen-blue.svg)](docs/)
[![Sphinx](https://img.shields.io/badge/docs-Sphinx-green.svg)](docs/)

SuperPoint 和 SuperGlue 的高性能 C++ 实现，用于特征检测和匹配。

[English](README.md) | 简体中文

## 📋 目录

- [项目概述](#项目概述)
- [功能特性](#功能特性)
- [架构设计](#架构设计)
- [系统要求](#系统要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [文档说明](#文档说明)
- [示例程序](#示例程序)
- [性能基准](#性能基准)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [致谢](#致谢)

## 🎯 项目概述

PointGlueCpp 是一个现代化的 C++20 库，提供了 SuperPoint 和 SuperGlue 神经网络的高效实现，用于特征检测和匹配任务。该库设计注重性能、灵活性和易用性，支持多种推理后端，包括 ONNX Runtime 和 Intel OpenVINO。

### 核心特性

- **高性能**: 优化的 C++20 实现，通过 TBB 支持多线程
- **灵活推理**: 支持 ONNX Runtime 和 OpenVINO 后端
- **生产就绪**: 完善的错误处理、日志记录和测试
- **文档完善**: 完整的 Doxygen 和 Sphinx 文档
- **易于集成**: 清晰的 API 设计，使用智能指针和 RAII

## ✨ 功能特性

### SuperPoint 模块
- 可配置置信度阈值的特征点检测
- 256 维描述符提取
- Fast-NMS（非极大值抑制）去除重复点
- 支持自定义输入尺寸
- 基于 YAML 的配置

### SuperGlue 模块
- 基于注意力机制的特征匹配
- 匹配置信度评分
- 匹配结果可视化工具
- 支持可变数量的关键点
- 基于 YAML 的配置

### 推理引擎
- **ONNX Runtime**: 支持 CPU 和 CUDA
- **OpenVINO**: 支持 CPU、GPU 和 VPU 加速
- 引擎间无缝切换
- 预热优化以获得稳定的推理延迟

### 工具类
- **TensorView**: 零拷贝张量包装器，支持所有权管理
- **ViewerTools**: 基于 OpenCV 的可视化工具有关关键点和匹配
- **DebugTools**: 张量数据导出和错误检查工具

## 🏗️ 架构设计

```
PointGlueCpp/
├── include/pointgluecpp/      # 公共 API 头文件
│   ├── SuperPoint/            # SuperPoint 实现
│   ├── SuperGlue/             # SuperGlue 实现
│   └── utils/                 # 工具类
├── src/                       # 实现文件
├── examples/                  # 示例程序
├── test/                      # 单元测试
├── res/                       # 资源文件（模型、配置、图像）
└── 3rdparty/npycpp/           # NumPy 文件 I/O 库
```

### 模块依赖

```
SuperPoint
├── SuperPointEngine (抽象基类)
│   ├── ONNXRuntimeEngine
│   └── OpenVINOEngine
└── Utils (TensorView, ViewerTools)

SuperGlue
├── SuperGlueEngine (抽象基类)
│   ├── ONNXRuntimeEngine
│   └── OpenVINOEngine
└── Utils (TensorView, ViewerTools)
```

## 📦 系统要求

### 系统要求
- **操作系统**: Linux（推荐 Ubuntu 20.04+）
- **编译器**: GCC 9+ / Clang 10+，支持 C++20
- **CMake**: 版本 3.10 或更高

### 依赖库
- **OpenCV**: 4.x（用于图像处理）
- **spdlog**: 1.x（用于日志记录）
- **TBB**: 2021.x（Intel 线程构建块）
- **ONNX Runtime**: 1.10+（可选，用于 ONNX Runtime 后端）
- **OpenVINO**: 2022.0+（可选，用于 OpenVINO 后端）
- **GTest**: 1.11+（用于测试）

### Python 依赖（用于文档）
- Python 3.8+
- Sphinx 4.x
- Breathe 4.x

## 🚀 安装指南

### 第一步：克隆仓库

```bash
git clone https://github.com/sunshanlu/PointGlueCpp.git
cd PointGlueCpp
git submodule update --init --recursive
```

### 第二步：安装依赖

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install cmake build-essential
sudo apt-get install libopencv-dev libspdlog-dev libtbb-dev
sudo apt-get install libgtest-dev
```

#### 安装 ONNX Runtime

使用 vcpkg:

```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install onnxruntime
```

或从源码编译: https://github.com/microsoft/onnxruntime

#### 安装 OpenVINO

```bash
sudo apt-get install openvino-dev
```

或从官网下载: https://docs.openvino.ai/

### 第三步：构建项目

```bash
mkdir build && cd build

# 基本配置
cmake .. -DCMAKE_BUILD_TYPE=Release

# 可选：选择推理引擎
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_ONNXRUNTIME=ON \
         -DUSE_OPENVINO=ON

# 可选：构建示例和测试
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_EXAMPLES=ON \
         -DBUILD_TESTING=ON

# 编译
cmake --build . --config Release

# 安装（可选）
cmake --install .
```

### 第四步：准备模型

下载 SuperPoint 和 SuperGlue 的 ONNX 模型，并将它们放置到 `res/weights/` 目录下。

在 `res/options/` 中配置 YAML 文件：

**SuperPoint 配置** (`res/options/SuperPointOptions.yaml`):
```yaml
model_path: "/path/to/superpoint.onnx"
width: 320
height: 320
border: 8
conf_threshold: 0.015
nms_dist: 4
```

**SuperGlue 配置** (`res/options/SuperGlueOptions.yaml`):
```yaml
model_path: "/path/to/superglue.onnx"
# 其他配置...
```

## 🎮 快速开始

### 基本用法

```cpp
#include "pointgluecpp/SuperPoint/SuperPoint.h"
#include "pointgluecpp/SuperGlue/SuperGlue.h"

// 加载图像
cv::Mat image0 = cv::imread("image0.png", cv::IMREAD_GRAYSCALE);
cv::Mat image1 = cv::imread("image1.png", cv::IMREAD_GRAYSCALE);

// 初始化 SuperPoint
auto sp_options = sp::SuperPoint::Options::CreateFromYaml(
    "res/options/SuperPointOptions.yaml"
);
sp::SuperPoint super_point(std::move(sp_options));

// 从两张图像中提取特征
auto [kpts0, desc0] = super_point.RunSession(image0);
auto [kpts1, desc1] = super_point.RunSession(image1);

// 初始化 SuperGlue
auto sg_options = sg::SuperGlue::Options::CreateFromYaml(
    "res/options/SuperGlueOptions.yaml"
);
sg::SuperGlue super_glue(std::move(sg_options));

// 匹配特征
sp::SuperPoint::SuperPointRet ret0{kpts0, desc0};
sp::SuperPoint::SuperPointRet ret1{kpts1, desc1};
auto matches = super_glue.RunSession(ret0, ret1, image0.size(), image1.size());

// 可视化结果
sg::SuperGlue::ShowMatches(image0, image1, matches, kpts0, kpts1);
```

### 运行示例

```bash
cd build/Release/bin

# SuperPoint 示例
./superpoint_example

# SuperGlue 示例
./superglue_example

# 完整流程示例
./pointglue_example
```

### 运行测试

```bash
cd build
ctest --output-on-failure

# 或使用自定义目标
make test_units

# 内存泄漏检测（需要 Valgrind）
make test_memory
```

## 📚 文档说明

### API 文档

生成 Doxygen XML 文件:

```bash
cd docs
doxygen Doxyfile
```

构建 Sphinx 文档网站:

```bash
cd docs/source
sphinx-build -b html ../build/html
```

在浏览器中打开 `docs/build/html/index.html` 查看文档。

### 文档结构

- **index.rst**: 项目概述和导航
- **api.rst**: 完整的 API 参考，集成 Doxygen
- **quickstart.rst**: 安装和使用指南
- **changelog.rst**: 版本历史和发布说明

## 💡 示例程序

`examples/` 目录包含三个示例程序：

1. **superpoint.cc**: 演示 SuperPoint 特征检测
2. **superglue.cc**: 演示 SuperGlue 特征匹配
3. **pointglue.cc**: 完整流程，结合两个模块

每个示例都包含详细注释，可作为自己应用程序的起点。

## 📊 性能基准

### 运行性能测试

项目包含使用 Google Benchmark 框架的性能测试，用于测量推理延迟。

#### 安装 Google Benchmark

```bash
# 使用 vcpkg
./vcpkg install benchmark

# 或从源码编译
git clone https://github.com/google/benchmark.git
cd benchmark
cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
make -j
sudo make install
```

#### 构建性能测试

```bash
cd build
cmake .. -DBUILD_BENCHMARKS=ON
cmake --build . --config Release
```

#### 运行性能测试

```bash
# 运行所有性能测试
cd build/Release/bin
./benchmark_superpoint
./benchmark_superglue
./benchmark_pointglue

# 或使用自定义目标
cd build
make run_benchmarks

# 使用特定选项运行
./benchmark_superpoint --benchmark_repetitions=10 --benchmark_out=results.json
```

#### 性能测试选项

- `--benchmark_repetitions=N`: 重复次数（默认：5）
- `--benchmark_out=FILE`: 将结果输出到 JSON 文件
- `--benchmark_filter=REGEX`: 使用正则表达式过滤测试
- `--benchmark_display_aggregates_only`: 仅显示聚合统计信息

#### 性能测试结果

性能测试测量以下场景：

**SuperPoint 性能测试** (`benchmark_superpoint`)：
- 640x480 图像推理
- 320x320 图像推理

**SuperGlue 性能测试** (`benchmark_superglue`)：
- 少量关键点匹配（< 100）
- 中等数量关键点匹配（100-500）
- 大量关键点匹配（> 500）

**完整流程性能测试** (`benchmark_pointglue`)：
- 640x480 图像完整流程
- 320x320 图像完整流程

*注：实际性能取决于硬件、模型配置和输入尺寸。请在目标系统上运行性能测试以获得准确的测量结果。*

### 优化建议

1. **使用 OpenVINO** 用于 Intel CPU：通常比 ONNX Runtime 快 20-30%
2. **启用 CUDA** 用于 NVIDIA GPU：加速 5-10 倍
3. **调整输入尺寸**：较小的图像更快，但可能降低精度
4. **批处理**：尽可能并行处理多个图像
5. **预热**：在测量性能前始终进行预热运行

## 🤝 贡献指南

欢迎贡献！请遵循以下准则：

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码风格

- 遵循现有的代码约定
- 在适当的地方使用 C++20 特性
- 为所有公共 API 添加 Doxygen 注释
- 为新功能编写单元测试
- 提交前确保所有测试通过

### 开发工作流程

```bash
# 安装开发依赖
sudo apt-get install clang-format clang-tidy

# 格式化代码
find . -name "*.cc" -o -name "*.h" | xargs clang-format -i

# 运行静态分析
clang-tidy src/*.cc include/**/*.h -- -std=c++20

# 运行测试
cd build
make test_units
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

PointGlueCpp 建立在以下优秀的研究和开源项目之上：

- **SuperPoint**: [SuperPoint: Self-Supervised Interest Point Detection and Description](https://github.com/magicleap/SuperPoint) by Magic Leap
- **SuperGlue**: [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://github.com/magicleap/SuperGlue) by Magic Leap
- **ONNX Runtime**: [ONNX Runtime](https://github.com/microsoft/onnxruntime) by Microsoft
- **OpenVINO**: [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino) by Intel
- **OpenCV**: [Open Source Computer Vision Library](https://opencv.org/)
- **spdlog**: [Fast C++ logging library](https://github.com/gabime/spdlog)

## 📧 联系方式

- **作者**: ssl
- **GitHub**: [sunshanlu/PointGlueCpp](https://github.com/sunshanlu/PointGlueCpp)
- **问题反馈**: [GitHub Issues](https://github.com/sunshanlu/PointGlueCpp/issues)

## 🔗 相关链接

- [文档](docs/)
- [示例](examples/)
- [API 参考](docs/build/html/api.html)
- [更新日志](docs/build/html/changelog.html)

---

**注意**: 本项目正在积极开发中。版本之间可能会发生 API 更改。请参考更新日志了解详细的发布说明。