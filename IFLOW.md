# PointGlueCpp 项目上下文

## 项目概述

PointGlueCpp 是一个用 C++ 编写的 SuperPoint 和 SuperGlue 推理项目。该项目提供了基于深度学习的特征点检测和匹配功能的 C++ 实现，支持多种推理引擎（ONNX Runtime 和 OpenVINO）。

### 核心功能

- **SuperPoint**: 特征点检测和描述符提取
- **SuperGlue**: 特征点匹配
- **多引擎支持**: ONNX Runtime 和 OpenVINO
- **可视化工具**: 匹配结果的可视化展示

### 技术栈

- **语言**: C++20
- **构建系统**: CMake (最低版本 3.10)
- **依赖库**:
  - OpenCV (图像处理)
  - spdlog (日志)
  - TBB (并行计算)
  - ONNX Runtime 或 OpenVINO (推理引擎)
  - GTest (单元测试)
- **第三方库**:
  - npycpp (NumPy 文件读写，作为子模块集成)
- **文档生成**:
  - Doxygen (API 文档)
  - Sphinx + Breathe (文档网站)

### 项目结构

```
PointGlueCpp/
├── include/pointgluecpp/      # 公共头文件
│   ├── SuperPoint/            # SuperPoint 相关头文件
│   ├── SuperGlue/             # SuperGlue 相关头文件
│   └── utils/                 # 工具类头文件
├── src/                       # 源代码实现
│   ├── SuperPoint/
│   ├── SuperGlue/
│   └── utils/
├── examples/                  # 示例程序
│   ├── superpoint.cc          # SuperPoint 单独使用示例
│   ├── superglue.cc           # SuperGlue 单独使用示例
│   └── pointglue.cc           # SuperPoint + SuperGlue 组合使用示例
├── test/                      # 单元测试
├── res/                       # 资源文件
│   ├── weights/               # 模型权重文件
│   ├── options/               # YAML 配置文件
│   └── images/                # 测试图像
├── docs/                      # 文档
│   ├── source/                # Sphinx 源文件
│   └── build/                 # 构建输出
└── 3rdparty/npycpp/           # 第三方依赖（Git 子模块）
```

## 构建和运行

### 前置要求

- CMake >= 3.10
- C++20 编译器 (GCC, Clang 或 MSVC)
- OpenCV
- spdlog
- TBB
- ONNX Runtime 或 OpenVINO (至少一个)
- GTest (用于测试)

### 构建选项

```bash
# 基本构建
mkdir build && cd build
cmake ..

# 可选的 CMake 选项
- USE_ONNXRUNTIME=ON/OFF    # 使用 ONNX Runtime 引擎（默认 ON）
- USE_OPENVINO=ON/OFF       # 使用 OpenVINO 引擎（默认 ON）
- BUILD_EXAMPLES=ON/OFF     # 构建示例程序（默认 ON）
- BUILD_TESTING=ON/OFF      # 构建测试（默认 ON）
```

### 构建步骤

```bash
# 1. 克隆仓库并初始化子模块
git clone git@github.com:sunshanlu/PointGlueCpp.git
cd PointGlueCpp
git submodule update --init --recursive

# 2. 创建构建目录
mkdir build && cd build

# 3. 配置 CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 4. 编译
cmake --build . --config Release

# 5. 安装（可选）
cmake --install .
```

### 运行示例

```bash
# 运行 SuperPoint 特征点检测示例
./build/Release/bin/superpoint_example

# 运行 SuperGlue 特征点匹配示例
./build/Release/bin/superglue_example

# 运行完整的 PointGlue 流程示例（SuperPoint + SuperGlue）
./build/Release/bin/pointglue_example
```

### 运行测试

```bash
# 运行所有单元测试
cd build
ctest --output-on-failure

# 或使用自定义目标
make test_units

# 内存泄漏检测（需要 Valgrind）
make test_memory
```

### 生成文档

```bash
# 使用 Doxygen 生成 XML
cd docs
doxygen Doxyfile

# 使用 Sphinx 构建文档网站
cd docs/source
sphinx-build -b html ../build/html

# 构建 PDF 文档
cd docs/source
sphinx-build -b latex ../build/latex
cd ../build/latex && make pdf
```

## 开发约定

### 代码风格

- 使用 C++20 标准
- 头文件使用 `#pragma once`
- 使用智能指针 (`std::unique_ptr`, `std::shared_ptr`) 管理资源
- 命名空间:
  - SuperPoint 相关: `sp`
  - SuperGlue 相关: `sg`
- 日志使用 spdlog，级别: `SPDLOG_INFO`, `SPDLOG_ERROR`, `SPDLOG_DEBUG`

### 配置管理

- 使用 YAML 文件管理模型配置
- 配置文件位于 `res/options/` 目录
- 通过 `Options::CreateFromYaml()` 方法加载配置

### 资源文件路径

- 资源文件通过 `RESOURCES_FILES_DIR` 宏定义路径
- 在 CMake 中通过 `add_definitions(-DRESOURCES_FILES_DIR="${PROJECT_SOURCE_DIR}/res")` 设置

### 调试支持

- Debug 模式下启用 `DEBUG` 宏
- 调试输出目录: `output/`
- 支持张量数据保存到文件用于调试

### 测试规范

- 使用 GTest 框架
- 每个测试源文件对应一个可执行文件
- 测试文件命名: `test_*.cpp`
- 使用 `add_test()` 注册测试

### 文档规范

- API 文档使用 Doxygen 注释
- 文档网站使用 Sphinx + Breathe
- 支持中文文档 (`language = 'zh_CN'`)
- 使用 ReadTheDocs 主题

## 核心类说明

### SuperPoint

位于 `include/pointgluecpp/SuperPoint/SuperPoint.h`

主要功能:
- 特征点检测
- 描述符提取
- 支持 ONNX Runtime 和 OpenVINO 引擎

关键方法:
- `CreateFromYaml()`: 从 YAML 文件加载配置
- `RunSession()`: 执行推理
- `PreProcess()`: 图像预处理
- `PostProcess()`: 后处理（包含 Fast-NMS）

### SuperGlue

位于 `include/pointgluecpp/SuperGlue/SuperGlue.h`

主要功能:
- 特征点匹配
- 匹配置信度计算
- 可视化支持

关键方法:
- `CreateFromYaml()`: 从 YAML 文件加载配置
- `RunSession()`: 执行匹配
- `ShowMatches()`: 可视化匹配结果

### 推理引擎

项目提供两种推理引擎实现:
- **ONNXRuntimeEngine**: 基于 ONNX Runtime
- **OpenVINOEngine**: 基于 Intel OpenVINO

引擎通过 CMake 选项动态选择和编译。

## 依赖管理

### 第三方库

项目使用 Git 子模块管理 npycpp 库:
```bash
git submodule update --init --recursive
```

### 安装依赖

#### Ubuntu/Debian
```bash
sudo apt-get install cmake build-essential
sudo apt-get install libopencv-dev libspdlog-dev libtbb-dev
sudo apt-get install libgtest-dev
```

#### ONNX Runtime
```bash
# 使用 vcpkg
vcpkg install onnxruntime

# 或从源码编译
# 参考: https://github.com/microsoft/onnxruntime
```

#### OpenVINO
```bash
# 使用 apt
sudo apt-get install openvino-dev

# 或从官网下载
# 参考: https://docs.openvino.ai/
```

## 常见问题

### 编译错误

1. **找不到 OpenCV**: 确保 OpenCV 正确安装并设置 `OpenCV_DIR`
2. **找不到 ONNX Runtime**: 确保 `onnxruntime_DIR` 正确设置
3. **CMake 版本过低**: 升级到 3.10 或更高版本

### 运行时错误

1. **模型文件未找到**: 检查 YAML 配置中的 `model_path` 是否正确
2. **图像加载失败**: 确保图像路径正确且格式支持
3. **推理引擎初始化失败**: 检查引擎依赖是否正确安装

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码审查要点

- 遵循现有代码风格
- 添加必要的单元测试
- 更新相关文档
- 确保所有测试通过

## 许可证

请参阅 LICENSE 文件了解详细信息。

## 联系方式

- GitHub: https://github.com/sunshanlu/PointGlueCpp
- 作者: ssl