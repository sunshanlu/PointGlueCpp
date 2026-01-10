# PointGlueCpp 文档系统规划

## 目录
- [1. 项目概述](#1-项目概述)
- [2. 文档结构设计](#2-文档结构设计)
- [3. 核心配置文件设计](#3-核心配置文件设计)
- [4. API文档自动生成策略](#4-api文档自动生成策略)
- [5. 内容编写指南和最佳实践](#5-内容编写指南和最佳实践)
- [6. 构建和部署策略](#6-构建和部署策略)
- [7. 实施步骤和时间线](#7-实施步骤和时间线)

---

## 1. 项目概述

### 1.1 项目背景
PointGlueCpp是一个基于C++20实现的SuperPoint和SuperGlue神经网络推理项目，用于图像特征检测和匹配。该项目支持多种推理引擎（ONNX Runtime、OpenVINO），采用模块化设计，提供高性能的计算机视觉功能。

项目特点：
- **模块化设计**：SuperPoint、SuperGlue、utils三大核心库
- **多引擎支持**：可配置使用ONNX Runtime或OpenVINO推理引擎
- **现代C++**：使用C++20标准，智能指针管理资源
- **高性能**：基于TBB并行化，支持多线程处理
- **错误处理**：统一的错误返回机制，便于调试

### 1.2 文档目标
- **主要用户**：软件开发者
- **语言选择**：纯中文文档
- **核心内容**：API文档、构建和安装指南、使用示例、性能优化指南、C++ API最佳实践、版本更新日志
- **更新策略**：版本发布时更新文档

### 1.3 文档系统特点
- 双引擎架构：Doxygen + Sphinx
- 完全自动化构建：通过.readthedocs.yml实现CI/CD
- 版本管理：支持多版本文档
- 中文优先：所有内容均为中文

---

## 2. 文档结构设计

### 2.1 目录结构
```
docs/
├── source/
│   ├── conf.py              # Sphinx主配置文件
│   ├── index.rst            # 文档首页
│   ├── api/                 # API文档目录
│   │   ├── api.rst
│   │   └── modules.rst
│   ├── guides/              # 使用指南
│   │   ├── installation.rst
│   │   ├── building.rst
│   │   └── performance.rst
│   ├── tutorials/           # 教程示例
│   │   ├── getting_started.rst
│   │   ├── superpoint_example.rst
│   │   └── superglue_example.rst
│   ├── architecture/        # 架构设计
│   │   ├── overview.rst
│   │   ├── core_components.rst
│   │   └── inference_engines.rst
│   └── development/         # 开发指南
│       ├── coding_style.rst
│       ├── contribution.rst
│       └── changelog.rst
├── build/                   # Doxygen生成的文档
├── .readthedocs.yml         # Read the Docs配置
├── requirements.txt         # Python依赖
└── doxygen/                 # Doxygen配置文件
    ├── Doxyfile
    └── Doxyfile.main
```

### 2.2 文档章节说明
- **API文档**：自动生成的C++ API参考
- **使用指南**：安装、构建、性能优化等
- **教程示例**：基于项目真实代码的使用示例
- **架构设计**：项目架构和组件说明
- **开发指南**：编码规范和贡献指南

---

## 3. 核心配置文件设计

### 3.1 Read the Docs配置 (.readthedocs.yml)

```yaml
# .readthedocs.yml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.10"

sphinx:
  configuration: docs/source/conf.py

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
      extra_requirements:
        - docs

# 构建前准备
build:
  jobs:
    post_create_environment:
      # 安装基础依赖
      - apt-get update && apt-get install -y doxygen graphviz
      # 编译项目以生成头文件
      - cd /home/ssl/Projects/PointGlueCpp
      - mkdir -p build && cd build
      - cmake -DUSE_ONNXRUNTIME=OFF -DUSE_OPENVINO=OFF -DBUILD_EXAMPLES=OFF \
              -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release ..
      - make -j$(nproc) point glue utils
      # 生成Doxygen文档
      - cd ../docs
      - doxygen Doxyfile
    post_build:
      # 构建Sphinx文档
      - sphinx-build -b html source build/html
      - sphinx-build -b latex source build/latex
      - cd build/latex && make pdf

formats:
  - pdf
  - epub
  - htmlzip

# 版本配置
versions:
  - 1.0.0
  - devel
```

### 3.2 Sphinx配置 (conf.py)

```python
# docs/source/conf.py
import os
import sys

# 项目信息
project = 'PointGlueCpp'
copyright = '2024, PointGlueCpp Team'
author = 'PointGlueCpp Team'
release = '1.0.0'

# 扩展配置
extensions = [
    'sphinx_rtd_theme',
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_inline_tabs',
    'myst_parser',
]

# Breathe配置（Doxygen集成）
breathe_projects = {
    "PointGlueCpp": "../build/xml"
}
breathe_default_project = "PointGlueCpp"

# 主题配置
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#3366cc',
}

# 源文件路径
sys.path.insert(0, os.path.abspath('../../include'))
```

### 3.3 Doxygen配置 (Doxyfile)

```makefile
# docs/doxygen/Doxyfile
PROJECT_NAME           = "PointGlueCpp"
PROJECT_NUMBER         = "1.0.0"
PROJECT_BRIEF          = "SuperPoint and SuperGlue feature detection and matching in C++"
OUTPUT_DIRECTORY       = ../build
CREATE_SUBDIRS         = YES

INPUT                  = ../../include ../../src
RECURSIVE              = YES
EXCLUDE                = ../../test
EXCLUDE_SYMLINKS       = YES
EXCLUDE_PATTERNS       = */CMakeFiles/* */build/* */3rdparty/*

OUTPUT_LANGUAGE        = Chinese
ENABLE_PREPROCESSING    = YES
MACRO_EXPANSION         = YES
EXPAND_ONLY_PREDEF      = NO
SEARCH_INCLUDES         = YES
INCLUDE_PATH            = ../../include
INCLUDE_FILE_PATTERNS   = *.h *.hpp
SOURCE_FILE_EXTENSION  = *.cpp *.cxx *.cc *.c *.hpp *.hxx *.hh

GENERATE_HTML          = NO
GENERATE_LATEX         = NO
GENERATE_XML           = YES
XML_OUTPUT             = xml
XML_SCHEMA             = yes
XML_DTD                = yes

EXTRACT_ALL            = YES
EXTRACT_PRIVATE         = YES
EXTRACT_PACKAGE         = YES
EXTRACT_STATIC          = YES
EXTRACT_LOCAL_CLASSES   = YES
EXTRACT_LOCAL_METHODS   = YES
EXTRACT_ANON_NSPACES    = YES

SORT_MEMBER_DOCS       = YES
SORT_BRIEF_DOCS        = YES
SORT_MEMBERS_CTORS_1ST = YES
SORT_GROUP_NAMES       = YES

GENERATE_TREEVIEW      = YES
TREEVIEW_WIDTH         = 250
GENERATE_TOC            = YES
TOC_INCLUDE_HEADINGS   = 5

HAVE_DOT               = YES
CLASS_DIAGRAMS         = YES
CALL_GRAPH             = YES
CALLER_GRAPH            = YES
```

### 3.4 Python依赖 (requirements.txt)

```txt
# docs/requirements.txt
sphinx==7.1.2
sphinx-rtd-theme==1.3.0rc1
breathe==6.0.0
sphinxcontrib-programoutput==0.17
myst-parser==2.0.0
sphinx-autodoc-typehints==1.24.0
sphinx-copybutton==0.5.2
sphinx-inline-tabs==2023.4.21
sphinx-kroki==0.10.0
```

---

## 4. API文档自动生成策略

### 4.1 文件扫描策略
Doxygen将扫描以下目录和文件类型：
- `include/pointgluecpp/` - 所有公共头文件（SuperPoint、SuperGlue、utils）
- `src/` - 实现文件（SuperPoint、SuperGlue的具体实现）
- 文件类型：`.h`, `.hpp`, `.cpp`, `.cxx`, `.cc`

具体扫描内容：
- **SuperPoint模块**：`include/pointgluecpp/SuperPoint/`下的所有头文件
- **SuperGlue模块**：`include/pointgluecpp/SuperGlue/`下的所有头文件
- **工具模块**：`include/pointgluecpp/utils/`下的ViewerTools等
- **引擎实现**：`src/`目录下的引擎实现文件（ONNXRuntime、OpenVINO）

### 4.2 Breathe集成处理
Sphinx通过Breathe扩展处理Doxygen生成的XML：
- 自动生成API导航
- 中文注释支持
- 类型引用解析
- 代码高亮显示

### 4.3 API文档组织结构

```rst
API文档
=======
.. toctree::
   :maxdepth: 2
   :caption: API参考

   api/modules
   api/superpoint
   api/superglue
   api/utils

SuperPoint模块
=============
.. doxygenmodule:: pointgluecpp::SuperPoint
   :members:
   :undoc-members:
   :show-inheritance:

SuperGlue模块
=============
.. doxygenmodule:: pointgluecpp::SuperGlue
   :members:
   :undoc-members:
   :show-inheritance:

工具模块
========
.. doxygenmodule:: pointgluecpp::utils
   :members:
   :undoc-members:
   :show-inheritance:
```

### 4.4 关键API文档示例
- SuperPoint类：关键点检测和描述子提取
- SuperGlue类：特征匹配
- Options结构体：配置参数
- 工具函数：可视化和数据处理

### 4.5 项目架构文档规划

**架构设计文档**将包含以下内容：

```rst
架构设计
=======

核心组件
--------

1. **SuperPoint模块** (`sp::SuperPoint`)
   - SuperPoint：主接口类，提供关键点检测功能
   - SuperPointEngine：推理引擎抽象基类
   - ONNXRuntimeEngine：ONNX Runtime实现
   - OpenVINOEngine：OpenVINO实现

2. **SuperGlue模块** (`sg::SuperGlue`)
   - SuperGlue：主接口类，提供特征匹配功能
   - SuperGlueEngine：推理引擎抽象基类
   - 匹配结果：Matches结构体，包含匹配关系和置信度

3. **工具模块** (`utils`)
   - ViewerTools：OpenCV可视化工具
   - TensorView：张量视图模板
   - 错误处理：统一的错误返回机制

数据流
-----

1. **关键点检测流程**
   输入图像 → 前处理 → 模型推理 → 后处理 → 输出关键点和描述子

2. **特征匹配流程**
   两图像的关键点/描述子 → 预处理 → 模型推理 → 后处理 → 输出匹配结果

3. **推理引擎切换**
   通过预处理器定义（-DONNXRUNTIMEENGINE/-DOPENVINOENGINE）选择实现
```

---

## 5. 内容编写指南和最佳实践

### 5.1 代码注释规范

**类注释规范**
```cpp
/**
 * @brief SuperPoint关键点检测器
 *
 * SuperPoint是一个用于图像关键点检测和描述子提取的深度学习模型。
 * 支持多种推理引擎（ONNX Runtime、OpenVINO），提供高性能的关键点检测。
 *
 * 主要功能：
 * - 检测图像中的关键点，返回位置和置信度
 * - 提取256维的描述子用于特征匹配
 * - 支持批量处理和多种输入格式
 *
 * @note 使用前需要确保模型文件存在且配置正确
 *
 * @see SuperGlue 用于特征匹配
 * @see Options 配置参数说明
 */
namespace sp
{
class SuperPoint
{
    // ...
};
}
```

**方法注释规范**
```cpp
/**
 * @brief 从YAML配置文件创建SuperPoint选项
 *
 * 该方法会读取指定的YAML文件，解析SuperPoint模型所需的配置参数，
 * 包括模型路径、输入输出维度、阈值等。
 *
 * @param yaml_path YAML配置文件的完整路径
 *                  文件应包含model_path、width、height等配置项
 * @return OptionsPtr 配置选项智能指针，失败时返回nullptr
 *
 * @example
 * auto options = sp::SuperPoint::Options::CreateFromYaml("config/SuperPointOptions.yaml");
 * if (!options) {
 *     SPDLOG_ERROR("Failed to load SuperPoint options");
 *     return;
 * }
 *
 * @see SuperPointOptions.yaml 配置文件模板
 */
static OptionsPtr CreateFromYaml(const std::string& yaml_path);
```

### 5.2 文档内容标准

**配置文档模板**
```rst
配置指南
=======

SuperPoint配置
-------------

SuperPoint通过YAML文件进行配置，配置文件包含以下参数：

.. list-table:: SuperPoint配置参数
   :widths: 20 15 10 55
   :header-rows: 1

   * - 参数名
     - 类型
     - 默认值
     - 说明
   * - model_path
     - string
     - 必填
     - ONNX模型文件的完整路径
   * - width
     - short
     - 320
     - 模型输入图像的宽度
   * - height
     - short
     - 320
     - 模型输入图像的高度
   * - conf_threshold
     - float
     - 0.015
     - 关键点检测置信度阈值(0-1)
   * - nms_dist
     - short
     - 4
     - 非极大值抑制的搜索半径

.. code-block:: yaml

   # SuperPointOptions.yaml
   model_path: "/path/to/superpoint.onnx"
   width: 320
   height: 320
   border: 8
   conf_threshold: 0.015
   nms_dist: 4

   input_names:
     - "grayscale image"

   input_dims:
     - 1
     - 1
     - 320
     - 320

   output_names:
     - "semi"
     - "desc"

   output_dims:
     - [1, 65, 40, 40]
     - [1, 256, 40, 40]
```

**使用示例模板**
```rst
快速开始
========

基本使用
--------

以下是一个使用SuperPoint进行关键点检测的完整示例：

.. code-block:: cpp

   #include <filesystem>
   #include <spdlog/spdlog.h>
   #include <opencv2/opencv.hpp>
   #include "pointgluecpp/SuperPoint/SuperPoint.h"

   int main() {
       // 1. 配置SuperPoint
       const std::filesystem::path yaml_path = "res/options/SuperPointOptions.yaml";
       auto options = sp::SuperPoint::Options::CreateFromYaml(yaml_path);
       if (!options) {
           SPDLOG_ERROR("Failed to load SuperPoint options");
           return -1;
       }

       sp::SuperPoint super_point(std::move(options));

       // 2. 读取图像
       cv::Mat input_image = cv::imread("test.jpg", cv::IMREAD_GRAYSCALE);
       if (input_image.empty()) {
           SPDLOG_ERROR("Failed to load image");
           return -1;
       }

       // 3. 运行推理
       SPDLOG_INFO("Running SuperPoint inference...");
       const auto [keypoints, keydesc] = super_point.RunSession(input_image);

       // 4. 处理结果
       SPDLOG_INFO("Detected {} keypoints", keypoints.size());

       // 可视化结果
       cv::Mat result_image = input_image.clone();
       for (const auto& kp : keypoints) {
           cv::circle(result_image, kp.pt, 2, cv::Scalar(0, 255, 0), -1);
       }

       cv::imshow("SuperPoint Result", result_image);
       cv::waitKey(0);

       return 0;
   }
```

### 5.3 性能优化指南

```rst
性能优化
=======

推理引擎选择
------------

PointGlueCpp支持多种推理引擎，各有优劣：

.. list-table:: 推理引擎对比
   :widths: 15 25 25 35
   :header-rows: 1

   * - 引擎
     - 优势
     - 劣势
     - 适用场景
   * - ONNX Runtime
     - 跨平台、部署简单
     - 性能一般
     - 开发测试、跨平台部署
   * - OpenVINO
     - Intel CPU上性能优异
     - 仅支持Intel硬件
     - Intel服务器部署、高性能需求

多线程优化
-----------

PointGlueCpp使用TBB进行多线程优化：

.. code-block:: bash
   # 安装TBB
   sudo apt-get install libtbb-dev

   # 编时启用多线程
   cmake -DUSE_TBB=ON -DCMAKE_BUILD_TYPE=Release ..

   # 运行时设置线程数
   export OMP_NUM_THREADS=8
```

---

## 6. 构建和部署策略

### 6.1 构建流程
1. **环境准备**：安装Python、Doxygen、Graphviz等依赖
2. **项目编译**：编译PointGlueCpp生成头文件
3. **Doxygen生成**：扫描头文件生成XML文档
4. **Sphinx构建**：生成HTML和PDF文档
5. **部署发布**：自动部署到Read the Docs

### 6.5 实际部署注意事项

**Read the Docs集成要点**：
1. **资源文件处理**：
   - 模型文件和配置文件需要单独处理
   - 通过`-DRESOURCES_FILES_DIR`指定资源文件目录
   - 在Read the Docs中可能需要额外配置资源文件访问

2. **构建优化**：
   - 关闭不必要的目标构建（如BUILD_EXAMPLES=OFF）
   - 只编译核心库，减少构建时间
   - 使用缓存机制避免重复编译

3. **测试验证**：
   - 在本地先验证Doxygen生成效果
   - 确保中文注释正确显示
   - 测试API导航的完整性

4. **潜在问题解决方案**：
   - 如果Doxygen扫描失败，检查头文件路径
   - 如果中文显示异常，设置正确的编码配置
   - 如果构建超时，优化构建目标数量

### 6.2 版本管理
- **活跃版本**：1.0.0（稳定版）
- **开发版本**：devel（最新开发版）
- **版本切换**：用户可以在Read the Docs网站上切换版本

### 6.3 文件格式输出
- **HTML**：在线阅读的主要格式
- **PDF**：可下载的完整文档
- **EPUB**：电子书格式
- **HTML ZIP**：离线HTML文档包

### 6.4 自动化特性
- CI/CD集成：代码提交后自动构建
- 依赖自动安装：Read the Docs自动管理所有依赖
- 多平台构建：支持Linux、Windows、macOS
- 缓存优化：构建结果缓存，提高构建速度

---

## 7. 实施步骤和时间线

### 7.1 准备阶段（第1周）
1. **完成基础配置**
   - 创建.readthedocs.yml
   - 配置requirements.txt
   - 设置Doxyfile

2. **建立文档结构**
   - 创建docs/source目录结构
   - 编写conf.py基础配置
   - 创建index.rst

3. **测试构建流程**
   - 本地测试Doxygen生成
   - 验证Sphinx集成
   - 确认Read the Docs连接

### 7.2 内容开发阶段（第2-3周）
1. **API文档生成**
   - 配置Doxygen扫描规则
   - 生成完整的API文档
   - 优化导航和链接

2. **编写使用指南**
   - 安装指南
   - 配置说明
   - 基础使用示例

3. **完善文档内容**
   - 性能优化指南
   - 架构设计文档
   - 常见问题解答

### 7.3 优化和发布阶段（第4周）
1. **文档优化**
   - 检查链接完整性
   - 优化文档结构
   - 添加搜索功能

2. **测试和验证**
   - 多格式输出测试
   - 移动端适配测试
   - 性能测试

3. **正式发布**
   - 配置版本发布
   - 设置域名和SEO
   - 监控文档访问情况

### 7.4 持续维护
1. **定期更新**
   - 版本发布时更新文档
   - 修正用户反馈的问题
   - 添加新功能说明

2. **社区反馈**
   - 收集用户意见
   - 改进文档质量
   - 响应技术问题

3. **技术维护**
   - 更新依赖版本
   - 优化构建流程
   - 采用新的文档技术

---

## 总结

本规划为PointGlueCpp项目提供了一个完整的Read the Docs文档系统解决方案。通过Doxygen和Sphinx的集成，实现了高质量的API文档自动生成；采用纯中文文档，更好地服务国内开发者；实施版本管理和自动化构建，确保文档的及时性和准确性。

该文档系统将显著提升项目的可用性和用户体验，为PointGlueCpp的推广和发展提供有力支持。