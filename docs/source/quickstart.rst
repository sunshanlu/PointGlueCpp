快速入门
========

本文档将帮助您快速开始使用 PointGlueCpp。

系统要求
--------

* **操作系统**: Linux (推荐 Ubuntu 20.04+)
* **编译器**: GCC 9+ 或 Clang 10+ (支持 C++20)
* **CMake**: 3.10 或更高版本
* **依赖库**:
  * OpenCV 4.x
  * spdlog
  * TBB (Intel Threading Building Blocks)
  * ONNX Runtime 或 OpenVINO (至少一个)
  * GTest (用于测试)

安装依赖
--------

Ubuntu/Debian
~~~~~~~~~~~~~

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install cmake build-essential
   sudo apt-get install libopencv-dev libspdlog-dev libtbb-dev
   sudo apt-get install libgtest-dev

安装 ONNX Runtime
~~~~~~~~~~~~~~~~~

使用 vcpkg 安装:

.. code-block:: bash

   # 安装 vcpkg
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.sh

   # 安装 onnxruntime
   ./vcpkg install onnxruntime

或者从源码编译，参考: https://github.com/microsoft/onnxruntime

安装 OpenVINO
~~~~~~~~~~~~~

.. code-block:: bash

   sudo apt-get install openvino-dev

或者从官网下载: https://docs.openvino.ai/

获取源码
--------

.. code-block:: bash

   git clone git@github.com:sunshanlu/PointGlueCpp.git
   cd PointGlueCpp
   git submodule update --init --recursive

构建项目
--------

创建构建目录:

.. code-block:: bash

   mkdir build && cd build

配置 CMake:

.. code-block:: bash

   # 基本配置
   cmake .. -DCMAKE_BUILD_TYPE=Release

   # 可选配置选项
   # - USE_ONNXRUNTIME=ON/OFF    # 使用 ONNX Runtime 引擎（默认 ON）
   # - USE_OPENVINO=ON/OFF       # 使用 OpenVINO 引擎（默认 ON）
   # - BUILD_EXAMPLES=ON/OFF     # 构建示例程序（默认 ON）
   # - BUILD_TESTING=ON/OFF      # 构建测试（默认 ON）

   # 示例：仅使用 ONNX Runtime
   cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_OPENVINO=OFF

编译:

.. code-block:: bash

   cmake --build . --config Release

安装（可选）:

.. code-block:: bash

   cmake --install .

准备模型
--------

下载 SuperPoint 和 SuperGlue 的 ONNX 模型，并将它们放置到 ``res/weights/`` 目录下。

配置 YAML 文件:

编辑 ``res/options/SuperPointOptions.yaml``:

.. code-block:: yaml

   model_path: "/path/to/superpoint.onnx"
   width: 320
   height: 320
   border: 8
   conf_threshold: 0.015
   nms_dist: 4

编辑 ``res/options/SuperGlueOptions.yaml``:

.. code-block:: yaml

   model_path: "/path/to/superglue.onnx"
   # 其他配置项...

运行示例
--------

PointGlueCpp 提供了三个示例程序：

1. **superpoint_example**: 仅使用 SuperPoint 进行特征点检测
2. **superglue_example**: 使用 SuperGlue 进行特征点匹配
3. **pointglue_example**: 完整流程（SuperPoint + SuperGlue）

运行完整示例:

.. code-block:: bash

   cd build/Release/bin
   ./pointglue_example

该示例将:

1. 加载两张测试图像
2. 使用 SuperPoint 提取特征点
3. 使用 SuperGlue 进行匹配
4. 可视化匹配结果

使用 API
--------

基本用法示例:

.. code-block:: cpp

   #include "pointgluecpp/SuperPoint/SuperPoint.h"
   #include "pointgluecpp/SuperGlue/SuperGlue.h"

   // 1. 创建 SuperPoint 配置
   auto superpoint_options = sp::SuperPoint::Options::CreateFromYaml(
       "res/options/SuperPointOptions.yaml"
   );

   // 2. 初始化 SuperPoint
   sp::SuperPoint super_point(std::move(superpoint_options));

   // 3. 读取图像
   cv::Mat image = cv::imread("image.png", cv::IMREAD_GRAYSCALE);

   // 4. 提取特征点
   auto [keypoints, descriptors] = super_point.RunSession(image);

   // 5. 创建 SuperGlue 配置
   auto superglue_options = sg::SuperGlue::Options::CreateFromYaml(
       "res/options/SuperGlueOptions.yaml"
   );

   // 6. 初始化 SuperGlue
   sg::SuperGlue super_glue(std::move(superglue_options));

   // 7. 进行特征匹配
   auto matches = super_glue.RunSession(
       keypoints0, keypoints1,
       image0.size(), image1.size()
   );

   // 8. 可视化匹配结果
   sg::SuperGlue::ShowMatches(
       image0, image1, matches, keypoints0, keypoints1
   );

运行测试
--------

运行所有单元测试:

.. code-block:: bash

   cd build
   ctest --output-on-failure

或使用自定义目标:

.. code-block:: bash

   make test_units

内存泄漏检测（需要 Valgrind）:

.. code-block:: bash

   make test_memory

生成文档
----------

生成 Doxygen XML 文档:

.. code-block:: bash

   cd docs
   doxygen Doxyfile

生成 Sphinx 文档网站:

.. code-block:: bash

   cd docs/source
   sphinx-build -b html ../build/html

构建完成后，在浏览器中打开 ``docs/build/html/index.html`` 查看文档。

常见问题
----------

编译错误
^^^^^^^^

**找不到 OpenCV**:

确保 OpenCV 正确安装并设置 ``OpenCV_DIR`` 环境变量。

**找不到 ONNX Runtime**:

确保 ``onnxruntime_DIR`` 正确设置，或使用 vcpkg 安装。

**CMake 版本过低**:

升级到 3.10 或更高版本。

运行时错误
^^^^^^^^^^

**模型文件未找到**:

检查 YAML 配置中的 ``model_path`` 是否正确。

**图像加载失败**:

确保图像路径正确且格式支持（推荐 PNG 或 JPG）。

**推理引擎初始化失败**:

检查引擎依赖是否正确安装。

下一步
--------

* 查看 :doc:`api` 了解详细的 API 文档
* 查看 ``examples/`` 目录中的示例代码
* 阅读 :doc:`changelog` 了解版本更新信息

获取帮助
----------

如有问题，请:

1. 查看 GitHub Issues: https://github.com/sunshanlu/PointGlueCpp/issues
2. 提交新的 Issue
3. 联系项目维护者