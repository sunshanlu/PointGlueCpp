.. PointGlueCpp documentation master file

PointGlueCpp Documentation
==========================

欢迎来到 PointGlueCpp 文档！这是一个 C++20 实现的 SuperPoint 和 SuperGlue 神经网络库，用于特征检测和匹配。

.. note::
   这是一个基于深度学习的特征点检测和匹配库，支持 ONNX Runtime 和 OpenVINO 推理引擎。

目录
----

.. toctree::
   :maxdepth: 2
   :caption: 内容:

   overview
   installation
   examples
   api/index

简介
-----

PointGlueCpp 是一个高性能的 C++ 库，实现了以下功能：

* **SuperPoint**: 关键点检测和描述子提取
  - 检测图像中的关键点
  - 计算 256 维描述子
  - 支持 CPU 和 GPU 推理

* **SuperGlue**: 特征点匹配
  - 基于注意力机制的特征匹配
  - 输出匹配关系和置信度
  - 适用于图像对匹配任务

主要特性
--------

* C++20 标准
* 支持多种推理引擎（ONNX Runtime、OpenVINO）
* 可插拔的引擎架构
* 零拷贝张量视图
* 完整的单元测试覆盖
* Doxygen 文档

快速开始
---------

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/yourusername/PointGlueCpp.git
   cd PointGlueCpp

   # 配置构建
   mkdir build && cd build
   cmake -DUSE_ONNXRUNTIME=ON -DBUILD_TESTING=ON ..

   # 编译
   make -j$(nproc)

   # 运行测试
   ctest

   # 运行示例
   ./examples/pointglue_example

文档
----

.. code-block:: cpp

   #include <pointgluecpp/SuperPoint/SuperPoint.h>
   #include <pointgluecpp/SuperGlue/SuperGlue.h>

   // 创建 SuperPoint
   auto options = sp::SuperPoint::Options::CreateFromYaml("config.yaml");
   sp::SuperPoint superpoint(std::move(options));

   // 检测特征点
   auto result = superpoint.RunSession(image);

   // 使用 SuperGlue 匹配
   auto sg_options = sg::SuperGlue::Options::CreateFromYaml("superglue_config.yaml");
   sg::SuperGlue superglue(std::move(sg_options));

   auto matches = superglue.RunSession(result0, result1, shape0, shape1);

相关链接
--------

* `GitHub 仓库 <https://github.com/yourusername/PointGlueCpp>`_
* `示例代码 <examples.html>`_
* `API 参考 <api/index.html>`_

索引
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`