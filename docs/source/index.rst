PointGlueCpp 文档
=================

欢迎来到 PointGlueCpp 项目文档！

.. toctree::
   :maxdepth: 2
   :caption: 目录:

   quickstart
   api
   changelog

项目简介
--------

PointGlueCpp 是一个用 C++ 编写的 SuperPoint 和 SuperGlue 推理项目，提供了基于深度学习的特征点检测和匹配功能。

核心功能
--------

* **SuperPoint**: 特征点检测和描述符提取
* **SuperGlue**: 基于注意力机制的特征点匹配
* **多引擎支持**: 支持 ONNX Runtime 和 OpenVINO 推理引擎
* **可视化工具**: 提供匹配结果的可视化展示

技术栈
------

* **语言**: C++20
* **构建系统**: CMake
* **依赖库**: OpenCV, spdlog, TBB
* **推理引擎**: ONNX Runtime / OpenVINO
* **文档生成**: Doxygen + Sphinx

项目结构
----------

.. code-block:: text

   PointGlueCpp/
   ├── include/pointgluecpp/      # 公共头文件
   │   ├── SuperPoint/            # SuperPoint 相关头文件
   │   ├── SuperGlue/             # SuperGlue 相关头文件
   │   └── utils/                 # 工具类头文件
   ├── src/                       # 源代码实现
   ├── examples/                  # 示例程序
   ├── test/                      # 单元测试
   └── res/                       # 资源文件

快速开始
----------

请查看 :doc:`quickstart` 了解如何安装和使用 PointGlueCpp。

API 文档
----------

完整的 API 文档请参考 :doc:`api`。

许可证
-------

本项目采用开源许可证，详情请参阅 LICENSE 文件。

联系作者
--------

* 作者: ssl
* GitHub: https://github.com/sunshanlu/PointGlueCpp

索引与表格
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`