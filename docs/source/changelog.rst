更新日志
========

本文档记录了 PointGlueCpp 项目的版本更新历史。

版本 1.0.0 (2026-01-10)
----------------------

新增功能
~~~~~~~~

* 初始版本发布
* 实现 SuperPoint 特征点检测和描述符提取功能
* 实现 SuperGlue 特征点匹配功能
* 支持 ONNX Runtime 推理引擎
* 支持 OpenVINO 推理引擎
* 提供完整的 Doxygen 注释文档
* 提供 Sphinx 文档网站
* 提供三个示例程序：
  * superpoint_example - SuperPoint 单独使用示例
  * superglue_example - SuperGlue 单独使用示例
  * pointglue_example - 完整流程示例

核心特性
~~~~~~~~

SuperPoint 模块
^^^^^^^^^^^^^^^

* 特征点检测
* 256 维描述符提取
* Fast-NMS 非极大值抑制
* 支持多种推理引擎（ONNX Runtime、OpenVINO）
* YAML 配置文件支持

SuperGlue 模块
^^^^^^^^^^^^^^

* 基于注意力机制的特征点匹配
* 匹配置信度计算
* 可视化支持（匹配线绘制）
* 支持多种推理引擎（ONNX Runtime、OpenVINO）
* YAML 配置文件支持

工具模块
^^^^^^^^

* TensorView - 零拷贝张量视图模板类
* ViewerTools - OpenCV 可视化工具函数
* DebugTools - 调试辅助工具

技术实现
~~~~~~~~

* 使用 C++20 标准编写
* CMake 构建系统
* 支持静态和动态链接
* 完整的单元测试覆盖
* 内存泄漏检测支持（Valgrind）

文档
~~~~

* 完整的 Doxygen API 文档
* Sphinx 文档网站
* 快速入门指南
* 示例代码文档

依赖项
~~~~~~

* OpenCV 4.x
* spdlog
* TBB
* ONNX Runtime 或 OpenVINO
* GTest

已知问题
~~~~~~~~

* 暂无

计划中的功能
~~~~~~~~~~~~

* 支持更多推理引擎（TensorRT、NCNN 等）
* 添加更多示例程序
* 性能优化
* 支持 Windows 和 macOS 平台
* 添加 Python 绑定

贡献者
~~~~~~

* ssl - 项目创建者和主要维护者

致谢
~~~~

感谢以下开源项目：

* SuperPoint: https://github.com/magicleap/SuperPoint
* SuperGlue: https://github.com/magicleap/SuperGlue
* ONNX Runtime: https://github.com/microsoft/onnxruntime
* OpenVINO: https://github.com/openvinotoolkit/openvino

许可证
~~~~~~

本项目采用开源许可证，详情请参阅 LICENSE 文件。

版本历史
----------

[1.0.0] - 2026-01-10
^^^^^^^^^^^^^^^^^^^^

* 首次公开发布
* 实现核心功能