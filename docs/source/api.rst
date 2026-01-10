API 文档
========

本文档详细介绍了 PointGlueCpp 的 API 接口。

SuperPoint 模块
---------------

SuperPoint 模块提供了特征点检测和描述符提取功能。

SuperPoint 类
~~~~~~~~~~~~~

.. doxygenclass:: sp::SuperPoint
  :project: pointgluecpp
  :members:

SuperPoint 引擎接口
~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: sp::SuperPointEngine
  :project: pointgluecpp
  :members:

ONNX Runtime 引擎实现
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: sp::ONNXRuntimeEngine
  :project: pointgluecpp
  :members:

OpenVINO 引擎实现
~~~~~~~~~~~~~~~~~~

.. doxygenclass:: sp::OpenVINOEngine
  :project: pointgluecpp
  :members:

SuperGlue 模块
--------------

SuperGlue 模块提供了基于注意力机制的特征点匹配功能。

SuperGlue 类
~~~~~~~~~~~~

.. doxygenclass:: sg::SuperGlue
  :project: pointgluecpp
  :members:

SuperGlue 引擎接口
~~~~~~~~~~~~~~~~~~

.. doxygenclass:: sg::SuperGlueEngine
  :project: pointgluecpp
  :members:

ONNX Runtime 引擎实现
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: sg::ONNXRuntimeEngine
  :project: pointgluecpp
  :members:

OpenVINO 引擎实现
~~~~~~~~~~~~~~~~~~

.. doxygenclass:: sg::OpenVINOEngine
  :project: pointgluecpp
  :members:

工具模块
---------

TensorView 工具类
~~~~~~~~~~~~~~~~~

张量视图模板类，提供高效的张量数据包装器。

.. doxygenstruct:: TensorView
  :project: pointgluecpp
  :members:

可视化工具
~~~~~~~~~~

提供关键点、匹配线等的可视化绘制功能。

注意：DrawCircles 和 DrawLines 函数有多个重载版本，具体使用方法请参考头文件 ``include/pointgluecpp/utils/ViewerTools.hpp``。

调试工具
~~~~~~~~

提供张量数据保存和错误检查等调试辅助功能。

保存数据到文件
^^^^^^^^^^^^^^

.. doxygenfunction:: SaveDataToFile
   :project: pointgluecpp

数据结构
--------

注意：以下结构体已在对应的类文档中详细说明，此处不再重复列出：

* ``sp::SuperPoint::SuperPointRet`` - SuperPoint 推理输出结果（见 SuperPoint 类文档）
* ``sp::SuperPoint::Options`` - SuperPoint 配置项（见 SuperPoint 类文档）
* ``sg::SuperGlue::Matches`` - SuperGlue 匹配结果（见 SuperGlue 类文档）
* ``sg::SuperGlue::Options`` - SuperGlue 配置项（见 SuperGlue 类文档）
* ``sg::SuperGlueEngine::SuperGlueInput`` - SuperGlue 输入数据结构（见 SuperGlueEngine 类文档）

命名空间
--------

sp 命名空间
~~~~~~~~~~~

SuperPoint 相关的类和函数都在 ``sp`` 命名空间中。

sg 命名空间
~~~~~~~~~~~

SuperGlue 相关的类和函数都在 ``sg`` 命名空间中。

类型定义
---------

.. doxygentypedef:: sp::SuperPoint::OptionsPtr
   :project: pointgluecpp

.. doxygentypedef:: sg::SuperGlue::OptionsPtr
   :project: pointgluecpp

.. doxygentypedef:: sp::SuperPointEngine::InferencePtr
   :project: pointgluecpp

.. doxygentypedef:: sg::SuperGlueEngine::InferencePtr
   :project: pointgluecpp