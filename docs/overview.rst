概述
====

PointGlueCpp 是一个现代化的 C++20 库，实现了 SuperPoint 和 SuperGlue 神经网络算法。

架构设计
--------

组件分层
~~~~~~~~

.. graphviz::

   digraph architecture {
       rankdir=TB;
       node [shape=box, style=rounded];

       Application [label="应用层\n示例程序"];
       Algorithms [label="算法层\nSuperPoint/SuperGlue"];
       Engines [label="引擎层\nONNXRuntime/OpenVINO"];
       Utils [label="工具层\nTensorView/ViewerTools"];

       Application -> Algorithms;
       Algorithms -> Engines;
       Algorithms -> Utils;
       Engines -> Utils;
   }

核心组件
~~~~~~~~

SuperPoint
^^^^^^^^^^

SuperPoint 是一个自监督的特征点检测和描述子提取网络。

功能：
- 检测图像中的关键点
- 计算每个关键点的 256 维描述子
- 使用 Fast-NMS 进行非极大值抑制

SuperGlue
^^^^^^^^^

SuperGlue 是一个基于注意力机制的特征点匹配网络。

功能：
- 匹配两幅图像之间的特征点
- 输出匹配关系和置信度
- 支持可变数量的特征点

推理引擎
^^^^^^^^

支持多种推理引擎：

* ONNX Runtime: 跨平台推理，支持 CUDA
* OpenVINO: Intel 优化推理

内存管理
--------

Zero-Copy 设计
~~~~~~~~~~~~~~~

TensorView 实现了零拷贝的张量视图：

* 拥有模式：管理内存生命周期
* 非拥有模式：包装外部指针
* 拷贝和移动语义

示例：

.. code-block:: cpp

   // 拥有模式
   TensorView<float> owned(1000);

   // 非拥有模式（零拷贝）
   float data[1000];
   TensorView<float> view(data, 1000);

   // 拷贝（深拷贝）
   TensorView<float> copy = owned;

   // 移动（转移所有权）
   TensorView<float> moved = std::move(owned);

RAII 原则
~~~~~~~~~~~

使用智能指针管理资源：

.. code-block:: cpp

   using InferencePtr = std::unique_ptr<SuperPointEngine>;
   InferencePtr engine = ONNXRuntimeEngine::CreateInstance(...);

线程安全
--------

并行化策略
~~~~~~~~~~

使用 C++17 并行算法和 TBB：

.. code-block:: cpp

   std::for_each(std::execution::par_unseq,
                 indices.begin(), indices.end(),
                 [&](const int& idx) {
       // 并行处理
   });

数据竞争防护
~~~~~~~~~~~~

* 只读数据共享
* 每个线程独立写入输出
* 使用原子操作和互斥锁保护共享状态

性能优化
--------

编译器优化
~~~~~~~~~~

* 编译时多态（模板）
* 内联函数
* 链接时优化 (LTO)

运行时优化
~~~~~~~~~~

* 内存池
* 对象复用
* 缓存友好访问模式