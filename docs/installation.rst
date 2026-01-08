安装指南
=========

依赖项
-------

必需依赖
^^^^^^^^

* C++20 编译器 (GCC 11+, Clang 13+, MSVC 2022+)
* CMake 3.20+
* OpenCV 4.0+
* spdlog
* TBB (Threading Building Blocks)

推理引擎（至少一个）
^^^^^^^^^^^^^^^^

* ONNX Runtime 1.12+
* OpenVINO 2022.1+

Ubuntu/Debian
^^^^^^^^^^^^^

.. code-block:: bash

   # 安装基础依赖
   sudo apt update
   sudo apt install -y \
       build-essential \
       cmake \
       git \
       libopencv-dev \
       libspdlog-dev \
       libtbb-dev

   # 安装 ONNX Runtime（推荐）
   # 从 https://github.com/microsoft/onnxruntime/releases 下载
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
   tar -xzf onnxruntime-linux-x64-1.16.0.tgz
   sudo cp -r onnxruntime-linux-x64-1.16.0/* /usr/local/

   # 或安装 OpenVINO
   wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-OPENVINO-2023.pub
   sudo apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2023.pub
   echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu20 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2023.list
   sudo apt update
   sudo apt install -y openvino-2023.0.0

macOS
^^^^^

.. code-block:: bash

   # 使用 Homebrew
   brew install cmake opencv spdlog tbb

   # 安装 ONNX Runtime
   brew install onnxruntime

Windows
^^^^^^^

.. code-block:: powershell

   # 使用 vcpkg
   vcpkg install opencv spdlog tbb onnxruntime

   # 或从源码编译
   # 下载并安装 CMake、Visual Studio 2022

编译安装
--------

从源码编译
^^^^^^^^^^

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/yourusername/PointGlueCpp.git
   cd PointGlueCpp

   # 配置构建
   mkdir build && cd build

   # 使用 ONNX Runtime
   cmake -DUSE_ONNXRUNTIME=ON \
         -DUSE_OPENVINO=OFF \
         -DBUILD_EXAMPLES=ON \
         -DBUILD_TESTING=ON \
         -DCMAKE_BUILD_TYPE=Release \
         ..

   # 或使用 OpenVINO
   cmake -DUSE_ONNXRUNTIME=OFF \
         -DUSE_OPENVINO=ON \
         -DBUILD_EXAMPLES=ON \
         -DBUILD_TESTING=ON \
         -DCMAKE_BUILD_TYPE=Release \
         ..

   # 或同时启用两个引擎
   cmake -DUSE_ONNXRUNTIME=ON \
         -DUSE_OPENVINO=ON \
         -DBUILD_EXAMPLES=ON \
         -DBUILD_TESTING=ON \
         -DCMAKE_BUILD_TYPE=Release \
         ..

   # 编译（使用所有核心）
   make -j$(nproc)

   # 运行测试
   ctest --output-on-failure

   # 安装（可选）
   sudo make install

CMake 选项
^^^^^^^^^^

+--------------------------+---------+---------------------------+
| 选项                     | 默认值  | 说明                      |
+==========================+=========+===========================+
| USE_ONNXRUNTIME          | ON      | 启用 ONNX Runtime 引擎   |
+--------------------------+---------+---------------------------+
| USE_OPENVINO             | ON      | 启用 OpenVINO 引擎        |
+--------------------------+---------+---------------------------+
| BUILD_EXAMPLES           | ON      | 构建示例程序              |
+--------------------------+---------+---------------------------+
| BUILD_TESTING            | ON      | 构建测试                  |
+--------------------------+---------+---------------------------+
| CMAKE_BUILD_TYPE         | Release | 构建类型                  |
+--------------------------+---------+---------------------------+

注意：至少需要启用一个推理引擎（ONNXRUNTIME 或 OPENVINO）。

安装位置
^^^^^^^^

默认安装位置：

* Linux: ``/usr/local/lib`` 和 ``/usr/local/include``
* Windows: ``C:/Program Files/PointGlueCpp``

自定义安装位置：

.. code-block:: bash

   cmake -DCMAKE_INSTALL_PREFIX=/opt/PointGlueCpp ..
   make
   sudo make install

模型文件
--------

下载预训练模型
^^^^^^^^^^^^^^

SuperPoint 和 SuperGlue 需要预训练的 ONNX 模型：

.. code-block:: bash

   # 创建模型目录
   mkdir -p models

   # 下载 SuperPoint 模型
   wget https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.onnx \
       -O models/superpoint.onnx

   # 下载 SuperGlue 模型
   wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/superglue.onnx \
       -O models/superglue.onnx

配置文件
^^^^^^^^

复制并编辑配置文件：

.. code-block:: bash

   cp res/options/SuperPointOptions.yaml.example res/options/SuperPointOptions.yaml
   cp res/options/SuperGlueOptions.yaml.example res/options/SuperGlueOptions.yaml

编辑 ``model_path`` 指向下载的模型文件。

验证安装
--------

运行示例程序验证安装：

.. code-block:: bash

   cd build
   ./examples/pointglue_example

如果看到特征点检测和匹配结果，说明安装成功！

常见问题
--------

找不到 OpenCV
^^^^^^^^^^^^

.. code-block:: bash

   # 设置 OpenCV_DIR
   export OpenCV_DIR=/path/to/opencv/build
   cmake ..

找不到 ONNX Runtime
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # 设置 ONNXRUNTIME_ROOT_DIR
   export ONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
   cmake -DONNXRUNTIME_ROOT_DIR=$ONNXRUNTIME_ROOT_DIR ..

编译错误
^^^^^^^^

确保使用 C++20 编译器：

.. code-block:: bash

   # 检查 GCC 版本
   g++ --version  # 需要 >= 11

   # 检查 Clang 版本
   clang++ --version  # 需要 >= 13