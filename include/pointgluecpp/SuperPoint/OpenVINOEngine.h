#pragma once

#include <openvino/openvino.hpp>

#include "pointgluecpp/SuperPoint/SuperPointEngine.h"

namespace sp
{
/**
 * @brief OpenVINO 推理引擎实现
 *
 * 使用 Intel OpenVINO 作为后端实现 SuperPoint 推理引擎接口
 * 支持 CPU、GPU、VPU 等多种硬件加速设备
 */
class OpenVINOEngine final : public SuperPointEngine
{
public:
  /**
   * @brief 运行 SuperPoint 模型推理
   *
   * @param data 输入图像数据指针（已预处理的浮点数组）
   * @return std::array<TensorView<float>, 2> 推理结果张量
   */
  std::array<TensorView<float>, 2> RunInference(float* data) override;

  /**
   * @brief 创建 OpenVINO 引擎实例
   *
   * 工厂方法，用于创建 OpenVINOEngine 对象
   *
   * @param model_path ONNX 模型文件路径
   * @param input_names 输入节点名称列表
   * @param output_names 输出节点名称列表
   * @param input_dims 输入张量维度
   * @param output_dims0 第一个输出张量维度
   * @param output_dims1 第二个输出张量维度
   * @return InferencePtr 引擎智能指针
   */
  static InferencePtr CreateInstance(const std::string& model_path,
                                     const std::vector<const char*>& input_names,
                                     const std::vector<const char*>& output_names,
                                     const std::vector<int64_t>& input_dims,
                                     const std::vector<int64_t>& output_dims0,
                                     const std::vector<int64_t>& output_dims1);

protected:
  /**
   * @brief 创建 OpenVINO 编译模型
   *
   * 配置并初始化 OpenVINO 推理引擎：
   * 1. 读取 ONNX 模型文件
   * 2. 编译模型为目标设备（默认 CPU）
   * 3. 创建推理请求对象
   *
   * @param model_path ONNX 模型文件路径
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* CreateSession(const std::string& model_path) override;

  /**
   * @brief 预热推理会话
   *
   * 执行一次 dummy 推理以初始化 OpenVINO 的内部状态
   * 确保后续推理操作的延迟稳定
   *
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* WarmUpSession() override;

private:
  ov::InferRequest infer_request_;   ///< OpenVINO 推理请求对象
  ov::CompiledModel compiled_model_; ///< OpenVINO 编译后的模型对象

  /**
   * @brief 构造 OpenVINO 引擎
   *
   * @param model_path ONNX 模型文件路径
   * @param input_names 输入节点名称列表
   * @param output_names 输出节点名称列表
   * @param input_dims 输入张量维度
   * @param output_dims0 第一个输出张量维度
   * @param output_dims1 第二个输出张量维度
   */
  OpenVINOEngine(const std::string& model_path,
                 const std::vector<const char*>& input_names,
                 const std::vector<const char*>& output_names,
                 const std::vector<int64_t>& input_dims,
                 const std::vector<int64_t>& output_dims0,
                 const std::vector<int64_t>& output_dims1);
};
}
