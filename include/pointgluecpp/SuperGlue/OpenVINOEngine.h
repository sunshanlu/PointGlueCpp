#pragma once

#include <openvino/openvino.hpp>

#include "pointgluecpp/SuperGlue/SuperGlueEngine.h"

namespace sg
{
/**
 * @brief OpenVINO 推理引擎实现
 *
 * 使用 Intel OpenVINO 作为后端实现 SuperGlue 推理引擎接口
 * 支持 CPU、GPU、VPU 等多种硬件加速设备
 */
class OpenVINOEngine final : public SuperGlueEngine
{
public:
  /**
   * @brief 创建 OpenVINO 引擎实例
   *
   * 工厂方法，用于创建 OpenVINOEngine 对象
   *
   * @param model_path ONNX 模型文件路径
   * @param input_names 输入节点名称列表
   * @param output_names 输出节点名称列表
   * @param desc_dim 描述子维度（通常为 256）
   * @param keypoint_dim 关键点维度（通常为 2）
   * @return InferencePtr 引擎智能指针
   */
  static InferencePtr CreateInstance(const std::string& model_path,
                                     const std::vector<const char*>& input_names,
                                     const std::vector<const char*>& output_names,
                                     const short& desc_dim,
                                     const short& keypoint_dim);

  /**
   * @brief 运行 SuperGlue 模型推理
   *
   * @param input SuperGlue 输入数据，包含两张图像的关键点、描述子和置信度分数
   * @return std::tuple<TensorView<int64_t>, TensorView<float>> 匹配结果：
   *         - 第一个元素：匹配索引数组（-1 表示无匹配）
   *         - 第二个元素：匹配置信度分数
   */
  std::tuple<TensorView<int64_t>, TensorView<float>> RunInference(SuperGlueInput input) override;

protected:
  /**
   * @brief 预热推理会话
   *
   * 执行一次 dummy 推理以初始化 OpenVINO 的内部状态
   * 确保后续推理操作的延迟稳定
   *
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* WarmupSession() override;

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

private:
  /**
   * @brief 构造 OpenVINO 引擎
   *
   * @param model_path ONNX 模型文件路径
   * @param input_names 输入节点名称列表
   * @param output_names 输出节点名称列表
   * @param desc_dim 描述子维度
   * @param keypoint_dim 关键点维度
   */
  OpenVINOEngine(const std::string& model_path,
                 const std::vector<const char*>& input_names,
                 const std::vector<const char*>& output_names,
                 const short& desc_dim,
                 const short& keypoint_dim);

  ov::InferRequest infer_request_;   ///< OpenVINO 推理请求对象
  ov::CompiledModel compiled_model_; ///< OpenVINO 编译后的模型对象
};
} // namespace sg
