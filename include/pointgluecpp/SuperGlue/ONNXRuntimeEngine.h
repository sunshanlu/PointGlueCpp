#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "pointgluecpp/SuperGlue/SuperGlueEngine.h"
#include "pointgluecpp/utils/TensorView.hpp"

namespace sg
{
/**
 * @brief ONNX Runtime 推理引擎实现
 *
 * 使用 ONNX Runtime 作为后端实现 SuperGlue 推理引擎接口
 * 支持 CPU 和 CUDA（如果可用）执行提供程序
 */
class ONNXRuntimeEngine final : public SuperGlueEngine
{
public:
  using SessionPtr = std::unique_ptr<Ort::Session>;

  /**
   * @brief 运行 SuperGlue 模型推理
   *
   * @param input SuperGlue 输入数据，包含两张图像的关键点、描述子和置信度分数
   * @return std::tuple<TensorView<int64_t>, TensorView<float>> 匹配结果：
   *         - 第一个元素：匹配索引数组（-1 表示无匹配）
   *         - 第二个元素：匹配置信度分数
   */
  std::tuple<TensorView<int64_t>, TensorView<float>> RunInference(SuperGlueInput input) override;

  /**
   * @brief 创建 ONNX Runtime 引擎实例
   *
   * 工厂方法，用于创建 ONNXRuntimeEngine 对象
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

protected:
  /**
   * @brief 预热推理会话
   *
   * 执行一次 dummy 推理以初始化 ONNX Runtime 的内部状态
   * 确保后续推理操作的延迟稳定
   *
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* WarmupSession() override;

  /**
   * @brief 创建 ONNX Runtime 推理会话
   *
   * 配置并初始化 ONNX Runtime 会话：
   * 1. 设置会话选项（图优化、线程数等）
   * 2. 检测并配置 CUDA 执行提供程序（如果可用）
   * 3. 创建 ONNX Runtime 会话对象
   *
   * @param model_path ONNX 模型文件路径
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* CreateSession(const std::string& model_path) override;

private:
  /**
   * @brief 构造 ONNX Runtime 引擎
   *
   * @param model_path ONNX 模型文件路径
   * @param input_names 输入节点名称列表
   * @param output_names 输出节点名称列表
   * @param desc_dim 描述子维度
   * @param keypoint_dim 关键点维度
   */
  ONNXRuntimeEngine(const std::string& model_path,
                    const std::vector<const char*>& input_names,
                    const std::vector<const char*>& output_names,
                    const short& desc_dim,
                    const short& keypoint_dim);

  Ort::Env env_;       ///< ONNX Runtime 环境对象
  SessionPtr session_; ///< ONNX Runtime 会话对象
};
} // namespace sg
