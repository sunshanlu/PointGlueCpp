#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "pointgluecpp/SuperPoint/SuperPointEngine.h"

namespace sp {
/**
 * @brief ONNX Runtime 推理引擎实现
 *
 * 使用 ONNX Runtime 作为后端实现 SuperPoint 推理引擎接口
 * 支持 CPU 和 CUDA（如果可用）执行提供程序
 */
class ONNXRuntimeEngine final : public SuperPointEngine {
public:
	using SessionPtr = std::unique_ptr<Ort::Session>;

	/**
	 * @brief 运行 SuperPoint 模型推理
	 *
	 * @param data 输入图像数据指针（已预处理的浮点数组）
	 * @return std::array<TensorView<float>, 2> 推理结果张量
	 */
	std::array<TensorView<float>, 2> RunInference(float *data) override;

	/**
	 * @brief 创建 ONNX Runtime 引擎实例
	 *
	 * 工厂方法，用于创建 ONNXRuntimeEngine 对象
	 *
	 * @param model_path ONNX 模型文件路径
	 * @param input_names 输入节点名称列表
	 * @param output_names 输出节点名称列表
	 * @param input_dims 输入张量维度
	 * @param output_dims0 第一个输出张量维度
	 * @param output_dims1 第二个输出张量维度
	 * @return InferencePtr 引擎智能指针
	 */
	static InferencePtr CreateInstance(const std::string &model_path,
	                                   const std::vector<const char *> &input_names,
	                                   const std::vector<const char *> &output_names,
	                                   const std::vector<int64_t> &input_dims,
	                                   const std::vector<int64_t> &output_dims0,
	                                   const std::vector<int64_t> &output_dims1);

protected:
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
	const char *CreateSession(const std::string &model_path) override;

	/**
	 * @brief 预热推理会话
	 *
	 * 执行一次 dummy 推理以初始化 ONNX Runtime 的内部状态
	 * 确保后续推理操作的延迟稳定
	 *
	 * @return const char* 错误信息，成功返回 nullptr
	 */
	const char *WarmUpSession() override;

private:
	/**
	 * @brief 构造 ONNX Runtime 引擎
	 *
	 * @param model_path ONNX 模型文件路径
	 * @param input_names 输入节点名称列表
	 * @param output_names 输出节点名称列表
	 * @param input_dims 输入张量维度
	 * @param output_dims0 第一个输出张量维度
	 * @param output_dims1 第二个输出张量维度
	 */
	ONNXRuntimeEngine(const std::string &model_path,
	                  const std::vector<const char *> &input_names,
	                  const std::vector<const char *> &output_names,
	                  const std::vector<int64_t> &input_dims,
	                  const std::vector<int64_t> &output_dims0,
	                  const std::vector<int64_t> &output_dims1);

	Ort::Env env_;       ///< ONNX Runtime 环境对象
	SessionPtr session_; ///< ONNX Runtime 会话对象
};
}
