#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "pointgluecpp/SuperPoint/SuperPointEngine.h"

namespace sp {
/// onnxruntime 推理引擎适配器
class ONNXRuntimeEngine final : public SuperPointEngine {
public:
	using SessionPtr = std::unique_ptr<Ort::Session>;


	/**
	 * 运行推理
	 *
	 * @param data 输入的待推理的数据指针
	 * @return 输出的‘裸’推理结果
	 */
	std::array<TensorView<float>, 2> RunInference(float *data) override;

	static InferencePtr CreateInstance(const std::string &model_path,
	                                   const std::vector<const char *> &input_names,
	                                   const std::vector<const char *> &output_names,
	                                   const std::vector<int64_t> &input_dims,
	                                   const std::vector<int64_t> &output_dims0,
	                                   const std::vector<int64_t> &output_dims1);

protected:
	/**
	 * 创建onnxruntime的session
	 *
	 * 1. session_options 配置
	 * 2. cuda_options 配置
	 * 3. 创建 onnxruntime session
	 *
	 * @param model_path 输入的模型路径
	 * @return error message
	 */
	const char *CreateSession(const std::string &model_path) override;

	/**
	 * 热处理底层软件的初始化
	 *
	 * @return error message
	 */
	const char *WarmUpSession() override;

private:
	/**
	 * 创建onnxruntime的推理session
	 *
	 * 1. 创建onnxruntime的session @see ONNXRuntimeEngine::CreateSession
	 * 2. 维护input nodes 和 ouput nodes属性
	 * 3. 热处理底层软件的初始化 @see ONNXRuntimeEngine::WarmSession
	 *
	 * @param model_path 输入 onnxruntime 模型路径
	 * @param input_names 输入节点名称
	 * @param output_names 输出节点名称
	 * @param input_dims 输入节点维度
	 * @param output_dims0 输出节点维度0
	 * @param output_dims1 输出节点维度1
	 */
	ONNXRuntimeEngine(const std::string &model_path,
	                  const std::vector<const char *> &input_names,
	                  const std::vector<const char *> &output_names,
	                  const std::vector<int64_t> &input_dims,
	                  const std::vector<int64_t> &output_dims0,
	                  const std::vector<int64_t> &output_dims1);

	Ort::Env env_;       //< onnxruntime 环境
	SessionPtr session_; //< onnxruntime session
};
}
