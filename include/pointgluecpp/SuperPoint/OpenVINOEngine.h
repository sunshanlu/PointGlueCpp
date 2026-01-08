#pragma once

#include <openvino/openvino.hpp>

#include "pointgluecpp/SuperPoint/SuperPointEngine.h"

namespace sp {
class OpenVINOEngine final : public SuperPointEngine {
public:
	std::array<TensorView<float>, 2> RunInference(float *data) override;

	static InferencePtr CreateInstance(const std::string &model_path,
	                                   const std::vector<const char *> &input_names,
	                                   const std::vector<const char *> &output_names,
	                                   const std::vector<int64_t> &input_dims,
	                                   const std::vector<int64_t> &output_dims0,
	                                   const std::vector<int64_t> &output_dims1);

protected:
	/**
	 * 创建openvino的compiled_model
	 *
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
	ov::InferRequest infer_request_;   //< OpenVINO inference request
	ov::CompiledModel compiled_model_; //< OpenVINO compiled model

	/**
	 * 创建openvino的推理compiled_engine
	 *
	 * 1. 创建onnxruntime的session @see OpenVINOEngine::CreateSession
	 * 2. 维护input nodes 和 ouput nodes属性
	 * 3. 热处理底层软件的初始化 @see OpenVINOEngine::WarmSession
	 *
	 * @param model_path 输入 onnxruntime 模型路径
	 * @param input_names 输入节点名称
	 * @param output_names 输出节点名称
	 * @param input_dims 输入节点维度
	 * @param output_dims0 输出节点维度0
	 * @param output_dims1 输出节点维度1
	 */
	OpenVINOEngine(const std::string &model_path,
	               const std::vector<const char *> &input_names,
	               const std::vector<const char *> &output_names,
	               const std::vector<int64_t> &input_dims,
	               const std::vector<int64_t> &output_dims0,
	               const std::vector<int64_t> &output_dims1);
};
}
