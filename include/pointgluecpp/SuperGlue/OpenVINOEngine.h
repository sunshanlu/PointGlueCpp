#pragma once

#include <openvino/openvino.hpp>

#include "pointgluecpp/SuperGlue/SuperGlueEngine.h"

namespace sg {
class OpenVINOEngine final : public SuperGlueEngine {
public:
	static InferencePtr CreateInstance(const std::string &model_path,
	                                   const std::vector<const char *> &input_names,
	                                   const std::vector<const char *> &output_names,
	                                   const short &desc_dim,
	                                   const short &keypoint_dim);

	std::tuple<TensorView<int64_t>, TensorView<float> > RunInference(SuperGlueInput input) override;

protected:
	const char *WarmupSession() override;

	const char *CreateSession(const std::string &model_path) override;

private:
	OpenVINOEngine(const std::string &model_path,
	               const std::vector<const char *> &input_names,
	               const std::vector<const char *> &output_names,
	               const short &desc_dim,
	               const short &keypoint_dim);

	ov::InferRequest infer_request_;
	ov::CompiledModel compiled_model_;
};
} // namespace sg
