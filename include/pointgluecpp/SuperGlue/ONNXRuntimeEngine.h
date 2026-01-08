#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "pointgluecpp/SuperGlue/SuperGlueEngine.h"
#include "pointgluecpp/utils/TensorView.hpp"

namespace sg {
class ONNXRuntimeEngine final : public SuperGlueEngine {
public:
	using SessionPtr = std::unique_ptr<Ort::Session>;

	std::tuple<TensorView<int64_t>, TensorView<float> > RunInference(SuperGlueInput input) override;

	static InferencePtr CreateInstance(const std::string &model_path,
	                                   const std::vector<const char *> &input_names,
	                                   const std::vector<const char *> &output_names,
	                                   const short &desc_dim,
	                                   const short &keypoint_dim);

protected:
	const char *WarmupSession() override;

	const char *CreateSession(const std::string &model_path) override;

private:
	ONNXRuntimeEngine(const std::string &model_path,
	                  const std::vector<const char *> &input_names,
	                  const std::vector<const char *> &output_names,
	                  const short &desc_dim,
	                  const short &keypoint_dim);

	Ort::Env env_;
	SessionPtr session_;
};
} // namespace sg
