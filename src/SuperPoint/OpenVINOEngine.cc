#include <array>

#include <spdlog//spdlog.h>

#include "pointgluecpp/SuperPoint/OpenVINOEngine.h"
#include "pointgluecpp/utils/TensorView.hpp"

#define RET_OK nullptr
#define CHECK(message)\
do {\
	if (message != RET_OK){\
		SPDLOG_ERROR("Error: {}, at {}:{}\n", message, __FILE__, __LINE__);\
		exit(EXIT_FAILURE);\
	}\
} while (0)

namespace sp {
const char *OpenVINOEngine::CreateSession(const std::string &model_path) {
	ov::Core core;
	const auto model = core.read_model(model_path);
	if (model->is_dynamic())
		model->reshape(input_dims_);

	compiled_model_ = core.compile_model(model, "AUTO");
	infer_request_ = compiled_model_.create_infer_request();

	return RET_OK;
}

const char *OpenVINOEngine::WarmUpSession() {
	const auto *warm_data = new float[input_dims_[2] * input_dims_[3]];
	ov::Tensor input_tensor = infer_request_.get_input_tensor();
	float *input_data = input_tensor.data<float>();

	memcpy(input_data, warm_data, input_dims_[2] * input_dims_[3] * sizeof(float));
	infer_request_.set_input_tensor(input_tensor);
	infer_request_.infer();

	delete[] warm_data;
	return RET_OK;
}

std::array<TensorView<float>, 2> OpenVINOEngine::RunInference(float *data) {
	ov::Tensor input_tensor = infer_request_.get_input_tensor();
	float *input_data = input_tensor.data<float>();
	memcpy(input_data, data, input_dims_[2] * input_dims_[3] * sizeof(float));
	infer_request_.set_input_tensor(input_tensor);
	infer_request_.infer();

	return {
		TensorView<float>{infer_request_.get_output_tensor(0).data<float>(), infer_request_.get_output_tensor(0).get_size()},
		TensorView<float>{infer_request_.get_output_tensor(1).data<float>(), infer_request_.get_output_tensor(1).get_size()}
	};
}

OpenVINOEngine::OpenVINOEngine(const std::string &model_path,
                               const std::vector<const char *> &input_names,
                               const std::vector<const char *> &output_names,
                               const std::vector<int64_t> &input_dims,
                               const std::vector<int64_t> &output_dims0,
                               const std::vector<int64_t> &output_dims1)
	: SuperPointEngine(model_path, input_names, output_names, input_dims, output_dims0, output_dims1) {
	CHECK(CreateSession(model_path_));
	CHECK(WarmUpSession());
}


SuperPointEngine::InferencePtr
OpenVINOEngine::CreateInstance(const std::string &model_path,
                               const std::vector<const char *> &input_names,
                               const std::vector<const char *> &output_names,
                               const std::vector<int64_t> &input_dims,
                               const std::vector<int64_t> &output_dims0,
                               const std::vector<int64_t> &output_dims1) {
	SuperPointEngine *ptr = new OpenVINOEngine(model_path, input_names, output_names,
	                                           input_dims, output_dims0, output_dims1);
	std::unique_ptr<SuperPointEngine> engine(std::move(ptr));
	return engine;
}
}
