#include <spdlog/spdlog.h>
#include <numeric>

#include "pointgluecpp/SuperPoint/ONNXRuntimeEngine.h"
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
std::array<TensorView<float>, 2>
ONNXRuntimeEngine::RunInference(float *data) {
	const auto input_tensor = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		data,
		input_dims_[2] * input_dims_[3],
		input_dims_.data(),
		input_dims_.size()
	);

	const size_t output_count0 = std::accumulate(
		output_dims0_.begin(), output_dims0_.end(),
		1, std::multiplies<size_t>());
	const size_t output_count1 = std::accumulate(
		output_dims1_.begin(), output_dims1_.end(),
		1, std::multiplies<size_t>());

	TensorView<float> output_data0(output_count0);
	TensorView<float> output_data1(output_count1);
	std::array<Ort::Value, 2> output_tensors;
	output_tensors[0] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		output_data0.data(),
		output_count0,
		output_dims0_.data(),
		output_dims0_.size()
	);
	output_tensors[1] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		output_data1.data(),
		output_count1,
		output_dims1_.data(),
		output_dims1_.size()
	);

	session_->Run(
		Ort::RunOptions{nullptr},
		input_names_.data(),
		&input_tensor,
		input_names_.size(),
		output_names_.data(),
		output_tensors.data(),
		output_names_.size()
	);

	return {std::move(output_data0), std::move(output_data1)};
}


SuperPointEngine::InferencePtr
ONNXRuntimeEngine::CreateInstance(const std::string &model_path,
                                  const std::vector<const char *> &input_names,
                                  const std::vector<const char *> &output_names,
                                  const std::vector<int64_t> &input_dims,
                                  const std::vector<int64_t> &output_dims0,
                                  const std::vector<int64_t> &output_dims1) {
	SuperPointEngine *ptr = new ONNXRuntimeEngine(model_path, input_names, output_names,
	                                              input_dims, output_dims0, output_dims1);
	std::unique_ptr<SuperPointEngine> engine(std::move(ptr));
	return engine;
}


ONNXRuntimeEngine::ONNXRuntimeEngine(const std::string &model_path,
                                     const std::vector<const char *> &input_names,
                                     const std::vector<const char *> &output_names,
                                     const std::vector<int64_t> &input_dims,
                                     const std::vector<int64_t> &output_dims0,
                                     const std::vector<int64_t> &output_dims1)
	: SuperPointEngine(model_path, input_names, output_names, input_dims, output_dims0, output_dims1) {
	CHECK(CreateSession(model_path_));
	CHECK(WarmUpSession());
}

// 创建onnxruntime的推理session
const char *ONNXRuntimeEngine::CreateSession(const std::string &model_path) {
	env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "superpoint");
	Ort::SessionOptions session_options;

	// 使用 GetAvailableProviders API 检查可用的执行提供程序
	// 避免在没有 CUDA 的系统中尝试加载 CUDA provider
	try {
		std::vector<std::string> available_providers = Ort::GetAvailableProviders();
		bool cuda_available = false;

		for (const auto &provider : available_providers) {
			if (provider == "CUDAExecutionProvider") {
				cuda_available = true;
				break;
			}
		}

		if (cuda_available) {
			OrtCUDAProviderOptions cuda_options;
			cuda_options.device_id = 0;
			session_options.AppendExecutionProvider_CUDA(cuda_options);
			SPDLOG_INFO("Using CUDA execution provider");
		} else {
			SPDLOG_INFO("CUDA execution provider not available, using CPU");
		}
	} catch (const std::exception &e) {
		SPDLOG_WARN("Failed to query available providers: {}, using CPU", e.what());
	}

	session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
	session_options.SetIntraOpNumThreads(1);
	session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
	return RET_OK;
}


// 热处理底层软件的初始化
const char *ONNXRuntimeEngine::WarmUpSession() {
	auto *warm_data = new float[input_dims_[2] * input_dims_[3]];

	Ort::Value input_tensor = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		warm_data,
		input_dims_[2] * input_dims_[3],
		input_dims_.data(),
		input_dims_.size()
	);

	session_->Run(
		Ort::RunOptions{nullptr},
		input_names_.data(),
		&input_tensor,
		input_names_.size(),
		output_names_.data(),
		output_names_.size()
	);

	delete[] warm_data;

	return RET_OK;
}

#undef RET_OK
#undef CHECK
}
