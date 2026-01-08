#include <filesystem>

#include <spdlog/spdlog.h>

#include "pointgluecpp/SuperGlue/ONNXRuntimeEngine.h"
#include "pointgluecpp/utils/DebugTools.hpp"

#define RET_OK nullptr
#define CHECK(message)\
do {\
	if (message != RET_OK){\
		SPDLOG_ERROR("Error: {}, at {}:{}\n", message, __FILE__, __LINE__);\
		exit(EXIT_FAILURE);\
	}\
} while (0)

namespace fs = std::filesystem;

namespace sg {
std::tuple<TensorView<int64_t>, TensorView<float> >
ONNXRuntimeEngine::RunInference(SuperGlueInput input) {
	const int &kp0_num = input.desc0.cols, &kp1_num = input.desc1.cols;

	const std::array<int64_t, 3> desc0_dims = {1, desc_dim_, kp0_num};
	const std::array<int64_t, 3> desc1_dims = {1, desc_dim_, kp1_num};
	const std::array<int64_t, 2> score0_dims = {1, kp0_num};
	const std::array<int64_t, 2> score1_dims = {1, kp1_num};
	const std::array<int64_t, 3> kp0_dims = {1, kp0_num, keypoint_dim_};
	const std::array<int64_t, 3> kp1_dims = {1, kp1_num, keypoint_dim_};
	const std::array<int64_t, 2> indices_dims = {1, kp0_num};
	const std::array<int64_t, 2> scores_dims = {1, kp0_num};

	std::array<Ort::Value, 6> input_values;
	input_values[0] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		input.desc0.ptr<float>(),
		input.desc0.rows * input.desc0.cols,
		desc0_dims.data(),
		desc0_dims.size()
	);

	input_values[1] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		input.desc1.ptr<float>(),
		input.desc1.rows * input.desc1.cols,
		desc1_dims.data(),
		desc1_dims.size()
	);

	input_values[2] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		input.scores0.data(),
		kp0_num,
		score0_dims.data(),
		score0_dims.size()
	);

	input_values[3] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		input.scores1.data(),
		kp1_num,
		score1_dims.data(),
		score1_dims.size()
	);

	input_values[4] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		input.kpts0.ptr<float>(),
		kp0_num * keypoint_dim_,
		kp0_dims.data(),
		kp0_dims.size()
	);

	input_values[5] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		input.kpts1.ptr<float>(),
		kp1_num * keypoint_dim_,
		kp1_dims.data(),
		kp1_dims.size()
	);

	TensorView<int64_t> output_indices(kp0_num);
	TensorView<float> output_scores(kp0_num);

	std::array<Ort::Value, 2> output_tensors;
	output_tensors[0] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		output_indices.data(),
		kp0_num,
		indices_dims.data(),
		indices_dims.size()
	);

	output_tensors[1] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		output_scores.data(),
		kp0_num,
		scores_dims.data(),
		scores_dims.size()
	);

	session_->Run(
		Ort::RunOptions{nullptr},
		input_names_.data(),
		input_values.data(),
		input_values.size(),
		output_names_.data(),
		output_tensors.data(),
		output_tensors.size()
	);

	return {std::move(output_indices), std::move(output_scores)};
}

SuperGlueEngine::InferencePtr
ONNXRuntimeEngine::CreateInstance(const std::string &model_path,
                                  const std::vector<const char *> &input_names,
                                  const std::vector<const char *> &output_names,
                                  const short &desc_dim,
                                  const short &keypoint_dim) {
	auto *ptr = new ONNXRuntimeEngine(model_path, input_names, output_names, desc_dim, keypoint_dim);
	return std::unique_ptr<ONNXRuntimeEngine>(std::move(ptr));
}

const char *ONNXRuntimeEngine::WarmupSession() {
	static constexpr short kp0_num = 100, kp1_num = 120;

	static const std::array<int64_t, 3> kp0_dims = {1, kp0_num, keypoint_dim_};
	static const std::array<int64_t, 3> kp1_dims = {1, kp1_num, keypoint_dim_};
	static const std::array<int64_t, 3> desc0_dims = {1, desc_dim_, kp0_num};
	static const std::array<int64_t, 3> desc1_dims = {1, desc_dim_, kp1_num};
	static constexpr std::array<int64_t, 2> score0_dims = {1, kp0_num};
	static constexpr std::array<int64_t, 2> score1_dims = {1, kp1_num};

	// kpt: [b, n, 2], desc: [b, 256, n], score: [b, n]
	TensorView<float> warm_data_kpt0(kp0_num * keypoint_dim_);
	TensorView<float> warm_data_kpt1(kp1_num * keypoint_dim_);
	TensorView<float> warm_data_desc0(kp0_num * desc_dim_);
	TensorView<float> warm_data_desc1(kp1_num * desc_dim_);
	TensorView<float> warm_data_score0(kp0_num);
	TensorView<float> warm_data_score1(kp1_num);

	std::array<Ort::Value, 6> input_values;
	input_values[0] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		warm_data_desc0.data(),
		kp0_num * desc_dim_,
		desc0_dims.data(),
		desc0_dims.size()
	);

	input_values[1] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		warm_data_desc1.data(),
		kp1_num * desc_dim_,
		desc1_dims.data(),
		desc1_dims.size()
	);

	input_values[2] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		warm_data_score0.data(),
		kp0_num,
		score0_dims.data(),
		score0_dims.size()
	);

	input_values[3] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		warm_data_score1.data(),
		kp1_num,
		score1_dims.data(),
		score1_dims.size()
	);

	input_values[4] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		warm_data_kpt0.data(),
		kp0_num * keypoint_dim_,
		kp0_dims.data(),
		kp0_dims.size()
	);

	input_values[5] = Ort::Value::CreateTensor(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		warm_data_kpt1.data(),
		kp1_num * keypoint_dim_,
		kp1_dims.data(),
		kp1_dims.size()
	);

	session_->Run(
		Ort::RunOptions{nullptr},
		input_names_.data(),
		input_values.data(),
		input_values.size(),
		output_names_.data(),
		output_names_.size()
	);

	return RET_OK;
}

const char *ONNXRuntimeEngine::CreateSession(const std::string &model_path) {
	env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SuperGlueEngine");
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

ONNXRuntimeEngine::ONNXRuntimeEngine(const std::string &model_path,
                                     const std::vector<const char *> &input_names,
                                     const std::vector<const char *> &output_names,
                                     const short &desc_dim,
                                     const short &keypoint_dim)
	: SuperGlueEngine(model_path, input_names, output_names, desc_dim, keypoint_dim) {
	CHECK(CreateSession(model_path_));
	CHECK(WarmupSession());
}
} // namespace sg

#undef RET_OK
#undef CHECK
