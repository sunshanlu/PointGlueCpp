#include <spdlog/spdlog.h>

#include "pointgluecpp/SuperGlue/OpenVINOEngine.h"

#define RET_OK nullptr
#define CHECK(message)\
do {\
	if (message != RET_OK){\
		SPDLOG_ERROR("Error: {}, at {}:{}\n", message, __FILE__, __LINE__);\
		exit(EXIT_FAILURE);\
	}\
} while (0)

namespace sg {
SuperGlueEngine::InferencePtr
OpenVINOEngine::CreateInstance(const std::string &model_path,
                               const std::vector<const char *> &input_names,
                               const std::vector<const char *> &output_names,
                               const short &desc_dim,
                               const short &keypoint_dim) {
	auto instance = new OpenVINOEngine(model_path, input_names, output_names, desc_dim, keypoint_dim);
	return std::unique_ptr<OpenVINOEngine>(std::move(instance));
}

std::tuple<TensorView<int64_t>, TensorView<float> >
OpenVINOEngine::RunInference(SuperGlueInput input) {
	short kpt0_num = input.desc0.cols, kpt1_num = input.desc1.cols;

	// 注意动态 tensor 的处理方法
	ov::Shape desc0_shape({1, static_cast<size_t>(desc_dim_), static_cast<size_t>(kpt0_num)});
	ov::Shape desc1_shape({1, static_cast<size_t>(desc_dim_), static_cast<size_t>(kpt1_num)});
	ov::Shape score1_shape({1, static_cast<size_t>(kpt1_num)});
	ov::Shape score0_shape({1, static_cast<size_t>(kpt0_num)});
	ov::Shape kpt0_shape({1, static_cast<size_t>(kpt0_num), static_cast<size_t>(keypoint_dim_)});
	ov::Shape kpt1_shape({1, static_cast<size_t>(kpt1_num), static_cast<size_t>(keypoint_dim_)});

	ov::Tensor input_desc0(ov::element::f32, desc0_shape);
	ov::Tensor input_desc1(ov::element::f32, desc1_shape);
	ov::Tensor input_score0(ov::element::f32, score0_shape);
	ov::Tensor input_score1(ov::element::f32, score1_shape);
	ov::Tensor input_kpt0(ov::element::f32, kpt0_shape);
	ov::Tensor input_kpt1(ov::element::f32, kpt1_shape);

	memcpy(input_desc0.data<float>(), input.desc0.ptr<float>(), input.desc0.total() * sizeof(float));
	memcpy(input_desc1.data<float>(), input.desc1.ptr<float>(), input.desc1.total() * sizeof(float));
	memcpy(input_score0.data<float>(), input.scores0.data(), input.scores0.size() * sizeof(float));
	memcpy(input_score1.data<float>(), input.scores1.data(), input.scores1.size() * sizeof(float));
	memcpy(input_kpt0.data<float>(), input.kpts0.ptr<float>(), input.kpts0.total() * sizeof(float));
	memcpy(input_kpt1.data<float>(), input.kpts1.ptr<float>(), input.kpts1.total() * sizeof(float));

	infer_request_.set_input_tensor(0, input_desc0);
	infer_request_.set_input_tensor(1, input_desc1);
	infer_request_.set_input_tensor(2, input_score0);
	infer_request_.set_input_tensor(3, input_score1);
	infer_request_.set_input_tensor(4, input_kpt0);
	infer_request_.set_input_tensor(5, input_kpt1);
	infer_request_.infer();

	return {
		TensorView<int64_t>(infer_request_.get_output_tensor(0).data<int64_t>(), infer_request_.get_output_tensor(0).get_size()),
		TensorView<float>(infer_request_.get_output_tensor(1).data<float>(), infer_request_.get_output_tensor(1).get_size())
	};
}

const char *OpenVINOEngine::WarmupSession() {
	static constexpr short kpt0_num = 100, kpt1_num = 120;

	TensorView<float> warm_data_kpt0(kpt0_num * keypoint_dim_);
	TensorView<float> warm_data_kpt1(kpt1_num * keypoint_dim_);
	TensorView<float> warm_data_desc0(kpt0_num * desc_dim_);
	TensorView<float> warm_data_desc1(kpt1_num * desc_dim_);
	TensorView<float> warm_data_score0(kpt0_num);
	TensorView<float> warm_data_score1(kpt1_num);

	ov::Shape desc0_shape({1, static_cast<size_t>(desc_dim_), static_cast<size_t>(kpt0_num)});
	ov::Shape desc1_shape({1, static_cast<size_t>(desc_dim_), static_cast<size_t>(kpt1_num)});
	ov::Shape score1_shape({1, static_cast<size_t>(kpt1_num)});
	ov::Shape score0_shape({1, static_cast<size_t>(kpt0_num)});
	ov::Shape kpt0_shape({1, static_cast<size_t>(kpt0_num), static_cast<size_t>(keypoint_dim_)});
	ov::Shape kpt1_shape({1, static_cast<size_t>(kpt1_num), static_cast<size_t>(keypoint_dim_)});

	ov::Tensor input_desc0(ov::element::f32, desc0_shape);
	ov::Tensor input_desc1(ov::element::f32, desc1_shape);
	ov::Tensor input_score0(ov::element::f32, score0_shape);
	ov::Tensor input_score1(ov::element::f32, score1_shape);
	ov::Tensor input_kpt0(ov::element::f32, kpt0_shape);
	ov::Tensor input_kpt1(ov::element::f32, kpt1_shape);

	memcpy(input_desc0.data<float>(), warm_data_desc0.data(), kpt0_num * desc_dim_ * sizeof(float));
	memcpy(input_desc1.data<float>(), warm_data_desc1.data(), kpt1_num * desc_dim_ * sizeof(float));
	memcpy(input_score0.data<float>(), warm_data_score0.data(), kpt0_num * sizeof(float));
	memcpy(input_score1.data<float>(), warm_data_score1.data(), kpt1_num * sizeof(float));
	memcpy(input_kpt0.data<float>(), warm_data_kpt0.data(), kpt0_num * keypoint_dim_ * sizeof(float));
	memcpy(input_kpt1.data<float>(), warm_data_kpt1.data(), kpt1_num * keypoint_dim_ * sizeof(float));

	infer_request_.set_input_tensor(0, input_desc0);
	infer_request_.set_input_tensor(1, input_desc1);
	infer_request_.set_input_tensor(2, input_score0);
	infer_request_.set_input_tensor(3, input_score1);
	infer_request_.set_input_tensor(4, input_kpt0);
	infer_request_.set_input_tensor(5, input_kpt1);

	infer_request_.infer();

	return RET_OK;
}

const char *OpenVINOEngine::CreateSession(const std::string &model_path) {
	ov::Core core;
	const auto model = core.read_model(model_path);

	compiled_model_ = core.compile_model(model, "AUTO");
	infer_request_ = compiled_model_.create_infer_request();

	return RET_OK;
}

OpenVINOEngine::OpenVINOEngine(const std::string &model_path,
                               const std::vector<const char *> &input_names,
                               const std::vector<const char *> &output_names,
                               const short &desc_dim,
                               const short &keypoint_dim)
	: SuperGlueEngine(model_path, input_names, output_names, desc_dim, keypoint_dim) {
	CHECK(CreateSession(model_path_));
	CHECK(WarmupSession());
}
}

#undef RET_OK
#undef CHECK
