#pragma once

#include <tuple>
#include <memory>

#include <opencv2/opencv.hpp>

#include "pointgluecpp/utils/TensorView.hpp"

namespace sg {
class SuperGlueEngine {
public:
	using InferencePtr = std::unique_ptr<SuperGlueEngine>;

	struct SuperGlueInput {
		explicit SuperGlueInput(size_t size0, size_t size1)
			: kpts0(size0, 2, CV_32F),
			  kpts1(size1, 2, CV_32F),
			  desc0(256, size0, CV_32F),
			  desc1(256, size1, CV_32F),
			  scores0(size0),
			  scores1(size1) {
		}

		SuperGlueInput() = default;

		cv::Mat kpts0;
		cv::Mat kpts1;
		cv::Mat desc0;
		cv::Mat desc1;
		std::vector<float> scores0;
		std::vector<float> scores1;
	};

	virtual std::tuple<TensorView<int64_t>, TensorView<float> > RunInference(SuperGlueInput input) = 0;


	virtual ~SuperGlueEngine() = default;

protected:
	SuperGlueEngine(const std::string &model_path,
	                const std::vector<const char *> &input_names,
	                const std::vector<const char *> &output_names,
	                const short &desc_dim,
	                const short &keypoint_dim);

	virtual const char *WarmupSession() = 0;

	virtual const char *CreateSession(const std::string &model_path) = 0;

	const std::string model_path_;           //< 模型路径
	std::vector<const char *> input_names_;  //< 输入名称
	std::vector<const char *> output_names_; //< 输出名称
	short desc_dim_;                         //< 描述维度
	short keypoint_dim_;                     //< 关键点维度
};
}
