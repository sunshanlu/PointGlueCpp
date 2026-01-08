#pragma once

#include <array>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "pointgluecpp/utils/TensorView.hpp"

namespace sp {
/// 推理引擎核心类
class SuperPointEngine {
public:
	using InferencePtr = std::unique_ptr<SuperPointEngine>;

	/// 推理输出结果
	struct SuperPointRet {
		std::vector<cv::KeyPoint> keypoints;          //< 关键点
		std::vector<std::array<float, 256> > keydesc; //< 描述子
	};

	/// 推理引擎枚举
	enum class Engine {
		ONNXRUNTIME, //< onnxruntime 推理引擎
		OPENVINO     //< openvino 推理引擎
	};

	virtual ~SuperPointEngine() = default;

	/// 推理操作要求数据
	virtual std::array<TensorView<float>, 2> RunInference(float *data) = 0;

protected:
	virtual const char *CreateSession(const std::string &model_path) =0;

	virtual const char *WarmUpSession() =0;

	SuperPointEngine(
		const std::string &model_path, std::vector<const char *> input_names,
		std::vector<const char *> output_names, std::vector<int64_t> input_dims,
		std::vector<int64_t> output_dims0, std::vector<int64_t> output_dims1
	);

	const std::string model_path_;
	std::vector<const char *> input_names_;  //< 输入节点名称
	std::vector<const char *> output_names_; //< 输出节点名称
	std::vector<int64_t> input_dims_;        //< 输入节点维度
	std::vector<int64_t> output_dims0_;      //< 输出节点维度0
	std::vector<int64_t> output_dims1_;      //< 输出节点维度1
};
}
