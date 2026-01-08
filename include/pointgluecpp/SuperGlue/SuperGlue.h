#pragma once

#include <memory>

#include "pointgluecpp/SuperPoint/SuperPoint.h"
#include "pointgluecpp/SuperGlue/SuperGlueEngine.h"

namespace sg {
class SuperGlueEngine;

class SuperGlue {
public:
	struct Options;
	using EnginePtr = std::unique_ptr<SuperGlueEngine>;
	using SuperPointRet = sp::SuperPoint::SuperPointRet;
	using OptionsPtr = std::unique_ptr<Options>;
	using SuperGlueInput = SuperGlueEngine::SuperGlueInput;

	explicit SuperGlue(OptionsPtr options);

	struct Matches {
		void clear() {
			matches.clear();
			confidence.clear();
			count = 0;
		}

		std::vector<int64_t> matches;  //< 匹配关系
		std::vector<float> confidence; //< 匹配置信度
		size_t count;                  //< 匹配数量
	};

	struct Options {
		static OptionsPtr CreateFromYaml(const std::string &yaml_path) {
			auto options = std::make_unique<Options>();

			const cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
			if (!fs.isOpened())
				throw std::runtime_error("Cannot open YAML file: " + yaml_path);

			fs["model_path"] >> options->model_path;
			fs["desc_dim"] >> options->desc_dim;
			fs["keypoint_dim"] >> options->keypoint_dim;

			// 读取输入节点名称
			const cv::FileNode input_names_node = fs["input_names"];
			for (const auto &node: input_names_node) {
				std::string name;
				node >> name;
				options->input_names.push_back(strdup(name.c_str()));
			}

			// 读取输出节点名称
			const cv::FileNode output_names_node = fs["output_names"];
			for (const auto &node: output_names_node) {
				std::string name;
				node >> name;
				options->output_names.push_back(strdup(name.c_str()));
			}

			return options;
		}

		std::string model_path;
		std::vector<const char *> input_names;
		std::vector<const char *> output_names;
		short desc_dim;
		short keypoint_dim;
	};

	Matches RunSession(const SuperPointRet &kpts0,
	                   const SuperPointRet &kpts1,
	                   cv::Size shape0, cv::Size shape1);

	Matches RunSession(float *kpts0,
	                   float *kpts1,
	                   float *desc0,
	                   float *desc1,
	                   float *scores0,
	                   float *scores1,
	                   const std::size_t &nums0,
	                   const std::size_t &nums1,
	                   const cv::Size &shape0,
	                   const cv::Size &shape1);

	static const char *MergeTwoImages(const cv::Mat &image0, const cv::Mat &image1, cv::Mat &merged_image);

	static const char *MappingColors(const std::vector<float> &conf, std::vector<cv::Scalar> &colors);

	static const char *AddTxtMessage(cv::Mat &merged_image, const std::size_t &num0, const std::size_t &num1, const int &count);

	static const char *ShowMatches(const cv::Mat &image0,
	                               const cv::Mat &image1,
	                               const Matches &matches,
	                               const std::vector<cv::KeyPoint> &kpts0,
	                               const std::vector<cv::KeyPoint> &kpts1);

	static const char *ShowMatches(const cv::Mat &image0,
	                               const cv::Mat &image1,
	                               const Matches &matches,
	                               const cv::Mat &kpts0,
	                               const cv::Mat &kpts1);

private:
	const char *PreProcess(const SuperPointRet &kpts0, const SuperPointRet &kpts1,
	                       const cv::Size &shape0, const cv::Size &shape1,
	                       SuperGlueInput &superglue_intput);

	const char *PreProcess(float *kpts0_data,
	                       float *kpts1_data,
	                       float *desc0_data,
	                       float *desc1_data,
	                       float *scores0_data,
	                       float *scores1_data,
	                       const std::size_t &nums0,
	                       const std::size_t &nums1,
	                       const cv::Size &shape0,
	                       const cv::Size &shape1,
	                       SuperGlueInput &superglue_input);

	const char *PostProcess(TensorView<int64_t> &indices0,
	                        TensorView<float> &mscores0,
	                        Matches &matches);

	const char *NormlizeKeypoints(cv::Mat &kpts, const cv::Size &shape);

	EnginePtr infer_engine_; //< 推理引擎
	OptionsPtr options_;     //< 参数
};
}
