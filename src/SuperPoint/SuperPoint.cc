#include <fstream>
#include <filesystem>
#include <execution>

#include <spdlog/spdlog.h>

#include "pointgluecpp/SuperPoint/SuperPoint.h"

#ifdef ONNXRUNTIMEENGINE
#include "pointgluecpp/SuperPoint/ONNXRuntimeEngine.h"
#endif

#ifdef OPENVINOENGINE
#include "pointgluecpp/SuperPoint/OpenVINOEngine.h"
#endif


#define RET_OK nullptr
#define CHECK(message)\
do {\
	if (message != RET_OK){\
		SPDLOG_ERROR("Error: {}, at {}:{}\n", message, __FILE__, __LINE__);\
		exit(EXIT_FAILURE);\
	}\
} while (0)

namespace fs = std::filesystem;

namespace sp {
SuperPoint::SuperPoint(OptionsPtr options)
	: options_(std::move(options)) {
#if defined(ONNXRUNTIMEENGINE)
	engine_ = ONNXRuntimeEngine::CreateInstance(options_->model_path,
	                                            options_->input_names,
	                                            options_->output_names,
	                                            options_->input_dims,
	                                            options_->output_dims0,
	                                            options_->output_dims1);
#elif defined(OPENVINOENGINE)
	engine_ = OpenVINOEngine::CreateInstance(options_->model_path,
	                                         options_->input_names,
	                                         options_->output_names,
	                                         options_->input_dims,
	                                         options_->output_dims0,
	                                         options_->output_dims1);
#else
	static_assert(false, "No inference engine defined. Please define either ONNXRUNTIMEENGINE or OPENVINOENGINE.")
#endif
}


// 推理SuperPoint
SuperPoint::SuperPointRet SuperPoint::RunSession(const cv::Mat &input_image) {
	cv::Mat onnx_image;
	SuperPointRet result;
	float xscale, yscale;
	CHECK(PreProcess(input_image, xscale, yscale, onnx_image));

	auto output_data = engine_->RunInference(onnx_image.ptr<float>());

#if defined(DEBUG) && defined(DEBUG_OUTPUT_DIR)
	const fs::path semi_tensor_file = fs::path(DEBUG_OUTPUT_DIR) / "semi_tensor.txt";
	const fs::path desc_tensor_file = fs::path(DEBUG_OUTPUT_DIR) / "desc_tensor.txt";
	CHECK(SaveDataToFile(output_data[0].data(),
		options_->output_dims0[0] * options_->output_dims0[1] *
		options_->output_dims0[2] * options_->output_dims0[3],
		semi_tensor_file));
	CHECK(SaveDataToFile(output_data[1].data(),
		options_->output_dims1[0] * options_->output_dims1[1] *
		options_->output_dims1[2] * options_->output_dims1[3],
		desc_tensor_file));
#endif

	CHECK(PostProcess(output_data, xscale, yscale, result));

	return result;
}

#ifdef DEBUG
// 使用iostream保存张量数据到文件，debug 工具函数
const char *SuperPoint::SaveDataToFile(const float *tensor_data, const size_t size,
                                       const std::string &file_path) {
	std::ofstream outfile(file_path);
	if (!outfile.is_open())
		return "Error: can not open file";

	for (size_t idx = 0; idx < size; ++idx)
		outfile << tensor_data[idx] << " ";

	outfile << std::endl;

	return RET_OK;
}
#endif

// 前处理操作
const char *SuperPoint::PreProcess(const cv::Mat &input_image, float &xscale,
                                   float &yscale, cv::Mat &engine_image) {
	input_image.convertTo(engine_image, CV_32F, 1.0 / 255.0);

#if defined(DEBUG) && defined(DEBUG_OUTPUT_DIR)
	CHECK(SaveDataToFile(
		engine_image.ptr<float>(),
		input_image.cols * input_image.rows,
		fs::path(DEBUG_OUTPUT_DIR) / "input_image.txt"));
#endif

	cv::resize(engine_image, engine_image, cv::Size(options_->width, options_->height), cv::INTER_LINEAR);
	xscale = static_cast<float>(input_image.cols) / static_cast<float>(options_->width);
	yscale = static_cast<float>(input_image.rows) / static_cast<float>(options_->height);

	return RET_OK;
}

// 后处理操作
const char *SuperPoint::PostProcess(std::array<TensorView<float>, 2> &output_data, const float &xscale,
                                    const float &yscale, SuperPointRet &result) {
	float *output0 = output_data[0].data();
	float *output1 = output_data[1].data();

	// step1. output0的softmax操作，并丢弃掉最后一列
	cv::Mat output_mat0(options_->output_dims0[1], options_->output_dims0[2] * options_->output_dims0[3], CV_32F, output0);
	cv::Mat row_sum, row_sum_repeat;
	cv::exp(output_mat0, output_mat0);
	cv::reduce(output_mat0, row_sum, 0, cv::REDUCE_SUM);
	cv::repeat(row_sum, options_->output_dims0[1], 1, row_sum_repeat);
	cv::divide(output_mat0, row_sum_repeat + 1e-5, output_mat0);
	output_mat0 = output_mat0.rowRange(0, 64).clone();

	std::vector<float> semi_output(options_->width * options_->height);
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 8; ++j)
			for (int k = 0; k < 40; ++k)
				for (int l = 0; l < 40; ++l) {
					const float &value = output_mat0.ptr<float>()[((i * 8 + j) * 40 + k) * 40 + l];
					semi_output[((k * 8 + i) * 40 + l) * 8 + j] = value;
				}

#if defined(DEBUG) && defined(DEBUG_OUTPUT_DIR)
	CHECK(SaveDataToFile(
		semi_output.data(), semi_output.size(),
		fs::path(DEBUG_OUTPUT_DIR) / "confidence.txt"));
#endif


	// step2. 将所有confidence中大于0.015位置的点保存下来
	cv::Mat confidence(options_->height, options_->width, CV_32F, semi_output.data()), condition;
	cv::compare(confidence, options_->conf_threshold, condition, cv::CMP_GT);
	std::vector<cv::Point> locations;
	cv::findNonZero(condition, locations);

	// step3. 使用Fast-NMS操作，获取选择的点
	auto keypoints = FastNMS(locations, confidence);

	// step4. 使用插值的方式获取指定特征点的desc
	std::vector<std::array<float, 256> > keydesc(keypoints.size());
	std::vector<int> indices(keypoints.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](const int &idx) {
		const auto &point = keypoints[idx];
		keydesc[idx] = bilinearInterpolate(output1, point.pt.x / 8.f, point.pt.y / 8.f);
		keypoints[idx].pt.x *= xscale;
		keypoints[idx].pt.y *= yscale;
		keypoints[idx].size *= (xscale + yscale) / 2.f;
	});

	result = SuperPointRet{keypoints, keydesc};
	return RET_OK;
}

std::vector<cv::KeyPoint>
SuperPoint::FastNMS(std::vector<cv::Point> &locations, const cv::Mat &confidence) {
	std::ranges::sort(locations, [&](const cv::Point &a, const cv::Point &b) {
		return confidence.at<float>(a) > confidence.at<float>(b);
	});
	cv::Mat flag(options_->height, options_->width, CV_8S, static_cast<int8_t>(0));
	std::vector<int> indices(locations.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::for_each(std::execution::par_unseq, indices.begin(),
	              indices.end(), [&](const int &idx) {
		              flag.at<int8_t>(locations[idx]) = 1;
	              });

	size_t count = 0;
	std::vector<cv::KeyPoint> keypoints;
	keypoints.reserve(locations.size());

	// 这里的border存在两种情况，一种是差值border问题，另一种是nms边界问题
	for (int idx = 0; idx < locations.size(); ++idx) {
		const auto &row = locations[idx].y;
		const auto &col = locations[idx].x;
		if (row < options_->border || row >= options_->height - options_->border ||
		    col < options_->border || col >= options_->width - options_->border)
			continue;

		if (flag.at<int8_t>(locations[idx]) == 1) {
			flag.rowRange(row - options_->nms_dist, row + options_->nms_dist + 1)
					.colRange(col - options_->nms_dist, col + options_->nms_dist + 1) = 0;
			flag.at<int8_t>(locations[idx]) = -1;

			cv::KeyPoint kp;
			kp.pt.x = static_cast<float>(col);
			kp.pt.y = static_cast<float>(row);
			kp.response = confidence.at<float>(locations[idx]);
			kp.size = options_->nms_dist;
			keypoints.push_back(kp);
			count++;
		}
	}

	return keypoints;
}

// 双线性差值工具函数，要求有指定的输入数据
std::array<float, 256>
SuperPoint::bilinearInterpolate(const float *data, const float &x, const float &y) {
	const int x1 = static_cast<int>(floor(x));
	const int y1 = static_cast<int>(floor(y));
	const int x2 = x1 + 1;
	const int y2 = y1 + 1;
	const float dx = x - x1;
	const float dy = y - y1;

	std::array<float, 256> result;
	std::array<int, 256> indices;
	std::iota(indices.begin(), indices.end(), 0);

	const int channel_gap = options_->output_dims1[2] * options_->output_dims1[3];
	const int row_gap = options_->output_dims1[3];
	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](const int &ch) {
		const float Q11 = data[ch * channel_gap + y1 * row_gap + x1];
		const float Q21 = data[ch * channel_gap + y1 * row_gap + x2];
		const float Q12 = data[ch * channel_gap + y2 * row_gap + x1];
		const float Q22 = data[ch * channel_gap + y2 * row_gap + x2];
		result[ch] = Q11 * (1 - dx) * (1 - dy) +
		             Q21 * dx * (1 - dy) +
		             Q12 * (1 - dx) * dy +
		             Q22 * dx * dy;
	});
	return result;
}

SuperPoint::OptionsPtr SuperPoint::Options::CreateFromYaml(const std::string &yaml_path) {
	// 创建选项对象
	auto options = std::make_unique<Options>();

	// 加载YAML文件
	const cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		throw std::runtime_error("Cannot open YAML file: " + yaml_path);
	}

	// 读取基本配置
	fs["model_path"] >> options->model_path;
	fs["width"] >> options->width;
	fs["height"] >> options->height;
	fs["border"] >> options->border;
	fs["conf_threshold"] >> options->conf_threshold;
	fs["nms_dist"] >> options->nms_dist;

	// 读取输入节点配置
	const cv::FileNode input_names_node = fs["input_names"];
	for (const auto &node: input_names_node) {
		std::string name;
		node >> name;
		options->input_names.push_back(strdup(name.c_str()));
	}

	const cv::FileNode input_dims_node = fs["input_dims"];
	for (const auto &node: input_dims_node) {
		int dim;
		node >> dim;
		options->input_dims.push_back(dim);
	}

	// 读取输出节点配置
	cv::FileNode output_names_node = fs["output_names"];
	for (const auto &node: output_names_node) {
		std::string name;
		node >> name;
		options->output_names.push_back(strdup(name.c_str()));
	}

	// 读取输出维度配置
	cv::FileNode output_dims_node = fs["output_dims"];
	int output_index = 0;
	for (const auto &node: output_dims_node) {
		for (const auto &dim_node: node) {
			int dim;
			dim_node >> dim;
			if (output_index == 0) {
				options->output_dims0.push_back(dim);
			} else {
				options->output_dims1.push_back(dim);
			}
		}
		output_index++;
	}

	return options;
}
}


#undef RET_OK
#undef CHECK
