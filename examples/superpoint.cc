#include <filesystem>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include "pointgluecpp/SuperPoint/SuperPoint.h"

int main(int argc, char **argv) {
	const std::filesystem::path yaml_path = std::filesystem::path(RESOURCES_FILES_DIR) / "options" / "SuperPointOptions.yaml";
	const std::string image_path = std::filesystem::path(RESOURCES_FILES_DIR) / "images" / "000000.png";

	// 配置SuperPoint
	auto options = sp::SuperPoint::Options::CreateFromYaml(yaml_path);
	sp::SuperPoint super_point(std::move(options));

	// 读取图片并推理
	SPDLOG_INFO("running superpoint model...");
	cv::Mat input_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	const auto [keypoints, keydesc] = super_point.RunSession(input_image);
	for (const auto &kp: keypoints)
		cv::circle(input_image, kp.pt, 2, cv::Scalar(0, 255, 0), -1);

	cv::imshow("superpoint display", input_image);
	cv::waitKey(0);
	cv::destroyAllWindows();

	SPDLOG_INFO("superpoint keypoins num: {}", keypoints.size());

	return EXIT_SUCCESS;
}
