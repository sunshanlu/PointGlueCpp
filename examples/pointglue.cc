#include <filesystem>
#include <chrono>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include "pointgluecpp/SuperPoint/SuperPoint.h"
#include "pointgluecpp/SuperGlue/SuperGlue.h"

int main(int argc, char **argv) {
	// 路径配置
	const std::filesystem::path superpoint_yaml_path =
		std::filesystem::path(RESOURCES_FILES_DIR) / "options" / "SuperPointOptions.yaml";
	const std::filesystem::path superglue_yaml_path =
		std::filesystem::path(RESOURCES_FILES_DIR) / "options" / "SuperGlueOptions.yaml";

	const std::string image0_path = std::filesystem::path(RESOURCES_FILES_DIR) / "images" / "000000.png";
	const std::string image1_path = std::filesystem::path(RESOURCES_FILES_DIR) / "images" / "000004.png";

	// 配置并初始化 SuperPoint
	SPDLOG_INFO("Initializing SuperPoint...");
	auto superpoint_options = sp::SuperPoint::Options::CreateFromYaml(superpoint_yaml_path.string());
	sp::SuperPoint super_point(std::move(superpoint_options));

	// 读取两张图像
	SPDLOG_INFO("Loading images...");
	cv::Mat image0 = cv::imread(image0_path, cv::IMREAD_GRAYSCALE);
	cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);

	if (image0.empty() || image1.empty()) {
		SPDLOG_ERROR("Failed to load images: {} or {}", image0_path, image1_path);
		return EXIT_FAILURE;
	}

	// 使用 SuperPoint 提取第一张图像的特征点
	SPDLOG_INFO("Extracting keypoints from image0 using SuperPoint...");
	auto start = std::chrono::high_resolution_clock::now();
	const auto [keypoints0, keydesc0] = super_point.RunSession(image0);
	auto end = std::chrono::high_resolution_clock::now();
	double sp_time0 = std::chrono::duration<double, std::milli>(end - start).count();
	SPDLOG_INFO("Image0: detected {} keypoints, time: {:.2f} ms", keypoints0.size(), sp_time0);

	// 使用 SuperPoint 提取第二张图像的特征点
	SPDLOG_INFO("Extracting keypoints from image1 using SuperPoint...");
	start = std::chrono::high_resolution_clock::now();
	const auto [keypoints1, keydesc1] = super_point.RunSession(image1);
	end = std::chrono::high_resolution_clock::now();
	double sp_time1 = std::chrono::duration<double, std::milli>(end - start).count();
	SPDLOG_INFO("Image1: detected {} keypoints, time: {:.2f} ms", keypoints1.size(), sp_time1);

	// 准备 SuperPoint 的结果结构用于 SuperGlue
	sp::SuperPoint::SuperPointRet superpoint_ret0;
	superpoint_ret0.keypoints = keypoints0;
	superpoint_ret0.keydesc = keydesc0;

	sp::SuperPoint::SuperPointRet superpoint_ret1;
	superpoint_ret1.keypoints = keypoints1;
	superpoint_ret1.keydesc = keydesc1;

	// 配置并初始化 SuperGlue
	SPDLOG_INFO("Initializing SuperGlue...");
	auto superglue_options = sg::SuperGlue::Options::CreateFromYaml(superglue_yaml_path.string());
	sg::SuperGlue super_glue(std::move(superglue_options));

	// 使用 SuperGlue 进行特征匹配
	SPDLOG_INFO("Matching keypoints using SuperGlue...");
	start = std::chrono::high_resolution_clock::now();
	auto matches = super_glue.RunSession(
		superpoint_ret0,
		superpoint_ret1,
		image0.size(),
		image1.size()
	);
	end = std::chrono::high_resolution_clock::now();
	double sg_time = std::chrono::duration<double, std::milli>(end - start).count();
	SPDLOG_INFO("SuperGlue matched {} keypoint pairs, time: {:.2f} ms", matches.count, sg_time);

	// 输出总耗时
	SPDLOG_INFO("Total SuperPoint time: {:.2f} ms (image0: {:.2f} ms, image1: {:.2f} ms)",
	            sp_time0 + sp_time1, sp_time0, sp_time1);
	SPDLOG_INFO("Total SuperGlue time: {:.2f} ms", sg_time);

	// 可视化匹配结果
	SPDLOG_INFO("Displaying matching results...");
	cv::Mat display_image0, display_image1;
	cv::cvtColor(image0, display_image0, cv::COLOR_GRAY2BGR);
	cv::cvtColor(image1, display_image1, cv::COLOR_GRAY2BGR);

	sg::SuperGlue::ShowMatches(display_image0, display_image1, matches, keypoints0, keypoints1);

	return EXIT_SUCCESS;
}
