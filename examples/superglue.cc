#include <filesystem>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <npycpp/npycpp.hpp>

#include "pointgluecpp/SuperGlue/SuperGlue.h"
#include "pointgluecpp/utils/TensorView.hpp"

int main(int argc, char** argv)
{
  std::filesystem::path yaml_path = std::filesystem::path(RESOURCES_FILES_DIR) / "options" / "SuperGlueOptions.yaml";
  std::string data_path = std::filesystem::path(RESOURCES_FILES_DIR) / "npz" / "input_data.npz";

  // 配置SuperGlue
  auto options = sg::SuperGlue::Options::CreateFromYaml(yaml_path);
  sg::SuperGlue super_glue(std::move(options));

  // 读取数据并推理
  SPDLOG_INFO("running superglue model...");
  npy::NpzReader input_data(data_path);
  npy::NpData keypoints0 = input_data["keypoints0"];
  npy::NpData keypoints1 = input_data["keypoints1"];
  npy::NpData descriptors0 = input_data["descriptors0"];
  npy::NpData descriptors1 = input_data["descriptors1"];
  npy::NpData scores0 = input_data["scores0"];
  npy::NpData scores1 = input_data["scores1"];
  npy::NpData image0 = input_data["image0"];
  npy::NpData image1 = input_data["image1"];

  cv::Mat pts0(keypoints0.Shape()[1], 2, CV_32F, keypoints0.Ptr<float>());
  cv::Mat pts1(keypoints1.Shape()[1], 2, CV_32F, keypoints1.Ptr<float>());

  TensorView kpts0(keypoints0.Ptr<float>(), keypoints0.Shape());
  TensorView kpts1(keypoints1.Ptr<float>(), keypoints1.Shape());
  TensorView desc0(descriptors0.Ptr<float>(), descriptors0.Shape());
  TensorView desc1(descriptors1.Ptr<float>(), descriptors1.Shape());
  TensorView scores0_tv(scores0.Ptr<float>(), scores0.Shape());
  TensorView scores1_tv(scores1.Ptr<float>(), scores1.Shape());

  cv::Size shape0 = cv::Size(image0.Shape()[3], image0.Shape()[2]);
  cv::Size shape1 = cv::Size(image1.Shape()[3], image1.Shape()[2]);

  cv::Mat image0_mat(shape0, CV_32F, image0.Ptr<float>());
  cv::Mat image1_mat(shape1, CV_32F, image1.Ptr<float>());
  image0_mat.convertTo(image0_mat, CV_8U, 255.0);
  image1_mat.convertTo(image1_mat, CV_8U, 255.0);


  auto matches = super_glue.RunSession(
    kpts0.data(),
    kpts1.data(),
    desc0.data(),
    desc1.data(),
    scores0_tv.data(),
    scores1_tv.data(),
    input_data["keypoints0"].Shape()[1],
    input_data["keypoints1"].Shape()[1],
    shape0,
    shape1
  );

  sg::SuperGlue::ShowMatches(image0_mat, image1_mat, matches, pts0, pts1);

  SPDLOG_INFO("superglue matches count: {}", matches.count);

  return EXIT_SUCCESS;
}
