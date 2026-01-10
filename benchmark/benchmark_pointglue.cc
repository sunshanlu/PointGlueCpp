#include <benchmark/benchmark.h>

#include <opencv2/opencv.hpp>

#include "pointgluecpp/SuperPoint/SuperPoint.h"
#include "pointgluecpp/SuperGlue/SuperGlue.h"

// 全局变量，用于存储测试数据
static cv::Mat test_image0;
static cv::Mat test_image1;
static sp::SuperPoint* super_point_instance = nullptr;
static sg::SuperGlue* super_glue_instance = nullptr;

// 初始化测试环境
static void BM_PointGlue_Setup(const benchmark::State& state)
{
  // 从真实图像文件加载测试数据
  const std::string image0_path = std::string(RESOURCES_FILES_DIR) + "/images/000000.png";
  const std::string image1_path = std::string(RESOURCES_FILES_DIR) + "/images/000004.png";

  test_image0 = cv::imread(image0_path, cv::IMREAD_GRAYSCALE);
  test_image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);

  if (test_image0.empty() || test_image1.empty()) {
    std::cerr << "Failed to load images: " << image0_path << " or " << image1_path << std::endl;
    return;
  }

  // 初始化 SuperPoint
  auto sp_options = sp::SuperPoint::Options::CreateFromYaml(
    std::string(RESOURCES_FILES_DIR) + "/options/SuperPointOptions.yaml"
  );
  super_point_instance = new sp::SuperPoint(std::move(sp_options));

  // 初始化 SuperGlue
  auto sg_options = sg::SuperGlue::Options::CreateFromYaml(
    std::string(RESOURCES_FILES_DIR) + "/options/SuperGlueOptions.yaml"
  );
  super_glue_instance = new sg::SuperGlue(std::move(sg_options));

  // 预热
  auto [kpts0, desc0] = super_point_instance->RunSession(test_image0);
  auto [kpts1, desc1] = super_point_instance->RunSession(test_image1);
  sp::SuperPoint::SuperPointRet ret0{kpts0, desc0};
  sp::SuperPoint::SuperPointRet ret1{kpts1, desc1};
  auto matches = super_glue_instance->RunSession(ret0, ret1, test_image0.size(), test_image1.size());
}

// 清理测试环境
static void BM_PointGlue_Teardown(const benchmark::State& state)
{
  delete super_point_instance;
  delete super_glue_instance;
}

// 完整流程性能测试 - 640x480 图像
static void BM_PointGlue_FullPipeline_640x480(benchmark::State& state)
{
  for (auto _ : state)
  {
    // SuperPoint 特征提取
    auto [kpts0, desc0] = super_point_instance->RunSession(test_image0);
    auto [kpts1, desc1] = super_point_instance->RunSession(test_image1);

    // SuperGlue 特征匹配
    sp::SuperPoint::SuperPointRet ret0{kpts0, desc0};
    sp::SuperPoint::SuperPointRet ret1{kpts1, desc1};
    auto matches = super_glue_instance->RunSession(ret0, ret1, test_image0.size(), test_image1.size());

    benchmark::DoNotOptimize(kpts0.data());
    benchmark::DoNotOptimize(kpts1.data());
    benchmark::DoNotOptimize(matches.matches.data());
    benchmark::DoNotOptimize(matches.confidence.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * test_image0.total() * 2);
}

// 完整流程性能测试 - 320x320 图像
static void BM_PointGlue_FullPipeline_320x320(benchmark::State& state)
{
  // 使用调整大小后的真实图像
  cv::Mat img0_320, img1_320;
  cv::resize(test_image0, img0_320, cv::Size(320, 320));
  cv::resize(test_image1, img1_320, cv::Size(320, 320));

  for (auto _ : state)
  {
    // SuperPoint 特征提取
    auto [kpts0, desc0] = super_point_instance->RunSession(img0_320);
    auto [kpts1, desc1] = super_point_instance->RunSession(img1_320);

    // SuperGlue 特征匹配
    sp::SuperPoint::SuperPointRet ret0{kpts0, desc0};
    sp::SuperPoint::SuperPointRet ret1{kpts1, desc1};
    auto matches = super_glue_instance->RunSession(ret0, ret1, img0_320.size(), img1_320.size());

    benchmark::DoNotOptimize(kpts0.data());
    benchmark::DoNotOptimize(kpts1.data());
    benchmark::DoNotOptimize(matches.matches.data());
    benchmark::DoNotOptimize(matches.confidence.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * img0_320.total() * 2);
}

// 注册性能测试
BENCHMARK(BM_PointGlue_FullPipeline_640x480)
    ->Setup(BM_PointGlue_Setup)
    ->Teardown(BM_PointGlue_Teardown)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5)
    ->Repetitions(5)
    ->DisplayAggregatesOnly(true)
    ->Name("PointGlue/FullPipeline_640x480");

BENCHMARK(BM_PointGlue_FullPipeline_320x320)
    ->Setup(BM_PointGlue_Setup)
    ->Teardown(BM_PointGlue_Teardown)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5)
    ->Repetitions(5)
    ->DisplayAggregatesOnly(true)
    ->Name("PointGlue/FullPipeline_320x320");

// 运行所有性能测试
BENCHMARK_MAIN();
