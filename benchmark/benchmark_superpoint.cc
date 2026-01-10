#include <benchmark/benchmark.h>

#include <opencv2/opencv.hpp>

#include "pointgluecpp/SuperPoint/SuperPoint.h"

// 全局变量，用于存储测试数据
static cv::Mat test_image_640x480;
static cv::Mat test_image_320x320;
static sp::SuperPoint* super_point_instance = nullptr;

// 初始化测试环境
static void BM_SuperPoint_Setup(const benchmark::State& state)
{
  // 从真实图像文件加载测试数据
  const std::string image_path = std::string(RESOURCES_FILES_DIR) + "/images/000000.png";
  test_image_640x480 = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

  if (test_image_640x480.empty()) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    return;
  }

  // 调整图像大小到 320x320
  cv::resize(test_image_640x480, test_image_320x320, cv::Size(320, 320));

  // 初始化 SuperPoint
  auto options = sp::SuperPoint::Options::CreateFromYaml(
    std::string(RESOURCES_FILES_DIR) + "/options/SuperPointOptions.yaml"
  );
  super_point_instance = new sp::SuperPoint(std::move(options));

  // 预热
  auto [kpts, desc] = super_point_instance->RunSession(test_image_640x480);
}

// 清理测试环境
static void BM_SuperPoint_Teardown(const benchmark::State& state) { delete super_point_instance; }

// SuperPoint 推理性能测试 - 640x480 图像
static void BM_SuperPoint_Inference_640x480(benchmark::State& state)
{
  for (auto _ : state)
  {
    auto [keypoints, descriptors] = super_point_instance->RunSession(test_image_640x480);
    benchmark::DoNotOptimize(keypoints.data());
    benchmark::DoNotOptimize(descriptors.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * test_image_640x480.total());
}

// SuperPoint 推理性能测试 - 320x320 图像
static void BM_SuperPoint_Inference_320x320(benchmark::State& state)
{
  for (auto _ : state)
  {
    auto [keypoints, descriptors] = super_point_instance->RunSession(test_image_320x320);
    benchmark::DoNotOptimize(keypoints.data());
    benchmark::DoNotOptimize(descriptors.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * test_image_320x320.total());
}

// 注册性能测试
BENCHMARK(BM_SuperPoint_Inference_640x480)
    ->Setup(BM_SuperPoint_Setup)
    ->Teardown(BM_SuperPoint_Teardown)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5)
    ->Repetitions(5)
    ->DisplayAggregatesOnly(true)
    ->Name("SuperPoint/Inference_640x480");

BENCHMARK(BM_SuperPoint_Inference_320x320)
    ->Setup(BM_SuperPoint_Setup)
    ->Teardown(BM_SuperPoint_Teardown)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5)
    ->Repetitions(5)
    ->DisplayAggregatesOnly(true)
    ->Name("SuperPoint/Inference_320x320");

// 运行所有性能测试
BENCHMARK_MAIN();
