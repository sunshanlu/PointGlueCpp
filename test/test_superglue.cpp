#include <memory>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

#include "pointgluecpp/SuperGlue/SuperGlue.h"

/**
 * SuperGlue 单元测试
 *
 * 测试重点：
 * 1. YAML 配置加载
 * 2. 内存泄漏风险（strdup）
 * 3. 边界条件
 * 4. 错误处理
 * 5. Matches 数据结构
 */
class SuperGlueTest : public ::testing::Test
{
protected:
  std::string test_yaml_path_;

  void SetUp() override
  {
    // 使用 CMake 定义的资源目录路径
#ifdef RESOURCES_FILES_DIR
    test_yaml_path_ = std::string(RESOURCES_FILES_DIR) + "/options/SuperGlueOptions.yaml";
#else
    // 如果未定义，尝试相对于构建目录的路径
    test_yaml_path_ = "../res/options/SuperGlueOptions.yaml";
#endif
  }

  void TearDown() override {}
};

// =============================================================================
// 测试 1: YAML 配置加载
// =============================================================================

TEST_F(SuperGlueTest, LoadYamlConfig)
{
  // 测试从 YAML 文件加载配置
  auto options = sg::SuperGlue::Options::CreateFromYaml(test_yaml_path_);

  EXPECT_FALSE(options->model_path.empty());
  EXPECT_GT(options->desc_dim, 0);
  EXPECT_GT(options->keypoint_dim, 0);

  // 验证输入输出节点名称不为空
  EXPECT_FALSE(options->input_names.empty());
  EXPECT_FALSE(options->output_names.empty());
}

TEST_F(SuperGlueTest, LoadYamlConfigNonExistentFile)
{
  // 测试加载不存在的 YAML 文件
  EXPECT_THROW(
    sg::SuperGlue::Options::CreateFromYaml("nonexistent.yaml"),
    std::runtime_error
  );
}

// =============================================================================
// 测试 2: 内存泄漏风险测试
// =============================================================================

TEST_F(SuperGlueTest, MemoryLeakRiskInOptions)
{
  for (int i = 0; i < 100; ++i)
    auto options = sg::SuperGlue::Options::CreateFromYaml(test_yaml_path_);
}

// =============================================================================
// 测试 3: SuperGlue 构造
// =============================================================================

TEST_F(SuperGlueTest, ConstructSuperGlue)
{
  // 测试 SuperGlue 对象构造
  auto options = sg::SuperGlue::Options::CreateFromYaml(test_yaml_path_);

  // 注意：这需要实际的模型文件存在
  if (std::filesystem::exists(options->model_path))
  {
    EXPECT_NO_THROW({
      sg::SuperGlue superglue(std::move(options));
      });
  }
  else { GTEST_SKIP() << "Model file not found: " << options->model_path; }
}

TEST_F(SuperGlueTest, ConstructSuperGlueWithInvalidModel)
{
  // 测试使用无效模型路径构造
  auto options = std::make_unique<sg::SuperGlue::Options>();
  options->model_path = "nonexistent_model.onnx";
  options->desc_dim = 256;
  options->keypoint_dim = 2;

  // 这应该抛出异常或返回错误
  EXPECT_THROW({sg::SuperGlue superglue(std::move(options));}, std::exception);
}

// =============================================================================
// 测试 4: Matches 数据结构测试
// =============================================================================

TEST_F(SuperGlueTest, MatchesStructure)
{
  // 测试 Matches 结构体的功能
  sg::SuperGlue::Matches matches;

  EXPECT_EQ(matches.matches.size(), 0);
  EXPECT_EQ(matches.confidence.size(), 0);
  EXPECT_EQ(matches.count, 0);

  // 填充测试数据
  matches.matches = {0, 1, 2, -1, 4};
  matches.confidence = {0.9f, 0.8f, 0.7f, 0.1f, 0.95f};
  matches.count = 4; // 除去 -1

  EXPECT_EQ(matches.matches.size(), 5);
  EXPECT_EQ(matches.confidence.size(), 5);
  EXPECT_EQ(matches.count, 4);

  // 测试 clear 方法
  matches.clear();
  EXPECT_EQ(matches.matches.size(), 0);
  EXPECT_EQ(matches.confidence.size(), 0);
  EXPECT_EQ(matches.count, 0);
}

// =============================================================================
// 测试 5: RunSession 边界测试
// =============================================================================

TEST_F(SuperGlueTest, RunSessionWithEmptyKeypoints)
{
  // 测试使用空关键点列表运行推理
  auto options = sg::SuperGlue::Options::CreateFromYaml(test_yaml_path_);

  if (std::filesystem::exists(options->model_path))
  {
    sg::SuperGlue superglue(std::move(options));

    sp::SuperPoint::SuperPointRet kpts0;
    sp::SuperPoint::SuperPointRet kpts1;
    cv::Size shape0(640, 480);
    cv::Size shape1(640, 480);

    auto matches = superglue.RunSession(kpts0, kpts1, shape0, shape1);
    EXPECT_EQ(matches.count, 0);
  }
  else { GTEST_SKIP() << "Model file not found"; }
}

TEST_F(SuperGlueTest, RunSessionWithMismatchedKeypoints)
{
  // 测试使用不匹配的关键点数量运行推理
  auto options = sg::SuperGlue::Options::CreateFromYaml(test_yaml_path_);

  if (std::filesystem::exists(options->model_path))
  {
    sg::SuperGlue superglue(std::move(options));

    // 创建不同数量的关键点
    sp::SuperPoint::SuperPointRet kpts0;
    kpts0.keypoints.resize(10);
    kpts0.keydesc.resize(10);

    sp::SuperPoint::SuperPointRet kpts1;
    kpts1.keypoints.resize(100);
    kpts1.keydesc.resize(100);

    cv::Size shape0(640, 480);
    cv::Size shape1(640, 480);

    EXPECT_NO_THROW({
      auto matches = superglue.RunSession(kpts0, kpts1, shape0, shape1);
      });
  }
  else { GTEST_SKIP() << "Model file not found"; }
}

TEST_F(SuperGlueTest, RunSessionWithRawData)
{
  // 测试使用原始数据指针运行推理
  auto options = sg::SuperGlue::Options::CreateFromYaml(test_yaml_path_);

  if (std::filesystem::exists(options->model_path))
  {
    sg::SuperGlue superglue(std::move(options));

    // 准备测试数据
    constexpr std::size_t nums0 = 10;
    constexpr std::size_t nums1 = 15;

    std::vector kpts0(nums0 * 2, 100.0f);
    std::vector kpts1(nums1 * 2, 200.0f);
    std::vector desc0(256 * nums0, 0.5f);
    std::vector desc1(256 * nums1, 0.5f);
    std::vector scores0(nums0, 0.9f);
    std::vector scores1(nums1, 0.8f);

    cv::Size shape0(640, 480);
    cv::Size shape1(640, 480);

    EXPECT_NO_THROW({
      auto matches = superglue.RunSession(
        kpts0.data(), kpts1.data(),
        desc0.data(), desc1.data(),
        scores0.data(), scores1.data(),
        nums0, nums1, shape0, shape1
      );
      });
  }
  else { GTEST_SKIP() << "Model file not found"; }
}

// =============================================================================
// 测试 6: 静态工具函数测试
// =============================================================================

TEST_F(SuperGlueTest, MergeTwoImages)
{
  // 测试图像合并功能
  cv::Mat image0(480, 640, CV_8UC1, cv::Scalar(100));
  cv::Mat image1(480, 640, CV_8UC1, cv::Scalar(200));
  cv::Mat merged_image;

  const char* result = sg::SuperGlue::MergeTwoImages(image0, image1, merged_image);

  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(merged_image.rows, 480);
  EXPECT_EQ(merged_image.cols, 1280);
}

TEST_F(SuperGlueTest, MergeTwoImagesWithEmptyInput)
{
  // 测试使用空图像合并
  cv::Mat image0;
  cv::Mat image1(480, 640, CV_8UC1, cv::Scalar(200));
  cv::Mat merged_image;

  const char* result = sg::SuperGlue::MergeTwoImages(image0, image1, merged_image);

  EXPECT_NE(result, nullptr);
  EXPECT_TRUE(merged_image.empty());
}

TEST_F(SuperGlueTest, MappingColors)
{
  // 测试颜色映射功能
  std::vector conf = {0.1f, 0.5f, 0.9f};
  std::vector<cv::Scalar> colors;

  const char* result = sg::SuperGlue::MappingColors(conf, colors);

  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(colors.size(), 3);

  // 验证颜色范围
  for (const auto& color : colors)
  {
    EXPECT_GE(color[0], 0);
    EXPECT_LE(color[0], 255);
    EXPECT_GE(color[1], 0);
    EXPECT_LE(color[1], 255);
    EXPECT_GE(color[2], 0);
    EXPECT_LE(color[2], 255);
  }
}

TEST_F(SuperGlueTest, MappingColorsWithEmptyConfidence)
{
  // 测试空置信度向量的颜色映射
  std::vector<float> conf;
  std::vector<cv::Scalar> colors;

  const char* result = sg::SuperGlue::MappingColors(conf, colors);

  EXPECT_NE(result, nullptr);
  EXPECT_EQ(colors.size(), 0);
}

TEST_F(SuperGlueTest, MappingColorsWithUniformConfidence)
{
  // 测试统一置信度的颜色映射
  std::vector conf(10, 0.5f);
  std::vector<cv::Scalar> colors;

  const char* result = sg::SuperGlue::MappingColors(conf, colors);

  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(colors.size(), 10);

  // 当置信度相同时，应该使用蓝色
  for (const auto& color : colors)
  {
    EXPECT_EQ(color[0], 255); // B
    EXPECT_EQ(color[1], 0);   // G
    EXPECT_EQ(color[2], 0);   // R
  }
}

// =============================================================================
// 测试 7: 并行化安全测试
// =============================================================================

TEST_F(SuperGlueTest, ParallelExecutionSafety)
{
  // 在 PreProcess 中使用了 std::execution::par_unseq
  // 需要确保线程安全性

  auto options = sg::SuperGlue::Options::CreateFromYaml(test_yaml_path_);

  if (std::filesystem::exists(options->model_path))
  {
    sg::SuperGlue superglue(std::move(options));

    // 准备测试数据
    sp::SuperPoint::SuperPointRet kpts0;
    sp::SuperPoint::SuperPointRet kpts1;
    kpts0.keypoints.resize(50);
    kpts0.keydesc.resize(50);
    kpts1.keypoints.resize(50);
    kpts1.keydesc.resize(50);

    cv::Size shape0(640, 480);
    cv::Size shape1(640, 480);

    // 多次运行以检测数据竞争
    for (int i = 0; i < 10; ++i)
      auto result = superglue.RunSession(kpts0, kpts1, shape0, shape1);
  }
  else { GTEST_SKIP() << "Model file not found"; }
}

// =============================================================================
// 测试 8: 配置验证
// =============================================================================

TEST_F(SuperGlueTest, ValidateConfigDimensions)
{
  // 测试配置维度的一致性
  auto options = sg::SuperGlue::Options::CreateFromYaml(test_yaml_path_);

  // 验证维度值
  EXPECT_EQ(options->desc_dim, 256);
  EXPECT_EQ(options->keypoint_dim, 2);
}

TEST_F(SuperGlueTest, ValidateInputOutputNodes)
{
  // 测试输入输出节点配置
  auto options = sg::SuperGlue::Options::CreateFromYaml(test_yaml_path_);

  // SuperGlue 应该有特定的输入输出节点
  EXPECT_GT(options->input_names.size(), 0);
  EXPECT_GT(options->output_names.size(), 0);
}
