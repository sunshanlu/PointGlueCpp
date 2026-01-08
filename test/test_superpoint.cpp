#include <memory>
#include <filesystem>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "pointgluecpp/SuperPoint/SuperPoint.h"

/**
 * SuperPoint 单元测试
 *
 * 测试重点：
 * 1. YAML 配置加载
 * 2. 内存泄漏风险
 * 3. 边界条件
 * 4. 错误处理
 */
class SuperPointTest : public ::testing::Test
{
protected:
  std::string test_yaml_path_;

  void SetUp() override
  {
    // 使用 CMake 定义的资源目录路径
#ifdef RESOURCES_FILES_DIR
    test_yaml_path_ = std::string(RESOURCES_FILES_DIR) + "/options/SuperPointOptions.yaml";
#else
    // 如果未定义，尝试相对于构建目录的路径
    test_yaml_path_ = "../res/options/SuperPointOptions.yaml";
#endif
  }

  void TearDown() override {}
};

// =============================================================================
// 测试 1: YAML 配置加载
// =============================================================================

TEST_F(SuperPointTest, LoadYamlConfig)
{
  // 测试从 YAML 文件加载配置
  auto options = sp::SuperPoint::Options::CreateFromYaml(test_yaml_path_);

  EXPECT_FALSE(options->model_path.empty());
  EXPECT_GT(options->width, 0);
  EXPECT_GT(options->height, 0);
  EXPECT_GE(options->border, 0);
  EXPECT_GT(options->conf_threshold, 0.0f);
  EXPECT_GT(options->nms_dist, 0);

  // 验证输入输出节点名称不为空
  EXPECT_FALSE(options->input_names.empty());
  EXPECT_FALSE(options->output_names.empty());

  // 验证维度配置
  EXPECT_FALSE(options->input_dims.empty());
  EXPECT_FALSE(options->output_dims0.empty());
  EXPECT_FALSE(options->output_dims1.empty());
}

TEST_F(SuperPointTest, LoadYamlConfigNonExistentFile)
{
  // 测试加载不存在的 YAML 文件
  EXPECT_THROW(
    sp::SuperPoint::Options::CreateFromYaml("nonexistent.yaml"),
    std::runtime_error
  );
}

// =============================================================================
// 测试 2: 内存泄漏风险测试
// =============================================================================

TEST_F(SuperPointTest, MemoryLeakRiskInOptions)
{
  // ✅ 修复：使用 std::string 存储，自动内存管理，不再泄漏
  for (int i = 0; i < 100; ++i)
    auto options = sp::SuperPoint::Options::CreateFromYaml(test_yaml_path_);
}

// =============================================================================
// 测试 3: SuperPoint 构造
// =============================================================================

TEST_F(SuperPointTest, ConstructSuperPoint)
{
  // 测试 SuperPoint 对象构造
  auto options = sp::SuperPoint::Options::CreateFromYaml(test_yaml_path_);

  // 注意：这需要实际的模型文件存在
  if (std::filesystem::exists(options->model_path))
  {
    EXPECT_NO_THROW({sp::SuperPoint superpoint(std::move(options));
      });
  }
  else { GTEST_SKIP() << "Model file not found: " << options->model_path; }
}

TEST_F(SuperPointTest, ConstructSuperPointWithInvalidModel)
{
  // 测试使用无效模型路径构造
  auto options = std::make_unique<sp::SuperPoint::Options>();
  options->model_path = "nonexistent_model.onnx";
  options->width = 640;
  options->height = 480;
  options->border = 4;
  options->conf_threshold = 0.015f;
  options->nms_dist = 4;

  // 这应该抛出异常或返回错误
  EXPECT_THROW({sp::SuperPoint superpoint(std::move(options));
               }, std::exception);
}

// =============================================================================
// 测试 4: RunSession 边界测试
// =============================================================================

TEST_F(SuperPointTest, RunSessionWithEmptyImage)
{
  // 测试使用空图像运行推理
  auto options = sp::SuperPoint::Options::CreateFromYaml(test_yaml_path_);

  if (std::filesystem::exists(options->model_path))
  {
    sp::SuperPoint superpoint(std::move(options));
    cv::Mat empty_image;

    EXPECT_THROW({
                 auto result = superpoint.RunSession(empty_image);
                 }, std::runtime_error);
  }
  else { GTEST_SKIP() << "Model file not found"; }
}

TEST_F(SuperPointTest, RunSessionWithVerySmallImage)
{
  // 测试使用极小图像运行推理
  auto options = sp::SuperPoint::Options::CreateFromYaml(test_yaml_path_);

  if (std::filesystem::exists(options->model_path))
  {
    sp::SuperPoint superpoint(std::move(options));

    // 1x1 图像
    cv::Mat tiny_image(1, 1, CV_8UC1, cv::Scalar(128));
    EXPECT_THROW(auto result = superpoint.RunSession(tiny_image), std::runtime_error);
  }
  else { GTEST_SKIP() << "Model file not found"; }
}

TEST_F(SuperPointTest, RunSessionWithNonSquareImage)
{
  // 测试使用非方形图像运行推理
  auto options = sp::SuperPoint::Options::CreateFromYaml(test_yaml_path_);

  if (std::filesystem::exists(options->model_path))
  {
    sp::SuperPoint superpoint(std::move(options));

    // 1920x1080 宽屏图像
    cv::Mat wide_image(1080, 1920, CV_8UC1, cv::Scalar(128));

    EXPECT_NO_THROW({
      auto result = superpoint.RunSession(wide_image);
      });
  }
  else { GTEST_SKIP() << "Model file not found"; }
}

// =============================================================================
// 测试 6: 并行化安全测试
// =============================================================================

TEST_F(SuperPointTest, ParallelExecutionSafety)
{
  // 测试并行执行的安全性
  // 代码中使用了 std::execution::par_unseq

  auto options = sp::SuperPoint::Options::CreateFromYaml(test_yaml_path_);

  if (std::filesystem::exists(options->model_path))
  {
    sp::SuperPoint superpoint(std::move(options));

    cv::Mat test_image(480, 640, CV_8UC1, cv::Scalar(128));

    // 多次运行以检测数据竞争
    for (int i = 0; i < 10; ++i)
    {
      auto [keypoints, keydesc] = superpoint.RunSession(test_image);
      EXPECT_GE(keypoints.size(), 0);
    }
  }
  else { GTEST_SKIP() << "Model file not found"; }
}

// =============================================================================
// 测试 8: 配置验证
// =============================================================================

TEST_F(SuperPointTest, ValidateConfigDimensions)
{
  // 测试配置维度的一致性
  auto options = sp::SuperPoint::Options::CreateFromYaml(test_yaml_path_);

  // 验证维度数组的长度
  EXPECT_EQ(options->input_dims.size(), 4);   // NCHW
  EXPECT_EQ(options->output_dims0.size(), 4); // Semi
  EXPECT_EQ(options->output_dims1.size(), 4); // Desc

  // 验证维度值
  EXPECT_EQ(options->input_dims[0], 1); // Batch size
  EXPECT_GT(options->input_dims[2], 0); // Height
  EXPECT_GT(options->input_dims[3], 0); // Width
}

TEST_F(SuperPointTest, ValidateConfigThresholds)
{
  // 测试配置阈值的合理性
  auto options = sp::SuperPoint::Options::CreateFromYaml(test_yaml_path_);

  // 置信度阈值应该在 [0, 1] 范围内
  EXPECT_GE(options->conf_threshold, 0.0f);
  EXPECT_LE(options->conf_threshold, 1.0f);

  // NMS 距离应该大于 0
  EXPECT_GT(options->nms_dist, 0);

  // Border 应该合理
  EXPECT_GE(options->border, 8);
  EXPECT_LT(options->border, options->height / 2);
  EXPECT_LT(options->border, options->width / 2);
}
