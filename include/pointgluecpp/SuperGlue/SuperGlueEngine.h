#pragma once

#include <tuple>
#include <memory>

#include <opencv2/opencv.hpp>

#include "pointgluecpp/utils/TensorView.hpp"

namespace sg
{
/**
 * @brief SuperGlue 推理引擎抽象基类
 *
 * 定义了 SuperGlue 特征匹配模型的推理引擎接口
 * 实现基于注意力机制的特征点匹配
 */
class SuperGlueEngine
{
public:
  using InferencePtr = std::unique_ptr<SuperGlueEngine>;

  /**
   * @brief SuperGlue 输入数据结构
   *
   * 包含两张图像的关键点、描述子和置信度分数
   */
  struct SuperGlueInput
  {
    /**
     * @brief 构造指定大小的 SuperGlue 输入
     *
     * @param size0 第一张图像的关键点数量
     * @param size1 第二张图像的关键点数量
     */
    explicit SuperGlueInput(size_t size0, size_t size1)
      : kpts0(size0, 2, CV_32F),
        kpts1(size1, 2, CV_32F),
        desc0(256, size0, CV_32F),
        desc1(256, size1, CV_32F),
        scores0(size0),
        scores1(size1) {}

    SuperGlueInput() = default;

    cv::Mat kpts0;              ///< 第一张图像的关键点坐标 [N x 2]
    cv::Mat kpts1;              ///< 第二张图像的关键点坐标 [M x 2]
    cv::Mat desc0;              ///< 第一张图像的描述子 [256 x N]
    cv::Mat desc1;              ///< 第二张图像的描述子 [256 x M]
    std::vector<float> scores0; ///< 第一张图像关键点的置信度分数 [N]
    std::vector<float> scores1; ///< 第二张图像关键点的置信度分数 [M]
  };

  /**
   * @brief 执行特征匹配推理
   *
   * @param input SuperGlue 输入数据
   * @return std::tuple<TensorView<int64_t>, TensorView<float>> 匹配结果：
   *         - 第一个元素：匹配索引数组（-1 表示无匹配）
   *         - 第二个元素：匹配置信度分数
   */
  virtual std::tuple<TensorView<int64_t>, TensorView<float>> RunInference(SuperGlueInput input) = 0;

  virtual ~SuperGlueEngine() = default;

protected:
  /**
   * @brief 构造 SuperGlue 引擎
   *
   * @param model_path ONNX 模型文件路径
   * @param input_names 输入节点名称列表
   * @param output_names 输出节点名称列表
   * @param desc_dim 描述子维度（通常为 256）
   * @param keypoint_dim 关键点维度（通常为 2，表示 x,y 坐标）
   */
  SuperGlueEngine(const std::string& model_path,
                  const std::vector<const char*>& input_names,
                  const std::vector<const char*>& output_names,
                  const short& desc_dim,
                  const short& keypoint_dim);

  /**
   * @brief 预热推理会话
   *
   * @return const char* 错误信息，成功返回 nullptr
   */
  virtual const char* WarmupSession() = 0;

  /**
   * @brief 创建推理会话
   *
   * @param model_path ONNX 模型文件路径
   * @return const char* 错误信息，成功返回 nullptr
   */
  virtual const char* CreateSession(const std::string& model_path) = 0;

  const std::string model_path_;          ///< ONNX 模型文件路径
  std::vector<const char*> input_names_;  ///< 输入节点名称列表
  std::vector<const char*> output_names_; ///< 输出节点名称列表
  short desc_dim_;                        ///< 描述子维度
  short keypoint_dim_;                    ///< 关键点维度
};
}
