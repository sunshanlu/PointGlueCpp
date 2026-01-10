#pragma once

#include <array>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "pointgluecpp/utils/TensorView.hpp"

namespace sp
{
/**
 * @brief SuperPoint 推理引擎抽象基类
 *
 * 定义了 SuperPoint 模型推理引擎的接口，支持多种后端实现（如 ONNX Runtime、OpenVINO）
 * 提供关键点检测和描述子提取功能
 */
class SuperPointEngine
{
public:
  using InferencePtr = std::unique_ptr<SuperPointEngine>;

  /**
   * @brief SuperPoint 推理输出结果
   *
   * 包含检测到的关键点和对应的描述子
   */
  struct SuperPointRet
  {
    std::vector<cv::KeyPoint> keypoints;         ///< 关键点列表
    std::vector<std::array<float, 256>> keydesc; ///< 每个关键点的 256 维描述子
  };

  /**
   * @brief 支持的推理引擎类型
   */
  enum class Engine
  {
    ONNXRUNTIME, ///< ONNX Runtime 推理引擎
    OPENVINO     ///< OpenVINO 推理引擎
  };

  virtual ~SuperPointEngine() = default;

  /**
   * @brief 执行推理操作
   *
   * @param data 输入图像数据指针（已预处理为浮点数组）
   * @return std::array<TensorView<float>, 2> 推理结果，包含两个张量：
   *         - [0] 半维度概率图 (semi-dense probability map)
   *         - [1] 描述子特征图 (descriptor feature map)
   */
  virtual std::array<TensorView<float>, 2> RunInference(float* data) = 0;

protected:
  /**
   * @brief 创建推理会话
   *
   * 初始化推理引擎的会话对象，配置执行提供程序等
   *
   * @param model_path ONNX 模型文件路径
   * @return const char* 错误信息，成功返回 nullptr
   */
  virtual const char* CreateSession(const std::string& model_path) = 0;

  /**
   * @brief 预热推理会话
   *
   * 执行一次推理操作以初始化推理引擎的内部状态
   *
   * @return const char* 错误信息，成功返回 nullptr
   */
  virtual const char* WarmUpSession() = 0;

  /**
   * @brief 构造 SuperPoint 引擎
   *
   * @param model_path ONNX 模型文件路径
   * @param input_names 输入节点名称列表
   * @param output_names 输出节点名称列表
   * @param input_dims 输入张量维度
   * @param output_dims0 第一个输出张量维度
   * @param output_dims1 第二个输出张量维度
   */
  SuperPointEngine(
    const std::string& model_path,
    std::vector<const char*> input_names,
    std::vector<const char*> output_names,
    std::vector<int64_t> input_dims,
    std::vector<int64_t> output_dims0,
    std::vector<int64_t> output_dims1
  );

  const std::string model_path_;          ///< ONNX 模型文件路径
  std::vector<const char*> input_names_;  ///< 输入节点名称列表
  std::vector<const char*> output_names_; ///< 输出节点名称列表
  std::vector<int64_t> input_dims_;       ///< 输入张量维度 (NCHW)
  std::vector<int64_t> output_dims0_;     ///< 第一个输出张量维度 (semi-dense)
  std::vector<int64_t> output_dims1_;     ///< 第二个输出张量维度 (descriptor)
};
}
