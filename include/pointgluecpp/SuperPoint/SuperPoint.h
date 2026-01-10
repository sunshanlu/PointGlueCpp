#pragma once

#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

#include "pointgluecpp/SuperPoint/SuperPointEngine.h"

namespace sp
{
/**
 * @brief SuperPoint 特征点检测和描述符提取类
 *
 * 提供 SuperPoint 网络的完整推理流程，包括：
 * - 图像预处理
 * - 模型推理
 * - 特征点检测和描述符提取
 * - Fast-NMS 后处理
 *
 * 支持多种推理引擎（ONNX Runtime、OpenVINO）
 */
class SuperPoint
{
public:
  /**
   * @brief 前向声明输出结果结构体
   */
  struct SuperPointRet;
  /**
   * @brief 前向声明配置项结构体
   */
  struct Options;
  using OptionsPtr = std::unique_ptr<Options>;

  /**
   * @brief SuperPoint 推理输出结果
   *
   * 包含检测到的关键点和对应的 256 维描述子
   */
  struct SuperPointRet
  {
    std::vector<cv::KeyPoint> keypoints;         ///< 检测到的关键点列表
    std::vector<std::array<float, 256>> keydesc; ///< 每个关键点的 256 维描述子
  };

  /**
   * @brief SuperPoint 配置项
   *
   * 定义 SuperPoint 模型的运行参数
   */
  struct Options
  {
    friend class SuperPoint;

    std::string model_path;                    ///< ONNX 模型文件路径
    short width;                               ///< 模型输入图像宽度
    short height;                              ///< 模型输入图像高度
    short border;                              ///< 后处理要求的边界大小
    float conf_threshold;                      ///< 关键点置信度阈值
    short nms_dist;                            ///< Fast-NMS 搜索半径
    std::vector<std::string> input_names_str;  ///< 输入节点名称（字符串存储）
    std::vector<std::string> output_names_str; ///< 输出节点名称（字符串存储）
    std::vector<const char*> input_names;      ///< 输入节点名称（指针，用于引擎接口）
    std::vector<const char*> output_names;     ///< 输出节点名称（指针，用于引擎接口）
    std::vector<int64_t> input_dims;           ///< 输入张量维度
    std::vector<int64_t> output_dims0;         ///< 第一个输出张量维度（semi-dense）
    std::vector<int64_t> output_dims1;         ///< 第二个输出张量维度（descriptor）

    /**
     * @brief 从 YAML 文件创建配置项
     *
     * @param yaml_path YAML 配置文件路径
     * @return OptionsPtr 配置项智能指针
     */
    static OptionsPtr CreateFromYaml(const std::string& yaml_path);

  private:
    /**
     * @brief 更新指针向量
     *
     * 从字符串向量生成 const char* 指针向量，用于引擎接口
     */
    void UpdatePointers();
  };

  /**
   * @brief 构造 SuperPoint 对象
   *
   * 初始化推理引擎，创建推理会话并进行预热
   *
   * @param options SuperPoint 配置项
   */
  explicit SuperPoint(OptionsPtr options);

  /**
   * @brief 执行 SuperPoint 推理
   *
   * 完整的推理流程包括：
   * 1. 图像预处理（缩放、归一化）
   * 2. 模型推理
   * 3. 后处理（Fast-NMS、描述子插值）
   *
   * @param input_image 输入图像（灰度图）
   * @return SuperPointRet 推理结果，包含关键点和描述子
   */
  SuperPointRet RunSession(const cv::Mat& input_image);

private:
  /**
   * @brief 图像预处理
   *
   * 1. 缩放图像至模型输入尺寸
   * 2. 转换为 float 类型并归一化
   * 3. 创建输入张量
   *
   * @param input_image 输入图像
   * @param xscale 输出的 x 轴缩放比例
   * @param yscale 输出的 y 轴缩放比例
   * @param engine_image 输出的预处理后图像（保证内存有效性）
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* PreProcess(const cv::Mat& input_image, float& xscale,
                         float& yscale, cv::Mat& engine_image);

  /**
   * @brief 后处理操作
   *
   * 1. 对输出进行 softmax 操作
   * 2. 过滤低置信度点
   * 3. 使用 Fast-NMS 去除重复点
   * 4. 使用双线性插值获取描述子
   *
   * @param output_data 推理输出张量
   * @param xscale x 轴缩放尺度
   * @param yscale y 轴缩放尺度
   * @param result 输出的后处理结果
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* PostProcess(std::array<TensorView<float>, 2>& output_data,
                          const float& xscale, const float& yscale, SuperPointRet& result);

  /**
   * @brief Fast-NMS 非极大值抑制
   *
   * @param locations 候选关键点位置
   * @param confidence 关键点置信度图
   * @return std::vector<cv::KeyPoint> 过滤后的关键点列表
   */
  std::vector<cv::KeyPoint>
  FastNMS(std::vector<cv::Point>& locations, const cv::Mat& confidence);

  /**
   * @brief 双线性插值
   *
   * 在描述子特征图上进行双线性插值，获取浮点坐标处的描述子
   *
   * @param data 描述子特征图数据 [c, h, w] 存储顺序
   * @param x 要插值的 x 坐标
   * @param y 要插值的 y 坐标
   * @return std::array<float, 256> 插值后的 256 维描述子
   */
  std::array<float, 256>
  bilinearInterpolate(const float* data, const float& x, const float& y);

#ifdef DEBUG
  /**
   * @brief 保存张量数据到文件（调试工具）
   *
   * @param tensor_data 要保存的张量数据
   * @param size 张量元素数量
   * @param file_path 输出文件路径
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* SaveDataToFile(const float* tensor_data, size_t size,
                             const std::string& file_path);
#endif

  OptionsPtr options_;                    ///< SuperPoint 配置项
  SuperPointEngine::InferencePtr engine_; ///< 推理引擎实例
};
}
