#pragma once

#include <memory>

#include "pointgluecpp/SuperPoint/SuperPoint.h"
#include "pointgluecpp/SuperGlue/SuperGlueEngine.h"

namespace sg
{
/**
 * @brief SuperGlue 特征点匹配类
 *
 * 提供基于注意力机制的特征点匹配功能，包括：
 * - 输入数据预处理
 * - 模型推理
 * - 匹配结果后处理
 * - 可视化支持
 *
 * 支持多种推理引擎（ONNX Runtime、OpenVINO）
 */
class SuperGlue
{
public:
  /**
   * @brief 前向声明配置项结构体
   */
  struct Options;
  using EnginePtr = std::unique_ptr<SuperGlueEngine>;
  using SuperPointRet = sp::SuperPoint::SuperPointRet;
  using OptionsPtr = std::unique_ptr<Options>;
  using SuperGlueInput = SuperGlueEngine::SuperGlueInput;

  /**
   * @brief 构造 SuperGlue 对象
   *
   * 初始化推理引擎，创建推理会话并进行预热
   *
   * @param options SuperGlue 配置项
   */
  explicit SuperGlue(OptionsPtr options);

  /**
   * @brief 匹配结果结构体
   *
   * 存储两张图像之间的特征点匹配关系
   */
  struct Matches
  {
    /**
     * @brief 清空匹配结果
     */
    void clear()
    {
      matches.clear();
      confidence.clear();
      count = 0;
    }

    std::vector<int64_t> matches;  ///< 匹配关系数组（matches[i] = j 表示图像0的第i个点匹配图像1的第j个点，-1表示无匹配）
    std::vector<float> confidence; ///< 匹配置信度分数
    size_t count = 0;              ///< 匹配数量
  };

  /**
   * @brief SuperGlue 配置项
   *
   * 定义 SuperGlue 模型的运行参数
   */
  struct Options
  {
    /**
     * @brief 从 YAML 文件创建配置项
     *
     * @param yaml_path YAML 配置文件路径
     * @return OptionsPtr 配置项智能指针
     */
    static OptionsPtr CreateFromYaml(const std::string& yaml_path);

    std::string model_path;                    ///< ONNX 模型文件路径
    std::vector<std::string> input_names_str;  ///< 输入节点名称（字符串存储）
    std::vector<std::string> output_names_str; ///< 输出节点名称（字符串存储）
    std::vector<const char*> input_names;      ///< 输入节点名称（指针，用于引擎接口）
    std::vector<const char*> output_names;     ///< 输出节点名称（指针，用于引擎接口）
    short desc_dim;                            ///< 描述子维度（通常为 256）
    short keypoint_dim;                        ///< 关键点维度（通常为 2）

  private:
    /**
     * @brief 更新指针向量
     *
     * 从字符串向量生成 const char* 指针向量，用于引擎接口
     */
    void UpdatePointers()
    {
      input_names.clear();
      output_names.clear();
      input_names.reserve(input_names_str.size());
      output_names.reserve(output_names_str.size());
      for (const auto& name : input_names_str) { input_names.push_back(name.c_str()); }
      for (const auto& name : output_names_str) { output_names.push_back(name.c_str()); }
    }

    friend class SuperGlue;
  };

  /**
   * @brief 执行特征匹配（SuperPointRet 格式）
   *
   * @param kpts0 第一张图像的 SuperPoint 结果
   * @param kpts1 第二张图像的 SuperPoint 结果
   * @param shape0 第一张图像的尺寸
   * @param shape1 第二张图像的尺寸
   * @return Matches 匹配结果
   */
  Matches RunSession(const SuperPointRet& kpts0,
                     const SuperPointRet& kpts1,
                     cv::Size shape0, cv::Size shape1);

  /**
   * @brief 执行特征匹配（原始指针格式）
   *
   * @param kpts0 第一张图像的关键点坐标指针
   * @param kpts1 第二张图像的关键点坐标指针
   * @param desc0 第一张图像的描述子指针
   * @param desc1 第二张图像的描述子指针
   * @param scores0 第一张图像的关键点置信度指针
   * @param scores1 第二张图像的关键点置信度指针
   * @param nums0 第一张图像的关键点数量
   * @param nums1 第二张图像的关键点数量
   * @param shape0 第一张图像的尺寸
   * @param shape1 第二张图像的尺寸
   * @return Matches 匹配结果
   */
  Matches RunSession(float* kpts0,
                     float* kpts1,
                     float* desc0,
                     float* desc1,
                     float* scores0,
                     float* scores1,
                     const std::size_t& nums0,
                     const std::size_t& nums1,
                     const cv::Size& shape0,
                     const cv::Size& shape1);

  /**
   * @brief 合并两张图像
   *
   * 将两张图像水平拼接，用于可视化匹配结果
   *
   * @param image0 第一张图像
   * @param image1 第二张图像
   * @param merged_image 输出的合并图像
   * @return const char* 错误信息，成功返回 nullptr
   */
  static const char* MergeTwoImages(const cv::Mat& image0,
                                    const cv::Mat& image1,
                                    cv::Mat& merged_image);

  /**
   * @brief 根据置信度映射颜色
   *
   * 将置信度分数映射为颜色，用于可视化
   *
   * @param conf 置信度分数列表
   * @param colors 输出的颜色列表
   * @return const char* 错误信息，成功返回 nullptr
   */
  static const char* MappingColors(const std::vector<float>& conf,
                                   std::vector<cv::Scalar>& colors);

  /**
   * @brief 在图像上添加文本信息
   *
   * 显示两张图像的关键点数量和匹配数量
   *
   * @param merged_image 合并后的图像
   * @param num0 第一张图像的关键点数量
   * @param num1 第二张图像的关键点数量
   * @param count 匹配数量
   * @return const char* 错误信息，成功返回 nullptr
   */
  static const char* AddTxtMessage(cv::Mat& merged_image,
                                   const std::size_t& num0,
                                   const std::size_t& num1,
                                   const int& count);

  /**
   * @brief 显示匹配结果（KeyPoint 格式）
   *
   * 在合并图像上绘制关键点和匹配线
   *
   * @param image0 第一张图像
   * @param image1 第二张图像
   * @param matches 匹配结果
   * @param kpts0 第一张图像的关键点
   * @param kpts1 第二张图像的关键点
   * @return const char* 错误信息，成功返回 nullptr
   */
  static const char* ShowMatches(const cv::Mat& image0,
                                 const cv::Mat& image1,
                                 const Matches& matches,
                                 const std::vector<cv::KeyPoint>& kpts0,
                                 const std::vector<cv::KeyPoint>& kpts1);

  /**
   * @brief 显示匹配结果（cv::Mat 格式）
   *
   * 在合并图像上绘制关键点和匹配线
   *
   * @param image0 第一张图像
   * @param image1 第二张图像
   * @param matches 匹配结果
   * @param kpts0 第一张图像的关键点矩阵
   * @param kpts1 第二张图像的关键点矩阵
   * @return const char* 错误信息，成功返回 nullptr
   */
  static const char* ShowMatches(const cv::Mat& image0,
                                 const cv::Mat& image1,
                                 const Matches& matches,
                                 const cv::Mat& kpts0,
                                 const cv::Mat& kpts1);

private:
  /**
   * @brief 预处理（SuperPointRet 格式）
   *
   * 将 SuperPoint 结果转换为 SuperGlue 输入格式
   *
   * @param kpts0 第一张图像的 SuperPoint 结果
   * @param kpts1 第二张图像的 SuperPoint 结果
   * @param shape0 第一张图像的尺寸
   * @param shape1 第二张图像的尺寸
   * @param superglue_intput 输出的 SuperGlue 输入
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* PreProcess(const SuperPointRet& kpts0,
                         const SuperPointRet& kpts1,
                         const cv::Size& shape0,
                         const cv::Size& shape1,
                         SuperGlueInput& superglue_intput);

  /**
   * @brief 预处理（原始指针格式）
   *
   * 将原始指针数据转换为 SuperGlue 输入格式
   *
   * @param kpts0_data 第一张图像的关键点坐标指针
   * @param kpts1_data 第二张图像的关键点坐标指针
   * @param desc0_data 第一张图像的描述子指针
   * @param desc1_data 第二张图像的描述子指针
   * @param scores0_data 第一张图像的关键点置信度指针
   * @param scores1_data 第二张图像的关键点置信度指针
   * @param nums0 第一张图像的关键点数量
   * @param nums1 第二张图像的关键点数量
   * @param shape0 第一张图像的尺寸
   * @param shape1 第二张图像的尺寸
   * @param superglue_input 输出的 SuperGlue 输入
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* PreProcess(float* kpts0_data,
                         float* kpts1_data,
                         float* desc0_data,
                         float* desc1_data,
                         float* scores0_data,
                         float* scores1_data,
                         const std::size_t& nums0,
                         const std::size_t& nums1,
                         const cv::Size& shape0,
                         const cv::Size& shape1,
                         SuperGlueInput& superglue_input);

  /**
   * @brief 后处理
   *
   * 处理推理输出，提取匹配结果和置信度
   *
   * @param indices0 匹配索引张量
   * @param mscores0 匹配置信度张量
   * @param matches 输出的匹配结果
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* PostProcess(TensorView<int64_t>& indices0,
                          TensorView<float>& mscores0,
                          Matches& matches);

  /**
   * @brief 归一化关键点坐标
   *
   * 将关键点坐标从像素空间归一化到 [0, 1] 区间
   *
   * @param kpts 关键点坐标矩阵
   * @param shape 图像尺寸
   * @return const char* 错误信息，成功返回 nullptr
   */
  const char* NormlizeKeypoints(cv::Mat& kpts, const cv::Size& shape);

  EnginePtr infer_engine_; ///< 推理引擎实例
  OptionsPtr options_;     ///< SuperGlue 配置项
};
}
