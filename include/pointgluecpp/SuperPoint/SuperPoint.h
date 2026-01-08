#pragma once

#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

#include "pointgluecpp/SuperPoint/SuperPointEngine.h"

namespace sp
{
class SuperPoint
{
public:
  struct SuperPointRet;
  struct Options;
  using OptionsPtr = std::unique_ptr<Options>;

  /// SuperPoint的输出结果
  struct SuperPointRet
  {
    std::vector<cv::KeyPoint> keypoints;
    std::vector<std::array<float, 256>> keydesc;
  };

  /// SuperPoint配置项
  struct Options
  {
    friend class SuperPoint;

    std::string model_path;                    //< 输入的模型路径
    short width;                               //< 模型input的宽
    short height;                              //< 模型input的高
    short border;                              //< 后处理要求的边界
    float conf_threshold;                      //< 关键点置信度阈值
    short nms_dist;                            //< fast-nms的搜索半径
    std::vector<std::string> input_names_str;  //< 输入节点名称（字符串存储）
    std::vector<std::string> output_names_str; //< 输出节点名称（字符串存储）
    std::vector<const char*> input_names;      //< 输入节点名称（指针，用于引擎接口）
    std::vector<const char*> output_names;     //< 输出节点名称（指针，用于引擎接口）
    std::vector<int64_t> input_dims;           //< 输入节点维度
    std::vector<int64_t> output_dims0;         //< 输出节点维度0
    std::vector<int64_t> output_dims1;         //< 输出节点维度1

    /**
     * 根据输入的 yaml 文件创建配置项
     *
     * @param yaml_path 输入的yaml文件路径
     * @return 输出的配置项指针
     */
    static OptionsPtr CreateFromYaml(const std::string& yaml_path);

  private:
    /**
     * 更新指针向量：从字符串向量生成 const char* 指针向量
     * 用于引擎接口
     */
    void UpdatePointers();
  };

  /**
   * 创建 onnxruntime 的推理session
   *
   * 1. 创建onnxruntime的session @see SuperPoint::CreateSession
   * 2. 维护input nodes 和 ouput nodes属性
   * 3. 热处理底层软件的初始化 @see SuperPoint::WarmSession
   *
   * @param options 输入配置项
   */
  explicit SuperPoint(OptionsPtr options);

  /**
   * 推理SuperPoint
   *
   * 1. 前处理 @see SuperPoint::PreProcess
   * 2. 使用指定 engine 进行模型推理 cuda
   * 3. 后处理 @see SuperPoint::PostProcess
   *
   * @param input_image 输入的图片
   * @return SuperPointRet 输出推理结果
   */
  SuperPointRet RunSession(const cv::Mat& input_image);

private:
  /**
   * 前处理操作
   *
   * 1. 缩放图像并convert float至 onnxruntime 模型输入要求
   * 2. 创建 input_tersor
   *
   * @param input_image		输入的图像
   * @param xscale				输出的前处理图像x轴缩放比例
   * @param yscale				输出的前处理图像y轴缩放比例
   * @param engine_image		为了保证input_tensor的数据有效性，输出的onnx_image
   * @return error message
   */
  const char* PreProcess(const cv::Mat& input_image, float& xscale,
                         float& yscale, cv::Mat& engine_image);

  /**
   * 后处理操作
   *
   * 1. output0的softmax操作，并丢弃掉最后一列
   * 2. 将所有confidence中大于conf_threshold的点保存下来
   * 3. 使用Fast-NMS操作，获取选择的点
   * 4. 使用插值的方式获取指定特征点的desc
   *
   * @param output_data	onnxruntime 推理输出张量
   * @param xscale					x 轴缩放尺度
   * @param yscale					y 轴缩放尺度
   * @param result					输出的后处理结果 @see SuperPointRet
   * @return error message
   */
  const char* PostProcess(std::array<TensorView<float>, 2>& output_data,
                          const float& xscale, const float& yscale, SuperPointRet& result);


  std::vector<cv::KeyPoint>
  FastNMS(std::vector<cv::Point>& locations, const cv::Mat& confidence);

  /**
   * 双线性差值工具函数，要求有指定的输入数据
   *
   * @param data	<c, h, w>存储顺序的 float 数组
   * @param x			要插值的 x 轴坐标
   * @param y			要插值的 y 轴坐标
   * @return std::array<float, 256> 插值后的结果
   */
  std::array<float, 256>
  bilinearInterpolate(const float* data, const float& x, const float& y);

#ifdef DEBUG
  /**
   * 使用iostream保存张量数据到文件，debug 工具函数
   *
   * @param tensor_data 要保存的张量数据
   * @param size				要保存的张量尺寸大小
   * @param file_path		要保存的文件路径
   * @return error message
   */
  const char* SaveDataToFile(const float* tensor_data, size_t size,
                             const std::string& file_path);
#endif

  OptionsPtr options_;                    //< 输入配置项
  SuperPointEngine::InferencePtr engine_; //< 推理引擎
};
}
