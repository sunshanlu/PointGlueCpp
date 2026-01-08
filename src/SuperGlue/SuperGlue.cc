#include <execution>
#include <numeric>

#include <spdlog/spdlog.h>
#include <npycpp/npycpp.hpp>

#include "pointgluecpp/utils/ViewerTools.hpp"
#include "pointgluecpp/SuperGlue/SuperGlue.h"
#include "pointgluecpp/SuperGlue/SuperGlueEngine.h"

#ifdef ONNXRUNTIMEENGINE
#include "pointgluecpp/SuperGlue/ONNXRuntimeEngine.h"
#endif

#ifdef OPENVINOENGINE
#include "pointgluecpp/SuperGlue/OpenVINOEngine.h"
#endif


#define RET_OK nullptr
#define CHECK(message)\
do {\
	if (message != RET_OK){\
		SPDLOG_ERROR("Error: {}, at {}:{}\n", message, __FILE__, __LINE__);\
		exit(EXIT_FAILURE);\
	}\
} while (0)

namespace sg
{
SuperGlue::SuperGlue(OptionsPtr options)
  : options_(std::move(options))
{
#if defined(ONNXRUNTIMEENGINE)
  infer_engine_ = ONNXRuntimeEngine::CreateInstance(options_->model_path,
                                                    options_->input_names,
                                                    options_->output_names,
                                                    options_->desc_dim,
                                                    options_->keypoint_dim);
#elif defined(OPENVINOENGINE)
  infer_engine_ = OpenVINOEngine::CreateInstance(options_->model_path,
                                                 options_->input_names,
                                                 options_->output_names,
                                                 options_->desc_dim,
                                                 options_->keypoint_dim);
#else
  static_assert(false, " No inference engine defined. Please define either ONNXRUNTIMEENGINE or OPENVINOENGINE");
#endif
}

SuperGlue::Matches
SuperGlue::RunSession(const SuperPointRet& kpts0,
                      const SuperPointRet& kpts1,
                      cv::Size shape0, cv::Size shape1)
{
  SuperGlueInput superglue_input;
  Matches matches;
  CHECK(PreProcess(kpts0, kpts1, shape0, shape1, superglue_input));

  auto [indices0, mscores0] = infer_engine_->RunInference(superglue_input);

  CHECK(PostProcess(indices0, mscores0, matches));
  return matches;
}

SuperGlue::Matches
SuperGlue::RunSession(float* kpts0,
                      float* kpts1,
                      float* desc0,
                      float* desc1,
                      float* scores0,
                      float* scores1,
                      const std::size_t& nums0,
                      const std::size_t& nums1,
                      const cv::Size& shape0,
                      const cv::Size& shape1)
{
  SuperGlueInput superglue_input;
  Matches matches;

#if defined(DEBUG) && defined(DEBUG_OUTPUT_DIR)
  // before preprocess
  namespace fs = std::filesystem;
  {
    npy::NpzWriter npz_writer(fs::path(DEBUG_OUTPUT_DIR) / "before_preprocess.npz");
    npz_writer.AddNpyData("kpts0", kpts0, {static_cast<int64_t>(nums0), 2});
    npz_writer.AddNpyData("kpts1", kpts1, {static_cast<int64_t>(nums1), 2});
    npz_writer.AddNpyData("desc0", desc0, {256, static_cast<int64_t>(nums0)});
    npz_writer.AddNpyData("desc1", desc1, {256, static_cast<int64_t>(nums1)});
    npz_writer.AddNpyData("scores0", scores0, {static_cast<int64_t>(nums0)});
    npz_writer.AddNpyData("scores1", scores1, {static_cast<int64_t>(nums1)});
  }
#endif

  CHECK(PreProcess(kpts0, kpts1, desc0, desc1, scores0,
    scores1, nums0, nums1, shape0, shape1, superglue_input));

#if defined(DEBUG) && defined(DEBUG_OUTPUT_DIR)
  // end process
  // before preprocess
  namespace fs = std::filesystem;
  {
    npy::NpzWriter npz_writer(fs::path(DEBUG_OUTPUT_DIR) / "end_preprocess.npz");
    npz_writer.AddNpyData("kpts0", superglue_input.kpts0.ptr<float>(), {static_cast<int64_t>(nums0), 2});
    npz_writer.AddNpyData("kpts1", superglue_input.kpts1.ptr<float>(), {static_cast<int64_t>(nums1), 2});
    npz_writer.AddNpyData("desc0", superglue_input.desc0.ptr<float>(), {256, static_cast<int64_t>(nums0)});
    npz_writer.AddNpyData("desc1", superglue_input.desc1.ptr<float>(), {256, static_cast<int64_t>(nums1)});
    npz_writer.AddNpyData("scores0", superglue_input.scores0.data(), {static_cast<int64_t>(nums0)});
    npz_writer.AddNpyData("scores1", superglue_input.scores1.data(), {static_cast<int64_t>(nums1)});
  }
#endif

  auto [indices0, mscores0] = infer_engine_->RunInference(superglue_input);

#if defined(DEBUG) && defined(DEBUG_OUTPUT_DIR)
  // end process
  // before preprocess
  namespace fs = std::filesystem;
  {
    npy::NpzWriter npz_writer(fs::path(DEBUG_OUTPUT_DIR) / "end_inference.npz");
    npz_writer.AddNpyData("indices0", indices0.data(), {static_cast<int64_t>(nums0)});
    npz_writer.AddNpyData("mscores0", mscores0.data(), {static_cast<int64_t>(nums0)});
  }
#endif

  CHECK(PostProcess(indices0, mscores0, matches));
  return matches;
}

const char* SuperGlue::PreProcess(const SuperPointRet& kpts0,
                                  const SuperPointRet& kpts1,
                                  const cv::Size& shape0,
                                  const cv::Size& shape1,
                                  SuperGlueInput& superglue_intput)
{
  superglue_intput = SuperGlueInput(kpts0.keypoints.size(), kpts1.keypoints.size());
  std::vector<int> indices0(kpts0.keypoints.size());
  std::vector<int> indices1(kpts1.keypoints.size());
  std::iota(indices0.begin(), indices0.end(), 0);
  std::iota(indices1.begin(), indices1.end(), 0);

  cv::Mat desc0(kpts0.keypoints.size(), 256, CV_32F);
  cv::Mat desc1(kpts1.keypoints.size(), 256, CV_32F);

  std::for_each(std::execution::par_unseq, indices0.begin(), indices0.end(), [&](const int& idx)
  {
    superglue_intput.kpts0.at<float>(idx, 0) = kpts0.keypoints[idx].pt.x;
    superglue_intput.kpts0.at<float>(idx, 1) = kpts0.keypoints[idx].pt.y;
    superglue_intput.scores0[idx] = kpts0.keypoints[idx].response;
    memcpy(desc0.row(idx).ptr<float>(), kpts0.keydesc[idx].data(), 256 * sizeof(float));
  });

  std::for_each(std::execution::par_unseq, indices1.begin(), indices1.end(), [&](const int& idx)
  {
    superglue_intput.kpts1.at<float>(idx, 0) = kpts1.keypoints[idx].pt.x;
    superglue_intput.kpts1.at<float>(idx, 1) = kpts1.keypoints[idx].pt.y;
    superglue_intput.scores1[idx] = kpts1.keypoints[idx].response;
    memcpy(desc1.row(idx).ptr<float>(), kpts1.keydesc[idx].data(), 256 * sizeof(float));
  });
  superglue_intput.desc0 = desc0.t();
  superglue_intput.desc1 = desc1.t();

  CHECK(NormlizeKeypoints(superglue_intput.kpts0, shape0));
  CHECK(NormlizeKeypoints(superglue_intput.kpts1, shape1));

  return RET_OK;
}

const char* SuperGlue::PreProcess(float* kpts0_data,
                                  float* kpts1_data,
                                  float* desc0_data,
                                  float* desc1_data,
                                  float* scores0_data,
                                  float* scores1_data,
                                  const std::size_t& nums0,
                                  const std::size_t& nums1,
                                  const cv::Size& shape0,
                                  const cv::Size& shape1,
                                  SuperGlueInput& superglue_input)
{
  superglue_input = SuperGlueInput(nums0, nums1);
  superglue_input.desc0 = cv::Mat(256, nums0, CV_32F, desc0_data);
  superglue_input.desc1 = cv::Mat(256, nums1, CV_32F, desc1_data);
  cv::Mat(nums0, 2, CV_32F, kpts0_data).copyTo(superglue_input.kpts0);
  cv::Mat(nums1, 2, CV_32F, kpts1_data).copyTo(superglue_input.kpts1);
  superglue_input.scores0 = std::vector(scores0_data, scores0_data + nums0);
  superglue_input.scores1 = std::vector(scores1_data, scores1_data + nums1);

  CHECK(NormlizeKeypoints(superglue_input.kpts0, shape0));
  CHECK(NormlizeKeypoints(superglue_input.kpts1, shape1));

  return RET_OK;
}

const char* SuperGlue::PostProcess(TensorView<int64_t>& indices0,
                                   TensorView<float>& mscores0,
                                   Matches& matches)
{
  matches.clear();
  matches.matches = std::vector(indices0.data(), indices0.data() + indices0.size_);
  matches.confidence = std::vector(mscores0.data(), mscores0.data() + mscores0.size_);

  for (int idx = 0; idx < indices0.size_; ++idx)
  {
    if (indices0.data()[idx] != -1)
      matches.count++;
  }
  return RET_OK;
}

const char* SuperGlue::NormlizeKeypoints(cv::Mat& kpts, const cv::Size& shape)
{
  const float& scaline = std::max(shape.width, shape.height) * 0.7f;
  cv::Mat center(1, 2, CV_32F), repeated_center;
  center.at<float>(0, 0) = shape.width / 2.0f;
  center.at<float>(0, 1) = shape.height / 2.0f;
  cv::repeat(center, kpts.rows, 1, repeated_center);

  kpts -= repeated_center;
  kpts /= scaline;

  return RET_OK;
}

const char* SuperGlue::MergeTwoImages(const cv::Mat& image0, const cv::Mat& image1, cv::Mat& merged_image)
{
  // 检查输入图像是否为空
  if (image0.empty() || image1.empty())
    return "Input images are empty";

  // 创建合并后的图像画布
  const int total_width = image0.cols + image1.cols;
  const int total_height = std::max(image0.rows, image1.rows);
  merged_image = cv::Mat(total_height, total_width, CV_8UC3, cv::Scalar(0, 0, 0));

  // 复制图像到合并画布
  cv::Mat roi0 = merged_image(cv::Rect(0, 0, image0.cols, image0.rows));
  if (image0.channels() == 1)
    cv::cvtColor(image0, roi0, cv::COLOR_GRAY2BGR);
  else
    image0.copyTo(roi0);

  cv::Mat roi1 = merged_image(cv::Rect(image0.cols, 0, image1.cols, image1.rows));
  if (image1.channels() == 1)
    cv::cvtColor(image1, roi1, cv::COLOR_GRAY2BGR);
  else
    image1.copyTo(roi1);

  return RET_OK;
}

const char* SuperGlue::MappingColors(const std::vector<float>& conf, std::vector<cv::Scalar>& colors)
{
  // 找到置信度的最大最小值，用于颜色映射
  float min_conf = 1.0f, max_conf = 0.0f;
  if (!conf.empty())
  {
    min_conf = *std::ranges::min_element(conf);
    max_conf = *std::ranges::max_element(conf);
  }

  std::vector<size_t> indices(conf.size());
  colors = std::vector<cv::Scalar>(conf.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                [&](const auto& idx)
                {
                  cv::Scalar& color = colors[idx];
                  if (min_conf != max_conf)
                  {
                    const float norm_conf = (conf[idx] - min_conf) / (max_conf - min_conf);

                    // 使用色谱映射：红色(低置信度) -> 绿色(高置信度)
                    const int red = static_cast<int>((1.0f - norm_conf) * 255);
                    const int green = static_cast<int>(norm_conf * 255);
                    color = cv::Scalar(0, green, red);
                  }
                  else
                  {
                    // 无法使用confidence时，是蓝色
                    color = cv::Scalar(255, 0, 0);
                  }
                });

  return RET_OK;
}

const char* SuperGlue::AddTxtMessage(cv::Mat& merged_image, const std::size_t& num0, const std::size_t& num1,
                                     const int& count)
{
  // 添加显示文字
  const std::vector small_text = {
    fmt::format("SuperGlue Matches"),
    fmt::format("Keypoints: {}:{}", num0, num1),
    fmt::format("Matches: {}", count)
  };

  const float sc = std::min(merged_image.rows, merged_image.cols) / 1000.f;
  const int Ht = static_cast<int>(sc * 40);
  const int top_margin = static_cast<int>(sc * 10);

  const cv::Scalar txt_color_fg(255, 255, 255);
  const cv::Scalar txt_color_bg(0, 0, 0);

  for (size_t i = 0; i < small_text.size(); ++i)
  {
    const auto& txt = small_text[i];
    const int ypos = static_cast<int>(top_margin + (i + 1) * Ht);

    cv::putText(merged_image, txt, cv::Point(static_cast<int>(8 * sc), ypos),
                cv::FONT_HERSHEY_DUPLEX, 1.2 * sc, txt_color_bg, 2, cv::LINE_AA);
    cv::putText(merged_image, txt, cv::Point(static_cast<int>(8 * sc), ypos),
                cv::FONT_HERSHEY_DUPLEX, 1.2 * sc, txt_color_fg, 1, cv::LINE_AA);
  }

  return RET_OK;
}

const char* SuperGlue::ShowMatches(const cv::Mat& image0,
                                   const cv::Mat& image1,
                                   const Matches& matches,
                                   const std::vector<cv::KeyPoint>& kpts0,
                                   const std::vector<cv::KeyPoint>& kpts1)
{
  cv::Mat merged_image;
  CHECK(MergeTwoImages(image0, image1, merged_image));

  // 绘制两幅图像上的特征点
  CHECK(DrawCircles(merged_image, kpts0, cv::Scalar(0, 255, 0)));
  CHECK(DrawCircles(merged_image, kpts1, cv::Scalar(255, 0, 0), {static_cast<float>(image0.cols), 0.f}));

  std::vector<cv::Scalar> colors(matches.matches.size());
  CHECK(MappingColors(matches.confidence, colors));
  CHECK(DrawLines(merged_image, kpts0, kpts1, matches.matches, colors, {static_cast<float>(image0.cols), 0.f}));
  CHECK(AddTxtMessage(merged_image, kpts0.size(), kpts1.size(), matches.count));

  // 显示图像
  cv::imshow("SuperGlue Matches", merged_image);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return RET_OK;
}

const char* SuperGlue::ShowMatches(const cv::Mat& image0,
                                   const cv::Mat& image1,
                                   const Matches& matches,
                                   const cv::Mat& kpts0,
                                   const cv::Mat& kpts1)
{
  cv::Mat merged_image;
  CHECK(MergeTwoImages(image0, image1, merged_image));

  // 绘制两幅图像上的特征点
  CHECK(DrawCircles(merged_image, kpts0, cv::Scalar(0, 255, 0)));
  CHECK(DrawCircles(merged_image, kpts1, cv::Scalar(255, 0, 0), {static_cast<float>(image0.cols), 0.f}));

  cv::imshow("SuperGlue Matches", merged_image);
  cv::waitKey(0);
  cv::destroyAllWindows();

  std::vector<cv::Scalar> colors(matches.matches.size());
  CHECK(MappingColors(matches.confidence, colors));
  CHECK(DrawLines(merged_image, kpts0, kpts1, matches.matches, colors, {static_cast<float>(image0.cols), 0.f}));
  CHECK(AddTxtMessage(merged_image, kpts0.rows, kpts1.rows, matches.count));

  // 显示图像
  cv::imshow("SuperGlue Matches", merged_image);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return RET_OK;
}
}

#undef RET_OK
#undef CHECK
