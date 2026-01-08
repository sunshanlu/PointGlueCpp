示例代码
========

本章节提供了 PointGlueCpp 的使用示例。

SuperPoint 示例
----------------

基础用法
^^^^^^^^

.. code-block:: cpp

   #include <pointgluecpp/SuperPoint/SuperPoint.h>
   #include <opencv2/opencv.hpp>

   int main() {
       // 1. 从 YAML 文件加载配置
       auto options = sp::SuperPoint::Options::CreateFromYaml(
           "res/options/SuperPointOptions.yaml"
       );

       // 2. 创建 SuperPoint 对象
       sp::SuperPoint superpoint(std::move(options));

       // 3. 读取图像
       cv::Mat image = cv::imread("test.jpg", cv::IMREAD_GRAYSCALE);

       // 4. 运行推理
       auto result = superpoint.RunSession(image);

       // 5. 访问结果
       std::cout << "Detected " << result.keypoints.size() << " keypoints\n";

       // 6. 可视化
       cv::Mat output;
       cv::drawKeypoints(image, result.keypoints, output);
       cv::imwrite("keypoints.jpg", output);

       return 0;
   }

批量处理图像
^^^^^^^^^^^^

.. code-block:: cpp

   #include <pointgluecpp/SuperPoint/SuperPoint.h>
   #include <opencv2/opencv.hpp>
   #include <vector>
   #include <filesystem>

   namespace fs = std::filesystem;

   int main() {
       auto options = sp::SuperPoint::Options::CreateFromYaml("config.yaml");
       sp::SuperPoint superpoint(std::move(options));

       std::string image_dir = "images/";
       std::string output_dir = "output/";

       // 遍历图像
       for (const auto& entry : fs::directory_iterator(image_dir)) {
           if (entry.path().extension() == ".jpg" ||
               entry.path().extension() == ".png") {

               cv::Mat image = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
               auto result = superpoint.RunSession(image);

               // 保存结果
               cv::Mat output;
               cv::drawKeypoints(image, result.keypoints, output);

               std::string out_path = output_dir + entry.path().filename().string();
               cv::imwrite(out_path, output);

               std::cout << "Processed: " << entry.path() << "\n";
           }
       }

       return 0;
   }

SuperGlue 示例
---------------

匹配两幅图像
^^^^^^^^^^^^

.. code-block:: cpp

   #include <pointgluecpp/SuperPoint/SuperPoint.h>
   #include <pointgluecpp/SuperGlue/SuperGlue.h>
   #include <opencv2/opencv.hpp>

   int main() {
       // 1. 创建 SuperPoint
       auto sp_options = sp::SuperPoint::Options::CreateFromYaml(
           "res/options/SuperPointOptions.yaml"
       );
       sp::SuperPoint superpoint(std::move(sp_options));

       // 2. 创建 SuperGlue
       auto sg_options = sg::SuperGlue::Options::CreateFromYaml(
           "res/options/SuperGlueOptions.yaml"
       );
       sg::SuperGlue superglue(std::move(sg_options));

       // 3. 读取两幅图像
       cv::Mat image0 = cv::imread("image0.jpg", cv::IMREAD_GRAYSCALE);
       cv::Mat image1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);

       // 4. 检测特征点
       auto result0 = superpoint.RunSession(image0);
       auto result1 = superpoint.RunSession(image1);

       // 5. 匹配特征点
       auto matches = superglue.RunSession(
           result0, result1,
           image0.size(), image1.size()
       );

       // 6. 显示结果
       std::cout << "Found " << matches.count << " matches\n";

       // 7. 可视化匹配
       cv::Mat merged_image;
       sg::SuperGlue::ShowMatches(
           image0, image1, matches,
           result0.keypoints, result1.keypoints
       );

       return 0;
   }

视频匹配
^^^^^^^^

.. code-block:: cpp

   #include <pointgluecpp/SuperPoint/SuperPoint.h>
   #include <pointgluecpp/SuperGlue/SuperGlue.h>
   #include <opencv2/opencv.hpp>

   int main() {
       auto sp_options = sp::SuperPoint::Options::CreateFromYaml("config.yaml");
       sp::SuperPoint superpoint(std::move(sp_options));

       auto sg_options = sg::SuperGlue::Options::CreateFromYaml("superglue_config.yaml");
       sg::SuperGlue superglue(std::move(sg_options));

       cv::VideoCapture cap("video.mp4");
       cv::Mat prev_frame, prev_result;

       bool first_frame = true;

       while (true) {
           cv::Mat frame;
           cap >> frame;
           if (frame.empty()) break;

           cv::Mat gray;
           cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

           // 检测特征点
           auto result = superpoint.RunSession(gray);

           if (!first_frame) {
               // 匹配相邻帧
               auto matches = superglue.RunSession(
                   prev_result, result,
                   prev_frame.size(), gray.size()
               );

               std::cout << "Matches: " << matches.count << "\n";

               // 可视化
               cv::Mat vis;
               sg::SuperGlue::ShowMatches(
                   prev_frame, frame, matches,
                   prev_result.keypoints, result.keypoints
               );
               cv::imshow("Matches", vis);

               if (cv::waitKey(1) == 27) break;  // ESC to exit
           }

           prev_frame = gray;
           prev_result = result;
           first_frame = false;
       }

       return 0;
   }

高级用法
--------

自定义配置
^^^^^^^^^^

.. code-block:: cpp

   #include <pointgluecpp/SuperPoint/SuperPoint.h>

   int main() {
       // 从 YAML 加载
       auto options = sp::SuperPoint::Options::CreateFromYaml("config.yaml");

       // 或手动配置
       auto options = std::make_unique<sp::SuperPoint::Options>();
       options->model_path = "models/superpoint.onnx";
       options->width = 640;
       options->height = 480;
       options->border = 4;
       options->conf_threshold = 0.015f;
       options->nms_dist = 4;

       sp::SuperPoint superpoint(std::move(options));
       return 0;
   }

使用原始数据
^^^^^^^^^^^^

.. code-block:: cpp

   #include <pointgluecpp/SuperGlue/SuperGlue.h>

   int main() {
       auto options = sg::SuperGlue::Options::CreateFromYaml("config.yaml");
       sg::SuperGlue superglue(std::move(options));

       // 准备数据
       const size_t n_kpts0 = 100;
       const size_t n_kpts1 = 150;

       std::vector<float> kpts0(n_kpts0 * 2, 0.5f);  // [N x 2]
       std::vector<float> kpts1(n_kpts1 * 2, 0.5f);  // [M x 2]
       std::vector<float> desc0(256 * n_kpts0, 0.1f);  // [256 x N]
       std::vector<float> desc1(256 * n_kpts1, 0.1f);  // [256 x M]
       std::vector<float> scores0(n_kpts0, 0.9f);
       std::vector<float> scores1(n_kpts1, 0.8f);

       // 运行推理
       auto matches = superglue.RunSession(
           kpts0.data(), kpts1.data(),
           desc0.data(), desc1.data(),
           scores0.data(), scores1.data(),
           n_kpts0, n_kpts1,
           cv::Size(640, 480), cv::Size(640, 480)
       );

       std::cout << "Matches: " << matches.count << "\n";
       return 0;
   }

错误处理
^^^^^^^^

.. code-block:: cpp

   #include <pointgluecpp/SuperPoint/SuperPoint.h>
   #include <iostream>

   int main() {
       try {
           auto options = sp::SuperPoint::Options::CreateFromYaml("config.yaml");
           sp::SuperPoint superpoint(std::move(options));

           cv::Mat image = cv::imread("test.jpg", cv::IMREAD_GRAYSCALE);

           if (image.empty()) {
               throw std::runtime_error("Failed to load image");
           }

           auto result = superpoint.RunSession(image);

       } catch (const std::exception& e) {
           std::cerr << "Error: " << e.what() << std::endl;
           return 1;
       }

       return 0;
   }

性能优化
--------

多线程处理
^^^^^^^^^^

.. code-block:: cpp

   #include <thread>
   #include <vector>
   #include <pointgluecpp/SuperPoint/SuperPoint.h>

   void process_image(const std::string& path) {
       static auto options = sp::SuperPoint::Options::CreateFromYaml("config.yaml");
       static sp::SuperPoint superpoint(std::move(options));

       cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
       auto result = superpoint.RunSession(image);
   }

   int main() {
       std::vector<std::string> images = {"1.jpg", "2.jpg", "3.jpg", "4.jpg"};
       std::vector<std::thread> threads;

       for (const auto& img : images) {
           threads.emplace_back(process_image, img);
       }

       for (auto& t : threads) {
           t.join();
       }

       return 0;
   }

GPU 推理
^^^^^^^^^

SuperPoint 和 SuperGlue 会自动使用可用的 GPU：

.. code-block:: cpp

   // 确保安装了 CUDA 版本的 ONNX Runtime
   // 程序会自动检测并使用 GPU

   auto options = sp::SuperPoint::Options::CreateFromYaml("config.yaml");
   sp::SuperPoint superpoint(std::move(options));

   // 推理会自动在 GPU 上运行
   auto result = superpoint.RunSession(image);