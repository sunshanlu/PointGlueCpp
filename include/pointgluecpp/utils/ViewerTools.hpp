#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

/**
 * @file ViewerTools.hpp
 * @brief OpenCV 可视化工具函数集合
 *
 * 提供关键点、匹配线等的可视化绘制功能
 */

/**
 * @brief 在图像上绘制圆圈（单色，cv::Mat 格式关键点）
 *
 * @param image 目标图像
 * @param kpts 关键点坐标矩阵 [N x 2]
 * @param color 绘制颜色
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawCircles(cv::Mat& image,
                        const cv::Mat& kpts,
                        const cv::Scalar& color,
                        const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制圆圈（多色，cv::Mat 格式关键点）
 *
 * @param image 目标图像
 * @param kpts 关键点坐标矩阵 [N x 2]
 * @param colors 每个关键点的颜色列表
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawCircles(cv::Mat& image,
                        const cv::Mat& kpts,
                        const std::vector<cv::Scalar>& colors,
                        const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制圆圈（单色，KeyPoint 向量格式）
 *
 * @param image 目标图像
 * @param kpts 关键点向量
 * @param color 绘制颜色
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawCircles(cv::Mat& image,
                        const std::vector<cv::KeyPoint>& kpts,
                        const cv::Scalar& color,
                        const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制圆圈（多色，KeyPoint 向量格式）
 *
 * @param image 目标图像
 * @param kpts 关键点向量
 * @param colors 每个关键点的颜色列表
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawCircles(cv::Mat& image,
                        const std::vector<cv::KeyPoint>& kpts,
                        const std::vector<cv::Scalar>& colors,
                        const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制圆圈（单色，原始指针格式）
 *
 * @param image 目标图像
 * @param pts 关键点指针数组
 * @param num 关键点数量
 * @param color 绘制颜色
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawCircles(cv::Mat& image,
                        const cv::Point2f* pts,
                        const std::size_t& num,
                        const cv::Scalar& color,
                        const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制圆圈（多色，原始指针格式）
 *
 * @param image 目标图像
 * @param pts 关键点指针数组
 * @param num 关键点数量
 * @param colors 每个关键点的颜色列表
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawCircles(cv::Mat& image,
                        const cv::Point2f* pts,
                        const std::size_t& num,
                        const std::vector<cv::Scalar>& colors,
                        const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制匹配线（多色，原始指针格式）
 *
 * @param image 目标图像
 * @param pts0 第一组关键点指针数组
 * @param pts1 第二组关键点指针数组
 * @param matches 匹配索引数组（matches[i] = j 表示 pts0[i] 匹配 pts1[j]，-1 表示无匹配）
 * @param colors 每条匹配线的颜色列表
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawLines(cv::Mat& image,
                      const cv::Point2f* pts0,
                      const cv::Point2f* pts1,
                      const std::vector<int64_t>& matches,
                      const std::vector<cv::Scalar>& colors,
                      const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制匹配线（单色，原始指针格式）
 *
 * @param image 目标图像
 * @param pts0 第一组关键点指针数组
 * @param pts1 第二组关键点指针数组
 * @param matches 匹配索引数组
 * @param color 绘制颜色
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawLines(cv::Mat& image,
                      const cv::Point2f* pts0,
                      const cv::Point2f* pts1,
                      const std::vector<int64_t>& matches,
                      const cv::Scalar& color,
                      const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制匹配线（多色，KeyPoint 向量格式）
 *
 * @param image 目标图像
 * @param kpts0 第一组关键点向量
 * @param pts1 第二组关键点向量
 * @param matches 匹配索引数组
 * @param colors 每条匹配线的颜色列表
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawLines(cv::Mat& image,
                      const std::vector<cv::KeyPoint>& kpts0,
                      const std::vector<cv::KeyPoint>& pts1,
                      const std::vector<int64_t>& matches,
                      const std::vector<cv::Scalar>& colors,
                      const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制匹配线（单色，KeyPoint 向量格式）
 *
 * @param image 目标图像
 * @param kpts0 第一组关键点向量
 * @param pts1 第二组关键点向量
 * @param matches 匹配索引数组
 * @param color 绘制颜色
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawLines(cv::Mat& image,
                      const std::vector<cv::KeyPoint>& kpts0,
                      const std::vector<cv::KeyPoint>& pts1,
                      const std::vector<int64_t>& matches,
                      const cv::Scalar& color,
                      const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制匹配线（多色，cv::Mat 格式）
 *
 * @param image 目标图像
 * @param kpts0 第一组关键点矩阵 [N x 2]
 * @param kpts1 第二组关键点矩阵 [M x 2]
 * @param matches 匹配索引数组
 * @param colors 每条匹配线的颜色列表
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawLines(cv::Mat& image,
                      const cv::Mat& kpts0,
                      const cv::Mat& kpts1,
                      const std::vector<int64_t>& matches,
                      const std::vector<cv::Scalar>& colors,
                      const cv::Point2f& offset = {0.0f, 0.0f});

/**
 * @brief 在图像上绘制匹配线（单色，cv::Mat 格式）
 *
 * @param image 目标图像
 * @param kpts0 第一组关键点矩阵 [N x 2]
 * @param kpts1 第二组关键点矩阵 [M x 2]
 * @param matches 匹配索引数组
 * @param color 绘制颜色
 * @param offset 位置偏移量，默认为 (0, 0)
 * @return const char* 错误信息，成功返回 nullptr
 */
const char* DrawLines(cv::Mat& image,
                      const cv::Mat& kpts0,
                      const cv::Mat& kpts1,
                      const std::vector<int64_t>& matches,
                      const cv::Scalar& color,
                      const cv::Point2f& offset = {0.0f, 0.0f});
