#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

const char *DrawCircles(cv::Mat &image,
                        const cv::Mat &kpts,
                        const cv::Scalar &color,
                        const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawCircles(cv::Mat &image,
                        const cv::Mat &kpts,
                        const std::vector<cv::Scalar> &colors,
                        const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawCircles(cv::Mat &image,
                        const std::vector<cv::KeyPoint> &kpts,
                        const cv::Scalar &color,
                        const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawCircles(cv::Mat &image,
                        const std::vector<cv::KeyPoint> &kpts,
                        const std::vector<cv::Scalar> &colors,
                        const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawCircles(cv::Mat &image,
                        const cv::Point2f *pts,
                        const std::size_t &num,
                        const cv::Scalar &color,
                        const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawCircles(cv::Mat &image,
                        const cv::Point2f *pts,
                        const std::size_t &num,
                        const std::vector<cv::Scalar> &colors,
                        const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawLines(cv::Mat &image,
                      const cv::Point2f *pts0,
                      const cv::Point2f *pts1,
                      const std::vector<int64_t> &matches,
                      const std::vector<cv::Scalar> &colors,
                      const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawLines(cv::Mat &image,
                      const cv::Point2f *pts0,
                      const cv::Point2f *pts1,
                      const std::vector<int64_t> &matches,
                      const cv::Scalar &color,
                      const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawLines(cv::Mat &image,
                      const std::vector<cv::KeyPoint> &kpts0,
                      const std::vector<cv::KeyPoint> &pts1,
                      const std::vector<int64_t> &matches,
                      const std::vector<cv::Scalar> &colors,
                      const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawLines(cv::Mat &image,
                      const std::vector<cv::KeyPoint> &kpts0,
                      const std::vector<cv::KeyPoint> &pts1,
                      const std::vector<int64_t> &matches,
                      const cv::Scalar &color,
                      const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawLines(cv::Mat &image,
                      const cv::Mat &kpts0,
                      const cv::Mat &kpts1,
                      const std::vector<int64_t> &matches,
                      const std::vector<cv::Scalar> &colors,
                      const cv::Point2f &offset = {0.0f, 0.0f});

const char *DrawLines(cv::Mat &image,
                      const cv::Mat &kpts0,
                      const cv::Mat &kpts1,
                      const std::vector<int64_t> &matches,
                      const cv::Scalar &color,
                      const cv::Point2f &offset = {0.0f, 0.0f});
