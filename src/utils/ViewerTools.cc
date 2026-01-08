#include <execution>

#include "pointgluecpp/utils/ViewerTools.hpp"

#define RET_OK nullptr;

const char *DrawCircles(cv::Mat &image,
                        const cv::Point2f *pts,
                        const std::size_t &num,
                        const cv::Scalar &color,
                        const cv::Point2f &offset) {
	if (image.empty())
		return "image is empty";

	if (!num)
		return RET_OK;

	std::vector<std::size_t> indices(num);
	std::iota(indices.begin(), indices.end(), 0);

	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](const std::size_t &idx) {
		const cv::Point2f &pt = pts[idx] + offset;
		cv::circle(image, pt, 1, color, -1);
	});

	return RET_OK;
}

const char *DrawCircles(cv::Mat &image,
                        const cv::Point2f *pts,
                        const std::size_t &num,
                        const std::vector<cv::Scalar> &colors,
                        const cv::Point2f &offset) {
	if (image.empty())
		return "image is empty";

	if (num == 0)
		return RET_OK;

	std::vector<std::size_t> indices(num);
	std::iota(indices.begin(), indices.end(), 0);
	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
	              [&](const std::size_t &idx) {
		              const cv::Point2f &pt = pts[idx] + offset;
		              cv::circle(image, pt, 1, colors[idx], -1);
	              });

	return RET_OK;
}


const char *DrawCircles(cv::Mat &image,
                        const cv::Mat &kpts,
                        const cv::Scalar &color,
                        const cv::Point2f &offset) {
	const int num = kpts.rows;
	const cv::Point2f *pts = kpts.ptr<cv::Point2f>();
	return DrawCircles(image, pts, num, color, offset);
}

const char *DrawCircles(cv::Mat &image,
                        const cv::Mat &kpts,
                        const std::vector<cv::Scalar> &colors,
                        const cv::Point2f &offset) {
	const int num = kpts.rows;
	const cv::Point2f *pts = kpts.ptr<cv::Point2f>();
	return DrawCircles(image, pts, num, colors, offset);
}

const char *DrawCircles(cv::Mat &image,
                        const std::vector<cv::KeyPoint> &kpts,
                        const cv::Scalar &color,
                        const cv::Point2f &offset) {
	if (image.empty())
		return "image is empty";

	if (kpts.empty())
		return RET_OK;

	std::for_each(std::execution::par_unseq, kpts.begin(), kpts.end(),
	              [&](const cv::KeyPoint &kpt) {
		              cv::circle(image, kpt.pt + offset, 1, color, -1);
	              }
	);

	return RET_OK;
}

const char *DrawCircles(cv::Mat &image,
                        const std::vector<cv::KeyPoint> &kpts,
                        const std::vector<cv::Scalar> &colors,
                        const cv::Point2f &offset) {
	if (image.empty())
		return "image is empty";

	if (kpts.empty())
		return RET_OK;

	std::vector<int> indices(kpts.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
	              [&](const int &idx) {
		              cv::circle(image, kpts[idx].pt + offset, 1, colors[idx], -1);
	              }
	);
	return RET_OK;
}

const char *DrawLines(cv::Mat &image,
                      const cv::Point2f *pts0,
                      const cv::Point2f *pts1,
                      const std::vector<int64_t> &matches,
                      const std::vector<cv::Scalar> &colors,
                      const cv::Point2f &offset) {
	if (image.empty())
		return "image is empty";

	if (matches.empty())
		return RET_OK;

	std::vector<std::size_t> indices(matches.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
	              [&](const std::size_t &idx) {
		              if (matches[idx] == -1)
			              return;

		              const cv::Point2f &pt0 = pts0[idx];
		              const cv::Point2f &pt1 = pts1[matches[idx]];
		              cv::line(image, pt0, pt1 + offset, colors[idx], 1, cv::LINE_AA);
	              });

	return RET_OK;
}

const char *DrawLines(cv::Mat &image,
                      const cv::Point2f *pts0,
                      const cv::Point2f *pts1,
                      const std::vector<int64_t> &matches,
                      const cv::Scalar &color,
                      const cv::Point2f &offset) {
	if (image.empty())
		return "image is empty";

	if (matches.empty())
		return RET_OK;

	std::vector<std::size_t> indices(matches.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
	              [&](const std::size_t &idx) {
		              if (matches[idx] == -1)
			              return;

		              const cv::Point2f &pt0 = pts0[idx];
		              const cv::Point2f &pt1 = pts1[matches[idx]];
		              cv::line(image, pt0, pt1 + offset, color, 1, cv::LINE_AA);
	              });

	return RET_OK;
}

const char *DrawLines(cv::Mat &image,
                      const std::vector<cv::KeyPoint> &kpts0,
                      const std::vector<cv::KeyPoint> &pts1,
                      const std::vector<int64_t> &matches,
                      const std::vector<cv::Scalar> &colors,
                      const cv::Point2f &offset) {
	if (image.empty())
		return "image is empty";

	if (matches.empty())
		return RET_OK;

	std::vector<std::size_t> indices(matches.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
	              [&](const std::size_t &idx) {
		              if (matches[idx] == -1)
			              return;

		              const cv::Point2f &pt0 = kpts0[idx].pt;
		              const cv::Point2f &pt1 = pts1[matches[idx]].pt;
		              cv::line(image, pt0, pt1 + offset, colors[idx], 1, cv::LINE_AA);
	              });

	return RET_OK;
}

const char *DrawLines(cv::Mat &image,
                      const std::vector<cv::KeyPoint> &kpts0,
                      const std::vector<cv::KeyPoint> &pts1,
                      const std::vector<int64_t> &matches,
                      const cv::Scalar &color,
                      const cv::Point2f &offset) {
	if (image.empty())
		return "image is empty";

	if (matches.empty())
		return RET_OK;

	std::vector<std::size_t> indices(matches.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
	              [&](const std::size_t &idx) {
		              if (matches[idx] == -1)
			              return;

		              const cv::Point2f &pt0 = kpts0[idx].pt;
		              const cv::Point2f &pt1 = pts1[matches[idx]].pt;
		              cv::line(image, pt0, pt1 + offset, color, 1, cv::LINE_AA);
	              });

	return RET_OK;
}

const char *DrawLines(cv::Mat &image,
                      const cv::Mat &kpts0,
                      const cv::Mat &kpts1,
                      const std::vector<int64_t> &matches,
                      const std::vector<cv::Scalar> &colors,
                      const cv::Point2f &offset) {
	const cv::Point2f *pts0 = kpts0.ptr<cv::Point2f>();
	const cv::Point2f *pts1 = kpts1.ptr<cv::Point2f>();
	return DrawLines(image, pts0, pts1, matches, colors, offset);
}

const char *DrawLines(cv::Mat &image,
                      const cv::Mat &kpts0,
                      const cv::Mat &kpts1,
                      const std::vector<int64_t> &matches,
                      const cv::Scalar &color,
                      const cv::Point2f &offset) {
	const cv::Point2f *pts0 = kpts0.ptr<cv::Point2f>();
	const cv::Point2f *pts1 = kpts1.ptr<cv::Point2f>();
	return DrawLines(image, pts0, pts1, matches, color, offset);
}
