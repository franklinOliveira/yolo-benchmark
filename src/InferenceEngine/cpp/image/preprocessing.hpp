#include <opencv2/opencv.hpp>
#include <vector>

namespace ImagePreprocessing
{
    cv::Mat format(cv::Mat image, const cv::Size inputShape, bool litertModel);
    cv::Mat quantize(cv::Mat image, float scale, float zeroPoint, std::string type);
}