#include <opencv2/opencv.hpp>
#include <vector>
#include <nlohmann/json.hpp>

namespace ImagePreprocessing
{
    cv::Mat format(cv::Mat image, const nlohmann::json formatDetails);
    cv::Mat quantize(cv::Mat image, float scale, float zeroPoint, std::string type);
    //cv::Mat normalize(cv::Mat image, ...);
}