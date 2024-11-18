#include <opencv2/opencv.hpp>
#include <vector>

namespace ImagePreprocessing
{

    std::vector<cv::Mat> format(std::vector<cv::Mat> images, const cv::Size input_shape, bool litert_model = true);

}