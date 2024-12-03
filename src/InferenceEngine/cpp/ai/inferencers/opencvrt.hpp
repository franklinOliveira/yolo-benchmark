#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace OpencvRT
{
    extern nlohmann::json inputDetails;
    extern nlohmann::json outputDetails;

    void load(std::string modelPath);
    cv::Mat forward(const cv::Mat& image);
}