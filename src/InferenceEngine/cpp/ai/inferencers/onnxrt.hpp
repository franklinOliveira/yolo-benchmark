#include <thread>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

namespace OnnxRT
{
    extern nlohmann::json inputDetails;
    extern nlohmann::json outputDetails;

    void load(std::string modelPath, std::string cpuCores);
    cv::Mat forward(const cv::Mat& image);
}