#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <thread>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/c/c_api.h>

namespace LiteRT
{
    extern nlohmann::json inputDetails;
    extern nlohmann::json outputDetails;

    void load(std::string modelPath);
    cv::Mat forward(const cv::Mat& image);
}