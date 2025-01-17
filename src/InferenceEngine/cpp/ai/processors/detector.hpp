#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>

#include "litert.hpp"
#include "onnxrt.hpp"
#include "ultralyticsyolo.hpp"

namespace Detector
{
    
    extern int preprocessTime;
    extern int inferenceTime;
    extern int postprocessTime;

    void init(std::string modelPath, std::string cpuCores, float scoreThresh, float confidenceThresh, float iouThresh);
    std::vector<Detection> run(const cv::Mat& image);

}