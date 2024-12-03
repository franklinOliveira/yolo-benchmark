#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>

#include "litert.hpp"
#include "opencvrt.hpp"
#include "ultralyticsyolo.hpp"

namespace Detector
{
    
    extern int preprocessTime;
    extern int inferenceTime;
    extern int postprocessTime;

    void init(std::string modelPath, float scoreThresh, float confidenceThresh, float iouThresh, std::string onnxInferencer);
    std::vector<Detection> run(const cv::Mat& image);

}