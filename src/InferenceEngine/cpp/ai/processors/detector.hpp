#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "litert.hpp"
#include "uyolo.hpp"

namespace Detector
{
    
    extern int preprocessTime;
    extern int inferenceTime;
    extern int postprocessTime;

    void init(std::string modelPath, float scoreThresh, float confidenceThresh, float iouThresh, std::string onnxInferencer);
    //std::vector<Detection> run(cv::Mat image);

}