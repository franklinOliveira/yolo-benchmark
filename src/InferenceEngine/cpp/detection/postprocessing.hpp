#include <opencv2/opencv.hpp>
#include <vector>
#include "detection.hpp"
#include <iostream>

namespace ImagePostprocessing
{
    std::vector<Detection> applyNMS(std::vector<int> predictedClasses, std::vector<float> predictedScores, std::vector<cv::Rect> predictedBoxes, const float confidenceThresh, const float iouThresh);
}