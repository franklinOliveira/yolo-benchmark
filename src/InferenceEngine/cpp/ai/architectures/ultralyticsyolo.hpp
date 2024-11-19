#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "preprocessing.hpp"
#include "detection.hpp"

class UltralyticsYOLO
{

public:
    nlohmann::json inputDetails;
    float scoreThresh;
    float confidenceThresh;
    float iouThresh;

    UltralyticsYOLO();
    UltralyticsYOLO(nlohmann::json inputDetails, float scoreThresh, float confidenceThresh, float iouThresh);
    cv::Mat preProcess(cv::Mat image, bool litertModel, bool opencvrtInferencer);
    
};