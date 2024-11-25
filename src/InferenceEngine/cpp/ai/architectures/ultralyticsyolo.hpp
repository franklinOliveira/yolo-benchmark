#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "preprocessing.hpp"
#include "postprocessing.hpp"

class UltralyticsYOLO
{

private:
    nlohmann::json formatDetails;
    nlohmann::json inputDetails;

    float scoreThresh;
    float confidenceThresh;
    float iouThresh;

    std::vector<std::string> classes{
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"};

public:
    UltralyticsYOLO();
    UltralyticsYOLO(nlohmann::json inputDetails, std::string modelInferencer, float scoreThresh, float confidenceThresh, float iouThresh);
    
    cv::Mat preProcess(cv::Mat image);
    std::vector<Detection> postProcess(cv::Mat outputs, cv::Mat image);
};