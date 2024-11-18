#include "uyolo.hpp"

UltralyticsYOLO::UltralyticsYOLO(){}
UltralyticsYOLO::UltralyticsYOLO(nlohmann::json inputDetails, float scoreThresh, float confidenceThresh, float iouThresh) : inputDetails(inputDetails), scoreThresh(scoreThresh), confidenceThresh(confidenceThresh), iouThresh(iouThresh) {}

cv::Mat UltralyticsYOLO::preProcess(cv::Mat image, bool litertModel, bool opencvInferencer)
{
    cv::Mat inputImage;
    image.copyTo(inputImage);
    cv::Size inputSize(this->inputDetails["shape"][2], this->inputDetails["shape"][1]);

    if (opencvInferencer)
    {
        inputImage = cv::dnn::blobFromImage(
            inputImage, 
            (1.0 / 255.0), 
            inputSize, 
            cv::Scalar(this->inputDetails["mean"]), 
            true, 
            false
        );
    }

    else
    {
        std::vector<cv::Mat> inputImages = {inputImage};
        inputImage = ImagePreprocessing::format(inputImages, inputSize, litertModel)[0];
    }
}
