#include "ultralyticsyolo.hpp"

UltralyticsYOLO::UltralyticsYOLO(){}
UltralyticsYOLO::UltralyticsYOLO(nlohmann::json inputDetails, float scoreThresh, float confidenceThresh, float iouThresh) : inputDetails(inputDetails), scoreThresh(scoreThresh), confidenceThresh(confidenceThresh), iouThresh(iouThresh) {}

cv::Mat UltralyticsYOLO::preProcess(cv::Mat image, bool litertModel, bool opencvrtInferencer)
{
    cv::Mat inputImage;
    image.copyTo(inputImage);
    cv::Size inputSize(this->inputDetails["shape"][2], this->inputDetails["shape"][1]);

    if (opencvrtInferencer)
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
        inputImage = ImagePreprocessing::format(inputImage, inputSize, litertModel);
    }

    if (this->inputDetails["type"] == "INT8")
    {
        inputImage = ImagePreprocessing::quantize(
            inputImage,
            this->inputDetails["scale"],
            this->inputDetails["zeroPoint"],
            this->inputDetails["type"]
        );
    }

    return inputImage;
}
