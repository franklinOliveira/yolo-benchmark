#include "ultralyticsyolo.hpp"

UltralyticsYOLO::UltralyticsYOLO() {}
UltralyticsYOLO::UltralyticsYOLO(nlohmann::json inputDetails, std::string modelInferencer, float scoreThresh, float confidenceThresh, float iouThresh) : inputDetails(inputDetails), scoreThresh(scoreThresh), confidenceThresh(confidenceThresh), iouThresh(iouThresh)
{
    this->formatDetails["size"] = {this->inputDetails["shape"][2], this->inputDetails["shape"][1]};
    this->formatDetails["inferencer"] = modelInferencer;

    if (this->inputDetails.contains("mean"))
    {
        this->formatDetails["mean"] = this->inputDetails["mean"];
    }
}

cv::Mat UltralyticsYOLO::preProcess(const cv::Mat& image)
{
    cv::Mat inputImage = ImagePreprocessing::format(image, this->formatDetails);
    if (this->formatDetails["inferencer"] == "litert")
    {
        if (this->inputDetails["type"] == "INT8")
        {
            inputImage = ImagePreprocessing::quantize(
                inputImage,
                this->inputDetails["scale"],
                this->inputDetails["zeroPoint"],
                this->inputDetails["type"]);
        }
        else if (this->inputDetails["type"] == "FLOAT32")
        {
            inputImage = ImagePreprocessing::normalize(inputImage);
        }
    }

    return inputImage;
}

std::vector<Detection> UltralyticsYOLO::postProcess(cv::Mat outputs, const cv::Mat& image)
{
    std::vector<int> predictedClasses;
    std::vector<float> predictedScores;
    std::vector<cv::Rect> predictedBoxes;

    std::vector<int> sizeVec = this->formatDetails["size"].get<std::vector<int>>();
    cv::Size inputSize(sizeVec[0], sizeVec[1]);
    cv::Size imageSize = image.size();

    const float inputFactorW = imageSize.width / inputSize.width;
    const float inputFactorH = imageSize.height / inputSize.height;
    const int rows = outputs.size[2];
    const int dimensions = outputs.size[1];

    cv::Mat processedOutputs = outputs.reshape(1, dimensions);
    cv::transpose(processedOutputs, processedOutputs);

    float *data = (float *)processedOutputs.data;
    for (int i = 0; i < rows; ++i)
    {
        float *classesScores = data + 4;
        cv::Mat scores(1, 80, CV_32FC1, classesScores);
        cv::Point classId;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &classId);
        if (maxClassScore > UltralyticsYOLO::scoreThresh)
        {
            predictedScores.push_back(maxClassScore);
            predictedClasses.push_back(classId.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * inputFactorW);
            int top = int((y - 0.5 * h) * inputFactorH);

            int width = int(w * inputFactorW);
            int height = int(h * inputFactorH);

            predictedBoxes.push_back(cv::Rect(left, top, width, height));
        }
        data += dimensions;
    }

    return ImagePostprocessing::applyNMS(
        predictedClasses,
        predictedScores,
        predictedBoxes,
        UltralyticsYOLO::confidenceThresh,
        UltralyticsYOLO::iouThresh);
}
