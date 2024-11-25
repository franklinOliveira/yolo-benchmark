#include "preprocessing.hpp"

namespace ImagePreprocessing
{
    cv::Mat format(cv::Mat image, const nlohmann::json formatDetails)
    {
        std::string modelInferencer = formatDetails["inferencer"].get<std::string>();
        std::vector<int> sizeVec = formatDetails["size"].get<std::vector<int>>();
        cv::Size inputSize(sizeVec[0], sizeVec[1]);
        cv::Mat formatedImage;

        if (modelInferencer == "opencvrt")
        {   
            formatedImage = cv::dnn::blobFromImage(
                image,
                (1.0 / 255.0),
                inputSize,
                cv::Scalar(),
                true,
                false);
        }

        else if (modelInferencer == "litert")
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            cv::resize(image, image, inputSize, cv::INTER_CUBIC);
            formatedImage = image;
        }

        return formatedImage;
    }

    cv::Mat quantize(cv::Mat image, float scale, float zeroPoint, std::string type)
    {
        cv::Mat quantizedImage(image.rows, image.cols, CV_8UC3);
        int8_t *quantizedData = reinterpret_cast<int8_t *>(quantizedImage.data);
        int totalPixels = image.cols * image.rows * image.channels();

        for (int i = 0; i < totalPixels; i++)
        {
            uint8_t rgbValue = static_cast<uint8_t>(image.data[i]);
            float normalizedValue = static_cast<float>(rgbValue) / 255.0;
            int8_t quantizedValue = (normalizedValue / scale) + zeroPoint;
            quantizedData[i] = quantizedValue;
        }

        return quantizedImage;
    }

    //cv::Mat normalize(cv::Mat image, ...)
}