#include "preprocessing.hpp"

namespace ImagePreprocessing
{
    cv::Mat format(cv::Mat image, const cv::Size inputShape, bool litertModel)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        cv::resize(image, image, inputShape, cv::INTER_CUBIC);
        int totalPixels = image.cols * image.rows * image.channels();

        cv::Mat formatedImage(image.rows, image.cols, CV_32FC3);
        float *formatedData = reinterpret_cast<float *>(formatedImage.data);

        for (int i = 0; i < totalPixels; i++)
        {
            float normalizedValue = ((float)image.data[i] / 255.0);
            formatedData[i] = normalizedValue;
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
            int8_t quantizedValue = ((float)image.data[i] / scale) + zeroPoint;
            quantizedData[i] = quantizedValue;
        }

        return quantizedImage;
    }
}