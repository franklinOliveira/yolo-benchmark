#include "preprocessing.hpp"

namespace ImagePreprocessing
{
    std::vector<cv::Mat> format(std::vector<cv::Mat> images, const cv::Size inputShape, bool litert_model)
    {
        for (auto &image : images)
        {
            cv::resize(image, image, inputShape, 0, 0, cv::INTER_LINEAR);
        }

        std::vector<cv::Mat> input(images.size());
        for (size_t i = 0; i < images.size(); ++i)
        {
            cv::Mat rgbImage;
            cv::cvtColor(images[i], rgbImage, cv::COLOR_BGR2RGB);
            rgbImage.convertTo(rgbImage, CV_32F, 1.0 / 255.0);
            input[i] = rgbImage;
            input[i] = cv::dnn::blobFromImage(rgbImage);
        }

        if (litert_model)
        {
            for (auto &img : input)
            {
                img = img.reshape(1, {img.size[0], img.size[2], img.size[3], img.size[1]});
            }
        }

        return input;
    }
}