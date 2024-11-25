#include "opencvrt.hpp"

namespace OpencvRT
{
    static void loadInputDetails(void);
    static void loadOutputDetails(void);

    static cv::dnn::Net interpreter;

    nlohmann::json inputDetails;
    nlohmann::json outputDetails;

    void load(std::string modelPath)
    {
        OpencvRT::interpreter = cv::dnn::readNetFromONNX(modelPath);
        OpencvRT::interpreter.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        OpencvRT::interpreter.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        OpencvRT::loadInputDetails();
        OpencvRT::loadOutputDetails();
        
    }

    cv::Mat forward(cv::Mat image)
    {
        std::vector<cv::Mat> outputs;
        OpencvRT::interpreter.setInput(image);
        OpencvRT::interpreter.forward(outputs, OpencvRT::interpreter.getUnconnectedOutLayersNames());
        return outputs[0];
    }

    static void loadInputDetails(void)
    {
        OpencvRT::inputDetails["name"] = OpencvRT::interpreter.getLayerNames()[0];
        OpencvRT::inputDetails["type"] = "FLOAT32";
        OpencvRT::inputDetails["shape"] = {1, 416, 416, 3};
        OpencvRT::inputDetails["mean"] = {0.485, 0.456, 0.406};
        OpencvRT::inputDetails["scale"] = 1.0;
        OpencvRT::inputDetails["zeroPoint"] = 0.0;
    }

    static void loadOutputDetails(void)
    {
        OpencvRT::outputDetails["name"] = OpencvRT::interpreter.getUnconnectedOutLayersNames()[0];
        OpencvRT::outputDetails["type"] = "FLOAT32";
        OpencvRT::outputDetails["scale"] = 1.0;
        OpencvRT::outputDetails["zeroPoint"] = 0.0;
    }
}