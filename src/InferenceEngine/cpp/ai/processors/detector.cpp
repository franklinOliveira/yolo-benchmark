#include "detector.hpp"

namespace Detector
{
    static nlohmann::json startInferencer(std::string modelPath, std::string onnxInferencer);
    static void loadArchitecture(std::string modelPath, nlohmann::json inputDetails, float scoreThresh, float confidenceThresh, float iouThresh);

    static std::string architectureFormat;
    static UltralyticsYOLO architecture;

    int preprocessTime;
    int inferenceTime;
    int postprocessTime;

    void init(std::string modelPath, float scoreThresh, float confidenceThresh, float iouThresh, std::string onnxInferencer)
    {
        nlohmann::json inputDetails = Detector::startInferencer(modelPath, onnxInferencer);
        loadArchitecture(modelPath, inputDetails, scoreThresh, confidenceThresh, iouThresh);
    }

    std::vector<Detection> run(cv::Mat image)
    {
        std::vector<Detection> detections;

        auto startTs = std::chrono::high_resolution_clock::now();
        cv::Mat input = Detector::architecture.preProcess(image, true, false);
        auto duration = std::chrono::high_resolution_clock::now() - (startTs);
        Detector::preprocessTime = (int)std::chrono::duration<double, std::milli>(duration).count();

        return detections;
    }

    static nlohmann::json startInferencer(std::string modelPath, std::string onnxInferencer)
    {
        nlohmann::json inputDetails;

        if (modelPath.find(".tflite") != std::string::npos)
        {
            LiteRT::load(modelPath);
            inputDetails = LiteRT::inputDetails;
            Detector::architectureFormat = "litert";
        }

        return inputDetails;
    }

    static void loadArchitecture(std::string modelPath, nlohmann::json inputDetails, float scoreThresh, float confidenceThresh, float iouThresh)
    {
        if ((modelPath.find("yolov5") != std::string::npos) || 
            (modelPath.find("yolov8") != std::string::npos) || 
            (modelPath.find("yolo11") != std::string::npos))
        {
            Detector::architecture = UltralyticsYOLO(inputDetails, scoreThresh, confidenceThresh, iouThresh);
        }
    }

}