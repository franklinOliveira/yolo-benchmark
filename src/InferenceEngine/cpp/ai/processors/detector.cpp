#include "detector.hpp"

namespace Detector
{
    static nlohmann::json startInferencer(std::string modelPath, std::string onnxInferencer);
    static void loadArchitecture(std::string modelPath, nlohmann::json inputDetails, float scoreThresh, float confidenceThresh, float iouThresh);

    static std::string modelInferencer;
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
        cv::Mat outputs;
        std::vector<Detection> detections;

        auto startTs = std::chrono::high_resolution_clock::now();
        cv::Mat input = Detector::architecture.preProcess(image);
        auto duration = std::chrono::high_resolution_clock::now() - (startTs);
        Detector::preprocessTime = (int)std::chrono::duration<double, std::milli>(duration).count();

        startTs = std::chrono::high_resolution_clock::now();
        if (Detector::modelInferencer == "litert")
        {
            outputs = LiteRT::forward(input);
        }

        else if (Detector::modelInferencer == "opencvrt")
        {
            outputs = OpencvRT::forward(input);
        }

        duration = std::chrono::high_resolution_clock::now() - (startTs);
        Detector::inferenceTime = (int)std::chrono::duration<double, std::milli>(duration).count();

        startTs = std::chrono::high_resolution_clock::now();
        detections = Detector::architecture.postProcess(outputs, image);
        duration = std::chrono::high_resolution_clock::now() - (startTs);
        Detector::postprocessTime = (int)std::chrono::duration<double, std::milli>(duration).count();

        return detections;
    }

    static nlohmann::json startInferencer(std::string modelPath, std::string onnxInferencer)
    {
        nlohmann::json inputDetails;

        if (modelPath.find(".tflite") != std::string::npos)
        {
            LiteRT::load(modelPath);
            inputDetails = LiteRT::inputDetails;
            Detector::modelInferencer = "litert";
        }

        else if ((modelPath.find(".onnx") != std::string::npos) && (onnxInferencer == "opencvrt"))
        {
            OpencvRT::load(modelPath);
            inputDetails = OpencvRT::inputDetails;
            Detector::modelInferencer = "opencvrt";
        }

        return inputDetails;
    }

    static void loadArchitecture(std::string modelPath, nlohmann::json inputDetails, float scoreThresh, float confidenceThresh, float iouThresh)
    {
        if ((modelPath.find("yolov5") != std::string::npos) || 
            (modelPath.find("yolov8") != std::string::npos) || 
            (modelPath.find("yolo11") != std::string::npos))
        {
            Detector::architecture = UltralyticsYOLO(inputDetails, Detector::modelInferencer, scoreThresh, confidenceThresh, iouThresh);
        }
    }

}