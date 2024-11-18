#include "detector.hpp"

int main()
{
    std::string modelPath = "/home/pi/yolo-benchmark/data/models/tflite/yolov8n_full_integer_quant.tflite";
    float scoreThresh = 0.5;
    float confidenceThresh = 0.5;
    float iouThresh = 0.5;
    std::string onnxInferencer = "onnxrt";
    
    Detector::init(modelPath, scoreThresh, confidenceThresh, iouThresh, onnxInferencer);

    return 0;
}