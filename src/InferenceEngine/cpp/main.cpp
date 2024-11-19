#include "detector.hpp"

int main()
{
    std::string modelPath = "/home/pi/yolo-benchmark/data/models/tflite/yolov8n_full_integer_quant.tflite";
    std::string imagePath = "/home/pi/yolo-benchmark/data/datasets/coco128/images/train2017/000000000257.jpg";

    float scoreThresh = 0.5;
    float confidenceThresh = 0.5;
    float iouThresh = 0.5;
    std::string onnxInferencer = "onnxrt";
    
    Detector::init(modelPath, scoreThresh, confidenceThresh, iouThresh, onnxInferencer);

    cv::Mat image = cv::imread(imagePath);
    Detector::run(image);

    std::cout << "Preprocessing time: " << Detector::preprocessTime << "ms" << std::endl;
    std::cout << "Inference time: " << Detector::inferenceTime << "ms" << std::endl;

    return 0;
}