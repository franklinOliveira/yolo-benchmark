#include "detector.hpp"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <modelPath>" << std::endl;
        return -1;
    }

    // Get modelPath and imagePath from command-line arguments
    std::string modelPath = argv[1];
    std::string imagePath = "/home/pi/yolo-benchmark/data/datasets/coco128/images/train2017/000000000257.jpg";

    // Define thresholds and inferencer
    float scoreThresh = 0.5;
    float confidenceThresh = 0.5;
    float iouThresh = 0.5;
    std::string onnxInferencer = "opencvrt";

    // Initialize the detector
    Detector::init(modelPath, scoreThresh, confidenceThresh, iouThresh, onnxInferencer);

    // Read the image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Unable to read image at " << imagePath << std::endl;
        return -1;
    }

    // Run the detector
    int detectionTimeSum = 0;
    float detectionTimeAvg = 0.0;
    int nExecutions = 100;
    for (int i = 0; i < nExecutions; i++)
    {
        Detector::run(image);
        int detectionTime = Detector::preprocessTime + Detector::inferenceTime + Detector::postprocessTime;
        detectionTimeSum += detectionTime;

        std::cout << "Detection #" << (i + 1) << " executed in " << detectionTime << "ms\n" << std::endl;
    }
    detectionTimeAvg = detectionTimeSum / nExecutions;

    std::cout << "AVERAGE DETECTION TIME: " << (int) detectionTimeAvg << " ms" << std::endl;

    return 0;
}
