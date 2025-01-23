#include "detector.hpp"
#include "mqttproducer.hpp"
#include "plotter.hpp"

#include <filesystem>
#include <indicators/progress_bar.hpp>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    /**
     * STAGE 1: Inference engine setup
     */
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << "<imagesFolder> <modelPath> <cpuCores> <outputFolder>" << std::endl;
        return -1;
    }

    std::string imagesFolder = argv[1];
    std::string modelPath = argv[2];
    std::string cpuCores = argv[3];
    std::string outputFolder = argv[4];
    float scoreThresh = 0.5;
    float confidenceThresh = 0.5;
    float iouThresh = 0.5;

    Detector::init(modelPath, cpuCores, scoreThresh, confidenceThresh, iouThresh);
    MQTTProducer mqttProducer("tcp://localhost:1883", "inference");
    mqttProducer.start();

    /**
     * STAGE 2: Continuous inferencing
    */
    nlohmann::json statusMsg;
    statusMsg["active"] = true;
    mqttProducer.produce("inferenceEngine/status", statusMsg);

    indicators::ProgressBar bar{
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::End{"]"},
        indicators::option::ForegroundColor{indicators::Color::white},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::PrefixText{"[INF. ENGINE] Inferencing images: "}};

    int i = 0;
    std::filesystem::path imagesDir = imagesFolder;
    if (!std::filesystem::exists(outputFolder))
    {
        std::filesystem::create_directories(outputFolder);
    }
    for (const auto &entry : std::filesystem::directory_iterator(imagesDir))
    {
        std::string imagePath = entry.path();
        std::string imageName = std::filesystem::path(imagePath).filename().string();

        cv::Mat image = cv::imread(imagePath);
        std::vector<Detection> detections = Detector::run(image);

        for (Detection detection : detections)
        {
            ImagePlotter::drawDetections(image, detection);
        }
        //cv::imwrite(outputFolder + "/" + imageName, image);

        nlohmann::json dataMsg;
        dataMsg["pre_processing_time"] = Detector::preprocessTime;
        dataMsg["inference_time"] = Detector::inferenceTime;
        dataMsg["post_processing_time"] = Detector::postprocessTime;
        mqttProducer.produce("inferenceEngine/data", dataMsg);
        bar.set_progress(i);
        i++;
    }

    /**
     * Stop inferencing and alerting
     */
    statusMsg["active"] = false;
    mqttProducer.produce("inferenceEngine/status", statusMsg);

    return 0;
}
