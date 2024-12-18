#include <iostream>
#include <thread>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

namespace OnnxRT
{
    static void loadInputDetails(void);
    static void loadOutputDetails(void);

    static Ort::Env __env{ORT_LOGGING_LEVEL_WARNING, "ONNXRT"};
    static Ort::SessionOptions __sessionOptions;
    static Ort::Session __inferencer{nullptr};
    static Ort::MemoryInfo __memoryInfo{nullptr};

    nlohmann::json inputDetails;
    nlohmann::json outputDetails;

    void load(std::string modelPath)
    {
        int numberOfCpus = std::thread::hardware_concurrency();
        OnnxRT::__sessionOptions.SetInterOpNumThreads(numberOfCpus);
        OnnxRT::__sessionOptions.SetIntraOpNumThreads(numberOfCpus);
        OnnxRT::__sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

        OnnxRT::__inferencer = Ort::Session{OnnxRT::__env, modelPath.c_str(), OnnxRT::__sessionOptions};
        OnnxRT::__memoryInfo = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        OnnxRT::loadInputDetails();
        OnnxRT::loadOutputDetails();
    }

    cv::Mat forward(const cv::Mat &image)
    {
        std::vector<std::vector<int64_t>> inputsShape;
        std::vector<Ort::Value> inputTensor;
        size_t inputTensorSize = image.total();

        inputsShape.push_back(OnnxRT::__inferencer.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape());
        inputTensor.emplace_back(
            Ort::Value::CreateTensor<float>(
                OnnxRT::__memoryInfo,
                (float *)image.data,
                inputTensorSize,
                inputsShape[0].data(),
                inputsShape[0].size()));

        std::vector<const char *> inputNodeName = {"images"};
        std::vector<const char *> outputNodeName = {"output0"};

        std::vector<Ort::Value> outputTensor = OnnxRT::__inferencer.Run(
            Ort::RunOptions{nullptr},
            inputNodeName.data(),
            inputTensor.data(),
            inputTensor.size(),
            outputNodeName.data(),
            outputNodeName.size());

        float *outputBuffer = outputTensor[0].GetTensorMutableData<float>();
        size_t outputsSize = outputTensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
        cv::Mat outputs = cv::Mat(3, outputsSize, CV_32F);

        // Print the output values
        for (size_t i = 0; i < outputsSize; ++i)
        {
            outputs.at<float>(i) = static_cast<float>(outputBuffer[i]);
        }
        return outputs;
    }

    static void loadInputDetails(void)
    {
        std::string inputName = OnnxRT::__inferencer.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
        std::vector<int64_t> inputShape = OnnxRT::__inferencer.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        OnnxRT::inputDetails["name"] = inputName;
        OnnxRT::inputDetails["type"] = "FLOAT32";
        OnnxRT::inputDetails["shape"] = inputShape;
        OnnxRT::inputDetails["scale"] = 1.0;
        OnnxRT::inputDetails["zeroPoint"] = 0.0;
    }

    static void loadOutputDetails(void)
    {
        std::string outputName = OnnxRT::__inferencer.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
        std::vector<int64_t> outputShape = OnnxRT::__inferencer.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        OnnxRT::outputDetails["name"] = outputName;
        OnnxRT::outputDetails["type"] = "FLOAT32";
        OnnxRT::outputDetails["shape"] = outputShape;
        OnnxRT::outputDetails["scale"] = 1.0;
        OnnxRT::outputDetails["zeroPoint"] = 0.0;
    }
}

void functionCall()
{
    // Example of loading an ONNX model
    std::string modelPath = "model.onnx";
    std::string imagePath = "image.jpg";

    OnnxRT::load(modelPath);
    std::cout << "Model loaded" << std::endl;

    cv::Mat image = cv::imread(imagePath);
    image = cv::dnn::blobFromImage(
        image,
        (1.0 / 255.0),
        cv::Size(416, 416),
        cv::Scalar(),
        true,
        false);
    std::cout << "Image read" << std::endl;

    OnnxRT::forward(image);
    std::cout << "Model forwarded" << std::endl;
}

int main()
{
    functionCall();
    return 0;
}