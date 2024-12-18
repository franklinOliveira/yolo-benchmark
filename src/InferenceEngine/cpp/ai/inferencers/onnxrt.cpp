#include "onnxrt.hpp"

namespace OnnxRT
{
    static void loadInputDetails(void);
    static void loadOutputDetails(void);

    static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ONNXRT"};
    static Ort::SessionOptions sessionOptions;
    static Ort::Session inferencer{nullptr};
    static Ort::MemoryInfo memoryInfo{nullptr};

    nlohmann::json inputDetails;
    nlohmann::json outputDetails;

    void load(std::string modelPath)
    {
        int numberOfCpus = std::thread::hardware_concurrency();
        OnnxRT::sessionOptions.SetInterOpNumThreads(numberOfCpus);
        OnnxRT::sessionOptions.SetIntraOpNumThreads(numberOfCpus);
        OnnxRT::sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

        OnnxRT::inferencer = Ort::Session{OnnxRT::env, modelPath.c_str(), OnnxRT::sessionOptions};
        OnnxRT::memoryInfo = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        OnnxRT::loadInputDetails();
        OnnxRT::loadOutputDetails();
    }

    cv::Mat forward(const cv::Mat &image)
    {
        std::vector<std::vector<int64_t>> inputsShape;
        std::vector<Ort::Value> inputTensor;
        size_t inputTensorSize = image.total();

        inputsShape.push_back(OnnxRT::inferencer.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape());
        inputTensor.emplace_back(
            Ort::Value::CreateTensor<float>(
                OnnxRT::memoryInfo,
                (float *)image.data,
                inputTensorSize,
                inputsShape[0].data(),
                inputsShape[0].size()));

        std::vector<const char *> inputNodeName = {"images"};
        std::vector<const char *> outputNodeName = {"output0"};

        std::vector<Ort::Value> outputTensor = OnnxRT::inferencer.Run(
            Ort::RunOptions{nullptr},
            inputNodeName.data(),
            inputTensor.data(),
            inputTensor.size(),
            outputNodeName.data(),
            outputNodeName.size());

        float *outputBuffer = outputTensor[0].GetTensorMutableData<float>();
        size_t outputsSize = outputTensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
        cv::Mat outputs = cv::Mat(3, outputsSize, CV_32F);
        for (size_t i = 0; i < outputsSize; ++i)
        {
            outputs.at<float>(i) = static_cast<float>(outputBuffer[i]);
        }
        return outputs;
    }

    static void loadInputDetails(void)
    {
        std::string inputName = OnnxRT::inferencer.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
        std::vector<int64_t> inputShape = OnnxRT::inferencer.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        OnnxRT::inputDetails["name"] = inputName;
        OnnxRT::inputDetails["type"] = "FLOAT32";
        OnnxRT::inputDetails["shape"] = inputShape;
        OnnxRT::inputDetails["scale"] = 1.0;
        OnnxRT::inputDetails["zeroPoint"] = 0.0;
    }

    static void loadOutputDetails(void)
    {
        std::string outputName = OnnxRT::inferencer.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
        std::vector<int64_t> outputShape = OnnxRT::inferencer.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        OnnxRT::outputDetails["name"] = outputName;
        OnnxRT::outputDetails["type"] = "FLOAT32";
        OnnxRT::outputDetails["shape"] = outputShape;
        OnnxRT::outputDetails["scale"] = 1.0;
        OnnxRT::outputDetails["zeroPoint"] = 0.0;
    }
}