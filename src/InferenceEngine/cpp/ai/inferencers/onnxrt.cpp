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
        std::vector<const char *> inputNodeName = {"images"};
        std::vector<std::vector<int64_t>> inputsShape;
        std::vector<Ort::Value> inputTensor;
        size_t inputTensorSize = image.total();
        cv::Mat inputs;

        std::vector<const char *> outputNodeName = {"output0"};
        const int outputBatches = OnnxRT::outputDetails["batchs"].get<int>();
        const int outputRows = OnnxRT::outputDetails["rows"].get<int>();
        const int outputColumns = OnnxRT::outputDetails["columns"].get<int>();
        int outputsSize[] = {outputBatches, outputRows, outputColumns};
        cv::Mat outputs = cv::Mat(3, outputsSize, CV_32F);

        inputsShape.push_back(OnnxRT::inferencer.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape());
        if (OnnxRT::inputDetails["type"] == "FLOAT32")
        {
            image.convertTo(inputs, CV_32F);
            inputTensor.emplace_back(
                Ort::Value::CreateTensor(
                    OnnxRT::memoryInfo,
                    (float *)inputs.data,
                    inputTensorSize,
                    inputsShape[0].data(),
                    inputsShape[0].size()));
        }

        else if (OnnxRT::inputDetails["type"] == "FLOAT16")
        {
            image.convertTo(inputs, CV_16F);
            inputTensor.emplace_back(
                Ort::Value::CreateTensor(
                    OnnxRT::memoryInfo,
                    reinterpret_cast<Ort::Float16_t *>(inputs.data),
                    inputTensorSize,
                    inputsShape[0].data(),
                    inputsShape[0].size()));
        }

        std::vector<Ort::Value> outputTensor = OnnxRT::inferencer.Run(
            Ort::RunOptions{nullptr},
            inputNodeName.data(),
            inputTensor.data(),
            inputTensor.size(),
            outputNodeName.data(),
            outputNodeName.size());
        
        if (OnnxRT::inputDetails["type"] == "FLOAT32")
        {
            float *outputBuffer = outputTensor[0].GetTensorMutableData<float>();
            for (size_t i = 0; i < outputs.total(); ++i)
            {
                outputs.at<float>(i) = static_cast<float>(outputBuffer[i]);
            }
        }
        else if (OnnxRT::inputDetails["type"] == "FLOAT16")
        {
            Ort::Float16_t *outputBuffer = outputTensor[0].GetTensorMutableData<Ort::Float16_t>();
            for (size_t i = 0; i < outputs.total(); ++i)
            {
                outputs.at<float>(i) = static_cast<float>(outputBuffer[i]);
            }
        }

        return outputs;
    }

    static void loadInputDetails(void)
    {
        std::string inputName = OnnxRT::inferencer.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
        std::vector<int64_t> inputShape = OnnxRT::inferencer.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        int inputType = OnnxRT::inferencer.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();

        OnnxRT::inputDetails["name"] = inputName;
        if (inputType == 1)
        {
            OnnxRT::inputDetails["type"] = "FLOAT32";
        }
        else if (inputType == 10)
        {
            OnnxRT::inputDetails["type"] = "FLOAT16";
        }
        OnnxRT::inputDetails["shape"] = inputShape;
        OnnxRT::inputDetails["scale"] = 1.0;
        OnnxRT::inputDetails["zeroPoint"] = 0.0;
    }

    static void loadOutputDetails(void)
    {
        std::string outputName = OnnxRT::inferencer.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
        std::vector<int64_t> outputShape = OnnxRT::inferencer.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        int outputType = OnnxRT::inferencer.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();

        OnnxRT::outputDetails["name"] = outputName;
        if (outputType == 1)
        {
            OnnxRT::outputDetails["type"] = "FLOAT32";
        }
        else if (outputType == 10)
        {
            OnnxRT::outputDetails["type"] = "FLOAT16";
        }
        OnnxRT::outputDetails["shape"] = outputShape;
        OnnxRT::outputDetails["batchs"] = outputShape[0];
        OnnxRT::outputDetails["rows"] = outputShape[1];
        OnnxRT::outputDetails["columns"] = outputShape[2];
        OnnxRT::outputDetails["scale"] = 1.0;
        OnnxRT::outputDetails["zeroPoint"] = 0.0;
    }
}