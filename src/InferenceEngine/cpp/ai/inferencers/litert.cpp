#include "litert.hpp"

namespace LiteRT
{
    static void loadInputDetails(void);
    static void loadOutputDetails(void);

    static std::unique_ptr<tflite::FlatBufferModel> model;
    static tflite::ops::builtin::BuiltinOpResolver resolver;
    static std::unique_ptr<tflite::Interpreter> interpreter;

    nlohmann::json inputDetails;
    nlohmann::json outputDetails;

    void load(std::string modelPath)
    {
        int numberOfCpus = std::thread::hardware_concurrency();
        LiteRT::model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
        tflite::InterpreterBuilder(*LiteRT::model, LiteRT::resolver)(&LiteRT::interpreter);
        LiteRT::interpreter->SetNumThreads(numberOfCpus);
        LiteRT::interpreter->AllocateTensors();
        LiteRT::loadInputDetails();
        LiteRT::loadOutputDetails();
    }

    cv::Mat forward(const cv::Mat &image)
    {
        const int inputSize = LiteRT::inputDetails["size"].get<int>();
        const int outputBatches = LiteRT::outputDetails["batchs"].get<int>();
        const int outputRows = LiteRT::outputDetails["rows"].get<int>();
        const int outputColumns = LiteRT::outputDetails["columns"].get<int>();
        int outputsSize[] = {outputBatches, outputRows, outputColumns};
        cv::Mat outputs = cv::Mat(3, outputsSize, CV_32F);

        if (LiteRT::inputDetails["type"] == "INT8")
        {
            int8_t *inputBuffer = LiteRT::interpreter->typed_input_tensor<int8_t>(LiteRT::interpreter->inputs()[0]);
            std::memcpy(inputBuffer, reinterpret_cast<int8_t *>(image.data), inputSize * sizeof(int8_t));
        }
        else if (LiteRT::inputDetails["type"] == "FLOAT32")
        {
            float *inputBuffer = LiteRT::interpreter->typed_input_tensor<float>(LiteRT::interpreter->inputs()[0]);
            std::memcpy(inputBuffer, reinterpret_cast<float *>(image.data), inputSize * sizeof(int8_t));
        }

        if (LiteRT::interpreter->Invoke() != kTfLiteOk)
        {
            return cv::Mat();
        }

        if (LiteRT::outputDetails["type"] == "INT8")
        {
            int8_t *outputBuffer = LiteRT::interpreter->typed_output_tensor<int8_t>(LiteRT::interpreter->inputs()[0]);
            const int zeroPoint = LiteRT::outputDetails["zeroPoint"].get<int>();
            const float scale = LiteRT::outputDetails["scale"].get<float>();

            for (size_t i = 0; i < outputs.total(); ++i)
            {
                outputs.at<float>(i) = (static_cast<float>(outputBuffer[i]) - ((float)zeroPoint)) * scale;
            }
        }
        else if (LiteRT::inputDetails["type"] == "FLOAT32")
        {
            float *outputBuffer = LiteRT::interpreter->typed_output_tensor<float>(LiteRT::interpreter->inputs()[0]);
            for (size_t i = 0; i < outputs.total(); ++i)
            {
                outputs.at<float>(i) = static_cast<float>(outputBuffer[i]);
            }
        }

        cv::Size inputShape(LiteRT::inputDetails["shape"][2], LiteRT::inputDetails["shape"][1]);
        for (int i = 0; i < outputs.size[2]; ++i)
        {
            outputs.at<float>(0, 0, i) *= inputShape.width;
            outputs.at<float>(0, 2, i) *= inputShape.width;
            outputs.at<float>(0, 1, i) *= inputShape.height;
            outputs.at<float>(0, 3, i) *= inputShape.height;
        }

        return outputs;
    }

    static void loadInputDetails(void)
    {
        TfLiteTensor *inputTensor = LiteRT::interpreter->tensor(LiteRT::interpreter->inputs()[0]);

        TfLiteType inputType = inputTensor->type;
        if (inputType == kTfLiteFloat32)
        {
            LiteRT::inputDetails["type"] = "FLOAT32";
        }
        else if (inputType == kTfLiteInt8)
        {
            LiteRT::inputDetails["type"] = "INT8";
        }
        std::vector<int> shape = {
            inputTensor->dims->data[0],
            inputTensor->dims->data[1],
            inputTensor->dims->data[2]};
        LiteRT::inputDetails["size"] = inputTensor->bytes;
        LiteRT::inputDetails["shape"] = shape;
        LiteRT::inputDetails["scale"] = inputTensor->params.scale;
        LiteRT::inputDetails["zeroPoint"] = inputTensor->params.zero_point;
    }

    static void loadOutputDetails(void)
    {
        TfLiteTensor *outputTensor = LiteRT::interpreter->output_tensor(LiteRT::interpreter->inputs()[0]);

        TfLiteType outputType = outputTensor->type;
        if (outputType == kTfLiteFloat32)
        {
            LiteRT::outputDetails["type"] = "FLOAT32";
        }
        else if (outputType == kTfLiteInt8)
        {
            LiteRT::outputDetails["type"] = "INT8";
        }
        LiteRT::outputDetails["size"] = outputTensor->bytes;
        LiteRT::outputDetails["batchs"] = outputTensor->dims->data[0];
        LiteRT::outputDetails["rows"] = outputTensor->dims->data[1];
        LiteRT::outputDetails["columns"] = outputTensor->dims->data[2];
        LiteRT::outputDetails["channels"] = outputTensor->dims->data[3];
        LiteRT::outputDetails["scale"] = outputTensor->params.scale;
        LiteRT::outputDetails["zeroPoint"] = outputTensor->params.zero_point;
    }
}