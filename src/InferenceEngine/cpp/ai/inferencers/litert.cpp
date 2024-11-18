#include "litert.hpp"

namespace LiteRT
{
    static void loadInputDetails(void);
    static void loadOutputDetails(void);

    static std::unique_ptr<tflite::FlatBufferModel> model;
    static tflite::ops::builtin::BuiltinOpResolver resolver;
    static std::unique_ptr<tflite::Interpreter> interpreter;
    static int8_t *inputBuffer;
    static int8_t *outputBuffer;

    nlohmann::json inputDetails;
    nlohmann::json outputDetails;

    void load(std::string modelPath)
    {
        LiteRT::model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
        tflite::InterpreterBuilder(*LiteRT::model, LiteRT::resolver)(&LiteRT::interpreter);
        LiteRT::interpreter->AllocateTensors();
        LiteRT::loadInputDetails();
        LiteRT::loadOutputDetails();
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
            inputTensor->dims->data[2]
        };
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