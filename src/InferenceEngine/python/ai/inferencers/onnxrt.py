import numpy as np
import onnxruntime as ort

class OnnxRT:

    __inferencer: ort.InferenceSession = None
    input_details: dict = dict()
    output_details: dict = dict()

    @staticmethod
    def load(model_path: str) -> None:
        # Initialize the ONNX runtime session
        OnnxRT.__inferencer = ort.InferenceSession(model_path)
        
        # Load input and output details
        OnnxRT.__load_input_details()
        OnnxRT.__load_output_details()

    @staticmethod
    def forward(input: np.ndarray) -> np.ndarray:
        # Run inference
        input_name = OnnxRT.input_details["name"]
        output_name = OnnxRT.output_details["name"]
        
        outputs = OnnxRT.__inferencer.run([output_name], {input_name: input})
        
        # Extract and process the output tensor
        output = outputs[0]
        if OnnxRT.output_details["type"] != np.float32:
            output = (output.astype(np.float32) - OnnxRT.output_details["zero_point"]) * OnnxRT.output_details["scale"]

        return output

    @staticmethod
    def __load_input_details() -> None:
        input_meta = OnnxRT.__inferencer.get_inputs()[0]
        OnnxRT.input_details["name"] = input_meta.name
        OnnxRT.input_details["type"] = OnnxRT.__map_onnx_dtype(input_meta.type)
        OnnxRT.input_details["shape"] = np.array(input_meta.shape)[[0, 2, 3, 1]]
        OnnxRT.input_details["scale"] = 1.0
        OnnxRT.input_details["zero_point"] = 0.0

    @staticmethod
    def __load_output_details() -> None:
        output_meta = OnnxRT.__inferencer.get_outputs()[0]
        OnnxRT.output_details["name"] = output_meta.name
        OnnxRT.output_details["type"] = OnnxRT.__map_onnx_dtype(output_meta.type)
        OnnxRT.output_details["scale"] = 1.0
        OnnxRT.output_details["zero_point"] = 0.0

    @staticmethod
    def __map_onnx_dtype(onnx_dtype: str) -> type:
        # Maps ONNX tensor data types to numpy data types
        if onnx_dtype == "tensor(float)":
            return np.float32
        elif onnx_dtype == "tensor(float16)":
            return np.float16
        else:
            raise TypeError(f"Unsupported data type: {onnx_dtype}")
