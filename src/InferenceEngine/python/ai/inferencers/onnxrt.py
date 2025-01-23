import numpy as np
import onnxruntime as ort
import multiprocessing

class OnnxRT:
    """
    OnnxRT provides a runtime interface for loading and running ONNX models using the ONNX Runtime.
    This class supports loading model input and output details and performs inference with customizable input scaling.

    Attributes:
        __inferencer (ort.InferenceSession): 
            Private attribute holding the ONNX Runtime session instance for model inference.
        input_details (dict): Dictionary containing model input details, such as name, type, shape, scale, and zero_point.
        output_details (dict): Dictionary containing model output details, including name, type, scale, and zero_point.
    """

    __inferencer: ort.InferenceSession = None
    input_details: dict = dict()
    output_details: dict = dict()

    @staticmethod
    def load(model_path: str, half_cores: bool) -> None:
        """
        Loads an ONNX model from the specified path and initializes the ONNX runtime session.
        Populates input and output details for the model.

        Args:
            model_path (str): Path to the ONNX model file.
            half_cores (bool): Use only half of CPU cores for inference

        Raises:
            RuntimeError: If the model file path is invalid or the session fails to load.
        """
        
        num_cores = multiprocessing.cpu_count()
        if half_cores:
            num_cores = num_cores // 2
        
        print(f"Number of cores: {num_cores}")
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_cores

        OnnxRT.__inferencer = ort.InferenceSession(model_path, sess_options=session_options)
        
        OnnxRT.__load_input_details()
        OnnxRT.__load_output_details()

    @staticmethod
    def forward(input: np.ndarray) -> np.ndarray:
        """
        Performs inference on the given input tensor using the loaded ONNX model.

        Args:
            input (np.ndarray): Preprocessed input data to be fed to the model.

        Returns:
            np.ndarray: Model output tensor, optionally scaled based on output quantization parameters.

        Raises:
            ValueError: If inference is attempted without a loaded model.
        """
        if OnnxRT.__inferencer is None:
            raise ValueError("Model not loaded. Please call OnnxRT.load() before inference.")

        input_name = OnnxRT.input_details["name"]
        output_name = OnnxRT.output_details["name"]
        
        outputs = OnnxRT.__inferencer.run([output_name], {input_name: input})
        
        output = outputs[0]
        if OnnxRT.output_details["type"] != np.float32:
            output = (output.astype(np.float32) - OnnxRT.output_details["zero_point"]) * OnnxRT.output_details["scale"]

        return output

    @staticmethod
    def __load_input_details() -> None:
        """
        Loads the input details of the ONNX model into the `input_details` dictionary.
        Extracted details include input name, data type, shape, scale, and zero_point.

        Notes:
            Shape order is adjusted to match TensorFlow conventions (batch, height, width, channels).
        """
        input_meta = OnnxRT.__inferencer.get_inputs()[0]
        OnnxRT.input_details["name"] = input_meta.name
        OnnxRT.input_details["type"] = OnnxRT.__map_onnx_dtype(input_meta.type)
        OnnxRT.input_details["shape"] = np.array(input_meta.shape)[[0, 2, 3, 1]]  # Adjust for (batch, height, width, channels)
        OnnxRT.input_details["scale"] = 1.0
        OnnxRT.input_details["zero_point"] = 0.0

    @staticmethod
    def __load_output_details() -> None:
        """
        Loads the output details of the ONNX model into the `output_details` dictionary.
        Extracted details include output name, data type, scale, and zero_point.
        """
        output_meta = OnnxRT.__inferencer.get_outputs()[0]
        OnnxRT.output_details["name"] = output_meta.name
        OnnxRT.output_details["type"] = OnnxRT.__map_onnx_dtype(output_meta.type)
        OnnxRT.output_details["scale"] = 1.0
        OnnxRT.output_details["zero_point"] = 0.0

    @staticmethod
    def __map_onnx_dtype(onnx_dtype: str) -> type:
        """
        Maps ONNX tensor data types to numpy data types.

        Args:
            onnx_dtype (str): ONNX data type string.

        Returns:
            type: Corresponding numpy data type.

        Raises:
            TypeError: If the provided ONNX data type is not supported.
        """
        if onnx_dtype == "tensor(float)":
            return np.float32
        elif onnx_dtype == "tensor(float16)":
            return np.float16
        else:
            raise TypeError(f"Unsupported data type: {onnx_dtype}")
