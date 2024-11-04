import cv2
import numpy as np

class OpencvRT:
    """
    OpencvRT provides an interface for loading and running ONNX models using OpenCV's DNN module.
    This class is designed to load model input and output details, and perform inference with support for 
    quantized scaling if needed.

    Attributes:
        __inferencer (cv2.dnn_Net): 
            Private attribute holding the OpenCV DNN model instance for inference.
        input_details (dict): 
            Dictionary containing model input details such as name, type, shape, mean, scale, and zero_point.
        output_details (dict): 
            Dictionary containing model output details including name, type, scale, and zero_point.
    """

    __inferencer: cv2.dnn_Net = None
    input_details: dict = dict()
    output_details: dict = dict()

    @staticmethod
    def load(model_path: str) -> None:
        """
        Loads an ONNX model from the specified path and initializes the OpenCV DNN model instance.
        Populates input and output details for the model.

        Args:
            model_path (str): Path to the ONNX model file.

        Raises:
            RuntimeError: If the model file path is invalid or loading fails.
        """
        OpencvRT.__inferencer = cv2.dnn.readNetFromONNX(model_path)
        
        OpencvRT.__load_input_details()
        OpencvRT.__load_output_details()

    @staticmethod
    def forward(input: np.ndarray) -> np.ndarray:
        """
        Performs inference on the given input tensor using the loaded OpenCV model.

        Args:
            input (np.ndarray): Preprocessed input data to be fed to the model, typically in the shape 
            (1, height, width, channels).

        Returns:
            np.ndarray: Model output tensor, optionally scaled based on output quantization parameters.

        Raises:
            ValueError: If inference is attempted without a loaded model.
        """
        if OpencvRT.__inferencer is None:
            raise ValueError("Model not loaded. Please call OpencvRT.load() before inference.")

        OpencvRT.__inferencer.setInput(input)
        output = OpencvRT.__inferencer.forward()
        
        output = np.array([cv2.transpose(output[0])])
        output = output.transpose(0, 2, 1)

        if OpencvRT.output_details["type"] != np.float32:
            output = (output.astype(np.float32) - OpencvRT.output_details["zero_point"]) * OpencvRT.output_details["scale"]

        return output

    @staticmethod
    def __load_input_details() -> None:
        """
        Loads the input details of the OpenCV model into the `input_details` dictionary.
        Extracted details include input name, data type, shape, mean, scale, and zero_point.

        Notes:
            Default values for mean and scale are set for normalization.
        """
        OpencvRT.input_details["name"] = OpencvRT.__inferencer.getLayerNames()[0]
        OpencvRT.input_details["type"] = np.float32
        OpencvRT.input_details["shape"] = np.array((1, 3, 416, 416))[[0, 2, 3, 1]]  # Adjust for (batch, height, width, channels)
        OpencvRT.input_details["mean"] = [0.485, 0.456, 0.406]
        OpencvRT.input_details["scale"] = 1.0
        OpencvRT.input_details["zero_point"] = 0.0

    @staticmethod
    def __load_output_details() -> None:
        """
        Loads the output details of the OpenCV model into the `output_details` dictionary.
        Extracted details include output name, data type, scale, and zero_point.
        """
        OpencvRT.output_details["name"] = OpencvRT.__inferencer.getUnconnectedOutLayersNames()[0]
        OpencvRT.output_details["type"] = np.float32
        OpencvRT.output_details["scale"] = 1.0
        OpencvRT.output_details["zero_point"] = 0.0
