import numpy as np
import multiprocessing

try:
    from tflite_runtime import interpreter as tflite
    from tflite_runtime.interpreter import load_delegate
    runtime_available: str = "tflite_runtime"
except ImportError:
    import tensorflow as tf
    runtime_available: str = "tensorflow"

class LiteRT:
    """
    LiteRT provides a runtime interface for loading and running TensorFlow Lite models.
    This class supports both the TensorFlow Lite runtime (tflite_runtime) and the full 
    TensorFlow library, with optional Edge TPU delegate loading for hardware acceleration.

    Attributes:
        __inferencer ('tflite.Interpreter' or 'tf.lite.Interpreter', optional): 
            Private attribute holding the interpreter instance for model inference.
        input_details (dict): Dictionary containing model input details, such as 
            index, type, shape, scale, and zero_point.
        output_details (dict): Dictionary containing model output details, including 
            index, type, scale, and zero_point.
    """

    __inferencer: 'tflite.Interpreter' or 'tf.lite.Interpreter' = None
    input_details: dict = dict()
    output_details: dict = dict()

    @staticmethod
    def load(model_path: str) -> None:
        """
        Loads a TensorFlow Lite model from the specified path. If the model file name 
        contains 'edgetpu', the Edge TPU delegate is loaded for hardware acceleration.

        Args:
            model_path (str): Path to the TensorFlow Lite model file.

        Raises:
            RuntimeError: If the model file path is invalid or the interpreter fails to load.
        """
        if runtime_available == "tflite_runtime":
            if "edgetpu" in model_path:
                LiteRT.__inferencer = tflite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[load_delegate('libedgetpu.so.1')]
                )
            else:
                LiteRT.__inferencer = tflite.Interpreter(model_path=model_path)
        else:
            LiteRT.__inferencer = tf.lite.Interpreter(model_path=model_path)

        LiteRT.__inferencer.SetNumThreads(multiprocessing.cpu_count())
        LiteRT.__inferencer.allocate_tensors()
        LiteRT.__load_input_details()
        LiteRT.__load_output_details()

    @staticmethod
    def forward(input: np.ndarray) -> np.ndarray:
        """
        Performs inference on the given input tensor using the loaded TensorFlow Lite model.
        
        Args:
            input (np.ndarray): Preprocessed input data to be fed to the model.

        Returns:
            np.ndarray: Model output tensor, optionally scaled based on output quantization parameters.

        Raises:
            ValueError: If inference is attempted without a loaded model.
        """
        if LiteRT.__inferencer is None:
            raise ValueError("Model not loaded. Please call LiteRT.load() before inference.")

        LiteRT.__inferencer.set_tensor(LiteRT.input_details["index"], input)
        LiteRT.__inferencer.invoke()

        output: np.ndarray = LiteRT.__inferencer.get_tensor(LiteRT.output_details["index"])
        if LiteRT.output_details["type"] != np.float32:
            output = (output.astype(np.float32) - LiteRT.output_details["zero_point"]) * LiteRT.output_details["scale"]

        output[:, [0, 2]] *= LiteRT.input_details["shape"][1]  # Scale x-coordinates
        output[:, [1, 3]] *= LiteRT.input_details["shape"][2]  # Scale y-coordinates

        return output

    @staticmethod
    def __load_input_details() -> None:
        """
        Loads the input details of the TensorFlow Lite model into `input_details` dictionary.
        Extracted details include index, data type, shape, scale, and zero_point.
        """
        litert_input_details: list = LiteRT.__inferencer.get_input_details()
        LiteRT.input_details["index"] = litert_input_details[0]["index"]
        LiteRT.input_details["type"] = litert_input_details[0]["dtype"]
        LiteRT.input_details["shape"] = litert_input_details[0]["shape"]
        LiteRT.input_details["scale"] = litert_input_details[0]["quantization"][0]
        LiteRT.input_details["zero_point"] = litert_input_details[0]["quantization"][1]

    @staticmethod
    def __load_output_details() -> None:
        """
        Loads the output details of the TensorFlow Lite model into `output_details` dictionary.
        Extracted details include index, data type, scale, and zero_point.
        """
        litert_output_details: list = LiteRT.__inferencer.get_output_details()
        LiteRT.output_details["index"] = litert_output_details[0]["index"]
        LiteRT.output_details["type"] = litert_output_details[0]["dtype"]
        LiteRT.output_details["scale"] = litert_output_details[0]["quantization"][0]
        LiteRT.output_details["zero_point"] = litert_output_details[0]["quantization"][1]
