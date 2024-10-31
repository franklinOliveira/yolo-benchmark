import numpy as np
try:
    from tflite_runtime import interpreter as tflite
    from tflite_runtime.interpreter import load_delegate
    runtime_available: str = "tflite_runtime"
except ImportError:
    import tensorflow as tf
    runtime_available: str = "tensorflow"

class LiteRT:

    __inferencer: 'tflite.Interpreter' or 'tf.lite.Interpreter' = None

    input_details: dict = dict()
    output_details: dict = dict()

    @staticmethod
    def load(model_path: str) -> None:
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

        LiteRT.__inferencer.allocate_tensors()
        LiteRT.__load_input_details()
        LiteRT.__load_output_details()

    @staticmethod
    def forward(input: np.ndarray) -> np.ndarray:

        LiteRT.__inferencer.set_tensor(LiteRT.input_details["index"], input)
        LiteRT.__inferencer.invoke()

        output: np.ndarray = LiteRT.__inferencer.get_tensor(LiteRT.output_details["index"])
        if LiteRT.output_details["type"] != np.float32:
            output = (output.astype(np.float32) - LiteRT.output_details["zero_point"]) * LiteRT.output_details["scale"]

        output[:, [0, 2]] *= LiteRT.input_details["shape"][1]  
        output[:, [1, 3]] *= LiteRT.input_details["shape"][2]  

        return output
    
    @staticmethod
    def __load_input_details() -> None:
        litert_input_details: list = LiteRT.__inferencer.get_input_details()
        LiteRT.input_details["index"] = litert_input_details[0]["index"]
        LiteRT.input_details["type"] = litert_input_details[0]["dtype"]
        LiteRT.input_details["shape"] = litert_input_details[0]["shape"]
        LiteRT.input_details["scale"] = litert_input_details[0]["quantization"][0]
        LiteRT.input_details["zero_point"] = litert_input_details[0]["quantization"][1]

    @staticmethod
    def __load_output_details() -> None:
        litert_output_details: list = LiteRT.__inferencer.get_output_details()
        LiteRT.output_details["index"] = litert_output_details[0]["index"]
        LiteRT.output_details["type"] = litert_output_details[0]["dtype"]
        LiteRT.output_details["scale"] = litert_output_details[0]["quantization"][0]
        LiteRT.output_details["zero_point"] = litert_output_details[0]["quantization"][1]
