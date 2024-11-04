import cv2
import numpy as np

class OpencvRT:
    
    __inferencer: cv2.dnn_Net = None
    input_details: dict = dict()
    output_details: dict = dict()

    @staticmethod
    def load(model_path: str) -> None:
        OpencvRT.__inferencer = cv2.dnn.readNetFromONNX(model_path)
        OpencvRT.__load_input_details()
        OpencvRT.__load_output_details()

    @staticmethod
    def forward(input: np.ndarray) -> np.ndarray:
        OpencvRT.__inferencer.setInput(input)
        output = OpencvRT.__inferencer.forward()
        output = np.array([cv2.transpose(output[0])])
        output = output.transpose(0, 2, 1)
        
        if OpencvRT.output_details["type"] != np.float32:
            output = (output.astype(np.float32) - OpencvRT.output_details["zero_point"]) * OpencvRT.output_details["scale"]

        return output

    @staticmethod
    def __load_input_details() -> None:
        OpencvRT.input_details["name"] = OpencvRT.__inferencer.getLayerNames()[0]
        OpencvRT.input_details["type"] = np.float32
        OpencvRT.input_details["shape"] = np.array((1, 3, 416, 416))[[0, 2, 3, 1]]
        OpencvRT.input_details["mean"] = [0.485, 0.456, 0.406]
        OpencvRT.input_details["scale"] = 1.0
        OpencvRT.input_details["zero_point"] = 0.0

    @staticmethod
    def __load_output_details() -> None:
        OpencvRT.output_details["name"] = OpencvRT.__inferencer.getUnconnectedOutLayersNames()[0]
        OpencvRT.output_details["type"] = np.float32
        OpencvRT.output_details["scale"] = 1.0
        OpencvRT.output_details["zero_point"] = 0.0
