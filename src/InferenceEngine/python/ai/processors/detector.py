from ai.inferencers.litert import LiteRT
from ai.inferencers.onnxrt import OnnxRT
from ai.inferencers.opencvrt import OpencvRT
from ai.architectures.yolov5 import YOLOv5
from ai.architectures.yolov8 import YOLOv8
from ai.architectures.yolo11 import YOLO11
import numpy as np
import time

class Detector:
    """
    The Detector class serves as an interface for initializing, processing, and running inference on YOLO models.
    It supports multiple inference backends, including LiteRT, OnnxRT, and OpenCVRT, and can use various YOLO architectures
    (YOLOv5, YOLOv8, YOLO11). Detector tracks the time taken for preprocessing, inference, and postprocessing.

    Attributes:
        __architecture_format (str): 
            Specifies the architecture format ('litert' or 'onnx') for the current model.
        __architecture (YOLO11, YOLOv8, or YOLOv5): 
            An instance of the appropriate YOLO architecture used for the current model.
        pre_process_time (int): 
            Duration in milliseconds taken for preprocessing.
        inference_time (int): 
            Duration in milliseconds taken for inference.
        post_process_time (int): 
            Duration in milliseconds taken for postprocessing.
    """

    __architecture_format: str
    __architecture: 'YOLO11' or 'YOLOv8' or 'YOLOv5'

    pre_process_time: int
    inference_time: int
    post_process_time: int

    @staticmethod
    def init(
        model_path: str,
        score_thresh: float,
        confidence_thresh: float,
        iou_thresh: float,
        onnx_inferencer: str
    ):
        """
        Initializes the Detector with the model path, thresholds, and specified inference backend.
        Configures the architecture and input details.

        Args:
            model_path (str): Path to the model file (.onnx or .tflite).
            score_thresh (float): Score threshold for the YOLO model.
            confidence_thresh (float): Confidence threshold for filtering detections.
            iou_thresh (float): Intersection-over-Union threshold for Non-Maximum Suppression.
            onnx_inferencer (str): ONNX inference engine to use ('onnxrt' or 'opencvrt').

        Raises:
            ValueError: If an invalid model file extension is provided.
        """
        input_details = Detector.__start_inferencer(
            model_path=model_path,
            onnx_inferencer=onnx_inferencer
        )

        Detector.__load_architecture(
            model_path=model_path,
            input_details=input_details,
            score_thresh=score_thresh,
            confidence_thresh=confidence_thresh,
            iou_thresh=iou_thresh
        )

    @staticmethod
    def run(image):
        """
        Runs inference on a given image, returning detected bounding boxes, class IDs, and scores.
        Times each step of the detection process.

        Args:
            image (np.ndarray): Input image to be processed.

        Returns:
            tuple: Contains lists of bounding boxes (np.ndarray), class IDs (int), and scores (float).
        """
        boxes: list[np.ndarray]
        classes_ids: list[int]
        scores: list[float]

        input: np.ndarray
        output: np.ndarray

        # Preprocess step
        start_ts = time.time()
        if Detector.__architecture_format == "litert":
            input = Detector.__architecture.pre_process(image=image, litert_model=True)
        elif Detector.__architecture_format == "onnx":
            if Detector.__onnx_inferencer == "onnxrt":
                input = Detector.__architecture.pre_process(image=image, litert_model=False)
            elif Detector.__onnx_inferencer == "opencvrt":
                input = Detector.__architecture.pre_process(image=image, litert_model=False, opencvrt_inferencer=True)
        Detector.pre_process_time = int((time.time() - start_ts) * 1000)

        # Inference step
        start_ts = time.time()
        if Detector.__architecture_format == "litert":
            output = LiteRT.forward(input=input)
        elif Detector.__architecture_format == "onnx":
            if Detector.__onnx_inferencer == "onnxrt":
                output = OnnxRT.forward(input=input)
            elif Detector.__onnx_inferencer == "opencvrt":
                output = OpencvRT.forward(input=input)
        Detector.inference_time = int((time.time() - start_ts) * 1000)

        # Postprocess step
        start_ts = time.time()
        boxes, classes_ids, scores = Detector.__architecture.post_process(output=output, image=image)
        Detector.post_process_time = int((time.time() - start_ts) * 1000)

        return boxes, classes_ids, scores

    @staticmethod
    def __start_inferencer(model_path: str, onnx_inferencer: str) -> dict:
        """
        Initializes the inference backend based on the model file type (.tflite or .onnx).
        Loads the model using the specified inferencer and extracts input details.

        Args:
            model_path (str): Path to the model file.
            onnx_inferencer (str): ONNX inference engine ('onnxrt' or 'opencvrt').

        Returns:
            dict: Dictionary containing input details of the loaded model.
        """
        input_details: dict = dict()

        if ".tflite" in model_path:
            LiteRT.load(model_path=model_path)
            Detector.__architecture_format = "litert"
            input_details = LiteRT.input_details
            
        elif ".onnx" in model_path:
            Detector.__architecture_format = "onnx"
            Detector.__onnx_inferencer = onnx_inferencer

            if onnx_inferencer == "onnxrt":
                OnnxRT.load(model_path=model_path)
                input_details = OnnxRT.input_details
            if onnx_inferencer == "opencvrt":
                OpencvRT.load(model_path=model_path)
                input_details = OpencvRT.input_details

        return input_details
    
    @staticmethod
    def __load_architecture(model_path: str, input_details: dict, score_thresh: float, confidence_thresh: float, iou_thresh: float):
        """
        Loads the appropriate YOLO architecture based on the model filename, configuring it with thresholds.

        Args:
            model_path (str): Path to the model file.
            input_details (dict): Input details extracted from the inference backend.
            score_thresh (float): Score threshold for detections.
            confidence_thresh (float): Confidence threshold for filtering.
            iou_thresh (float): IoU threshold for Non-Maximum Suppression.
        """
        if "yolo11" in model_path:
            Detector.__architecture = YOLO11(
                input_details=input_details,
                score_thresh=score_thresh,
                confidence_thresh=confidence_thresh,
                iou_thresh=iou_thresh,
            )
        elif "yolov8" in model_path:
            Detector.__architecture = YOLOv8(
                input_details=input_details,
                score_thresh=score_thresh,
                confidence_thresh=confidence_thresh,
                iou_thresh=iou_thresh,
            )
        elif "yolov5" in model_path:
            Detector.__architecture = YOLOv5(
                input_details=input_details,
                score_thresh=score_thresh,
                confidence_thresh=confidence_thresh,
                iou_thresh=iou_thresh,
            )
