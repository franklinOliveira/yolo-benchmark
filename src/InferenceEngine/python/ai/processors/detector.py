from ai.inferencers.litert import LiteRT
from ai.architectures.yolov5 import YOLOv5
from ai.architectures.yolov8 import YOLOv8
from ai.architectures.yolo11 import YOLO11
import numpy as np
import time

class Detector:

    __architecture_type: str
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
    ):
        input_details: dict = Detector.__start_inferencer(model_path=model_path)

        Detector.__load_architecture(
            model_path=model_path,
            input_details=input_details,
            score_thresh=score_thresh,
            confidence_thresh=confidence_thresh,
            iou_thresh=iou_thresh
        )

    @staticmethod
    def run(image):
        boxes: list[np.ndarray]
        classes_ids: list[int]
        scores: list[float]

        input: np.ndarray
        output: np.ndarray

        start_ts = time.time()
        input = Detector.__architecture.pre_process(image=image)
        Detector.pre_process_time = int((time.time() - start_ts) * 1000)
        
        start_ts = time.time()
        if Detector.__architecture_type == "tflite":
            output = LiteRT.forward(input=input)

        elif Detector.__architecture_type == "onnx":
            pass
        Detector.inference_time = int((time.time() - start_ts) * 1000)

        start_ts = time.time()
        boxes, classes_ids, scores = Detector.__architecture.post_process(
            output=output, 
            image=image
        )
        Detector.post_process_time = int((time.time() - start_ts) * 1000)

        return boxes, classes_ids, scores

    @staticmethod
    def __start_inferencer(model_path: str) -> dict:
        input_details: dict = dict()

        if ".tflite" in model_path:
            LiteRT.load(model_path=model_path)
            Detector.__architecture_type = "tflite"
            input_details = LiteRT.input_details
            

        elif ".onnx" in model_path:
            pass

        return input_details
    
    @staticmethod
    def __load_architecture(model_path: str, input_details: dict, score_thresh: float, confidence_thresh: float, iou_thresh: float):
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