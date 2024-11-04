import numpy as np
import cv2
from typing import Tuple, List, Dict
from image.preprocessing import ImagePreprocessing
from detection.postprocessing import DetectionPostprocessing


class YOLOv5:
    input_details: Dict[str, any]
    score_thresh: float
    confidence_thresh: float
    iou_thresh: float

    def __init__(
        self,
        input_details: Dict[str, any],
        score_thresh: float,
        confidence_thresh: float,
        iou_thresh: float,
    ):

        self.input_details = input_details
        self.score_thresh = score_thresh
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh

    def pre_process(self, image: np.ndarray, litert_model: bool = True, opencvrt_inferencer: bool = False) -> np.ndarray:
        
        if opencvrt_inferencer:
            input_image = image
            input_image = cv2.dnn.blobFromImage(
                input_image, 
                scalefactor=1.0/255.0, 
                size=self.input_details["shape"][1:3], 
                mean=self.input_details["mean"], 
                swapRB=True, 
                crop=False
            )

        else:
            input_image: np.ndarray = ImagePreprocessing.format(
                images=[image], 
                input_shape=self.input_details["shape"][1:3],
                litert_model=litert_model
            )

        if self.input_details["type"] != np.float32:
            input_image = ImagePreprocessing.quantize(
                input=input_image,
                scale=self.input_details["scale"],
                zero_point=self.input_details["zero_point"],
                type=self.input_details["type"],
            )

        return input_image

    def post_process(
        self, output: np.ndarray, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[int], List[float]]:
        
        output: List[np.ndarray] = [np.transpose(pred) for pred in output]

        input_factor: Tuple[float, float] = (
            image.shape[0] / self.input_details["shape"][1],
            image.shape[1] / self.input_details["shape"][2],
        )

        boxes: List[np.ndarray] = []
        scores: List[float] = []
        classes_ids: List[int] = []

        for pred in output:
            x: np.ndarray = pred[:, 0]
            y: np.ndarray = pred[:, 1]
            w: np.ndarray = pred[:, 2]
            h: np.ndarray = pred[:, 3]
            boxes.extend(np.column_stack([x, y, w, h]))

            idx: np.ndarray = np.argmax(pred[:, 4:], axis=1)
            scores.extend(pred[np.arange(pred.shape[0]), idx + 4])
            classes_ids.extend(idx)

        boxes, scores, classes_ids = DetectionPostprocessing.apply_nms(
            boxes=boxes,
            scores=scores,
            classes_ids=classes_ids,
            input_factor=input_factor,
            confidence_thresh=self.confidence_thresh,
            iou_thresh=self.iou_thresh,
            score_thresh=self.score_thresh,
        )

        return boxes, scores, classes_ids