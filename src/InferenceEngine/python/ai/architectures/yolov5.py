import numpy as np
import cv2
from typing import Tuple, List, Dict
from image.preprocessing import ImagePreprocessing
from detection.postprocessing import DetectionPostprocessing

class YOLOv5:
    """
    YOLOv5 is a wrapper class designed for object detection using a Ultralytics YOLO-based model.
    It includes methods for preprocessing images before inference and postprocessing
    the model output to extract bounding boxes, class IDs, and confidence scores.

    Attributes:
        input_details (Dict[str, any]): Configuration details of the model input, 
            containing keys like 'shape', 'mean', 'scale', 'zero_point', and 'type'.
        score_thresh (float): Threshold for filtering boxes based on score.
        confidence_thresh (float): Minimum confidence threshold for detections.
        iou_thresh (float): Intersection over Union (IoU) threshold used for Non-Maximum Suppression (NMS).
    """

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
        """
        Initializes the YOLOv5 class with model input details and thresholds.

        Args:
            input_details (Dict[str, any]): Model input configuration, including shape, mean,
                scale, zero point, and data type (e.g., np.float32).
            score_thresh (float): Score threshold for filtering detections.
            confidence_thresh (float): Minimum confidence threshold for detected objects.
            iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS).
        """
        self.input_details = input_details
        self.score_thresh = score_thresh
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh

    def pre_process(
        self, image: np.ndarray, litert_model: bool = True, opencvrt_inferencer: bool = False
    ) -> np.ndarray:
        """
        Preprocesses an input image for YOLO inference based on the input model configuration.

        Args:
            image (np.ndarray): Input image array in BGR format.
            litert_model (bool, optional): Indicates if the model uses TensorFlow Lite runtime.
                Defaults to True.
            opencvrt_inferencer (bool, optional): If True, preprocesses the image using OpenCV
                DNN module. Defaults to False.

        Returns:
            np.ndarray: Preprocessed image ready for model inference.
        """
        if opencvrt_inferencer:
            input_image = image
            input_image = cv2.dnn.blobFromImage(
                input_image, 
                scalefactor=1.0 / 255.0, 
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
        """
        Postprocesses model output to extract bounding boxes, confidence scores, and class IDs.

        Args:
            output (np.ndarray): Model output, usually containing bounding box predictions
                and class scores for each detected object.
            image (np.ndarray): Original input image, used to calculate scaling factors.

        Returns:
            Tuple[List[np.ndarray], List[int], List[float]]:
                - List of bounding boxes, where each box is represented as an np.ndarray.
                - List of integer class IDs corresponding to detected objects.
                - List of confidence scores for each detected object.
        """
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
