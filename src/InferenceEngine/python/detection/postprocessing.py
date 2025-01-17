import cv2
import numpy as np
from typing import List, Tuple
from model.detection import Detection

class DetectionPostprocessing:
    """
    A class for post-processing detection results, including applying Non-Maximum Suppression (NMS)
    and reverting letterbox transformations on bounding boxes.

    Methods
    -------
    apply_nms(boxes: List[np.ndarray], scores: List[float], classes_ids: List[int],
              input_factor: Tuple[float, float], confidence_thresh: float,
              iou_thresh: float, score_thresh: float) -> Tuple[List[np.ndarray], List[int], List[float]]:
        Applies NMS to filter out overlapping bounding boxes and scales the bounding boxes
        to the input dimensions.
        
    __revert_letterbox(boxes: List[np.ndarray], image_shape: Tuple[int, int],
                       input_shape: Tuple[int, int]) -> List[np.ndarray]:
        Adjusts bounding boxes to revert any letterboxing applied, restoring their
        coordinates to the original image dimensions.
    """

    @staticmethod
    def apply_nms(
        boxes: List[np.ndarray],
        scores: List[float],
        classes_ids: List[int],
        input_factor: Tuple[float, float],
        confidence_thresh: float,
        iou_thresh: float,
        score_thresh: float
    ) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """
        Applies Non-Maximum Suppression (NMS) on detection boxes to remove overlapping boxes
        with lower confidence scores, and rescales boxes based on the provided input factor.

        Parameters
        ----------
        boxes : List[np.ndarray]
            A list of bounding boxes in the format [x_center, y_center, width, height].
        scores : List[float]
            Confidence scores for each bounding box.
        classes_ids : List[int]
            Class IDs associated with each bounding box.
        input_factor : Tuple[float, float]
            Scaling factor for width and height to match the original image size.
        confidence_thresh : float
            Minimum confidence score for a box to be retained.
        iou_thresh : float
            IoU threshold for NMS; boxes with IoU > iou_thresh are suppressed.
        score_thresh : float
            Minimum score threshold; boxes with scores below this value are removed.

        Returns
        -------
        List[Detection]
            List of computed detections with bbox, score and class ID.
        """
        detections: List[Detection] = []
        indices: np.ndarray = cv2.dnn.NMSBoxes(
            boxes, scores, confidence_thresh, iou_thresh
        )

        if len(indices) > 0:
            for i in indices.flatten():
                box: np.ndarray = boxes[i]
                box[0] = int((box[0] - 0.5 * box[2]) * input_factor[1])  
                box[1] = int((box[1] - 0.5 * box[3]) * input_factor[0])  
                box[2] = box[0] + int(box[2] * input_factor[1])  
                box[3] = box[1] + int(box[3] * input_factor[0]) 
                
                score: float = scores[i]
                class_id: int = classes_ids[i]

                if score > score_thresh:
                    detection = Detection(class_id=class_id, locations=box, score=score)
                    detections.append(detection)
                    
        return detections
    
    @staticmethod
    def __revert_letterbox(
        boxes: List[np.ndarray],
        image_shape: Tuple[int, int],
        input_shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """
        Reverts the letterboxing effect on bounding boxes, mapping them back to the
        original image dimensions.

        Parameters
        ----------
        boxes : List[np.ndarray]
            List of bounding boxes in the format [x_center, y_center, width, height].
        image_shape : Tuple[int, int]
            Height and width of the original image.
        input_shape : Tuple[int, int]
            Height and width of the input image used by the model.

        Returns
        -------
        List[np.ndarray]
            A list of bounding boxes with coordinates adjusted to match the original
            image dimensions.
        """
        reverted_boxes: List[np.ndarray] = []

        for box in boxes:
            gain = min(image_shape[0] / input_shape[0], image_shape[1] / input_shape[1])
            pad = (
                round((image_shape[0] - input_shape[0] * gain) / 2 - 0.1),
                round((image_shape[1] - input_shape[1] * gain) / 2 - 0.1),
            )
            box[0] = (box[0] - pad[0]) / gain
            box[1] = (box[1] - pad[1]) / gain
            box[2] = box[2] / gain
            box[3] = box[3] / gain

            reverted_boxes.append(box)

        return reverted_boxes
