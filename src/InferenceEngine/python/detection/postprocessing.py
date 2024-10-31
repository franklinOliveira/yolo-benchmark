import cv2
import numpy as np

class DetectionPostprocessing:

    @staticmethod
    def apply_nms(boxes, scores, classes_ids, input_factor, confidence_thresh, iou_thresh, score_thresh):
        final_boxes: list[np.ndarray] = []
        final_scores: list[float] = []
        final_classes_ids: list[int] = []

        indices: np.ndarray = cv2.dnn.NMSBoxes(
            boxes, scores, confidence_thresh, iou_thresh
        )

        if len(indices) > 0:
            for i in indices.flatten():
                box: np.ndarray = boxes[i]
                box[0] = int(
                    (box[0] - 0.5 * box[2]) * input_factor[1]
                )  
                box[1] = int(
                    (box[1] - 0.5 * box[3]) * input_factor[0]
                )  
                box[2] = int(box[2] * input_factor[1])  
                box[3] = int(box[3] * input_factor[0]) 
                
                score: float = scores[i]
                class_id: int = classes_ids[i]

                if score > score_thresh:
                    final_boxes.append(box)
                    final_classes_ids.append(class_id)
                    final_scores.append(score)

        return final_boxes, final_classes_ids, final_scores
    
    @staticmethod
    def __revert_letterbox(boxes, image_shape, input_shape):
        reverted_boxes = list()

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