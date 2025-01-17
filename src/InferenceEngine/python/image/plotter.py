import numpy as np
import cv2
from model.detection import Detection

class ImagePlotter:
    """
    A class for drawing detection boxes on an image with class labels and scores.

    Attributes
    ----------
    classes : List[str]
        A list of class names corresponding to detection class IDs.
    color_palette : np.ndarray
        An array of RGB color values used to represent each class visually.

    Methods
    -------
    draw_detections(image: np.ndarray, box: Tuple[int, int, int, int], score: float, class_id: int) -> None
        Draws a detection bounding box with the class label and confidence score on the image.
    """

    classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", 
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
        "toothbrush"
    ]

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detection: Detection
    ) -> None:
        """
        Draws a bounding box around the detected object with the class label and confidence score.
        
        The label box and text are scaled based on the bounding box dimensions for clarity.

        Parameters
        ----------
        image : np.ndarray
            The image on which to draw the bounding box and label.
        box : Tuple[int, int, int, int]
            A tuple containing the bounding box coordinates in the format (x, y, width, height).
        score : float
            Confidence score for the detected object, used in the label.
        class_id : int
            The ID of the detected object's class, which indexes the `classes` list and `color_palette`.

        Returns
        -------
        None
        """
        bbox = detection.get_bounding_box()
        class_id = detection.get_class_id()
        score = detection.get_score() * 100
                
        # Draw the bounding box
        cv2.rectangle(image, (bbox.xMin, bbox.yMin), (bbox.xMax, bbox.yMax), (64, 203, 255), 2)

        # Draw the label background
        label = f"{ImagePlotter.classes[class_id]}: {score:.0f}%"
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        text_width, _ = text_size
                
        cv2.rectangle(
            image,
            (bbox.xMin, (bbox.yMin - 15)),
            (bbox.xMin + text_width + 2, bbox.yMin),
            (64, 203, 255),
            cv2.FILLED,
        )
        
        # Draw the label text
        cv2.putText(
            image,
            label,
            ((bbox.xMin + 2), bbox.yMin),
            cv2.FONT_HERSHEY_DUPLEX,
            0.50,
            (0, 0, 0),
            1        
        )