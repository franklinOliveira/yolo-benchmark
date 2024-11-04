import numpy as np
import cv2
from typing import List, Tuple

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

    classes: List[str] = None
    color_palette: np.ndarray = None

    def __init__(self, classes: List[str]):
        """
        Initializes the ImagePlotter with a list of class names and generates a color palette
        for each class.

        Parameters
        ----------
        classes : List[str]
            A list of class names for the detected objects. Each class ID in a detection
            should correspond to an index in this list.
        """
        self.classes = classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        score: float,
        class_id: int
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
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        
        label = f"{self.classes[class_id]}: {score:.2f}"
        font_scale: float = max(0.5, min(1, w / 200))  
        thickness: int = max(1, int(w / 150))  
        
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_width = max(int(w), label_width)
        
        label_x: int = x1
        label_y: int = y1 - 10 if y1 - 10 > label_height else y1 + h + label_height + 10

        # Draw the bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Draw the label background
        cv2.rectangle(
            image,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y)),
            color,
            cv2.FILLED,
        )

        # Draw the label text
        cv2.putText(
            image,
            label,
            (int(label_x), int(label_y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )
