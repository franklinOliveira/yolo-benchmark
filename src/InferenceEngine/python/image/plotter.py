import numpy as np
import cv2

class ImagePlotter:
    
    classes: list[str] = None
    color_palette: np.ndarray = None

    def __init__(self, classes: list[str]):
        self.classes = classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, image: np.ndarray, box: tuple[int, int, int, int], score: float, class_id: int) -> None:
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        
        label = f"{self.classes[class_id]}: {score:.2f}"
        font_scale: float = max(0.5, min(1, w / 200))  
        thickness: int = max(1, int(w / 150))  
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_width = max(int(w), label_width)
        label_x: int = x1
        label_y: int = y1 - 10 if y1 - 10 > label_height else y1 + h + label_height + 10

        cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        cv2.rectangle(
            image,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y)),
            color,
            cv2.FILLED,
        )
        cv2.putText(image, label, (int(label_x), int(label_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
