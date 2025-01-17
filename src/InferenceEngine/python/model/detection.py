from dataclasses import dataclass

@dataclass
class BoundingBox:
    xMin: int
    yMin: int
    xMax: int
    yMax: int

class Detection:
    def __init__(self, class_id: int, score: float, locations: list[int]):
        self.class_id = class_id
        self.score = score
        self.bbox = BoundingBox(
            xMin=int(locations[0]),
            yMin=int(locations[1]),
            xMax=int(locations[2]),
            yMax=int(locations[3])
        )

    def get_class_id(self) -> int:
        return self.class_id

    def get_score(self) -> float:
        return self.score

    def get_bounding_box(self) -> BoundingBox:
        return self.bbox