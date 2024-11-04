
import os
import cv2
import argparse

from ai.processors.detector import Detector
from image.plotter import ImagePlotter

COCO_CLASSES = [
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

def process_images(images_folder: str, model_path: str, output_folder: str, onnx_inferencer: str):
    Detector.init(
        model_path=model_path,
        score_thresh=0.25,
        confidence_thresh=0.5,
        iou_thresh=0.5,
        onnx_inferencer=onnx_inferencer
    )
    
    plotter = ImagePlotter(classes=COCO_CLASSES)
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    os.makedirs(output_folder, exist_ok=True)

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)

        boxes, classes_ids, scores = Detector.run(image=image)

        duration = Detector.pre_process_time + Detector.inference_time + Detector.post_process_time
        print(f"Image {image_file} processed in {duration}ms")
        print(f"  Pre processing in {Detector.pre_process_time}ms")
        print(f"  Inference in {Detector.inference_time}ms")
        print(f"  Post processing in {Detector.post_process_time}ms\n")

        for idx in range(len(boxes)):
            plotter.draw_detections(
                image=image,
                box=boxes[idx],
                score=scores[idx],
                class_id=classes_ids[idx]
            )

        output_path = os.path.join(output_folder, f"output_{image_file}")
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with YOLOv8 model.")
    parser.add_argument(
        "--images_folder",
        type=str,
        required=True,
        help="Path to the folder containing images."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the YOLOv8 model file."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/output",
        help="Path to the folder where output images will be saved."
    )

    parser.add_argument(
        "--onnx_inferencer",
        type=str,
        default="opencvrt",
        help="If model is ONNX, choose the RT (onnxrt or opencvrt)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    process_images(
        images_folder=args.images_folder, 
        model_path=args.model_path, 
        output_folder=args.output_folder,
        onnx_inferencer=args.onnx_inferencer
    )