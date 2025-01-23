
import os
import cv2
import argparse
from tqdm import tqdm 

from ai.processors.detector import Detector
from image.plotter import ImagePlotter
from interface.mqttproducer import MQTTProducer

def start_inferencing(images_folder: str, model_path: str, half_cores: bool, output_folder: str):
    '''
    STAGE 1: Inference engine setup
    '''
    Detector.init(
        model_path=model_path,
        score_thresh=0.25,
        confidence_thresh=0.5,
        iou_thresh=0.5,
        half_cores=half_cores
    )

    mqtt_producer = MQTTProducer(
        server={
            "address": "localhost",
            "port": 1883
        },
        client_id="inference"
    )
    mqtt_producer.start()
    
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    os.makedirs(output_folder, exist_ok=True)

    '''
    STAGE 2: Continuous inferencing
    '''
    mqtt_producer.produce(
        topic="inferenceEngine/status",
        msg={"active": True}
    )
    
    for image_file in tqdm(image_files, desc="[INF. ENGINE] Inferencing images "):
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)

        detections = Detector.run(image=image)        
        mqtt_producer.produce(
            topic="inferenceEngine/data",
            msg={
                "pre_processing_time": Detector.pre_process_time,
                "inference_time": Detector.inference_time,
                "post_processing_time": Detector.post_process_time,
            }
        )
        
        for detection in detections:
            ImagePlotter.draw_detections(
                image=image,
                detection=detection
            )

        #output_path = os.path.join(output_folder, image_file)
        #cv2.imwrite(output_path, image)
    
    '''
    STAGE 3: Stop inferencing and alerting
    '''
    mqtt_producer.produce(
        topic="inferenceEngine/status",
        msg={"active": False}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--half_cores",
        action="store_true",
        help="Use the half or full number of CPU cores."
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/output",
        help="Path to the folder where output images will be saved."
    )    
    
    # Parse arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    start_inferencing(
        images_folder=args.images_folder, 
        model_path=args.model_path, 
        half_cores=args.half_cores,
        output_folder=args.output_folder
    )