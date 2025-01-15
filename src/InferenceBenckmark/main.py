import argparse
import os
from time import sleep

from monitor.performancemetrics import PerformanceMetrics

def start_benchmarking(images_folder: str, model_path: str, half_cores: bool, language: str):
    inferencer_cmd: str = ""
    if language == "python":
        inferencer_cmd = "python3 /home/pi/yolo-benchmark/src/InferenceEngine/python/main.py"
        inferencer_cmd += f" --images_folder {images_folder}"
        inferencer_cmd += f" --model_path {model_path}"
        if half_cores:
            inferencer_cmd += " --half_cores"

    elif language == "cpp":
        pass
    os.system(f"{inferencer_cmd} &")

    print("Starting benchmarking")
    
    PerformanceMetrics.init()
    while True:
        PerformanceMetrics.update()
        sleep(0.1)
        if not PerformanceMetrics.is_active() and PerformanceMetrics.get_metrics() is not None:
            break
        
    print("Finishing benchmarking")

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
        help="Path to the YOLO model file."
    )
    parser.add_argument(
        "--half_cores",
        action="store_true",
        help="Use the half or full number of CPU cores."
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Python or C++ inferencer."
    )

    args = parser.parse_args()

    start_benchmarking(
        images_folder=args.images_folder, 
        model_path=args.model_path, 
        half_cores=args.half_cores,
        language=args.language
    )