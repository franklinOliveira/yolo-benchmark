import argparse
import os
from time import sleep
from datetime import datetime
from report.table import generate_table, export_table

from monitor.performancemetrics import PerformanceMetrics
from monitor.consumptionmetrics import ConsumptionMetrics

def start_benchmarking(images_folder: str, model_path: str, half_cores: bool, language: str):
    '''
    STAGE 1: Benchmark activation
    '''
    with open("/proc/device-tree/model", "r") as file:
        board_name = file.read().strip().replace("\x00", "")
        
    model_name = model_path.split("/")[-1].replace('.', '_').split("_")
    current_datetime = datetime.now()
    current_datetime = current_datetime.strftime("%Y%m%dT%H%M%S")
    experiment_specs = {
        "datetime": current_datetime,
        "board": board_name,
        "architecture": model_name[0],
        "type": model_name[1],
        "format": model_name[2],
        "cores": "half" if half_cores else "full",
        "language": language
    }
    
    print(
        f"[INF. BENCHMARK] Experiment {experiment_specs['datetime']} started\n"
        f"    Board specs\n"
        f"         Name: {experiment_specs['board']}\n"
        f"    Model specs\n"
        f"         Architecture: {experiment_specs['architecture']}\n"
        f"         Data type: {experiment_specs['type']}\n"
        f"         Format: {experiment_specs['format']}\n"
        f"    Inference specs\n"
        f"         Language: {experiment_specs['language']}\n"
        f"         CPU cores: {experiment_specs['cores']}"
    )
    
    output_path = f"/home/pi/yolo-benchmark/data/output/{experiment_specs['board'].replace(' ', '_')}/"
    output_path += f"{experiment_specs['architecture']}_{experiment_specs['type']}_"
    output_path += f"{experiment_specs['format']}_{experiment_specs['language']}_"
    output_path += f"{experiment_specs['cores']}/{experiment_specs['datetime']}"
    os.makedirs(output_path, exist_ok=True)
    
    PerformanceMetrics.init()
    ConsumptionMetrics.init()
    
    '''
    STAGE 2: Inference engine activation
    '''
    inferencer_cmd: str = ""
    if language == "python":
        inferencer_cmd = "python3 /home/pi/yolo-benchmark/src/InferenceEngine/python/main.py"
        inferencer_cmd += f" --images_folder {images_folder}"
        inferencer_cmd += f" --model_path {model_path}"
        if half_cores:
            inferencer_cmd += " --half_cores"
        inferencer_cmd += f" --output {output_path}/detections"

    elif language == "cpp":
        inferencer_cmd = "cd /home/pi/yolo-benchmark/src/InferenceEngine/cpp/ && sudo ./yolo_benchmark"
        inferencer_cmd += f" {images_folder} {model_path}"
        if half_cores:
            inferencer_cmd += " half"
        else:
            inferencer_cmd += " full"
        inferencer_cmd += f" {output_path}/detections"
    
    os.system(f"{inferencer_cmd} &")
    
    '''
    STAGE 3: Benchmark monitoring
    '''
    while True:
        ConsumptionMetrics.update()
        PerformanceMetrics.update()
        sleep(0.1)
        if not PerformanceMetrics.is_active() and PerformanceMetrics.get_measures() is not None:
            break
    
    min_current_consumption = int(input(f"[INF. BENCHMARK] Enter with the minimum current consumption in mA: "))
    max_current_consumption = int(input(f"[INF. BENCHMARK] Enter with the maximum current consumption in mA: "))
    ConsumptionMetrics.compute_current_levels(min_current=min_current_consumption, max_current=max_current_consumption)
    
    '''
    STAGE 4: Report generation
    '''
    print("[INF. BENCHMARK] Generating the report")
    pre_processing_times, inference_times, post_processing_times = PerformanceMetrics.get_measures()
    cpu_usage_levels, cpu_temperature_levels, ram_usage_levels, current_usage_levels = ConsumptionMetrics.get_measures()
    performance_table = generate_table(
        fields_names=["Sample", "Preprocessing time (ms)", "Inference time (ms)", "Post processing time (ms)"],
        rows=zip(pre_processing_times, inference_times, post_processing_times),
    )
    consumption_table = generate_table(
        fields_names=["Sample", "CPU usage (%)", "CPU temperature (°C)", "RAM usage (MB)" , "Current consumption (mA)"],
        rows=zip(cpu_usage_levels, cpu_temperature_levels, ram_usage_levels, current_usage_levels),
    )
    print("\n######################## PERFORMANCE METRICS ########################")
    print(performance_table)
    export_table(
        table=performance_table, 
        file_path=f"{output_path}/performance.csv"
    )
    print("\n########################  CONSUMPTION METRICS  ########################")
    print(consumption_table)
    export_table(
        table=consumption_table, 
        file_path=f"{output_path}/consumption.csv"
    )
    print(f"[INF. BENCHMARK] Experiment {experiment_specs['datetime']} finished")

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