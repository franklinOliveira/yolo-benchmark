from interface.mqttconsumer import MQTTConsumer
import json

class PerformanceMetrics:
    __is_active: bool = False
    __mqtt_consumer: MQTTConsumer = None

    __pre_process_times: list = list()
    __inference_times: list = list()
    __post_process_times: list = list()
    
    @staticmethod
    def init() -> None:
        PerformanceMetrics.__mqtt_consumer = MQTTConsumer(
            server={
                "address": "localhost",
                "port": 1883
            },
            client_id="benchmark",
            topics=[
                "inferenceEngine/status",
                "inferenceEngine/data"
            ]
        )
        PerformanceMetrics.__mqtt_consumer.start()

    @staticmethod
    def update() -> None:
        topic, msg = PerformanceMetrics.__mqtt_consumer.consume()
        
        if msg is not None:
            msg = json.loads(msg)
            
        if topic == "inferenceEngine/status":
            PerformanceMetrics.__is_active = bool(msg['active'])
            
        elif topic == "inferenceEngine/data":
            PerformanceMetrics.__pre_process_times.append(int(msg['pre_processing_time']))
            PerformanceMetrics.__inference_times.append(int(msg['inference_time']))
            PerformanceMetrics.__post_process_times.append(int(msg['post_processing_time']))

    @staticmethod
    def get_measures() -> list:
        if PerformanceMetrics.__pre_process_times:
            return PerformanceMetrics.__pre_process_times, PerformanceMetrics.__inference_times, PerformanceMetrics.__post_process_times
        else:
            return None
    
    @staticmethod
    def is_active() -> bool:
        return PerformanceMetrics.__is_active


