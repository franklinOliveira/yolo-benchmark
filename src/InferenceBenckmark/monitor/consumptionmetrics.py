import psutil

class ConsumptionMetrics:
    __cpu_usage_levels: list = list()
    __cpu_temperature_levels: list = list()
    __ram_usage_levels: list = list()
    __current_usage_levels: list = list()

    @staticmethod
    def init() -> None:
        pass

    @staticmethod
    def update() -> None:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as file:
            ConsumptionMetrics.__cpu_temperature_levels.append(float(file.read()) / 1000.0)
        ConsumptionMetrics.__cpu_usage_levels.append(psutil.cpu_percent())
        ConsumptionMetrics.__ram_usage_levels.append(psutil.virtual_memory().used / (1024 * 1024))
    
    @staticmethod
    def compute_current_levels(min_current: int, max_current: int) -> None:
        ConsumptionMetrics.__current_usage_levels = [
            min_current + (max_current - min_current) * (cpu_usage / 100) for cpu_usage in ConsumptionMetrics.__cpu_usage_levels
        ]


    @staticmethod
    def get_measures() -> list:
        if ConsumptionMetrics.__cpu_usage_levels:
            return ConsumptionMetrics.__cpu_usage_levels, ConsumptionMetrics.__cpu_temperature_levels, ConsumptionMetrics.__ram_usage_levels, ConsumptionMetrics.__current_usage_levels
        else:
            return None 
        
    

