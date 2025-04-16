import psutil
import subprocess

class ConsumptionMetrics:
    __internal_current_sensor: bool

    __cpu_usage_levels: list = list()
    __cpu_temperature_levels: list = list()
    __ram_usage_levels: list = list()
    __current_usage_levels: list = list()

    @staticmethod
    def init(internal_current_sensor: bool) -> None:
        ConsumptionMetrics.__internal_current_sensor = internal_current_sensor

    @staticmethod
    def update() -> None:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as file:
            ConsumptionMetrics.__cpu_temperature_levels.append(
                float(file.read()) / 1000.0
            )
        ConsumptionMetrics.__cpu_usage_levels.append(psutil.cpu_percent())
        ConsumptionMetrics.__ram_usage_levels.append(
            psutil.virtual_memory().used / (1024 * 1024)
        )

        if ConsumptionMetrics.__internal_current_sensor:
            ConsumptionMetrics.__current_usage_levels.append(
                ConsumptionMetrics.__get_current_consumption()
            )

    @staticmethod
    def compute_current_levels(
        min_current: int, max_current: int, half_cores: bool
    ) -> None:
        if not half_cores:
            cpu_max = 100
        else:
            cpu_max = 50

        ConsumptionMetrics.__current_usage_levels = [
            min_current + (max_current - min_current) * (cpu_usage / cpu_max)
            for cpu_usage in ConsumptionMetrics.__cpu_usage_levels
        ]

    @staticmethod
    def get_measures() -> list:
        if ConsumptionMetrics.__cpu_usage_levels:
            return (
                ConsumptionMetrics.__cpu_usage_levels,
                ConsumptionMetrics.__cpu_temperature_levels,
                ConsumptionMetrics.__ram_usage_levels,
                ConsumptionMetrics.__current_usage_levels,
            )
        else:
            return None
        
    @staticmethod
    def __get_current_consumption() -> float:
        cmd_result = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True)
        measures_str = cmd_result.stdout.strip()
        measures_str = measures_str.replace(" ", "")
        measures_list = measures_str.split("\n")

        rails_currents = dict()
        rails_voltages = dict()
        for measure in measures_list:
            measure_type, measure_value = measure.split("=")
            measure_type = "_".join(measure_type.split("_")[:-1])
            
            if "A" in measure_value:
                measure_value = float(measure_value.replace("A", ""))
                rails_currents[measure_type] = measure_value

            elif "V" in measure_value:
                measure_value = float(measure_value.replace("V", ""))
                rails_voltages[measure_type] = measure_value

        total_5v_current = 0
        for rail_name, rail_current in rails_currents.items():
            rail_power = rail_current * rails_voltages[rail_name]

            if rail_power > 0:
                rail_5v_current = (rail_power / 5.0)
            else:
                rail_5v_current = 0

            total_5v_current += rail_5v_current
        
        return (total_5v_current * 1000)
