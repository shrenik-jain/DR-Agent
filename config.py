"""
Configuration settings for DRAgent
"""

# Model Configuration
MODEL_NAME = "gpt-4-turbo"
TEMPERATURE = 0  # Set to 0 for deterministic optimization
MAX_ITERATIONS = 10  # Maximum agent iterations

# Optimization Settings
OPTIMIZATION_HORIZON_HOURS = 24
DEFAULT_HOUSEHOLD_PEAK_KW = 15.0
SOLVER = "ECOS"  # Options: ECOS, SCS, OSQP

# Price Data Settings (SDG&E EV-TOU-5)
PRICE_SUPER_OFF_PEAK = 0.27  # 12am-6am
PRICE_OFF_PEAK = 0.36        # 6am-4pm, 9pm-12am
PRICE_ON_PEAK = 0.52         # 4pm-9pm

# Carbon Intensity Patterns (CAISO typical)
CARBON_NIGHT_LOW = 250       # lbs/MWh
CARBON_MORNING_RAMP = 450
CARBON_SOLAR_PEAK = 200
CARBON_EVENING_HIGH = 550

# Evaluation Settings
EVAL_SCENARIOS = [
    "sufficient_info",
    "insufficient_info",
    "redundant_info",
    "carbon_optimization"
]

# Common Appliance Defaults (kWh and kW)
APPLIANCE_DEFAULTS = {
    "EV_MODEL3": {
        "energy_kwh": 16.0,
        "max_power_kw": 11.0,
        "typical_window": (22, 7)  # 10 PM to 7 AM
    },
    "DISHWASHER": {
        "energy_kwh": 3.6,
        "max_power_kw": 2.0,
        "typical_window": (20, 23)  # 8 PM to 11 PM
    },
    "DRYER": {
        "energy_kwh": 4.5,
        "max_power_kw": 4.0,
        "typical_window": (21, 24)  # 9 PM to midnight
    },
    "WATER_HEATER": {
        "energy_kwh": 3.5,
        "max_power_kw": 4.5,
        "typical_window": (22, 6)  # 10 PM to 6 AM
    },
    "POOL_PUMP": {
        "energy_kwh": 8.0,
        "max_power_kw": 2.0,
        "typical_window": (0, 23)  # Flexible all day
    }
}

# Logging
VERBOSE_MODE = True
SAVE_OPTIMIZATION_RESULTS = True
RESULTS_DIR = "/home/claude/results"
