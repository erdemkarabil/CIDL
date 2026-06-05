"""
VoltOptimizer - Global Configuration File
==========================================
Constants and hyperparameters shared across all modules.
"""

import os

# ============================================================
# Project Directories
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create directories if they don't exist
for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Synthetic Data Parameters
# ============================================================
DATA_CONFIG = {
    "num_batteries": 50,          # Number of simulated batteries
    "sequence_length": 100,       # Length of each time-series window
    "num_features": 5,            # Voltage, Current, Temperature, Speed, Slope
    "noise_std": 0.02,            # Noise standard deviation
    "train_ratio": 0.8,           # Train / test split ratio
    "random_seed": 42,
}

# ============================================================
# Deep Learning Model Hyperparameters
# ============================================================
MODEL_CONFIG = {
    # Two CNN stages: 32 filters capture low-level waveform shapes,
    # 64 filters in the second block combine them into richer patterns
    "cnn_filters": [32, 64],
    # kernel_size=3 is a common default for 1D time-series; larger kernels
    # risk blurring short transients (e.g. a single-step voltage spike)
    "cnn_kernel_size": 3,
    "gru_hidden_size": 128,
    "gru_num_layers": 2,
    # 0.3 dropout provides regularisation without collapsing learning speed;
    # chosen via the hyperparameter sweep reported in Table 3 of the paper
    "dropout": 0.3,
    "fc_hidden": 64,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    # weight_decay (L2) is small but non-zero to discourage large weights
    # without suppressing the model's capacity on this relatively small dataset
    "weight_decay": 1e-5,
    "patience": 10,
    # Halve LR every 15 epochs; empirically, the validation curve plateaus
    # around epoch 15-20, so this step aligns with the natural convergence knee
    "lr_scheduler_step": 15,
    "lr_scheduler_gamma": 0.5,
}

# ============================================================
# Battery Physical Limits
# ============================================================
BATTERY_LIMITS = {
    "max_voltage": 4.2,           # Volts
    "min_voltage": 2.5,           # Volts
    "nominal_voltage": 3.7,       # Volts
    "max_current": 150.0,         # Amperes
    "max_temperature": 60.0,      # °C (Critical upper limit)
    "warning_temperature": 45.0,  # °C (Warning threshold)
    "optimal_temperature": 25.0,  # °C
    "max_charge_soc": 100.0,      # Max charge limit (%)
    "safe_charge_soc": 80.0,      # Safe charge limit (%)
    "min_soc": 10.0,              # Minimum SoC (%)
    "critical_rul": 20.0,         # Critical RUL threshold (%)
    "warning_rul": 40.0,          # Warning RUL threshold (%)
}

# ============================================================
# Grid Tariff Parameters (Simulation)
# ============================================================
GRID_CONFIG = {
    "currency": "TL",
    "peak_hours": [(8, 12), (18, 22)],                    # Peak hours
    "off_peak_hours": [(0, 8), (12, 18), (22, 24)],       # Off-peak hours
    "peak_price_kwh": 4.50,                               # TL/kWh (Peak)
    "off_peak_price_kwh": 1.80,                           # TL/kWh (Off-peak)
    "super_off_peak_price_kwh": 0.90,                     # TL/kWh (Super off-peak)
    "super_off_peak_hours": [(1, 5)],                     # Super off-peak hours
}

# ============================================================
# Route Planning Parameters (Simulation)
# ============================================================
ROUTE_CONFIG = {
    "ev_range_km": 400,                     # EV range (km)
    "ev_consumption_kwh_per_km": 0.18,      # Energy consumption (kWh/km)
    "ev_battery_capacity_kwh": 75.0,        # Battery capacity (kWh)
    "charge_speed_kw": {                    # Charging speeds
        "fast_dc": 150.0,
        "normal_dc": 50.0,
        "ac": 22.0,
    },
    "temperature_efficiency_factor": {
        "cold": 0.75,    # <5°C → 25% efficiency loss
        "cool": 0.90,    # 5-15°C
        "optimal": 1.00, # 15-30°C
        "warm": 0.95,    # 30-40°C
        "hot": 0.80,     # >40°C → 20% efficiency loss
    },
    "elevation_factor_per_100m": 0.03,      # 100m climb → 3% extra consumption
}

# ============================================================
# Agent Settings
# ============================================================
AGENT_CONFIG = {
    "verbose": True,
    "max_reasoning_steps": 5,   # Max reasoning steps
    "self_correction_enabled": True,
}
