"""
VoltOptimizer - Synthetic Data Generator (Mock Data Generator)
==============================================================
Simulates time-series data from electric vehicle battery sensors.

Generated Features:
    1. Cell Voltage (V)     : 2.5 - 4.2V degradation curve
    2. Current (A)          : Charge/discharge cycles
    3. Temperature (°C)     : Ambient + internal heating model
    4. Vehicle Speed (km/h) : Driving profile simulation
    5. Road Gradient (%)    : Topographic variation

Target:
    - RUL (Remaining Useful Life): Continuous value from 0% to 100%
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG


class BatteryDataset(Dataset):
    """
    PyTorch Dataset class.
    Each sample: a time series of shape (sequence_length × num_features)
    and the corresponding RUL (%) target value.
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Numpy array of shape (N, seq_len, features)
            targets: RUL target values of shape (N,), range [0-100]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def generate_battery_degradation_data(
    num_batteries: int = None,
    sequence_length: int = None,
    noise_std: float = None,
    random_seed: int = None,
) -> tuple:
    """
    Generates synthetic battery degradation data.

    A different ageing profile is created for each battery.
    Based on real-world physics models:
      - Voltage: Non-linear drop dependent on capacity loss
      - Current: Charge/discharge cyclic variation
      - Temperature: Ambient + current-dependent heating
      - Speed/Gradient: Random driving profile

    Returns:
        (sequences, targets) → numpy arrays
    """
    num_batteries = num_batteries or DATA_CONFIG["num_batteries"]
    sequence_length = sequence_length or DATA_CONFIG["sequence_length"]
    noise_std = noise_std or DATA_CONFIG["noise_std"]
    random_seed = random_seed or DATA_CONFIG["random_seed"]

    np.random.seed(random_seed)

    all_sequences = []
    all_targets = []

    for i in range(num_batteries):
        # ── Random age per battery (0.0 = new, 1.0 = end-of-life)
        battery_age = np.random.uniform(0.0, 1.0)

        # ── RUL target: higher age → lower RUL
        rul = max(0.0, min(100.0, (1.0 - battery_age) * 100.0))

        # ── Multiple windows per battery
        num_windows = np.random.randint(5, 15)

        for w in range(num_windows):
            t = np.linspace(0, 1, sequence_length)

            # --- Feature 1: Cell Voltage (V) ---
            # Aged cells have lower nominal voltage and higher internal
            # resistance, causing larger voltage sag under load. We model
            # this as a linear shift of the open-circuit voltage curve
            # (4.2 V new → ~3.0 V at end-of-life) plus a sinusoidal
            # charge/discharge cycle and a within-window drift term.
            base_voltage = 4.2 - battery_age * 1.2  # 4.2V → 3.0V
            voltage_cycle = 0.3 * np.sin(2 * np.pi * t * 3)
            degradation = -0.2 * battery_age * t
            voltage = (base_voltage + voltage_cycle + degradation
                       + np.random.normal(0, noise_std * 0.5, sequence_length))
            voltage = np.clip(voltage, 2.5, 4.2)

            # --- Feature 2: Current (A) ---
            # Charge (+) and discharge (-) cycles
            current_base = 30 * np.sin(2 * np.pi * t * 2)
            current_noise = np.random.normal(0, 5, sequence_length)
            current = current_base + current_noise
            current = np.clip(current, -150, 150)

            # --- Feature 3: Temperature (°C) ---
            # Joule heating (∝ |I|) dominates; aged cells also show higher
            # internal resistance, contributing an extra aging_heat offset
            ambient = np.random.uniform(15, 35)
            heat_from_current = 0.15 * np.abs(current)
            aging_heat = 5 * battery_age
            temperature = (ambient + heat_from_current + aging_heat
                           + np.random.normal(0, noise_std * 10,
                                              sequence_length))
            temperature = np.clip(temperature, -10, 70)

            # --- Feature 4: Vehicle Speed (km/h) ---
            speed = (60 * np.abs(np.sin(2 * np.pi * t * 1.5))
                     + np.random.normal(0, 10, sequence_length))
            speed = np.clip(speed, 0, 180)

            # --- Feature 5: Road Gradient (%) ---
            elevation = (5 * np.sin(2 * np.pi * t * 0.5)
                         + np.random.normal(0, 2, sequence_length))
            elevation = np.clip(elevation, -15, 15)

            # ── Stack features: (seq_len, 5)
            sequence = np.stack(
                [voltage, current, temperature, speed, elevation], axis=-1
            )
            all_sequences.append(sequence)

            # ── Per-window RUL variation (small fluctuation)
            window_rul = rul + np.random.normal(0, 2)
            window_rul = np.clip(window_rul, 0, 100)
            all_targets.append(window_rul)

    sequences = np.array(all_sequences, dtype=np.float32)
    targets = np.array(all_targets, dtype=np.float32)

    print(f"  ⚡ Synthetic data generated: {sequences.shape[0]} samples, "
          f"seq_len={sequences.shape[1]}, features={sequences.shape[2]}")
    print(f"  📊 RUL distribution: min={targets.min():.1f}%, "
          f"max={targets.max():.1f}%, mean={targets.mean():.1f}%")

    return sequences, targets


def create_dataloaders(
    batch_size: int = None,
    train_ratio: float = None,
) -> tuple:
    """
    Generates synthetic data and converts it to PyTorch DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, feature_stats)
    """
    from config import TRAIN_CONFIG

    batch_size = batch_size or TRAIN_CONFIG["batch_size"]
    train_ratio = train_ratio or DATA_CONFIG["train_ratio"]

    sequences, targets = generate_battery_degradation_data()

    # Feature normalisation (Z-Score)
    feature_means = sequences.mean(axis=(0, 1))
    feature_stds = sequences.std(axis=(0, 1))
    # Guard against constant features (std == 0) that would cause division by
    # zero; replacing with 1.0 leaves those features unscaled
    feature_stds[feature_stds == 0] = 1.0
    sequences = (sequences - feature_means) / feature_stds

    # Target normalisation: [0-100] → [0-1]
    targets = targets / 100.0

    feature_stats = {
        "means": feature_means,
        "stds": feature_stds,
    }

    # Train / Validation / Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, targets, train_size=train_ratio,
        random_state=DATA_CONFIG["random_seed"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=DATA_CONFIG["random_seed"]
    )

    print(f"  📂 Train: {len(X_train)}, Validation: {len(X_val)}, "
          f"Test: {len(X_test)}")

    train_loader = DataLoader(
        BatteryDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        BatteryDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        BatteryDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, feature_stats


def generate_single_realtime_sample(
    battery_age: float = 0.7,
    ambient_temp: float = 45.0,
    current_soc: float = 25.0,
) -> dict:
    """
    Generates a single instantaneous battery state sample.
    Used by agents for real-time simulation.

    Args:
        battery_age: Battery age [0-1]
        ambient_temp: Ambient temperature (°C)
        current_soc: Current state of charge (%)

    Returns:
        Battery state dictionary
    """
    np.random.seed(None)  # True randomness

    voltage = 4.2 - battery_age * 1.0 - (1 - current_soc / 100) * 0.8
    voltage += np.random.normal(0, 0.02)
    voltage = np.clip(voltage, 2.5, 4.2)

    current = np.random.uniform(-50, 80)

    temperature = (ambient_temp + 0.1 * abs(current) + 3 * battery_age
                   + np.random.normal(0, 1))

    return {
        "voltage": round(float(voltage), 3),
        "current": round(float(current), 2),
        "temperature": round(float(temperature), 1),
        "soc": round(float(current_soc), 1),
        "battery_age": round(float(battery_age), 3),
        "speed": round(float(np.random.uniform(0, 120)), 1),
        "elevation": round(float(np.random.normal(0, 3)), 1),
    }
