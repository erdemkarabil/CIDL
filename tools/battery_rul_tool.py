"""
VoltOptimizer - Battery RUL Prediction Tool
=============================================
Wraps the deep learning model as a "tool" so agents can call the model
like a function and interpret its output.

This module loads the trained 1D-CNN + GRU model and:
  - Predicts RUL from real-time battery data
  - Performs anomaly detection
  - Generates a battery health report
"""

import os
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATTERY_LIMITS, DATA_CONFIG, MODEL_DIR,
)
from models.cnn_gru_model import HybridCNNGRU
from data.mock_data_generator import generate_single_realtime_sample
from utils.logger import logger


class BatteryRULTool:
    """
    Battery RUL Prediction Tool usable by agents.

    Exposes the deep learning model through a function interface.
    Battery Guardian Agent calls this tool to:
      - Get an instantaneous RUL estimate
      - Check anomaly / temperature status
      - Perform safety assessment
    """

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Trained model file path (.pth)
                        If None, a fresh model is instantiated.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HybridCNNGRU()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.feature_stats = None

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device,
                                    weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.log("tool", f"BatteryRULTool: Model loaded → {model_path}")
        else:
            logger.log("tool",
                       "BatteryRULTool: No trained model found, "
                       "simulation mode active")

    def set_feature_stats(self, stats: dict):
        """Sets normalisation statistics."""
        self.feature_stats = stats

    def predict_rul(self, battery_state: dict) -> dict:
        """
        Predicts RUL from an instantaneous battery state.

        Can be called directly by agents.

        Args:
            battery_state: {
                "voltage": float,      # Cell voltage (V)
                "current": float,      # Instantaneous current (A)
                "temperature": float,  # Temperature (°C)
                "soc": float,          # State of charge (%)
                "battery_age": float,  # Battery age [0-1]
                "speed": float,        # Vehicle speed (km/h)
                "elevation": float,    # Road gradient (%)
            }

        Returns:
            {
                "rul_percentage": float,     # RUL estimate (%)
                "health_status": str,        # "GOOD", "WARNING", "CRITICAL"
                "anomaly_detected": bool,    # Anomaly present?
                "anomaly_details": list,     # Anomaly details
                "temperature_status": str,   # Temperature status
                "recommendations": list,     # Recommendations
                "raw_data": dict,            # Raw input data
            }
        """
        logger.log("tool", "BatteryRULTool called → predicting RUL...")

        # ── Convert input to model-compatible format ──
        features = np.array([
            battery_state["voltage"],
            battery_state["current"],
            battery_state["temperature"],
            battery_state.get("speed", 60.0),
            battery_state.get("elevation", 0.0),
        ], dtype=np.float32)

        # The model expects a full (seq_len, features) window, but at inference
        # time we only have a single snapshot. Tiling the snapshot is a common
        # approximation used when a rolling buffer is unavailable; the added
        # Gaussian noise breaks the artificial periodicity so the CNN/GRU
        # does not produce degenerate activations
        seq_len = DATA_CONFIG["sequence_length"]
        sequence = np.tile(features, (seq_len, 1))
        noise = np.random.normal(0, 0.01, sequence.shape)
        sequence = sequence + noise

        # Normalisation
        if self.feature_stats:
            sequence = ((sequence - self.feature_stats["means"])
                        / self.feature_stats["stds"])

        # Model inference
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor)
            rul_raw = prediction.item()

        # [0,1] → [0,100] with physical constraints
        rul_percentage = np.clip(rul_raw * 100, 0, 100)

        # ── Anomaly detection ──
        anomalies = []
        temp = battery_state["temperature"]
        volt = battery_state["voltage"]
        soc = battery_state["soc"]

        if temp > BATTERY_LIMITS["max_temperature"]:
            anomalies.append(
                f"CRITICAL TEMPERATURE: {temp:.1f}°C "
                f"(limit: {BATTERY_LIMITS['max_temperature']}°C)"
            )
        elif temp > BATTERY_LIMITS["warning_temperature"]:
            anomalies.append(
                f"High temperature warning: {temp:.1f}°C "
                f"(warning limit: {BATTERY_LIMITS['warning_temperature']}°C)"
            )

        if volt < BATTERY_LIMITS["min_voltage"]:
            anomalies.append(
                f"Low voltage: {volt:.3f}V "
                f"(min: {BATTERY_LIMITS['min_voltage']}V)"
            )

        if soc < BATTERY_LIMITS["min_soc"]:
            anomalies.append(
                f"Critically low charge: {soc:.1f}% "
                f"(min: {BATTERY_LIMITS['min_soc']}%)"
            )

        # Health classification: two or more concurrent anomalies (e.g.
        # low voltage AND high temperature) are treated as CRITICAL regardless
        # of the predicted RUL, because compound failures escalate non-linearly
        if rul_percentage < BATTERY_LIMITS["critical_rul"] or len(anomalies) > 1:
            health_status = "CRITICAL"
        elif rul_percentage < BATTERY_LIMITS["warning_rul"] or len(anomalies) > 0:
            health_status = "WARNING"
        else:
            health_status = "GOOD"

        # ── Temperature status ──
        if temp > BATTERY_LIMITS["max_temperature"]:
            temp_status = "CRITICAL_HOT"
        elif temp > BATTERY_LIMITS["warning_temperature"]:
            temp_status = "HIGH"
        elif temp < 5:
            temp_status = "COLD"
        else:
            temp_status = "NORMAL"

        # ── Recommendations ──
        recommendations = []
        if health_status == "CRITICAL":
            recommendations.append(
                "Reduce charge current immediately (max. 30A)")
            recommendations.append(
                "Lower charge ceiling to 80%")
            recommendations.append(
                "Check the nearest service point")
        elif health_status == "WARNING":
            recommendations.append(
                "Reduce charge current to normal level (max. 80A)")
            recommendations.append(
                "Monitor battery temperature")

        if temp_status in ("CRITICAL_HOT", "HIGH"):
            recommendations.append(
                "Pause charging or reduce current to allow cooling")
        elif temp_status == "COLD":
            recommendations.append(
                "Battery pre-heating required, efficiency is reduced")

        result = {
            "rul_percentage": round(rul_percentage, 2),
            "health_status": health_status,
            "anomaly_detected": len(anomalies) > 0,
            "anomaly_details": anomalies,
            "temperature_status": temp_status,
            "temperature_celsius": round(temp, 1),
            "voltage": round(volt, 3),
            "soc": round(soc, 1),
            "recommendations": recommendations,
            "raw_data": battery_state,
        }

        logger.log("tool",
                    f"RUL Estimate: {rul_percentage:.1f}% | "
                    f"Status: {health_status} | "
                    f"Temperature: {temp_status} ({temp:.1f}°C)")

        return result

    def get_realtime_battery_assessment(
        self,
        battery_age: float = 0.7,
        ambient_temp: float = 45.0,
        current_soc: float = 25.0,
    ) -> dict:
        """
        Real-time battery assessment.
        Simulates instantaneous sensor data with the mock data generator
        and predicts RUL with the DL model.

        Args:
            battery_age: Battery age [0-1]
            ambient_temp: Ambient temperature (°C)
            current_soc: Current state of charge (%)

        Returns:
            RUL prediction result dictionary
        """
        battery_state = generate_single_realtime_sample(
            battery_age=battery_age,
            ambient_temp=ambient_temp,
            current_soc=current_soc,
        )

        return self.predict_rul(battery_state)
