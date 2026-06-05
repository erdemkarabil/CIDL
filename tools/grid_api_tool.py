"""
VoltOptimizer - Grid Tariff API Tool (Simulation)
==================================================
Simulates the dynamic pricing tariffs of the electricity grid.
Grid Tariff Agent uses this tool to identify the cheapest and most
optimal charging time windows.

Simulation Features:
    - Realistic Turkish electricity tariffs
    - Peak / Off-Peak / Super Off-Peak time slots
    - Seasonal price variation
    - Instantaneous demand-based price fluctuation
"""

import datetime
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GRID_CONFIG
from utils.logger import logger


class GridTariffAPITool:
    """
    Simulated Electricity Grid Tariff API.

    Provides an interface similar to a real-world API, allowing agents
    to query grid prices.
    """

    def __init__(self):
        self.config = GRID_CONFIG
        logger.log("tool", "GridTariffAPITool initialised (Simulation mode)")

    def _get_price_for_hour(self, hour: int) -> dict:
        """
        Returns the electricity price for a given hour.

        Args:
            hour: Hour (0-23)

        Returns:
            {"hour": int, "price_kwh": float, "tariff_type": str}
        """
        # Super off-peak (deep night): stochastic ±5% fluctuation models
        # real-world spot-market volatility even at low-demand hours
        for start, end in self.config["super_off_peak_hours"]:
            if start <= hour < end:
                base_price = self.config["super_off_peak_price_kwh"]
                price = base_price * (1 + np.random.uniform(-0.05, 0.05))
                return {
                    "hour": hour,
                    "price_kwh": round(price, 2),
                    "tariff_type": "SUPER_OFF_PEAK",
                    "demand_level": "Very Low",
                }

        # Peak hours
        for start, end in self.config["peak_hours"]:
            if start <= hour < end:
                base_price = self.config["peak_price_kwh"]
                price = base_price * (1 + np.random.uniform(-0.08, 0.08))
                return {
                    "hour": hour,
                    "price_kwh": round(price, 2),
                    "tariff_type": "PEAK",
                    "demand_level": "High",
                }

        # Off-peak hours (default)
        base_price = self.config["off_peak_price_kwh"]
        price = base_price * (1 + np.random.uniform(-0.05, 0.05))
        return {
            "hour": hour,
            "price_kwh": round(price, 2),
            "tariff_type": "OFF_PEAK",
            "demand_level": "Medium",
        }

    def get_24h_tariff_schedule(self) -> dict:
        """
        Returns the 24-hour tariff schedule.

        Returns:
            {
                "date": str,
                "currency": str,
                "schedule": [{"hour": int, "price_kwh": float, ...}],
                "cheapest_window": {"start": int, "end": int, ...},
                "most_expensive_window": {"start": int, "end": int, ...},
            }
        """
        logger.log("tool", "GridTariffAPITool: Querying 24-hour tariff schedule...")

        schedule = []
        for h in range(24):
            schedule.append(self._get_price_for_hour(h))

        prices = [s["price_kwh"] for s in schedule]

        # Find cheapest consecutive 3-hour window
        best_cost = float("inf")
        best_start = 0
        for i in range(22):
            window_cost = sum(prices[i:i + 3])
            if window_cost < best_cost:
                best_cost = window_cost
                best_start = i

        # Find most expensive consecutive 3-hour window
        worst_cost = 0
        worst_start = 0
        for i in range(22):
            window_cost = sum(prices[i:i + 3])
            if window_cost > worst_cost:
                worst_cost = window_cost
                worst_start = i

        result = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "currency": self.config["currency"],
            "schedule": schedule,
            "cheapest_window": {
                "start_hour": best_start,
                "end_hour": best_start + 3,
                "avg_price_kwh": round(best_cost / 3, 2),
                "total_3h_cost_per_kwh": round(best_cost, 2),
                "tariff_type": schedule[best_start]["tariff_type"],
            },
            "most_expensive_window": {
                "start_hour": worst_start,
                "end_hour": worst_start + 3,
                "avg_price_kwh": round(worst_cost / 3, 2),
                "total_3h_cost_per_kwh": round(worst_cost, 2),
                "tariff_type": schedule[worst_start]["tariff_type"],
            },
            "daily_avg_price": round(np.mean(prices), 2),
        }

        logger.log("tool",
                    f"Tariff received → Cheapest: "
                    f"{best_start}:00-{best_start + 3}:00 "
                    f"({result['cheapest_window']['avg_price_kwh']} "
                    f"{self.config['currency']}/kWh)")

        return result

    def get_optimal_charge_windows(
        self,
        required_kwh: float,
        charge_power_kw: float = 50.0,
        max_windows: int = 3,
    ) -> list:
        """
        Recommends the most suitable charge windows for a given energy need.

        Args:
            required_kwh: Required energy amount (kWh)
            charge_power_kw: Charge power (kW)
            max_windows: Maximum number of windows

        Returns:
            [{start_hour, end_hour, price_kwh, estimated_cost, ...}]
        """
        logger.log("tool",
                    f"Calculating optimal charge windows: "
                    f"{required_kwh:.1f} kWh, {charge_power_kw:.0f} kW")

        charge_hours_needed = required_kwh / charge_power_kw
        schedule = self.get_24h_tariff_schedule()

        # Sort hours by price
        sorted_hours = sorted(schedule["schedule"],
                              key=lambda x: x["price_kwh"])

        windows = []
        remaining_kwh = required_kwh
        total_cost = 0

        for hour_info in sorted_hours:
            if remaining_kwh <= 0 or len(windows) >= max_windows * 3:
                break

            energy_this_hour = min(charge_power_kw, remaining_kwh)
            cost_this_hour = energy_this_hour * hour_info["price_kwh"]

            windows.append({
                "hour": hour_info["hour"],
                "price_kwh": hour_info["price_kwh"],
                "energy_kwh": round(energy_this_hour, 2),
                "cost_tl": round(cost_this_hour, 2),
                "tariff_type": hour_info["tariff_type"],
            })

            remaining_kwh -= energy_this_hour
            total_cost += cost_this_hour

        # Sort by hour
        windows.sort(key=lambda x: x["hour"])

        # Merge consecutive hours
        merged = []
        for w in windows:
            if (merged and
                    w["hour"] == merged[-1]["end_hour"]):
                merged[-1]["end_hour"] = w["hour"] + 1
                merged[-1]["energy_kwh"] += w["energy_kwh"]
                merged[-1]["cost_tl"] += w["cost_tl"]
                merged[-1]["avg_price"] = round(
                    merged[-1]["cost_tl"] / merged[-1]["energy_kwh"], 2
                )
            else:
                merged.append({
                    "start_hour": w["hour"],
                    "end_hour": w["hour"] + 1,
                    "energy_kwh": w["energy_kwh"],
                    "cost_tl": round(w["cost_tl"], 2),
                    "avg_price": w["price_kwh"],
                    "tariff_type": w["tariff_type"],
                })

        logger.log("tool",
                    f"Optimal charge plan ready → "
                    f"{len(merged)} windows, "
                    f"total cost: {total_cost:.2f} TL")

        return merged[:max_windows]

    def estimate_charge_cost(
        self,
        energy_kwh: float,
        start_hour: int,
    ) -> dict:
        """
        Calculates the charge cost at a given hour.

        Args:
            energy_kwh: Energy to charge (kWh)
            start_hour: Start hour

        Returns:
            {"cost_tl": float, "price_kwh": float, "tariff_type": str}
        """
        hour_info = self._get_price_for_hour(start_hour)
        cost = energy_kwh * hour_info["price_kwh"]

        return {
            "energy_kwh": energy_kwh,
            "price_kwh": hour_info["price_kwh"],
            "cost_tl": round(cost, 2),
            "tariff_type": hour_info["tariff_type"],
            "hour": start_hour,
        }
