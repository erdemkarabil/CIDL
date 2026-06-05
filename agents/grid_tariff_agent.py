"""
VoltOptimizer - Grid Tariff Agent
==================================
Smart Grid and Energy Market Analyst Agent.

Tasks:
    1. Query dynamic electricity tariff schedules
    2. Analyse peak / off-peak hours
    3. Identify the cheapest and most optimal charging windows
    4. Provide cost-saving recommendations

Reasoning Process:
    Query tariff → Price analysis → Window comparison →
    Optimal time selection → Savings report
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.logger import logger


class GridTariffAgent(BaseAgent):
    """
    Smart Grid and Energy Market Analyst.

    Uses a simulated Web/API tool to analyse electricity prices
    and recommend the most suitable charging windows.
    """

    def __init__(self, grid_api_tool):
        super().__init__(
            role="Grid Tariff Agent",
            goal=("Analyse the dynamic tariffs of the electricity grid "
                  "to identify the cheapest and most efficient charging windows"),
            backstory=(
                "I am VoltOptimizer's Energy Market Analyst. "
                "I track and analyse the dynamic tariffs of Turkey's "
                "electricity grid in real time. My goal is to minimise "
                "the user's charging cost while balancing grid load. "
                "I provide smart recommendations by evaluating "
                "peak/off-peak time slots, seasonal price variations, "
                "and instantaneous demand levels."
            ),
            tools=[grid_api_tool],
        )

        self.grid_api_tool = grid_api_tool

    def execute(self, task_input: dict) -> dict:
        """
        Main task: analyse the energy market and present an optimal charge plan.

        Args:
            task_input: {
                "required_energy_kwh": float,
                "charge_power_kw": float,
                "max_charge_soc": float,
                "battery_capacity_kwh": float,
                "current_soc": float,
            }

        Returns:
            Energy analysis and optimal charge plan
        """
        logger.header("💰 GRID TARIFF AGENT")
        logger.log("grid_tariff",
                    "Starting energy market analysis...")

        required_kwh = task_input.get("required_energy_kwh", 50.0)
        charge_power = task_input.get("charge_power_kw", 50.0)
        max_soc = task_input.get("max_charge_soc", 80.0)
        capacity = task_input.get("battery_capacity_kwh", 75.0)
        current_soc = task_input.get("current_soc", 25.0)

        actual_energy_needed = capacity * ((max_soc - current_soc) / 100.0)
        required_kwh = max(required_kwh, actual_energy_needed)

        # ══════════════════════════════════════════════
        # STEP 1: Query 24-Hour Tariff Schedule
        # ══════════════════════════════════════════════
        self.reason({"step": "querying tariff"}, step=1)
        logger.agent_thinking(
            self.role,
            "Querying the 24-hour tariff schedule from the simulated grid API.",
            step=1,
        )

        schedule = self.use_tool(
            tool_name="GridTariffAPITool",
            tool_instance=self.grid_api_tool,
            method="get_24h_tariff_schedule",
        )

        # ══════════════════════════════════════════════
        # STEP 2: Price Analysis
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            f"Tariff received. Daily average: "
            f"{schedule['daily_avg_price']:.2f} TL/kWh. "
            f"Cheapest window: {schedule['cheapest_window']['start_hour']}:00-"
            f"{schedule['cheapest_window']['end_hour']}:00 "
            f"({schedule['cheapest_window']['avg_price_kwh']} TL/kWh). "
            f"Most expensive: "
            f"{schedule['most_expensive_window']['start_hour']}:00-"
            f"{schedule['most_expensive_window']['end_hour']}:00 "
            f"({schedule['most_expensive_window']['avg_price_kwh']} TL/kWh).",
            step=2,
        )

        # ══════════════════════════════════════════════
        # STEP 3: Identify Optimal Charge Windows
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            f"Required energy: {required_kwh:.1f} kWh, "
            f"charge power: {charge_power:.0f} kW. "
            f"Calculating most suitable charge windows.",
            step=3,
        )

        optimal_windows = self.use_tool(
            tool_name="GridTariffAPITool",
            tool_instance=self.grid_api_tool,
            method="get_optimal_charge_windows",
            required_kwh=required_kwh,
            charge_power_kw=charge_power,
        )

        # ══════════════════════════════════════════════
        # STEP 4: Savings Analysis
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            "Comparing peak-hour charging cost with optimal-hour charging cost.",
            step=4,
        )

        peak_cost = self.use_tool(
            tool_name="GridTariffAPITool",
            tool_instance=self.grid_api_tool,
            method="estimate_charge_cost",
            energy_kwh=required_kwh,
            start_hour=schedule["most_expensive_window"]["start_hour"],
        )

        optimal_total_cost = sum(w["cost_tl"] for w in optimal_windows)

        savings = peak_cost["cost_tl"] - optimal_total_cost
        savings_pct = ((savings / peak_cost["cost_tl"]) * 100
                       if peak_cost["cost_tl"] > 0 else 0)

        # ══════════════════════════════════════════════
        # STEP 5: Result Report
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            f"Savings analysis complete. Optimal charging saves "
            f"{savings:.2f} TL ({savings_pct:.1f}%).",
            step=5,
        )

        result = {
            "tariff_analysis": {
                "date": schedule["date"],
                "daily_avg_price_kwh": schedule["daily_avg_price"],
                "cheapest_window": schedule["cheapest_window"],
                "most_expensive_window": schedule["most_expensive_window"],
            },
            "optimal_charge_plan": {
                "required_energy_kwh": round(required_kwh, 2),
                "charge_power_kw": charge_power,
                "optimal_windows": optimal_windows,
                "optimal_total_cost_tl": round(optimal_total_cost, 2),
            },
            "cost_comparison": {
                "peak_cost_tl": peak_cost["cost_tl"],
                "optimal_cost_tl": round(optimal_total_cost, 2),
                "savings_tl": round(savings, 2),
                "savings_percentage": round(savings_pct, 1),
            },
            "recommendation": self._generate_recommendation(
                schedule, optimal_windows, savings, savings_pct
            ),
        }

        self.outputs["tariff_analysis"] = result

        logger.agent_result(
            self.role,
            f"Optimal Cost: {optimal_total_cost:.2f} TL | "
            f"Savings: {savings:.2f} TL ({savings_pct:.1f}%)"
        )

        return result

    def _generate_recommendation(
        self,
        schedule: dict,
        windows: list,
        savings: float,
        savings_pct: float,
    ) -> str:
        """Generates a user-friendly recommendation text."""
        if not windows:
            return "No charge window could be determined."

        best = windows[0]
        rec = (
            f"💡 Recommended Charge Plan: "
            f"Charge between {best['start_hour']}:00 and {best['end_hour']}:00 "
            f"({best['tariff_type']} tariff). "
            f"This saves approximately {savings:.2f} TL ({savings_pct:.1f}%). "
            f"Avoid charging during peak hours (08-12, 18-22)!"
        )
        return rec
