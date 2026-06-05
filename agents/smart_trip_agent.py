"""
VoltOptimizer - Smart Trip Agent
=================================
Route and Travel Planning Assistant Agent.

Tasks:
    1. Analyse the user's route, weather, and road gradient
    2. Optimise charge stops based on battery state and grid prices
    3. Plan which station, how long, and at what cost to charge
    4. Build a comprehensive travel plan integrating other agents' data

Reasoning Process:
    Route analysis → Battery Guardian data → Grid data →
    Charge stop optimisation → Comprehensive plan output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.logger import logger


class SmartTripAgent(BaseAgent):
    """
    Route and Travel Planning Assistant.

    Uses the Route Planner Tool (mock Google Maps) to build an optimal
    travel plan. Integrates information from Battery Guardian and
    Grid Tariff agents.
    """

    def __init__(self, route_planner_tool):
        super().__init__(
            role="Smart Trip Agent",
            goal=("Create the safest, cheapest, and fastest travel plan "
                  "by combining the user's route, battery state, and energy prices"),
            backstory=(
                "I am VoltOptimizer's Travel Planning Assistant. "
                "I use a Google Maps-like tool to analyse routes, "
                "evaluate weather conditions and road gradients. "
                "I integrate Battery Guardian Agent's safety data and "
                "Grid Tariff Agent's price analysis to present the user "
                "with a comprehensive travel plan."
            ),
            tools=[route_planner_tool],
        )

        self.route_planner_tool = route_planner_tool

    def execute(self, task_input: dict) -> dict:
        """
        Main task: build a comprehensive travel plan.

        Args:
            task_input: {
                "origin": str,
                "destination": str,
                "total_distance_km": float,
                "current_soc": float,
                "ambient_temperature": float,
                "battery_guardian_report": dict,
                "grid_tariff_report": dict,
            }

        Returns:
            Comprehensive travel plan
        """
        logger.header("🗺️  SMART TRIP AGENT")
        logger.log("smart_trip", "Starting travel planning...")

        origin = task_input.get("origin", "Izmir")
        destination = task_input.get("destination", "Istanbul")
        distance = task_input.get("total_distance_km", 600.0)
        current_soc = task_input.get("current_soc", 25.0)
        temp = task_input.get("ambient_temperature", 35.0)
        guardian_report = task_input.get("battery_guardian_report", {})
        grid_report = task_input.get("grid_tariff_report", {})

        # ══════════════════════════════════════════════
        # STEP 1: Evaluate data from other agents
        # ══════════════════════════════════════════════
        self.reason({"step": "evaluating agent data"}, step=1)

        charge_params = guardian_report.get("charge_parameters", {})
        max_charge_soc = charge_params.get("max_charge_soc", 80.0)
        max_current = charge_params.get("max_charge_current_a", 150.0)
        safety_level = charge_params.get("safety_level", "NORMAL")

        logger.agent_thinking(
            self.role,
            f"Constraints from Battery Guardian: "
            f"Max charge: {max_charge_soc:.0f}%, "
            f"Max current: {max_current:.0f}A, "
            f"Safety: {safety_level}",
            step=1,
        )

        cost_comp = grid_report.get("cost_comparison", {})
        optimal_cost = cost_comp.get("optimal_cost_tl", 0)

        logger.agent_thinking(
            self.role,
            f"Optimal cost from Grid Tariff: {optimal_cost:.2f} TL. "
            f"I will use this in charge stop planning.",
            step=1,
        )

        # ══════════════════════════════════════════════
        # STEP 2: Route planning (Mock Google Maps)
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            f"Starting route analysis: {origin} → {destination} "
            f"({distance:.0f} km, {temp:.0f}°C)",
            step=2,
        )

        trip_plan = self.use_tool(
            tool_name="RoutePlannerTool",
            tool_instance=self.route_planner_tool,
            method="optimize_trip",
            origin=origin,
            destination=destination,
            current_soc=current_soc,
            max_charge_soc=max_charge_soc,
            ambient_temperature=temp,
            total_distance_km=distance,
        )

        # ══════════════════════════════════════════════
        # STEP 3: Plan revision for safety constraints
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            "Revising plan according to safety constraints.",
            step=3,
        )

        revised_plan = self._revise_plan_for_safety(
            trip_plan, safety_level, max_current, max_charge_soc
        )

        # ══════════════════════════════════════════════
        # STEP 4: Build comprehensive plan
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            "Combining all information to build the comprehensive travel plan.",
            step=4,
        )

        final_plan = self._create_final_plan(
            revised_plan, guardian_report, grid_report,
            origin, destination, distance, temp
        )

        self.outputs["trip_plan"] = final_plan

        # ══════════════════════════════════════════════
        # STEP 5: Result Report
        # ══════════════════════════════════════════════
        summary = final_plan["trip_summary"]
        logger.agent_result(
            self.role,
            f"Travel Plan Ready → "
            f"{summary['total_charge_stops']} stops, "
            f"{summary['total_trip_time']} hours, "
            f"{summary['total_cost']} TL"
        )

        return final_plan

    def _revise_plan_for_safety(
        self,
        plan: dict,
        safety_level: str,
        max_current: float,
        max_charge_soc: float,
    ) -> dict:
        """
        Revises the travel plan according to the safety level.
        """
        revised = plan.copy()
        revisions = []

        if safety_level in ("CRITICAL", "EMERGENCY"):
            revisions.append(
                "⚠️ CRITICAL safety level: More frequent charge stops "
                "planned and charge current reduced."
            )
            if "charge_stops" in revised:
                for stop in revised["charge_stops"]:
                    original_power = stop.get("charge_power_kw", 150)
                    # Approximate pack power from current limit (A):
                    # P = I × V_nominal ≈ I × 400V → factor of 0.4 converts A → kW
                    limited_power = min(original_power,
                                        max_current * 0.4)  # ~400V pack
                    if limited_power < original_power:
                        ratio = original_power / limited_power
                        stop["charge_time_min"] = round(
                            stop["charge_time_min"] * ratio, 1
                        )
                        stop["charge_power_kw"] = limited_power
                        revisions.append(
                            f"  → {stop['station_name']}: Power "
                            f"{original_power:.0f}kW → "
                            f"{limited_power:.0f}kW, "
                            f"time: {stop['charge_time_min']:.0f} min"
                        )

        elif safety_level == "WARNING":
            revisions.append(
                "🟡 WARNING safety level: Charge SoC ceiling "
                f"restricted to {max_charge_soc:.0f}%."
            )

        for rev in revisions:
            logger.agent_action(self.role, rev)

        revised["safety_revisions"] = revisions
        return revised

    def _create_final_plan(
        self,
        trip_plan: dict,
        guardian_report: dict,
        grid_report: dict,
        origin: str,
        destination: str,
        distance: float,
        temperature: float,
    ) -> dict:
        """
        Combines all agent data to build the final travel plan.
        """
        charge_stops = trip_plan.get("charge_stops", [])
        summary = trip_plan.get("summary", {})

        stop_details = []
        for i, stop in enumerate(charge_stops, 1):
            stop_details.append({
                "stop_number": i,
                "station": stop.get("station_name", f"Station-{i}"),
                "location_km": stop.get("station_location_km", 0),
                "arrival_soc": f"{stop.get('current_soc', 0):.0f}%",
                "departure_soc": f"{stop.get('target_soc', 80):.0f}%",
                "energy_kwh": f"{stop.get('energy_kwh', 0):.1f} kWh",
                "duration": f"{stop.get('charge_time_min', 0):.0f} min",
                "power_kw": f"{stop.get('charge_power_kw', 0):.0f} kW",
                "cost": f"{stop.get('cost_tl', 0):.2f} TL",
            })

        guardian_actions = guardian_report.get("actions_taken", [])
        safety_info = guardian_report.get("charge_parameters", {})

        tariff_info = grid_report.get("tariff_analysis", {})
        tariff_rec = grid_report.get("recommendation", "")

        final_plan = {
            "route_info": {
                "origin": origin,
                "destination": destination,
                "distance_km": distance,
                "temperature": f"{temperature}°C",
            },
            "safety_status": {
                "level": safety_info.get("safety_level", "NORMAL"),
                "charge_limit": f"{safety_info.get('max_charge_soc', 80):.0f}%",
                "max_current": f"{safety_info.get('max_charge_current_a', 150):.0f}A",
                "actions": guardian_actions,
            },
            "energy_analysis": {
                "daily_avg_price": tariff_info.get("daily_avg_price_kwh", 0),
                "cheapest_window": tariff_info.get("cheapest_window", {}),
                "recommendation": tariff_rec,
            },
            "charge_stops_detail": stop_details,
            "trip_summary": {
                "total_distance_km": distance,
                "total_charge_stops": len(charge_stops),
                "total_charge_time_min": summary.get("total_charge_time_min", 0),
                "total_trip_time": summary.get("total_trip_time_hours", 0),
                "total_cost": summary.get("total_charge_cost_tl", 0),
                "arrival_soc": trip_plan.get("battery", {}).get("arrival_soc", 0),
            },
            "revisions": trip_plan.get("safety_revisions", []),
        }

        return final_plan
