"""
VoltOptimizer - Agent Orchestration Engine
==========================================
Main control unit managing the coordination and communication of all agents.

Follows a CrewAI-like orchestration pattern:
  1. Task definition and agent assignment
  2. Sequential task execution (pipeline)
  3. Inter-agent messaging and data sharing
  4. Consolidated result compilation

Flow:
    Battery Guardian → Grid Tariff → Smart Trip
    (Each agent's output flows as input to the next)
"""

import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import logger
from tools.battery_rul_tool import BatteryRULTool
from tools.grid_api_tool import GridTariffAPITool
from tools.route_planner_tool import RoutePlannerTool
from agents.battery_guardian_agent import BatteryGuardianAgent
from agents.grid_tariff_agent import GridTariffAgent
from agents.smart_trip_agent import SmartTripAgent


class CrewOrchestrator:
    """
    VoltOptimizer Agent Orchestration Engine.

    Builds a "Crew" (team) in the style of the CrewAI framework
    and executes tasks sequentially.

    Flow:
        1. Battery Guardian: Battery safety assessment
        2. Grid Tariff: Energy market analysis
        3. Smart Trip: Comprehensive travel plan
    """

    def __init__(self, model_path: str = None, feature_stats: dict = None):
        """
        Initialises the orchestrator: creates tools and agents.

        Args:
            model_path: Trained DL model file path
            feature_stats: Feature normalisation statistics (means, stds)
        """
        logger.header("⚡ VOLTOPTIMIZER - AGENT ORCHESTRATION")
        logger.log("orchestrator", "Initialising system...")

        logger.log("orchestrator", "Loading tools...")
        self.battery_rul_tool = BatteryRULTool(model_path=model_path)
        if feature_stats is not None:
            self.battery_rul_tool.set_feature_stats(feature_stats)
        self.grid_api_tool = GridTariffAPITool()
        self.route_planner_tool = RoutePlannerTool()

        logger.log("orchestrator", "Creating agents...")
        self.battery_guardian = BatteryGuardianAgent(self.battery_rul_tool)
        self.grid_tariff = GridTariffAgent(self.grid_api_tool)
        self.smart_trip = SmartTripAgent(self.route_planner_tool)

        logger.log("orchestrator",
                    "System ready! 3 agents active, 3 tools loaded.")

    def run_scenario(self, scenario: dict) -> dict:
        """
        Runs a scenario end-to-end.

        Triggers all agents in sequence and coordinates
        the communication between them.

        Args:
            scenario: {
                "name": str,
                "battery_age": float,
                "ambient_temp": float,
                "current_soc": float,
                "origin": str,
                "destination": str,
                "total_distance_km": float,
            }

        Returns:
            Consolidated scenario result
        """
        logger.separator("═")
        logger.header(f"🚗 SCENARIO: {scenario.get('name', 'Unnamed')}")
        logger.separator("═")

        start_time = time.time()

        scenario_name = scenario.get("name", "Unnamed Scenario")
        battery_age = scenario.get("battery_age", 0.7)
        ambient_temp = scenario.get("ambient_temp", 45.0)
        current_soc = scenario.get("current_soc", 25.0)
        origin = scenario.get("origin", "Izmir")
        destination = scenario.get("destination", "Istanbul")
        distance = scenario.get("total_distance_km", 600.0)

        logger.log("orchestrator", "Scenario parameters:")
        logger.log("orchestrator",
                    f"  🔋 Battery Age: {battery_age:.1%} "
                    f"(0=New, 1=End-of-life)")
        logger.log("orchestrator",
                    f"  🌡️  Ambient Temperature: {ambient_temp}°C")
        logger.log("orchestrator",
                    f"  ⚡ Current SoC: {current_soc}%")
        logger.log("orchestrator",
                    f"  📍 Route: {origin} → {destination} ({distance} km)")

        # ══════════════════════════════════════════════════
        # PHASE 1: Battery Guardian Agent
        # ══════════════════════════════════════════════════
        logger.separator("─")
        logger.log("orchestrator",
                    "PHASE 1/3: Triggering Battery Guardian Agent...")

        guardian_result = self.battery_guardian.execute({
            "battery_age": battery_age,
            "ambient_temp": ambient_temp,
            "current_soc": current_soc,
        })

        # Pass safety constraints downstream so Grid Tariff can calculate
        # the correct required energy (capacity × SoC delta) rather than
        # using a hardcoded 50 kWh estimate
        self.battery_guardian.send_message(
            self.grid_tariff,
            {
                "type": "safety_constraints",
                "max_charge_soc": guardian_result["charge_parameters"][
                    "max_charge_soc"],
                "max_current": guardian_result["charge_parameters"][
                    "max_charge_current_a"],
                "safety_level": guardian_result["charge_parameters"][
                    "safety_level"],
            }
        )

        # ══════════════════════════════════════════════════
        # PHASE 2: Grid Tariff Agent
        # ══════════════════════════════════════════════════
        logger.separator("─")
        logger.log("orchestrator",
                    "PHASE 2/3: Triggering Grid Tariff Agent...")

        max_soc = guardian_result["charge_parameters"]["max_charge_soc"]

        grid_result = self.grid_tariff.execute({
            "required_energy_kwh": 50.0,
            # Approximate pack power (kW) from current limit (A):
            # P = I × V_nominal ≈ I × 400V → factor of 0.4 converts A → kW
            "charge_power_kw": guardian_result["charge_parameters"][
                "max_charge_current_a"] * 0.4,
            "max_charge_soc": max_soc,
            "battery_capacity_kwh": 75.0,
            "current_soc": current_soc,
        })

        # Grid → Trip: Send message
        self.grid_tariff.send_message(
            self.smart_trip,
            {
                "type": "tariff_data",
                "optimal_cost": grid_result["optimal_charge_plan"][
                    "optimal_total_cost_tl"],
                "cheapest_window": grid_result["tariff_analysis"][
                    "cheapest_window"],
            }
        )

        # ══════════════════════════════════════════════════
        # PHASE 3: Smart Trip Agent
        # ══════════════════════════════════════════════════
        logger.separator("─")
        logger.log("orchestrator",
                    "PHASE 3/3: Triggering Smart Trip Agent...")

        trip_result = self.smart_trip.execute({
            "origin": origin,
            "destination": destination,
            "total_distance_km": distance,
            "current_soc": current_soc,
            "ambient_temperature": ambient_temp,
            "battery_guardian_report": guardian_result,
            "grid_tariff_report": grid_result,
        })

        # ══════════════════════════════════════════════════
        # CONSOLIDATED RESULT
        # ══════════════════════════════════════════════════
        elapsed = time.time() - start_time

        logger.separator("═")
        logger.header("📋 VOLTOPTIMIZER - CONSOLIDATED RESULT REPORT")

        self._print_final_report(
            scenario_name, guardian_result, grid_result,
            trip_result, elapsed
        )

        return {
            "scenario": scenario,
            "battery_guardian_result": guardian_result,
            "grid_tariff_result": grid_result,
            "smart_trip_result": trip_result,
            "execution_time_seconds": round(elapsed, 2),
        }

    def _print_final_report(
        self,
        scenario_name: str,
        guardian: dict,
        grid: dict,
        trip: dict,
        elapsed: float,
    ):
        """Prints the final report to the terminal."""
        print()
        print("╔" + "═" * 68 + "╗")
        print(f"║{'VoltOptimizer - Consolidated Result Report':^68}║")
        print(f"║{'Scenario: ' + scenario_name:^68}║")
        print("╠" + "═" * 68 + "╣")

        # Battery Safety
        cp = guardian["charge_parameters"]
        print(f"║ 🛡️  BATTERY SAFETY ASSESSMENT"
              f"{' ' * 37}║")
        print(f"║   RUL Estimate   : {guardian['rul_assessment']['rul_percentage']:<43.1f}%║")
        print(f"║   Health Status  : {guardian['rul_assessment']['health_status']:<46}║")
        print(f"║   Temperature    : {guardian['temperature']['current']:<40.1f}°C   ║")
        print(f"║   Safety Level   : {cp['safety_level']:<46}║")
        print(f"║   Charge Limit   : {cp['max_charge_soc']:<43.0f}%║")
        print(f"║   Max Current    : {cp['max_charge_current_a']:<43.0f}A   ║")

        print("╠" + "═" * 68 + "╣")

        # Energy Prices
        cc = grid.get("cost_comparison", {})
        print(f"║ 💰 ENERGY MARKET ANALYSIS"
              f"{' ' * 41}║")
        print(f"║   Optimal Cost   : {cc.get('optimal_cost_tl', 0):<42.2f} TL  ║")
        print(f"║   Peak Cost      : {cc.get('peak_cost_tl', 0):<42.2f} TL  ║")
        print(f"║   Savings        : {cc.get('savings_tl', 0):<35.2f} TL ({cc.get('savings_percentage', 0):.0f}%) ║")

        print("╠" + "═" * 68 + "╣")

        # Travel Plan
        ts = trip.get("trip_summary", {})
        print(f"║ 🗺️  TRAVEL PLAN"
              f"{' ' * 51}║")
        print(f"║   Total Distance : {ts.get('total_distance_km', 0):<42.0f} km  ║")
        print(f"║   Charge Stops   : {ts.get('total_charge_stops', 0):<46}║")
        print(f"║   Charge Time    : {ts.get('total_charge_time_min', 0):<42.0f} min ║")
        print(f"║   Total Time     : {ts.get('total_trip_time', 0):<40.1f} hours ║")
        print(f"║   Total Cost     : {ts.get('total_cost', 0):<42.2f} TL  ║")
        print(f"║   Arrival SoC    : {ts.get('arrival_soc', 0):<43.1f}%║")

        print("╠" + "═" * 68 + "╣")

        # Charge Stops
        stops = trip.get("charge_stops_detail", [])
        if stops:
            print(f"║ ⚡ CHARGE STOPS"
                  f"{' ' * 51}║")
            for stop in stops:
                name = stop.get("station", "Unknown")
                if len(name) > 40:
                    name = name[:37] + "..."
                print(f"║   {stop['stop_number']}. {name:<63}║"[:71] + "║")
                print(f"║      SoC: {stop['arrival_soc']} → {stop['departure_soc']} | "
                      f"{stop['duration']} | {stop['power_kw']} | "
                      f"{stop['cost']:<10}      ║")

            print("╠" + "═" * 68 + "╣")

        # Safety Actions
        actions = guardian.get("actions_taken", [])
        if actions:
            print(f"║ 📋 SAFETY ACTIONS TAKEN"
                  f"{' ' * 43}║")
            for action in actions[:5]:
                text = action[:64]
                print(f"║   {text:<65}║")

        print("╠" + "═" * 68 + "╣")
        print(f"║ ⏱️  Execution Time: {elapsed:.2f} seconds"
              f"{' ' * (45 - len(f'{elapsed:.2f}'))}║")
        print("╚" + "═" * 68 + "╝")
        print()
