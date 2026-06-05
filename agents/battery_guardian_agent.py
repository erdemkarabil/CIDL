"""
VoltOptimizer - Battery Guardian Agent
=======================================
Battery Health Guardian and Safety Controller Agent.

Tasks:
    1. Regularly read RUL and anomaly outputs from the DL model
    2. Dynamically limit charge parameters on critical temperature / low RUL
    3. Apply the 80% rule (battery lifetime preservation)
    4. Revise safety decisions via self-correction

Reasoning Process (ReAct):
    Observe → Temperature/RUL analysis → Risk assessment →
    Charge parameter adjustment → Report result
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from config import BATTERY_LIMITS
from utils.logger import logger


class BatteryGuardianAgent(BaseAgent):
    """
    Battery Health Guardian Agent.

    Uses the DL model as a Tool to monitor battery state
    and dynamically adjust safety parameters.
    """

    def __init__(self, battery_rul_tool):
        super().__init__(
            role="Battery Guardian Agent",
            goal=("Protect battery health, dynamically adjust safety limits, "
                  "and respond to anomalies"),
            backstory=(
                "I am VoltOptimizer's Battery Health Guardian. "
                "To maximise the lifetime of EV batteries and minimise safety "
                "risks, I interpret the predictions of the deep learning model "
                "and make real-time decisions. I analyse battery temperature, "
                "voltage, and RUL data to automatically adjust charge current "
                "and the charge SoC ceiling."
            ),
            tools=[battery_rul_tool],
        )

        self.battery_rul_tool = battery_rul_tool
        self.limits = BATTERY_LIMITS.copy()

        self.current_charge_limit_soc = self.limits["max_charge_soc"]
        self.current_max_charge_current = self.limits["max_current"]
        self.safety_level = "NORMAL"

    def execute(self, task_input: dict) -> dict:
        """
        Main task: assess battery state and adjust safety parameters.

        Args:
            task_input: {
                "battery_age": float,
                "ambient_temp": float,
                "current_soc": float,
            }

        Returns:
            Safety assessment and adjusted parameters
        """
        logger.header("🛡️  BATTERY GUARDIAN AGENT")
        logger.log("battery_guardian",
                    "Starting battery safety assessment...")

        battery_age = task_input.get("battery_age", 0.7)
        ambient_temp = task_input.get("ambient_temp", 45.0)
        current_soc = task_input.get("current_soc", 25.0)

        # ══════════════════════════════════════════════
        # STEP 1: Call DL Model as a Tool
        # ══════════════════════════════════════════════
        self.reason({"step": "calling DL model"}, step=1)
        logger.agent_thinking(
            self.role,
            "I will call the DL model to get a real-time RUL estimate. "
            "This prediction is critical for determining charge parameters.",
            step=1,
        )

        rul_result = self.use_tool(
            tool_name="BatteryRULTool",
            tool_instance=self.battery_rul_tool,
            method="get_realtime_battery_assessment",
            battery_age=battery_age,
            ambient_temp=ambient_temp,
            current_soc=current_soc,
        )

        # ══════════════════════════════════════════════
        # STEP 2: RUL and Anomaly Analysis
        # ══════════════════════════════════════════════
        rul_pct = rul_result["rul_percentage"]
        health = rul_result["health_status"]
        anomalies = rul_result["anomaly_details"]
        temp = rul_result["temperature_celsius"]
        temp_status = rul_result["temperature_status"]

        self.reason({
            "rul": rul_pct,
            "health": health,
            "anomalies": anomalies,
            "temperature": temp,
        }, step=2)

        logger.agent_thinking(
            self.role,
            f"RUL estimate: {rul_pct:.1f}% | Health: {health} | "
            f"Temperature: {temp:.1f}°C ({temp_status})",
            step=2,
        )

        if anomalies:
            for anomaly in anomalies:
                logger.agent_thinking(
                    self.role, f"⚠️ ANOMALY: {anomaly}", step=2
                )

        # ══════════════════════════════════════════════
        # STEP 3: Risk Assessment and Parameter Adjustment
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            "Performing risk assessment. I will adjust charge parameters "
            "based on temperature and RUL.",
            step=3,
        )

        charge_decisions = self._evaluate_and_adjust(
            rul_pct, health, temp, temp_status, anomalies
        )

        # ══════════════════════════════════════════════
        # STEP 4: Self-Correction
        # ══════════════════════════════════════════════
        charge_decisions = self._self_correct_decisions(
            charge_decisions, rul_pct, temp
        )

        # ══════════════════════════════════════════════
        # STEP 5: Result Report
        # ══════════════════════════════════════════════
        result = {
            "rul_assessment": {
                "rul_percentage": rul_pct,
                "health_status": health,
                "anomalies": anomalies,
            },
            "temperature": {
                "current": temp,
                "status": temp_status,
            },
            "charge_parameters": {
                "max_charge_soc": charge_decisions["max_charge_soc"],
                "max_charge_current_a": charge_decisions["max_current"],
                "safety_level": charge_decisions["safety_level"],
            },
            "actions_taken": charge_decisions["actions"],
            "recommendations": rul_result["recommendations"],
        }

        self.outputs["safety_assessment"] = result

        logger.agent_result(
            self.role,
            f"Safety Level: {charge_decisions['safety_level']} | "
            f"Charge Limit: {charge_decisions['max_charge_soc']:.0f}% | "
            f"Max Current: {charge_decisions['max_current']:.0f}A"
        )

        return result

    def _evaluate_and_adjust(
        self,
        rul: float,
        health: str,
        temp: float,
        temp_status: str,
        anomalies: list,
    ) -> dict:
        """
        Evaluates risk and adjusts charge parameters.

        Rule-based reasoning:
            - CRITICAL: Charge limit 60%, reduce current to 20%
            - WARNING:  Charge limit 80%, reduce current to 50%
            - NORMAL:   Full power
        """
        actions = []

        max_soc = self.limits["max_charge_soc"]
        max_current = self.limits["max_current"]
        safety_level = "NORMAL"

        # --- CRITICAL State ---
        if health == "CRITICAL":
            safety_level = "CRITICAL"
            max_soc = 60.0
            max_current = 30.0
            actions.append(
                f"🔴 CRITICAL: RUL={rul:.1f}%. "
                f"Charge limit reduced to 60%, current to 30A."
            )

            if temp > self.limits["max_temperature"]:
                max_current = 15.0
                actions.append(
                    f"🔴 OVERTEMPERATURE ({temp:.1f}°C): "
                    f"Current reduced to 15A. Cooling required."
                )

        # --- WARNING State ---
        elif health == "WARNING":
            safety_level = "WARNING"
            # 80% rule: lithium-ion cells age significantly faster above 80% SoC
            # due to lithium plating; capping here extends usable lifetime
            max_soc = 80.0
            max_current = 80.0
            actions.append(
                f"🟡 WARNING: RUL={rul:.1f}%. "
                f"80% rule applied to protect battery lifetime."
            )

            if temp > self.limits["warning_temperature"]:
                max_current = 50.0
                actions.append(
                    f"🟡 High temperature ({temp:.1f}°C): "
                    f"Current limited to 50A."
                )

        # --- NORMAL State ---
        else:
            safety_level = "NORMAL"
            max_soc = 90.0  # Conservative normal
            max_current = 150.0
            actions.append(
                f"🟢 NORMAL: RUL={rul:.1f}%. Full-power charging possible."
            )

        # Temperature-based additional adjustment
        if temp_status == "COLD":
            max_current = min(max_current, 50.0)
            actions.append(
                f"❄️ Cold weather ({temp:.1f}°C): Current limited to 50A, "
                f"battery pre-heating recommended."
            )

        for action in actions:
            logger.agent_action(self.role, action)

        self.current_charge_limit_soc = max_soc
        self.current_max_charge_current = max_current
        self.safety_level = safety_level

        return {
            "max_charge_soc": max_soc,
            "max_current": max_current,
            "safety_level": safety_level,
            "actions": actions,
        }

    def _self_correct_decisions(
        self, decisions: dict, rul: float, temp: float
    ) -> dict:
        """
        Self-correction mechanism.

        Re-examines the decisions made:
          - Are they overly restrictive?
          - Are they safe enough?
        """
        logger.agent_thinking(
            self.role,
            "Reviewing my decisions (self-correction)...",
            step=4,
        )

        corrected = False

        # Case 1: good RUL but elevated temperature — the initial pass may have
        # been overly conservative; relax SoC limit since the real risk here
        # is thermal, not capacity degradation
        if rul > 70 and temp > self.limits["warning_temperature"]:
            if decisions["max_charge_soc"] < 80:
                decisions["max_charge_soc"] = 85.0
                decisions["actions"].append(
                    "↩️ Correction: RUL is high ({:.1f}%), charge limit raised "
                    "to 85% (thermal risk only).".format(rul)
                )
                corrected = True

        # Case 2: Overly restrictive current (but temperature is normal)
        if (decisions["max_current"] < 30
                and temp < self.limits["warning_temperature"]):
            decisions["max_current"] = 50.0
            decisions["actions"].append(
                "↩️ Correction: Temperature normal, current raised to 50A."
            )
            corrected = True

        # Case 3: Everything critical but not restricted enough
        if rul < 10 and temp > 55:
            decisions["max_current"] = 10.0
            decisions["max_charge_soc"] = 50.0
            decisions["safety_level"] = "EMERGENCY"
            decisions["actions"].append(
                "🚨 EMERGENCY CORRECTION: Extremely low RUL and high temperature! "
                "Current 10A, charge limit 50%. SERVICE REQUIRED!"
            )
            corrected = True

        if corrected:
            logger.agent_thinking(
                self.role,
                "Decisions corrected and updated.",
                step=4,
            )
        else:
            logger.agent_thinking(
                self.role,
                "Decisions are consistent, no correction needed.",
                step=4,
            )

        return decisions
