"""
VoltOptimizer - Route Planner Tool (Mock Google Maps Tool)
==========================================================
A tool that simulates the route, weather conditions, and road gradient
for a given trip. Smart Trip Agent uses this tool to build the optimal
travel plan.

Simulation:
    - Route from origin to destination
    - Charging stations along the route
    - Ambient temperature and road gradient
    - Energy consumption estimate
"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROUTE_CONFIG
from utils.logger import logger


class RoutePlannerTool:
    """
    Mock Google Maps-like route planning tool.

    Uses physics-based simulation instead of real map data to:
      - Calculate distance and travel time
      - Simulate road gradient and weather
      - Locate charging stations
      - Estimate energy consumption
    """

    # Simulated charging station database
    CHARGING_STATIONS = [
        {
            "id": "CS-001", "name": "Izmir City Centre Fast Charger",
            "location_km": 0, "type": "fast_dc",
            "power_kw": 150, "price_kwh": 3.50, "available": True
        },
        {
            "id": "CS-002", "name": "Manisa Roadside Charger",
            "location_km": 85, "type": "normal_dc",
            "power_kw": 50, "price_kwh": 2.80, "available": True
        },
        {
            "id": "CS-003", "name": "Usak Rest Area Charger",
            "location_km": 195, "type": "fast_dc",
            "power_kw": 120, "price_kwh": 3.20, "available": True
        },
        {
            "id": "CS-004", "name": "Afyon Motorway Charger",
            "location_km": 310, "type": "fast_dc",
            "power_kw": 150, "price_kwh": 3.00, "available": True
        },
        {
            "id": "CS-005", "name": "Eskisehir City Charger",
            "location_km": 420, "type": "normal_dc",
            "power_kw": 50, "price_kwh": 2.50, "available": True
        },
        {
            "id": "CS-006", "name": "Bolu Mountain Pass Charger",
            "location_km": 530, "type": "fast_dc",
            "power_kw": 100, "price_kwh": 3.30, "available": True
        },
        {
            "id": "CS-007", "name": "Istanbul Entrance Charger",
            "location_km": 600, "type": "fast_dc",
            "power_kw": 150, "price_kwh": 3.80, "available": True
        },
    ]

    def __init__(self):
        self.config = ROUTE_CONFIG
        logger.log("tool", "RoutePlannerTool initialised (Simulation mode)")

    def _get_temperature_efficiency(self, temp: float) -> float:
        """Returns the efficiency factor based on temperature."""
        factors = self.config["temperature_efficiency_factor"]
        if temp < 5:
            return factors["cold"]
        elif temp < 15:
            return factors["cool"]
        elif temp <= 30:
            return factors["optimal"]
        elif temp <= 40:
            return factors["warm"]
        else:
            return factors["hot"]

    def plan_route(
        self,
        origin: str = "Izmir",
        destination: str = "Istanbul",
        total_distance_km: float = 600.0,
        ambient_temperature: float = 35.0,
        avg_elevation_change_m: float = 500.0,
    ) -> dict:
        """
        Creates a route plan between two points.

        Args:
            origin: Starting point
            destination: Destination
            total_distance_km: Total distance (km)
            ambient_temperature: Ambient temperature (°C)
            avg_elevation_change_m: Average elevation change (m)

        Returns:
            Detailed route information dictionary
        """
        logger.log("tool",
                    f"Planning route: {origin} → {destination} "
                    f"({total_distance_km:.0f} km)")

        temp_efficiency = self._get_temperature_efficiency(
            ambient_temperature)
        elevation_factor = (1 + self.config["elevation_factor_per_100m"]
                            * (avg_elevation_change_m / 100))

        # Actual energy consumption (kWh/km)
        base_consumption = self.config["ev_consumption_kwh_per_km"]
        actual_consumption = (base_consumption
                              * (1 / temp_efficiency)
                              * elevation_factor)

        total_energy_needed = actual_consumption * total_distance_km

        battery_capacity = self.config["ev_battery_capacity_kwh"]
        actual_range = battery_capacity / actual_consumption

        segments = self._create_route_segments(
            total_distance_km, ambient_temperature
        )

        relevant_stations = [
            s for s in self.CHARGING_STATIONS
            if s["location_km"] <= total_distance_km
        ]

        result = {
            "origin": origin,
            "destination": destination,
            "total_distance_km": total_distance_km,
            "estimated_duration_hours": round(
                total_distance_km / 85, 1),  # Avg 85 km/h
            "ambient_temperature": ambient_temperature,
            "temperature_efficiency": round(temp_efficiency, 2),
            "elevation_change_m": avg_elevation_change_m,
            "elevation_factor": round(elevation_factor, 3),
            "base_consumption_kwh_km": base_consumption,
            "actual_consumption_kwh_km": round(actual_consumption, 4),
            "total_energy_needed_kwh": round(total_energy_needed, 2),
            "battery_capacity_kwh": battery_capacity,
            "actual_range_km": round(actual_range, 1),
            "segments": segments,
            "charging_stations": relevant_stations,
            "needs_charging": total_distance_km > actual_range,
        }

        logger.log("tool",
                    f"Route ready → Range: {actual_range:.0f} km, "
                    f"Needed: {total_energy_needed:.1f} kWh, "
                    f"Charging required: {'Yes' if result['needs_charging'] else 'No'}")

        return result

    def _create_route_segments(
        self,
        total_distance: float,
        base_temp: float,
    ) -> list:
        """Divides the route into segments."""
        num_segments = max(4, int(total_distance / 100))
        segment_distance = total_distance / num_segments

        segments = []
        for i in range(num_segments):
            start_km = i * segment_distance
            end_km = (i + 1) * segment_distance

            elevation = np.random.normal(0, 200)

            # Temperature varies with altitude
            temp = base_temp - (abs(elevation) / 1000) * 6.5
            temp += np.random.normal(0, 2)

            segments.append({
                "segment_id": i + 1,
                "start_km": round(start_km, 1),
                "end_km": round(end_km, 1),
                "distance_km": round(segment_distance, 1),
                "elevation_change_m": round(elevation, 0),
                "temperature": round(temp, 1),
                "road_type": np.random.choice(
                    ["Motorway", "State Road", "Mountain Pass"],
                    p=[0.6, 0.3, 0.1]
                ),
            })

        return segments

    def calculate_charge_stop(
        self,
        station: dict,
        current_soc: float,
        target_soc: float,
        battery_capacity_kwh: float = None,
    ) -> dict:
        """
        Calculates charge stop details at a given station.

        Args:
            station: Charging station information
            current_soc: Current SoC (%)
            target_soc: Target SoC (%)
            battery_capacity_kwh: Battery capacity (kWh)

        Returns:
            {"charge_time_min": float, "energy_kwh": float, "cost_tl": float}
        """
        capacity = battery_capacity_kwh or self.config["ev_battery_capacity_kwh"]

        soc_diff = target_soc - current_soc
        energy_needed = capacity * (soc_diff / 100.0)

        # CC/CV charging model: constant-current (CC) up to 80% SoC, then
        # constant-voltage (CV) taper reduces effective power to ~40% to
        # protect the cells from lithium plating at high states of charge
        if target_soc > 80:
            energy_to_80 = max(0, capacity * ((80 - current_soc) / 100.0))
            energy_80_to_target = capacity * ((target_soc - 80) / 100.0)

            time_to_80 = energy_to_80 / station["power_kw"]
            time_80_to_target = energy_80_to_target / (
                station["power_kw"] * 0.4)

            charge_time_hours = time_to_80 + time_80_to_target
        else:
            charge_time_hours = energy_needed / station["power_kw"]

        charge_time_min = charge_time_hours * 60
        cost = energy_needed * station["price_kwh"]

        return {
            "station_name": station["name"],
            "station_id": station["id"],
            "station_location_km": station["location_km"],
            "current_soc": round(current_soc, 1),
            "target_soc": round(target_soc, 1),
            "energy_kwh": round(energy_needed, 2),
            "charge_time_min": round(charge_time_min, 1),
            "charge_power_kw": station["power_kw"],
            "cost_tl": round(cost, 2),
            "price_kwh": station["price_kwh"],
        }

    def optimize_trip(
        self,
        origin: str,
        destination: str,
        current_soc: float,
        battery_capacity_kwh: float = None,
        max_charge_soc: float = 80.0,
        ambient_temperature: float = 35.0,
        total_distance_km: float = 600.0,
    ) -> dict:
        """
        Creates the optimal travel plan: where, how much, and at what cost
        to charge.

        Args:
            origin: Starting point
            destination: Destination
            current_soc: Current state of charge (%)
            battery_capacity_kwh: Battery capacity
            max_charge_soc: Maximum charge SoC limit (%)
            ambient_temperature: Ambient temperature
            total_distance_km: Total distance

        Returns:
            Detailed travel plan dictionary
        """
        logger.log("tool",
                    f"Trip optimisation: {origin} → {destination}, "
                    f"SoC: {current_soc:.0f}%, "
                    f"Temperature: {ambient_temperature}°C")

        capacity = battery_capacity_kwh or self.config["ev_battery_capacity_kwh"]

        route = self.plan_route(
            origin, destination, total_distance_km,
            ambient_temperature,
        )

        consumption = route["actual_consumption_kwh_km"]
        current_energy = capacity * (current_soc / 100.0)

        # Simulation: advance km by km, stop when charging is needed
        charge_stops = []
        current_km = 0
        total_charge_cost = 0
        total_charge_time = 0

        # If initial range is too low, charge at departure point
        initial_range = current_energy / consumption
        first_reachable = [
            s for s in route["charging_stations"]
            if s["location_km"] > 0 and s["available"]
        ]
        needs_initial_charge = (
            first_reachable
            and initial_range < first_reachable[0]["location_km"] + 10
        )

        if needs_initial_charge:
            origin_stations = [
                s for s in route["charging_stations"]
                if s["location_km"] == 0 and s["available"]
            ]
            if origin_stations:
                station = origin_stations[0]
                arrival_soc = current_soc
                charge_info = self.calculate_charge_stop(
                    station, arrival_soc, max_charge_soc, capacity
                )
                charge_stops.append(charge_info)
                current_energy = capacity * (max_charge_soc / 100.0)
                total_charge_cost += charge_info["cost_tl"]
                total_charge_time += charge_info["charge_time_min"]
                logger.log("tool",
                           f"Initial charge required → "
                           f"{station['name']} ({charge_info['charge_time_min']:.0f} min)")

        while current_km < total_distance_km:
            remaining_range = current_energy / consumption

            next_stations = [
                s for s in route["charging_stations"]
                if s["location_km"] > current_km and s["available"]
            ]

            remaining_distance = total_distance_km - current_km
            if remaining_range >= remaining_distance + 20:  # +20 km safety margin
                current_km = total_distance_km
                current_energy -= remaining_distance * consumption
                break

            station_found = False
            for station in next_stations:
                dist_to_station = station["location_km"] - current_km
                if dist_to_station <= remaining_range - 10:  # 10 km safety margin
                    energy_used = dist_to_station * consumption
                    current_energy -= energy_used
                    current_km = station["location_km"]

                    arrival_soc = (current_energy / capacity) * 100
                    charge_info = self.calculate_charge_stop(
                        station, arrival_soc, max_charge_soc, capacity
                    )

                    charge_stops.append(charge_info)
                    current_energy = capacity * (max_charge_soc / 100.0)

                    total_charge_cost += charge_info["cost_tl"]
                    total_charge_time += charge_info["charge_time_min"]
                    station_found = True
                    break

            if not station_found:
                # Out of range → emergency warning
                logger.log("warning",
                            "Out of range! Cannot reach a charging station.")
                break

        arrival_soc = max(0, (current_energy / capacity) * 100)

        trip_plan = {
            "route": {
                "origin": origin,
                "destination": destination,
                "total_distance_km": total_distance_km,
                "estimated_drive_time_hours": route[
                    "estimated_duration_hours"],
            },
            "battery": {
                "start_soc": current_soc,
                "arrival_soc": round(arrival_soc, 1),
                "max_charge_limit": max_charge_soc,
                "capacity_kwh": capacity,
            },
            "conditions": {
                "temperature": ambient_temperature,
                "temperature_efficiency": route["temperature_efficiency"],
                "actual_consumption_kwh_km": route[
                    "actual_consumption_kwh_km"],
            },
            "charge_stops": charge_stops,
            "summary": {
                "num_charge_stops": len(charge_stops),
                "total_charge_time_min": round(total_charge_time, 1),
                "total_charge_cost_tl": round(total_charge_cost, 2),
                "total_trip_time_hours": round(
                    route["estimated_duration_hours"]
                    + total_charge_time / 60, 1
                ),
            },
        }

        logger.log("tool",
                    f"Trip plan ready → "
                    f"{len(charge_stops)} charge stops, "
                    f"total cost: {total_charge_cost:.2f} TL, "
                    f"total time: "
                    f"{trip_plan['summary']['total_trip_time_hours']:.1f} hours")

        return trip_plan
