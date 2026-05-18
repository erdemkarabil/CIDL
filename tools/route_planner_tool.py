"""
VoltOptimizer - Rota Planlama Aracı (Mock Google Maps Tool)
============================================================
Kullanıcının gitmek istediği rotayı, hava koşullarını ve yol eğimini
simüle eden bir araç. Smart Trip Agent bu aracı kullanarak optimum
seyahat planı oluşturur.

Simülasyon:
    - Başlangıç → Varış arası rota
    - Yol üzerindeki şarj istasyonları
    - Hava sıcaklığı ve yol eğimi
    - Enerji tüketim tahmini
"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROUTE_CONFIG
from utils.logger import logger


class RoutePlannerTool:
    """
    Mock Google Maps benzeri rota planlama aracı.

    Gerçek harita verisi yerine fizik tabanlı simülasyon kullanarak:
      - Mesafe ve süre hesaplar
      - Yol eğimi ve hava durumunu simüle eder
      - Şarj istasyonlarını konumlandırır
      - Enerji tüketim tahminleri yapar
    """

    # Simüle edilmiş şarj istasyonları veritabanı
    CHARGING_STATIONS = [
        {
            "id": "CS-001", "name": "İzmir Merkez Hızlı Şarj",
            "location_km": 0, "type": "fast_dc",
            "power_kw": 150, "price_kwh": 3.50, "available": True
        },
        {
            "id": "CS-002", "name": "Manisa Yol Üstü Şarj",
            "location_km": 85, "type": "normal_dc",
            "power_kw": 50, "price_kwh": 2.80, "available": True
        },
        {
            "id": "CS-003", "name": "Uşak Dinlenme Tesisi Şarj",
            "location_km": 195, "type": "fast_dc",
            "power_kw": 120, "price_kwh": 3.20, "available": True
        },
        {
            "id": "CS-004", "name": "Afyon Otoyol Şarj",
            "location_km": 310, "type": "fast_dc",
            "power_kw": 150, "price_kwh": 3.00, "available": True
        },
        {
            "id": "CS-005", "name": "Eskişehir Şehir Şarj",
            "location_km": 420, "type": "normal_dc",
            "power_kw": 50, "price_kwh": 2.50, "available": True
        },
        {
            "id": "CS-006", "name": "Bolu Dağ Geçidi Şarj",
            "location_km": 530, "type": "fast_dc",
            "power_kw": 100, "price_kwh": 3.30, "available": True
        },
        {
            "id": "CS-007", "name": "İstanbul Giriş Şarj",
            "location_km": 600, "type": "fast_dc",
            "power_kw": 150, "price_kwh": 3.80, "available": True
        },
    ]

    def __init__(self):
        self.config = ROUTE_CONFIG
        logger.log("tool", "RoutePlannerTool başlatıldı (Simülasyon modu)")

    def _get_temperature_efficiency(self, temp: float) -> float:
        """Sıcaklığa bağlı verimlilik faktörünü döndürür."""
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
        origin: str = "İzmir",
        destination: str = "İstanbul",
        total_distance_km: float = 600.0,
        ambient_temperature: float = 35.0,
        avg_elevation_change_m: float = 500.0,
    ) -> dict:
        """
        İki nokta arası rota planı oluşturur.

        Args:
            origin: Başlangıç noktası
            destination: Varış noktası
            total_distance_km: Toplam mesafe (km)
            ambient_temperature: Ortam sıcaklığı (°C)
            avg_elevation_change_m: Ortalama yükseklik değişimi (m)

        Returns:
            Detaylı rota bilgisi sözlüğü
        """
        logger.log("tool",
                    f"Rota planlanıyor: {origin} → {destination} "
                    f"({total_distance_km:.0f} km)")

        # Verimlilik hesabı
        temp_efficiency = self._get_temperature_efficiency(
            ambient_temperature)
        elevation_factor = (1 + self.config["elevation_factor_per_100m"]
                            * (avg_elevation_change_m / 100))

        # Gerçek enerji tüketimi (kWh/km)
        base_consumption = self.config["ev_consumption_kwh_per_km"]
        actual_consumption = (base_consumption
                              * (1 / temp_efficiency)
                              * elevation_factor)

        # Toplam enerji ihtiyacı
        total_energy_needed = actual_consumption * total_distance_km

        # Menzil hesabı
        battery_capacity = self.config["ev_battery_capacity_kwh"]
        actual_range = battery_capacity / actual_consumption

        # Yol segmentleri
        segments = self._create_route_segments(
            total_distance_km, ambient_temperature
        )

        # Uygun şarj istasyonlarını filtrele
        relevant_stations = [
            s for s in self.CHARGING_STATIONS
            if s["location_km"] <= total_distance_km
        ]

        result = {
            "origin": origin,
            "destination": destination,
            "total_distance_km": total_distance_km,
            "estimated_duration_hours": round(
                total_distance_km / 85, 1),  # Ort. 85 km/h
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
                    f"Rota hazır → Menzil: {actual_range:.0f} km, "
                    f"İhtiyaç: {total_energy_needed:.1f} kWh, "
                    f"Şarj gerekli: {'Evet' if result['needs_charging'] else 'Hayır'}")

        return result

    def _create_route_segments(
        self,
        total_distance: float,
        base_temp: float,
    ) -> list:
        """Rotayı segmentlere ayırır."""
        num_segments = max(4, int(total_distance / 100))
        segment_distance = total_distance / num_segments

        segments = []
        for i in range(num_segments):
            start_km = i * segment_distance
            end_km = (i + 1) * segment_distance

            # Yükseklik varyasyonu
            elevation = np.random.normal(0, 200)

            # Sıcaklık varyasyonu (yüksekliğe bağlı)
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
                    ["Otoyol", "Devlet Yolu", "Dağ Geçidi"],
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
        Belirli bir istasyonda şarj duraklama detaylarını hesaplar.

        Args:
            station: Şarj istasyonu bilgisi
            current_soc: Mevcut SoC (%)
            target_soc: Hedef SoC (%)
            battery_capacity_kwh: Batarya kapasitesi (kWh)

        Returns:
            {"charge_time_min": float, "energy_kwh": float, "cost_tl": float}
        """
        capacity = battery_capacity_kwh or self.config["ev_battery_capacity_kwh"]

        soc_diff = target_soc - current_soc
        energy_needed = capacity * (soc_diff / 100.0)

        # Şarj süresi (sabit güç varsayımı + %80 üstü yavaşlama)
        if target_soc > 80:
            # 80%'e kadar normal, sonrası yavaş
            energy_to_80 = max(0, capacity * ((80 - current_soc) / 100.0))
            energy_80_to_target = capacity * ((target_soc - 80) / 100.0)

            time_to_80 = energy_to_80 / station["power_kw"]
            # %80 üstü → güç %40'a düşer
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
        Optimal seyahat planı oluşturur: nerede, ne kadar, ne maliyetle
        şarj edilecek.

        Args:
            origin: Başlangıç noktası
            destination: Varış noktası
            current_soc: Mevcut şarj durumu (%)
            battery_capacity_kwh: Batarya kapasitesi
            max_charge_soc: Maksimum şarj limiti (%)
            ambient_temperature: Ortam sıcaklığı
            total_distance_km: Toplam mesafe

        Returns:
            Detaylı seyahat planı sözlüğü
        """
        logger.log("tool",
                    f"Seyahat optimizasyonu: {origin} → {destination}, "
                    f"SoC: %{current_soc:.0f}, "
                    f"Sıcaklık: {ambient_temperature}°C")

        capacity = battery_capacity_kwh or self.config["ev_battery_capacity_kwh"]

        # Rota bilgisi
        route = self.plan_route(
            origin, destination, total_distance_km,
            ambient_temperature,
        )

        consumption = route["actual_consumption_kwh_km"]
        current_energy = capacity * (current_soc / 100.0)

        # Simülasyon: km km ilerle, şarj gerektiğinde dur
        charge_stops = []
        current_km = 0
        total_charge_cost = 0
        total_charge_time = 0

        # Başlangıçta menzil çok düşükse, çıkış noktasında şarj et
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
            # Başlangıç istasyonunda (km 0) şarj et
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
                           f"Çıkış noktasında şarj gerekli → "
                           f"{station['name']} ({charge_info['charge_time_min']:.0f} dk)")

        while current_km < total_distance_km:
            # Mevcut enerji ile gidilebilecek mesafe
            remaining_range = current_energy / consumption

            # Bir sonraki şarj istasyonu
            next_stations = [
                s for s in route["charging_stations"]
                if s["location_km"] > current_km and s["available"]
            ]

            # Varışa ulaşabilir miyiz?
            remaining_distance = total_distance_km - current_km
            if remaining_range >= remaining_distance + 20:  # +20 km güvenlik
                current_km = total_distance_km
                current_energy -= remaining_distance * consumption
                break

            # En yakın ulaşılabilir istasyonda dur
            station_found = False
            for station in next_stations:
                dist_to_station = station["location_km"] - current_km
                if dist_to_station <= remaining_range - 10:  # 10 km güvenlik
                    # İstasyona git
                    energy_used = dist_to_station * consumption
                    current_energy -= energy_used
                    current_km = station["location_km"]

                    # Şarj et
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
                # Menzil dışı → Acil uyarı
                logger.log("warning",
                            "Menzil dışı! Şarj istasyonuna ulaşılamıyor.")
                break

        # Varış SoC
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
                    f"Seyahat planı hazır → "
                    f"{len(charge_stops)} şarj durağı, "
                    f"toplam maliyet: {total_charge_cost:.2f} TL, "
                    f"toplam süre: "
                    f"{trip_plan['summary']['total_trip_time_hours']:.1f} saat")

        return trip_plan
