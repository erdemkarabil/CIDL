"""
VoltOptimizer - Şebeke Tarife API Aracı (Simülasyon)
=====================================================
Elektrik şebekesinin dinamik fiyat tarifelerini simüle eden bir Web/API
aracı. Grid Tariff Agent bu aracı kullanarak en ucuz ve en optimize
şarj saat aralıklarını belirler.

Simülasyon Özellikleri:
    - Gerçekçi Türkiye elektrik tarifeleri
    - Peak / Off-Peak / Super Off-Peak saat dilimleri
    - Mevsimsel fiyat varyasyonu
    - Anlık talep bazlı fiyat dalgalanması
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
    Simüle edilmiş Elektrik Şebeke Tarife API'si.

    Gerçek dünya API'sine benzer bir arayüz sunarak ajanların
    şebeke fiyatlarını sorgulamasını sağlar.
    """

    def __init__(self):
        self.config = GRID_CONFIG
        logger.log("tool", "GridTariffAPITool başlatıldı (Simülasyon modu)")

    def _get_price_for_hour(self, hour: int) -> dict:
        """
        Verilen saat için elektrik fiyatını döndürür.

        Args:
            hour: Saat (0-23)

        Returns:
            {"hour": int, "price_kwh": float, "tariff_type": str}
        """
        # Süper sakin saatler (gece)
        for start, end in self.config["super_off_peak_hours"]:
            if start <= hour < end:
                base_price = self.config["super_off_peak_price_kwh"]
                # Rastgele dalgalanma (%5)
                price = base_price * (1 + np.random.uniform(-0.05, 0.05))
                return {
                    "hour": hour,
                    "price_kwh": round(price, 2),
                    "tariff_type": "SÜPER_SAKİN",
                    "demand_level": "Çok Düşük",
                }

        # Yoğun saatler
        for start, end in self.config["peak_hours"]:
            if start <= hour < end:
                base_price = self.config["peak_price_kwh"]
                price = base_price * (1 + np.random.uniform(-0.08, 0.08))
                return {
                    "hour": hour,
                    "price_kwh": round(price, 2),
                    "tariff_type": "YOĞUN",
                    "demand_level": "Yüksek",
                }

        # Sakin saatler (varsayılan)
        base_price = self.config["off_peak_price_kwh"]
        price = base_price * (1 + np.random.uniform(-0.05, 0.05))
        return {
            "hour": hour,
            "price_kwh": round(price, 2),
            "tariff_type": "SAKİN",
            "demand_level": "Orta",
        }

    def get_24h_tariff_schedule(self) -> dict:
        """
        24 saatlik tarife çizelgesini döndürür.

        Returns:
            {
                "date": str,
                "currency": str,
                "schedule": [{"hour": int, "price_kwh": float, ...}],
                "cheapest_window": {"start": int, "end": int, ...},
                "most_expensive_window": {"start": int, "end": int, ...},
            }
        """
        logger.log("tool", "GridTariffAPITool: 24 saatlik tarife sorgulanıyor...")

        schedule = []
        for h in range(24):
            schedule.append(self._get_price_for_hour(h))

        # En ucuz ve en pahalı pencereleri bul
        prices = [s["price_kwh"] for s in schedule]

        # En ucuz ardışık 3 saatlik pencereyi bul
        best_cost = float("inf")
        best_start = 0
        for i in range(22):  # 3 saatlik pencere
            window_cost = sum(prices[i:i + 3])
            if window_cost < best_cost:
                best_cost = window_cost
                best_start = i

        # En pahalı ardışık 3 saatlik pencereyi bul
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
                    f"Tarife alındı → En ucuz: "
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
        Belirli bir enerji ihtiyacı için en uygun şarj pencerelerini önerir.

        Args:
            required_kwh: Gerekli enerji miktarı (kWh)
            charge_power_kw: Şarj gücü (kW)
            max_windows: Maksimum pencere sayısı

        Returns:
            [{start_hour, end_hour, price_kwh, estimated_cost, ...}]
        """
        logger.log("tool",
                    f"Optimal şarj penceresi hesaplanıyor: "
                    f"{required_kwh:.1f} kWh, {charge_power_kw:.0f} kW")

        charge_hours_needed = required_kwh / charge_power_kw
        schedule = self.get_24h_tariff_schedule()

        # Saatleri fiyata göre sırala
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

        # Saate göre sırala
        windows.sort(key=lambda x: x["hour"])

        # Ardışık saatleri birleştir
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
                    f"Optimal şarj planı hazır → "
                    f"{len(merged)} pencere, "
                    f"toplam maliyet: {total_cost:.2f} TL")

        return merged[:max_windows]

    def estimate_charge_cost(
        self,
        energy_kwh: float,
        start_hour: int,
    ) -> dict:
        """
        Belirli bir saatte şarj maliyetini hesaplar.

        Args:
            energy_kwh: Şarj edilecek enerji (kWh)
            start_hour: Başlangıç saati

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
