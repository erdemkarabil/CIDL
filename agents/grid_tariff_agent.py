"""
VoltOptimizer - Grid Tariff Agent
==================================
Akıllı Şebeke ve Enerji Piyasası Analisti Ajanı.

Görevler:
    1. Dinamik elektrik fiyat tarifelerini sorgulamak
    2. Peak / Off-Peak saatleri analiz etmek
    3. En ucuz ve en optimize şarj zamanlarını belirlemek
    4. Maliyet tasarruf önerileri sunmak

Akıl Yürütme Süreci:
    Tarife sorgula → Fiyat analizi → Pencere karşılaştırma →
    Optimal zaman belirleme → Tasarruf raporu
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.logger import logger


class GridTariffAgent(BaseAgent):
    """
    Akıllı Şebeke ve Enerji Piyasası Analisti.

    Simüle edilmiş Web/API aracını kullanarak elektrik fiyatlarını
    analiz eder ve en uygun şarj zamanlarını önerir.
    """

    def __init__(self, grid_api_tool):
        super().__init__(
            role="Grid Tariff Agent",
            goal=("Elektrik şebekesinin dinamik tarifelerini analiz ederek "
                  "en ucuz ve en verimli şarj zamanlarını belirlemek"),
            backstory=(
                "Ben VoltOptimizer'ın Enerji Piyasası Analistiyim. "
                "Türkiye'nin elektrik şebekesi dinamik tarifelerini "
                "anlık olarak takip ediyor ve analiz ediyorum. Amacım "
                "kullanıcının şarj maliyetini minimize ederken şebeke "
                "yükünü dengelemektir. Peak/Off-Peak saat dilimlerini, "
                "mevsimsel fiyat değişimlerini ve anlık talep seviyesini "
                "değerlendirerek akıllı öneriler sunuyorum."
            ),
            tools=[grid_api_tool],
        )

        self.grid_api_tool = grid_api_tool

    def execute(self, task_input: dict) -> dict:
        """
        Ana görev: Enerji piyasasını analiz et ve optimal şarj planı sun.

        Args:
            task_input: {
                "required_energy_kwh": float,
                "charge_power_kw": float,
                "max_charge_soc": float,
                "battery_capacity_kwh": float,
                "current_soc": float,
            }

        Returns:
            Enerji analizi ve optimal şarj planı
        """
        logger.header("💰 GRID TARIFF AGENT")
        logger.log("grid_tariff",
                    "Enerji piyasası analizi başlatılıyor...")

        required_kwh = task_input.get("required_energy_kwh", 50.0)
        charge_power = task_input.get("charge_power_kw", 50.0)
        max_soc = task_input.get("max_charge_soc", 80.0)
        capacity = task_input.get("battery_capacity_kwh", 75.0)
        current_soc = task_input.get("current_soc", 25.0)

        # Gerçek enerji ihtiyacını hesapla
        actual_energy_needed = capacity * ((max_soc - current_soc) / 100.0)
        required_kwh = max(required_kwh, actual_energy_needed)

        # ══════════════════════════════════════════════
        # ADIM 1: 24 Saatlik Tarife Çizelgesini Sorgula
        # ══════════════════════════════════════════════
        self.reason({"step": "Tarife sorgulanıyor"}, step=1)
        logger.agent_thinking(
            self.role,
            "Simüle edilmiş şebeke API'sinden 24 saatlik tarife "
            "çizelgesini sorguluyorum.",
            step=1,
        )

        schedule = self.use_tool(
            tool_name="GridTariffAPITool",
            tool_instance=self.grid_api_tool,
            method="get_24h_tariff_schedule",
        )

        # ══════════════════════════════════════════════
        # ADIM 2: Fiyat Analizi
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            f"Tarife alındı. Günlük ortalama: "
            f"{schedule['daily_avg_price']:.2f} TL/kWh. "
            f"En ucuz pencere: {schedule['cheapest_window']['start_hour']}:00-"
            f"{schedule['cheapest_window']['end_hour']}:00 "
            f"({schedule['cheapest_window']['avg_price_kwh']} TL/kWh). "
            f"En pahalı pencere: "
            f"{schedule['most_expensive_window']['start_hour']}:00-"
            f"{schedule['most_expensive_window']['end_hour']}:00 "
            f"({schedule['most_expensive_window']['avg_price_kwh']} TL/kWh).",
            step=2,
        )

        # ══════════════════════════════════════════════
        # ADIM 3: Optimal Şarj Pencerelerini Belirle
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            f"Gereken enerji: {required_kwh:.1f} kWh, "
            f"şarj gücü: {charge_power:.0f} kW. "
            f"En uygun şarj pencerelerini hesaplıyorum.",
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
        # ADIM 4: Tasarruf Analizi
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            "Peak saatte şarj maliyeti ile optimal saatte şarj "
            "maliyetini karşılaştırıyorum.",
            step=4,
        )

        # Peak saatte şarj maliyeti
        peak_cost = self.use_tool(
            tool_name="GridTariffAPITool",
            tool_instance=self.grid_api_tool,
            method="estimate_charge_cost",
            energy_kwh=required_kwh,
            start_hour=schedule["most_expensive_window"]["start_hour"],
        )

        # Optimal maliyet
        optimal_total_cost = sum(w["cost_tl"] for w in optimal_windows)

        savings = peak_cost["cost_tl"] - optimal_total_cost
        savings_pct = ((savings / peak_cost["cost_tl"]) * 100
                       if peak_cost["cost_tl"] > 0 else 0)

        # ══════════════════════════════════════════════
        # ADIM 5: Sonuç Raporu
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            f"Tasarruf analizi tamamlandı. Optimal şarj ile "
            f"{savings:.2f} TL (%{savings_pct:.1f}) tasarruf sağlanabilir.",
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
            f"Optimal Maliyet: {optimal_total_cost:.2f} TL | "
            f"Tasarruf: {savings:.2f} TL (%{savings_pct:.1f})"
        )

        return result

    def _generate_recommendation(
        self,
        schedule: dict,
        windows: list,
        savings: float,
        savings_pct: float,
    ) -> str:
        """Kullanıcı dostu öneri metni üretir."""
        if not windows:
            return "Şarj penceresi belirlenemedi."

        best = windows[0]
        rec = (
            f"💡 Önerilen Şarj Planı: "
            f"Saat {best['start_hour']}:00-{best['end_hour']}:00 "
            f"arasında şarj edin ({best['tariff_type']} tarife). "
            f"Bu sayede yaklaşık {savings:.2f} TL (%{savings_pct:.1f}) "
            f"tasarruf sağlarsınız. "
            f"Yoğun saatlerde (08-12, 18-22) şarjdan kaçının!"
        )
        return rec
