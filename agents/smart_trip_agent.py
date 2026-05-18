"""
VoltOptimizer - Smart Trip Agent
=================================
Rota ve Seyahat Planlama Asistanı Ajanı.

Görevler:
    1. Kullanıcının rotasını, hava durumunu ve yol eğimini analiz etmek
    2. Batarya durumu ve şebeke fiyatlarına göre şarj duraklarını optimize
    3. Hangi istasyonda, ne kadar süre, ne maliyetle şarj edileceğini planlamak
    4. Diğer ajanlardan gelen bilgilerle bütünsel seyahat planı oluşturmak

Akıl Yürütme Süreci:
    Rota analizi → Batarya Guardian bilgisi → Grid bilgisi →
    Şarj durağı optimizasyonu → Bütünsel plan çıktısı
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.logger import logger


class SmartTripAgent(BaseAgent):
    """
    Rota ve Seyahat Planlama Asistanı.

    Route Planner Tool'u (mock Google Maps) kullanarak optimal
    seyahat planı oluşturur. Battery Guardian ve Grid Tariff
    ajanlarından gelen bilgileri entegre eder.
    """

    def __init__(self, route_planner_tool):
        super().__init__(
            role="Smart Trip Agent",
            goal=("Kullanıcının rotasını, batarya durumunu ve enerji "
                  "fiyatlarını birleştirerek en güvenli, en ucuz ve en "
                  "hızlı seyahat planını oluşturmak"),
            backstory=(
                "Ben VoltOptimizer'ın Seyahat Planlama Asistanıyım. "
                "Google Maps benzeri bir araç kullanarak rota analizi "
                "yapıyor, hava koşullarını ve yol eğimini değerlendiriyorum. "
                "Battery Guardian Agent'ın güvenlik verilerini ve "
                "Grid Tariff Agent'ın fiyat analizlerini entegre ederek "
                "kullanıcıya bütünsel bir seyahat planı sunuyorum."
            ),
            tools=[route_planner_tool],
        )

        self.route_planner_tool = route_planner_tool

    def execute(self, task_input: dict) -> dict:
        """
        Ana görev: Bütünsel seyahat planı oluştur.

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
            Bütünsel seyahat planı
        """
        logger.header("🗺️  SMART TRIP AGENT")
        logger.log("smart_trip",
                    "Seyahat planlaması başlatılıyor...")

        origin = task_input.get("origin", "İzmir")
        destination = task_input.get("destination", "İstanbul")
        distance = task_input.get("total_distance_km", 600.0)
        current_soc = task_input.get("current_soc", 25.0)
        temp = task_input.get("ambient_temperature", 35.0)
        guardian_report = task_input.get("battery_guardian_report", {})
        grid_report = task_input.get("grid_tariff_report", {})

        # ══════════════════════════════════════════════
        # ADIM 1: Diğer ajanlardan gelen bilgileri değerlendir
        # ══════════════════════════════════════════════
        self.reason({"step": "Ajan bilgileri değerlendiriliyor"}, step=1)

        # Battery Guardian'dan gelen kısıtlamalar
        charge_params = guardian_report.get("charge_parameters", {})
        max_charge_soc = charge_params.get("max_charge_soc", 80.0)
        max_current = charge_params.get("max_charge_current_a", 150.0)
        safety_level = charge_params.get("safety_level", "NORMAL")

        logger.agent_thinking(
            self.role,
            f"Battery Guardian'dan gelen kısıtlamalar: "
            f"Maks şarj: %{max_charge_soc:.0f}, "
            f"Maks akım: {max_current:.0f}A, "
            f"Güvenlik: {safety_level}",
            step=1,
        )

        # Grid Tariff'den gelen fiyat bilgileri
        cost_comp = grid_report.get("cost_comparison", {})
        optimal_cost = cost_comp.get("optimal_cost_tl", 0)

        logger.agent_thinking(
            self.role,
            f"Grid Tariff'den gelen optimal maliyet: "
            f"{optimal_cost:.2f} TL. "
            f"Bu bilgiyi şarj durak planlamasında kullanacağım.",
            step=1,
        )

        # ══════════════════════════════════════════════
        # ADIM 2: Rota planlaması (Mock Google Maps)
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            f"Rota analizi başlatılıyor: {origin} → {destination} "
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
        # ADIM 3: Güvenlik kısıtlamalarına göre plan revizyonu
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            "Güvenlik kısıtlamalarına göre planı revize ediyorum.",
            step=3,
        )

        revised_plan = self._revise_plan_for_safety(
            trip_plan, safety_level, max_current, max_charge_soc
        )

        # ══════════════════════════════════════════════
        # ADIM 4: Bütünsel plan oluşturma
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            "Tüm bilgileri birleştirerek bütünsel seyahat planını "
            "oluşturuyorum.",
            step=4,
        )

        final_plan = self._create_final_plan(
            revised_plan, guardian_report, grid_report,
            origin, destination, distance, temp
        )

        self.outputs["trip_plan"] = final_plan

        # ══════════════════════════════════════════════
        # ADIM 5: Sonuç Raporu
        # ══════════════════════════════════════════════
        summary = final_plan["trip_summary"]
        logger.agent_result(
            self.role,
            f"Seyahat Planı Hazır → "
            f"{summary['total_charge_stops']} durak, "
            f"{summary['total_trip_time']} saat, "
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
        Güvenlik seviyesine göre seyahat planını revize eder.
        """
        revised = plan.copy()
        revisions = []

        if safety_level in ("KRİTİK", "ACİL"):
            # Kritik durumda ek şarj durakları ekle
            revisions.append(
                "⚠️ KRİTİK güvenlik seviyesi: Daha sık şarj molaları "
                "planlandı ve şarj akımı düşürüldü."
            )
            # Şarj sürelerini güncelle (düşük akım = uzun şarj)
            if "charge_stops" in revised:
                for stop in revised["charge_stops"]:
                    original_power = stop.get("charge_power_kw", 150)
                    # Güç kısıtlamasını uygula (akım * yaklaşık voltaj)
                    limited_power = min(original_power,
                                        max_current * 0.4)  # ~400V pack
                    if limited_power < original_power:
                        ratio = original_power / limited_power
                        stop["charge_time_min"] = round(
                            stop["charge_time_min"] * ratio, 1
                        )
                        stop["charge_power_kw"] = limited_power
                        revisions.append(
                            f"  → {stop['station_name']}: Güç "
                            f"{original_power:.0f}kW → "
                            f"{limited_power:.0f}kW, "
                            f"süre: {stop['charge_time_min']:.0f} dk"
                        )

        elif safety_level == "UYARI":
            revisions.append(
                "🟡 UYARI güvenlik seviyesi: Şarj üst limiti "
                f"%{max_charge_soc:.0f} olarak kısıtlandı."
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
        Tüm ajan bilgilerini birleştirerek nihai seyahat planı oluşturur.
        """
        charge_stops = trip_plan.get("charge_stops", [])
        summary = trip_plan.get("summary", {})

        # Şarj durakları detayları
        stop_details = []
        for i, stop in enumerate(charge_stops, 1):
            stop_details.append({
                "durak_no": i,
                "istasyon": stop.get("station_name", f"İstasyon-{i}"),
                "konum_km": stop.get("station_location_km", 0),
                "giriş_soc": f"%{stop.get('current_soc', 0):.0f}",
                "çıkış_soc": f"%{stop.get('target_soc', 80):.0f}",
                "enerji_kwh": f"{stop.get('energy_kwh', 0):.1f} kWh",
                "süre": f"{stop.get('charge_time_min', 0):.0f} dakika",
                "güç_kw": f"{stop.get('charge_power_kw', 0):.0f} kW",
                "maliyet": f"{stop.get('cost_tl', 0):.2f} TL",
            })

        # Güvenlik durumu
        guardian_actions = guardian_report.get("actions_taken", [])
        safety_info = guardian_report.get("charge_parameters", {})

        # Enerji fiyat bilgisi
        tariff_info = grid_report.get("tariff_analysis", {})
        tariff_rec = grid_report.get("recommendation", "")

        final_plan = {
            "rota": {
                "başlangıç": origin,
                "varış": destination,
                "mesafe_km": distance,
                "sıcaklık": f"{temperature}°C",
            },
            "güvenlik_durumu": {
                "seviye": safety_info.get("safety_level", "NORMAL"),
                "şarj_limiti": f"%{safety_info.get('max_charge_soc', 80):.0f}",
                "maks_akım": f"{safety_info.get('max_charge_current_a', 150):.0f}A",
                "aksiyonlar": guardian_actions,
            },
            "enerji_analizi": {
                "günlük_ort_fiyat": tariff_info.get(
                    "daily_avg_price_kwh", 0),
                "en_ucuz_pencere": tariff_info.get("cheapest_window", {}),
                "öneri": tariff_rec,
            },
            "şarj_durakları": stop_details,
            "trip_summary": {
                "total_distance_km": distance,
                "total_charge_stops": len(charge_stops),
                "total_charge_time_min": summary.get(
                    "total_charge_time_min", 0),
                "total_trip_time": summary.get(
                    "total_trip_time_hours", 0),
                "total_cost": summary.get(
                    "total_charge_cost_tl", 0),
                "arrival_soc": trip_plan.get("battery", {}).get(
                    "arrival_soc", 0),
            },
            "revisions": trip_plan.get("safety_revisions", []),
        }

        return final_plan
