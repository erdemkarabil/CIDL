"""
VoltOptimizer - Ajan Orkestrasyon Motoru
=========================================
Tüm ajanların koordinasyonunu ve iletişimini yöneten ana kontrol birimi.

CrewAI benzeri bir orkestrasyon deseni ile:
  1. Görev tanımı ve ajan ataması
  2. Sıralı görev yürütme (pipeline)
  3. Ajan arası mesajlaşma ve veri paylaşımı
  4. Bütünsel sonuç derleme

Akış:
    Battery Guardian → Grid Tariff → Smart Trip
    (Her ajanın çıktısı bir sonrakine girdi olarak akar)
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
    VoltOptimizer Ajan Orkestrasyon Motoru.

    CrewAI framework'üne benzer şekilde bir "Crew" (ekip) oluşturur
    ve görevleri sıralı olarak yürütür.

    Akış:
        1. Battery Guardian: Batarya güvenlik değerlendirmesi
        2. Grid Tariff: Enerji piyasası analizi
        3. Smart Trip: Bütünsel seyahat planı
    """

    def __init__(self, model_path: str = None, feature_stats: dict = None):
        """
        Orkestratörü başlatır: Tool'ları ve Ajanları oluşturur.

        Args:
            model_path: Eğitilmiş DL model dosya yolu
            feature_stats: Öznitelik normalizasyon istatistikleri (means, stds)
        """
        logger.header("⚡ VOLTOPTIMIZER - AJAN ORKESTRASYONU")
        logger.log("orchestrator", "Sistem başlatılıyor...")

        # ── Tool'ları oluştur ──
        logger.log("orchestrator", "Araçlar (Tools) yükleniyor...")
        self.battery_rul_tool = BatteryRULTool(model_path=model_path)
        if feature_stats is not None:
            self.battery_rul_tool.set_feature_stats(feature_stats)
        self.grid_api_tool = GridTariffAPITool()
        self.route_planner_tool = RoutePlannerTool()

        # ── Ajanları oluştur ──
        logger.log("orchestrator", "Ajanlar oluşturuluyor...")
        self.battery_guardian = BatteryGuardianAgent(self.battery_rul_tool)
        self.grid_tariff = GridTariffAgent(self.grid_api_tool)
        self.smart_trip = SmartTripAgent(self.route_planner_tool)

        logger.log("orchestrator",
                    "Sistem hazır! 3 ajan aktif, 3 araç yüklendi.")

    def run_scenario(self, scenario: dict) -> dict:
        """
        Bir senaryoyu uçtan uca çalıştırır.

        Senaryo tüm ajanları sırayla tetikler ve
        aralarındaki iletişimi koordine eder.

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
            Bütünsel senaryo sonucu
        """
        logger.separator("═")
        logger.header(f"🚗 SENARYO: {scenario.get('name', 'Adsız')}")
        logger.separator("═")

        start_time = time.time()

        scenario_name = scenario.get("name", "Adsız Senaryo")
        battery_age = scenario.get("battery_age", 0.7)
        ambient_temp = scenario.get("ambient_temp", 45.0)
        current_soc = scenario.get("current_soc", 25.0)
        origin = scenario.get("origin", "İzmir")
        destination = scenario.get("destination", "İstanbul")
        distance = scenario.get("total_distance_km", 600.0)

        logger.log("orchestrator",
                    f"Senaryo parametreleri:")
        logger.log("orchestrator",
                    f"  🔋 Batarya Yaşı: {battery_age:.1%} "
                    f"(0=Yeni, 1=Ömür sonu)")
        logger.log("orchestrator",
                    f"  🌡️  Ortam Sıcaklığı: {ambient_temp}°C")
        logger.log("orchestrator",
                    f"  ⚡ Anlık SoC: %{current_soc}")
        logger.log("orchestrator",
                    f"  📍 Rota: {origin} → {destination} ({distance} km)")

        # ══════════════════════════════════════════════════
        # AŞAMA 1: Battery Guardian Agent
        # ══════════════════════════════════════════════════
        logger.separator("─")
        logger.log("orchestrator",
                    "AŞAMA 1/3: Battery Guardian Agent tetikleniyor...")

        guardian_result = self.battery_guardian.execute({
            "battery_age": battery_age,
            "ambient_temp": ambient_temp,
            "current_soc": current_soc,
        })

        # ── Guardian → Grid: Mesaj gönder ──
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
        # AŞAMA 2: Grid Tariff Agent
        # ══════════════════════════════════════════════════
        logger.separator("─")
        logger.log("orchestrator",
                    "AŞAMA 2/3: Grid Tariff Agent tetikleniyor...")

        max_soc = guardian_result["charge_parameters"]["max_charge_soc"]

        grid_result = self.grid_tariff.execute({
            "required_energy_kwh": 50.0,
            "charge_power_kw": guardian_result["charge_parameters"][
                "max_charge_current_a"] * 0.4,
            "max_charge_soc": max_soc,
            "battery_capacity_kwh": 75.0,
            "current_soc": current_soc,
        })

        # ── Grid → Trip: Mesaj gönder ──
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
        # AŞAMA 3: Smart Trip Agent
        # ══════════════════════════════════════════════════
        logger.separator("─")
        logger.log("orchestrator",
                    "AŞAMA 3/3: Smart Trip Agent tetikleniyor...")

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
        # BÜTÜNLEŞİK SONUÇ
        # ══════════════════════════════════════════════════
        elapsed = time.time() - start_time

        logger.separator("═")
        logger.header("📋 VOLTOPTIMIZER - BÜTÜNLEŞİK SONUÇ RAPORU")

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
        """Nihai raporu terminale yazdırır."""
        print()
        print("╔" + "═" * 68 + "╗")
        print(f"║{'VoltOptimizer - Bütünleşik Sonuç Raporu':^68}║")
        print(f"║{'Senaryo: ' + scenario_name:^68}║")
        print("╠" + "═" * 68 + "╣")

        # Batarya Güvenlik
        cp = guardian["charge_parameters"]
        print(f"║ 🛡️  BATARYA GÜVENLİK DEĞERLENDİRMESİ"
              f"{' ' * 29}║")
        print(f"║   RUL Tahmini    : %{guardian['rul_assessment']['rul_percentage']:<44.1f}║")
        print(f"║   Sağlık Durumu  : {guardian['rul_assessment']['health_status']:<46}║")
        print(f"║   Sıcaklık       : {guardian['temperature']['current']:<40.1f}°C   ║")
        print(f"║   Güvenlik Sev.  : {cp['safety_level']:<46}║")
        print(f"║   Şarj Limiti    : %{cp['max_charge_soc']:<44.0f}║")
        print(f"║   Maks Akım      : {cp['max_charge_current_a']:<43.0f}A   ║")

        print("╠" + "═" * 68 + "╣")

        # Enerji Fiyatları
        cc = grid.get("cost_comparison", {})
        print(f"║ 💰 ENERJİ PİYASASI ANALİZİ"
              f"{' ' * 40}║")
        print(f"║   Optimal Maliyet : {cc.get('optimal_cost_tl', 0):<42.2f} TL  ║")
        print(f"║   Peak Maliyeti   : {cc.get('peak_cost_tl', 0):<42.2f} TL  ║")
        print(f"║   Tasarruf        : {cc.get('savings_tl', 0):<35.2f} TL (%{cc.get('savings_percentage', 0):.0f}) ║")

        print("╠" + "═" * 68 + "╣")

        # Seyahat Planı
        ts = trip.get("trip_summary", {})
        print(f"║ 🗺️  SEYAHAT PLANI"
              f"{' ' * 49}║")
        print(f"║   Toplam Mesafe   : {ts.get('total_distance_km', 0):<42.0f} km  ║")
        print(f"║   Şarj Durakları  : {ts.get('total_charge_stops', 0):<46}║")
        print(f"║   Şarj Süresi     : {ts.get('total_charge_time_min', 0):<42.0f} dk  ║")
        print(f"║   Toplam Süre     : {ts.get('total_trip_time', 0):<40.1f} saat  ║")
        print(f"║   Toplam Maliyet  : {ts.get('total_cost', 0):<42.2f} TL  ║")
        print(f"║   Varış SoC       : %{ts.get('arrival_soc', 0):<44.1f}║")

        print("╠" + "═" * 68 + "╣")

        # Şarj Durakları
        stops = trip.get("şarj_durakları", [])
        if stops:
            print(f"║ ⚡ ŞARJ DURAKLARI"
                  f"{' ' * 49}║")
            for stop in stops:
                name = stop.get('istasyon', 'Bilinmiyor')
                # İstasyon adını 40 karakterle sınırla
                if len(name) > 40:
                    name = name[:37] + "..."
                print(f"║   {stop['durak_no']}. {name:<63}║"[:71] + "║")
                print(f"║      SoC: {stop['giriş_soc']} → {stop['çıkış_soc']} | "
                      f"{stop['süre']} | {stop['güç_kw']} | "
                      f"{stop['maliyet']:<10}      ║")

            print("╠" + "═" * 68 + "╣")

        # Güvenlik Aksiyonları
        actions = guardian.get("actions_taken", [])
        if actions:
            print(f"║ 📋 ALINAN GÜVENLİK AKSİYONLARI"
                  f"{' ' * 35}║")
            for action in actions[:5]:
                # Her satırı 66 karakterle sınırla
                text = action[:64]
                print(f"║   {text:<65}║")

        print("╠" + "═" * 68 + "╣")
        print(f"║ ⏱️  İşlem Süresi: {elapsed:.2f} saniye"
              f"{' ' * (46 - len(f'{elapsed:.2f}'))}║")
        print("╚" + "═" * 68 + "╝")
        print()
