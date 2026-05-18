"""
VoltOptimizer - Battery Guardian Agent
=======================================
Batarya Sağlık Koruyucusu ve Güvenlik Sorumlusu Ajanı.

Görevler:
    1. DL modelinin RUL ve anomali çıktılarını düzenli okuma
    2. Kritik sıcaklık/düşük RUL durumunda şarj parametrelerini
       dinamik olarak sınırlama
    3. %80 kuralı uygulama (batarya ömrü koruma)
    4. Kendi kendini düzeltme ile güvenlik kararlarını revize etme

Akıl Yürütme Süreci (ReAct):
    Gözlem → Sıcaklık/RUL analizi → Risk değerlendirmesi →
    Şarj parametresi ayarı → Sonuç bildirimi
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from config import BATTERY_LIMITS
from utils.logger import logger


class BatteryGuardianAgent(BaseAgent):
    """
    Batarya Sağlık Koruyucusu Ajanı.

    DL modelini bir Tool olarak kullanarak batarya durumunu izler
    ve güvenlik parametrelerini dinamik olarak ayarlar.
    """

    def __init__(self, battery_rul_tool):
        super().__init__(
            role="Battery Guardian Agent",
            goal=("Batarya sağlığını korumak, güvenlik limitlerini "
                  "dinamik olarak ayarlamak ve anomalilere müdahale etmek"),
            backstory=(
                "Ben VoltOptimizer'ın Batarya Sağlık Koruyucusuyum. "
                "Elektrikli araç bataryalarının ömrünü maksimize etmek ve "
                "güvenlik risklerini minimize etmek için derin öğrenme "
                "modelinin tahminlerini yorumlayarak anlık kararlar alıyorum. "
                "Batarya sıcaklığı, voltaj ve RUL verilerini analiz ederek "
                "şarj akımını ve şarj üst limitini otomatik ayarlıyorum."
            ),
            tools=[battery_rul_tool],
        )

        self.battery_rul_tool = battery_rul_tool
        self.limits = BATTERY_LIMITS.copy()

        # Dinamik olarak ayarlanabilir parametreler
        self.current_charge_limit_soc = self.limits["max_charge_soc"]
        self.current_max_charge_current = self.limits["max_current"]
        self.safety_level = "NORMAL"

    def execute(self, task_input: dict) -> dict:
        """
        Ana görev: Batarya durumunu değerlendir ve güvenlik parametrelerini
        ayarla.

        Args:
            task_input: {
                "battery_age": float,
                "ambient_temp": float,
                "current_soc": float,
            }

        Returns:
            Güvenlik değerlendirmesi ve ayarlanmış parametreler
        """
        logger.header("🛡️  BATTERY GUARDIAN AGENT")
        logger.log("battery_guardian",
                    "Batarya güvenlik değerlendirmesi başlatılıyor...")

        battery_age = task_input.get("battery_age", 0.7)
        ambient_temp = task_input.get("ambient_temp", 45.0)
        current_soc = task_input.get("current_soc", 25.0)

        # ══════════════════════════════════════════════
        # ADIM 1: DL Modelini Tool olarak çağır
        # ══════════════════════════════════════════════
        self.reason({"step": "DL modeli çağrılıyor"}, step=1)
        logger.agent_thinking(
            self.role,
            "DL modelini çağırarak anlık RUL tahmini alacağım. "
            "Bu tahmin, şarj parametrelerini belirlemek için kritik.",
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
        # ADIM 2: RUL ve Anomali Analizi
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
            f"RUL tahmini: %{rul_pct:.1f} | Sağlık: {health} | "
            f"Sıcaklık: {temp:.1f}°C ({temp_status})",
            step=2,
        )

        if anomalies:
            for anomaly in anomalies:
                logger.agent_thinking(
                    self.role, f"⚠️ ANOMALİ: {anomaly}", step=2
                )

        # ══════════════════════════════════════════════
        # ADIM 3: Risk Değerlendirmesi ve Parametre Ayarı
        # ══════════════════════════════════════════════
        logger.agent_thinking(
            self.role,
            "Risk değerlendirmesi yapıyorum. Sıcaklık ve RUL'a göre "
            "şarj parametrelerini ayarlayacağım.",
            step=3,
        )

        charge_decisions = self._evaluate_and_adjust(
            rul_pct, health, temp, temp_status, anomalies
        )

        # ══════════════════════════════════════════════
        # ADIM 4: Kendi Kendini Düzeltme
        # ══════════════════════════════════════════════
        charge_decisions = self._self_correct_decisions(
            charge_decisions, rul_pct, temp
        )

        # ══════════════════════════════════════════════
        # ADIM 5: Sonuç Raporu
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
            f"Güvenlik Seviyesi: {charge_decisions['safety_level']} | "
            f"Şarj Limiti: %{charge_decisions['max_charge_soc']:.0f} | "
            f"Maks Akım: {charge_decisions['max_current']:.0f}A"
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
        Risk değerlendirmesi yaparak şarj parametrelerini ayarlar.

        Kural tabanlı akıl yürütme:
            - KRİTİK: Şarj limiti %60, akım %20'ye düşür
            - UYARI: Şarj limiti %80, akım %50'ye düşür
            - NORMAL: Tam güç
        """
        actions = []

        # --- Varsayılan değerler ---
        max_soc = self.limits["max_charge_soc"]
        max_current = self.limits["max_current"]
        safety_level = "NORMAL"

        # --- KRİTİK Durum ---
        if health == "KRİTİK":
            safety_level = "KRİTİK"
            max_soc = 60.0
            max_current = 30.0
            actions.append(
                f"🔴 KRİTİK DURUM: RUL=%{rul:.1f}. "
                f"Şarj limiti %60'a ve akım 30A'e düşürüldü."
            )

            if temp > self.limits["max_temperature"]:
                max_current = 15.0
                actions.append(
                    f"🔴 AŞIRI SICAKLIK ({temp:.1f}°C): "
                    f"Akım 15A'e düşürüldü. Soğuma beklenmeli."
                )

        # --- UYARI Durumu ---
        elif health == "UYARI":
            safety_level = "UYARI"
            max_soc = 80.0  # %80 kuralı
            max_current = 80.0
            actions.append(
                f"🟡 UYARI DURUMU: RUL=%{rul:.1f}. "
                f"Batarya ömrünü korumak için %80 kuralı uygulandı."
            )

            if temp > self.limits["warning_temperature"]:
                max_current = 50.0
                actions.append(
                    f"🟡 Yüksek sıcaklık ({temp:.1f}°C): "
                    f"Akım 50A ile sınırlandırıldı."
                )

        # --- NORMAL Durum ---
        else:
            safety_level = "NORMAL"
            max_soc = 90.0  # Konservatif normal
            max_current = 150.0
            actions.append(
                f"🟢 NORMAL: RUL=%{rul:.1f}. Tam güç şarj mümkün."
            )

        # Sıcaklık bazlı ek ayarlama
        if temp_status == "SOĞUK":
            max_current = min(max_current, 50.0)
            actions.append(
                f"❄️ Soğuk hava ({temp:.1f}°C): Akım 50A ile sınırlandı, "
                f"batarya ön ısıtması önerilir."
            )

        # Logla
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
        Kendi kendini düzeltme mekanizması.

        Alınan kararları tekrar gözden geçirir:
          - Aşırı kısıtlayıcı mı?
          - Yeterince güvenli mi?
        """
        logger.agent_thinking(
            self.role,
            "Kararlarımı gözden geçiriyorum (self-correction)...",
            step=4,
        )

        corrected = False

        # Durum 1: RUL iyi ama sıcaklık yüksek → sadece akımı düşür
        if rul > 70 and temp > self.limits["warning_temperature"]:
            if decisions["max_charge_soc"] < 80:
                decisions["max_charge_soc"] = 85.0
                decisions["actions"].append(
                    "↩️ Düzeltme: RUL yüksek (%{:.1f}), şarj limiti "
                    "%85'e yükseltildi (yalnız sıcaklık riski).".format(rul)
                )
                corrected = True

        # Durum 2: Aşırı kısıtlayıcı akım (ama sıcaklık normal)
        if (decisions["max_current"] < 30
                and temp < self.limits["warning_temperature"]):
            decisions["max_current"] = 50.0
            decisions["actions"].append(
                "↩️ Düzeltme: Sıcaklık normal, akım 50A'e yükseltildi."
            )
            corrected = True

        # Durum 3: Her şey çok kötü ama yeterince kısıtlı değil
        if rul < 10 and temp > 55:
            decisions["max_current"] = 10.0
            decisions["max_charge_soc"] = 50.0
            decisions["safety_level"] = "ACİL"
            decisions["actions"].append(
                "🚨 ACİL DÜZELTME: Aşırı düşük RUL ve yüksek sıcaklık! "
                "Akım 10A, şarj limiti %50. SERVİS ÇAĞIRILMALI!"
            )
            corrected = True

        if corrected:
            logger.agent_thinking(
                self.role,
                "Kararlar düzeltildi ve güncellendi.",
                step=4,
            )
        else:
            logger.agent_thinking(
                self.role,
                "Kararlar tutarlı, düzeltme gerekmiyor.",
                step=4,
            )

        return decisions
