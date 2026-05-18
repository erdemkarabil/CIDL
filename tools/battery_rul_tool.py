"""
VoltOptimizer - Batarya RUL Tahmin Aracı (Tool)
=================================================
Derin Öğrenme modelini bir "araç" olarak sarmalayarak ajanların
modeli bir fonksiyon gibi çağırıp sonucunu yorumlayabilmesini sağlar.

Bu modül, eğitilmiş 1D-CNN + GRU modelini yükler ve:
  - Anlık batarya verileri ile RUL tahmini yapar
  - Anomali tespiti gerçekleştirir
  - Batarya sağlık raporu üretir
"""

import os
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATTERY_LIMITS, DATA_CONFIG, MODEL_DIR,
)
from models.cnn_gru_model import HybridCNNGRU
from data.mock_data_generator import generate_single_realtime_sample
from utils.logger import logger


class BatteryRULTool:
    """
    Ajanlar tarafından kullanılabilen Batarya RUL Tahmin Aracı.

    Derin öğrenme modelini bir fonksiyon arayüzü olarak sunar.
    Battery Guardian Agent bu aracı çağırarak:
      - Anlık RUL tahminini alır
      - Anomali/sıcaklık durumunu kontrol eder
      - Güvenlik değerlendirmesi yapar
    """

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Eğitilmiş model dosya yolu (.pth)
                        Eğer None ise, yeni model oluşturulur.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HybridCNNGRU()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.feature_stats = None

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device,
                                    weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.log("tool", f"BatteryRULTool: Model yüklendi → {model_path}")
        else:
            logger.log("tool",
                       "BatteryRULTool: Eğitilmiş model bulunamadı, "
                       "simülasyon modu aktif")

    def set_feature_stats(self, stats: dict):
        """Normalizasyon istatistiklerini ayarlar."""
        self.feature_stats = stats

    def predict_rul(self, battery_state: dict) -> dict:
        """
        Anlık batarya durumundan RUL tahmin eder.

        Bu fonksiyon ajanlar tarafından doğrudan çağrılabilir.

        Args:
            battery_state: {
                "voltage": float,      # Hücre voltajı (V)
                "current": float,      # Anlık akım (A)
                "temperature": float,  # Sıcaklık (°C)
                "soc": float,          # Şarj durumu (%)
                "battery_age": float,  # Batarya yaşı [0-1]
                "speed": float,        # Araç hızı (km/h)
                "elevation": float,    # Yol eğimi (%)
            }

        Returns:
            {
                "rul_percentage": float,     # RUL tahmini (%)
                "health_status": str,        # "İYİ", "UYARI", "KRİTİK"
                "anomaly_detected": bool,    # Anomali var mı?
                "anomaly_details": list,     # Anomali detayları
                "temperature_status": str,   # Sıcaklık durumu
                "recommendations": list,     # Öneriler
                "raw_data": dict,            # Ham veri
            }
        """
        logger.log("tool", "BatteryRULTool çağrıldı → RUL tahmin ediliyor...")

        # ── Girdi verisini modele uygun formata dönüştür ──
        features = np.array([
            battery_state["voltage"],
            battery_state["current"],
            battery_state["temperature"],
            battery_state.get("speed", 60.0),
            battery_state.get("elevation", 0.0),
        ], dtype=np.float32)

        # Sekans oluştur (tek adımı sequence_length'e çoğalt)
        seq_len = DATA_CONFIG["sequence_length"]
        sequence = np.tile(features, (seq_len, 1))

        # Gürültü ekle (gerçekçilik için)
        noise = np.random.normal(0, 0.01, sequence.shape)
        sequence = sequence + noise

        # Normalizasyon
        if self.feature_stats:
            sequence = ((sequence - self.feature_stats["means"])
                        / self.feature_stats["stds"])

        # Model tahmini
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor)
            rul_raw = prediction.item()

        # [0,1] → [0,100] ve fiziksel kısıtlar
        rul_percentage = np.clip(rul_raw * 100, 0, 100)

        # ── Anomali tespiti ──
        anomalies = []
        temp = battery_state["temperature"]
        volt = battery_state["voltage"]
        soc = battery_state["soc"]

        if temp > BATTERY_LIMITS["max_temperature"]:
            anomalies.append(
                f"KRİTİK SICAKLIK: {temp:.1f}°C "
                f"(limit: {BATTERY_LIMITS['max_temperature']}°C)"
            )
        elif temp > BATTERY_LIMITS["warning_temperature"]:
            anomalies.append(
                f"Yüksek sıcaklık uyarısı: {temp:.1f}°C "
                f"(uyarı limiti: {BATTERY_LIMITS['warning_temperature']}°C)"
            )

        if volt < BATTERY_LIMITS["min_voltage"]:
            anomalies.append(
                f"Düşük voltaj: {volt:.3f}V "
                f"(min: {BATTERY_LIMITS['min_voltage']}V)"
            )

        if soc < BATTERY_LIMITS["min_soc"]:
            anomalies.append(
                f"Kritik düşük şarj: %{soc:.1f} "
                f"(min: %{BATTERY_LIMITS['min_soc']})"
            )

        # ── Sağlık durumu belirleme ──
        if rul_percentage < BATTERY_LIMITS["critical_rul"] or len(anomalies) > 1:
            health_status = "KRİTİK"
        elif rul_percentage < BATTERY_LIMITS["warning_rul"] or len(anomalies) > 0:
            health_status = "UYARI"
        else:
            health_status = "İYİ"

        # ── Sıcaklık durumu ──
        if temp > BATTERY_LIMITS["max_temperature"]:
            temp_status = "KRİTİK_SICAK"
        elif temp > BATTERY_LIMITS["warning_temperature"]:
            temp_status = "YÜKSEK"
        elif temp < 5:
            temp_status = "SOĞUK"
        else:
            temp_status = "NORMAL"

        # ── Öneriler ──
        recommendations = []
        if health_status == "KRİTİK":
            recommendations.append(
                "Şarj akımını derhal düşürün (maks. 30A)")
            recommendations.append(
                "Şarj üst limitini %80'e indirin")
            recommendations.append(
                "En yakın servis noktasını kontrol edin")
        elif health_status == "UYARI":
            recommendations.append(
                "Şarj akımını normal seviyeye çekin (maks. 80A)")
            recommendations.append(
                "Batarya sıcaklığını monitör edin")

        if temp_status in ("KRİTİK_SICAK", "YÜKSEK"):
            recommendations.append(
                "Soğuma için şarjı duraklat veya akımı azalt")
        elif temp_status == "SOĞUK":
            recommendations.append(
                "Batarya ön ısıtması gerekli, verimlilik düşük")

        result = {
            "rul_percentage": round(rul_percentage, 2),
            "health_status": health_status,
            "anomaly_detected": len(anomalies) > 0,
            "anomaly_details": anomalies,
            "temperature_status": temp_status,
            "temperature_celsius": round(temp, 1),
            "voltage": round(volt, 3),
            "soc": round(soc, 1),
            "recommendations": recommendations,
            "raw_data": battery_state,
        }

        logger.log("tool",
                    f"RUL Tahmini: %{rul_percentage:.1f} | "
                    f"Durum: {health_status} | "
                    f"Sıcaklık: {temp_status} ({temp:.1f}°C)")

        return result

    def get_realtime_battery_assessment(
        self,
        battery_age: float = 0.7,
        ambient_temp: float = 45.0,
        current_soc: float = 25.0,
    ) -> dict:
        """
        Gerçek zamanlı batarya değerlendirmesi.
        Mock veri üretici ile anlık sensör verisi simüle eder ve
        DL modeli ile RUL tahmin eder.

        Args:
            battery_age: Batarya yaşı [0-1]
            ambient_temp: Ortam sıcaklığı (°C)
            current_soc: Anlık şarj durumu (%)

        Returns:
            RUL tahmin sonucu sözlüğü
        """
        battery_state = generate_single_realtime_sample(
            battery_age=battery_age,
            ambient_temp=ambient_temp,
            current_soc=current_soc,
        )

        return self.predict_rul(battery_state)
