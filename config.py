"""
VoltOptimizer - Genel Konfigürasyon Dosyası
============================================
Tüm modüllerin paylaştığı sabit değerler ve hiper-parametreler burada tanımlanır.
"""

import os

# ============================================================
# Proje Dizinleri
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Dizinleri oluştur
for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Sentetik Veri Parametreleri
# ============================================================
DATA_CONFIG = {
    "num_batteries": 50,          # Farklı batarya sayısı
    "sequence_length": 100,       # Her bir zaman serisi penceresi uzunluğu
    "num_features": 5,            # Voltaj, Akım, Sıcaklık, Hız, Eğim
    "noise_std": 0.02,            # Gürültü standart sapması
    "train_ratio": 0.8,           # Eğitim / test oranı
    "random_seed": 42,
}

# ============================================================
# Derin Öğrenme Model Hiper-Parametreleri
# ============================================================
MODEL_CONFIG = {
    "cnn_filters": [32, 64],      # 1D-CNN filtre sayıları
    "cnn_kernel_size": 3,         # CNN çekirdek boyutu
    "gru_hidden_size": 128,       # GRU gizli katman boyutu
    "gru_num_layers": 2,          # GRU katman sayısı
    "dropout": 0.3,               # Dropout oranı
    "fc_hidden": 64,              # Tam bağlantılı katman boyutu
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "patience": 10,               # Early stopping sabır değeri
    "lr_scheduler_step": 15,      # Öğrenme oranı azaltma adımı
    "lr_scheduler_gamma": 0.5,    # Öğrenme oranı azaltma oranı
}

# ============================================================
# Batarya Fiziksel Limitleri
# ============================================================
BATTERY_LIMITS = {
    "max_voltage": 4.2,           # Volt
    "min_voltage": 2.5,           # Volt
    "nominal_voltage": 3.7,       # Volt
    "max_current": 150.0,         # Amper
    "max_temperature": 60.0,      # °C (Kritik üst limit)
    "warning_temperature": 45.0,  # °C (Uyarı sıcaklığı)
    "optimal_temperature": 25.0,  # °C
    "max_charge_soc": 100.0,      # Şarj üst limiti (%)
    "safe_charge_soc": 80.0,      # Güvenli şarj limiti (%)
    "min_soc": 10.0,              # Minimum SoC (%)
    "critical_rul": 20.0,         # Kritik RUL (%)
    "warning_rul": 40.0,          # Uyarı RUL (%)
}

# ============================================================
# Şebeke Tarife Parametreleri (Simülasyon)
# ============================================================
GRID_CONFIG = {
    "currency": "TL",
    "peak_hours": [(8, 12), (18, 22)],      # Yoğun saatler
    "off_peak_hours": [(0, 8), (12, 18), (22, 24)],  # Sakin saatler
    "peak_price_kwh": 4.50,                 # TL/kWh (Yoğun)
    "off_peak_price_kwh": 1.80,             # TL/kWh (Sakin)
    "super_off_peak_price_kwh": 0.90,       # TL/kWh (Gece süper sakin)
    "super_off_peak_hours": [(1, 5)],       # Gece süper sakin saatler
}

# ============================================================
# Rota Planlama Parametreleri (Simülasyon)
# ============================================================
ROUTE_CONFIG = {
    "ev_range_km": 400,                     # EV menzili (km)
    "ev_consumption_kwh_per_km": 0.18,      # Enerji tüketimi (kWh/km)
    "ev_battery_capacity_kwh": 75.0,        # Batarya kapasitesi (kWh)
    "charge_speed_kw": {                    # Şarj hızları
        "fast_dc": 150.0,
        "normal_dc": 50.0,
        "ac": 22.0,
    },
    "temperature_efficiency_factor": {
        "cold": 0.75,    # <5°C → %25 verimlilik kaybı
        "cool": 0.90,    # 5-15°C
        "optimal": 1.00, # 15-30°C
        "warm": 0.95,    # 30-40°C
        "hot": 0.80,     # >40°C → %20 verimlilik kaybı
    },
    "elevation_factor_per_100m": 0.03,      # 100m yükselme → %3 ek tüketim
}

# ============================================================
# Ajan Ayarları
# ============================================================
AGENT_CONFIG = {
    "verbose": True,
    "max_reasoning_steps": 5,   # Maks. akıl yürütme adımı
    "self_correction_enabled": True,
}
