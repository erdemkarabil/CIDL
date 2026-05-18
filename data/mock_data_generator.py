"""
VoltOptimizer - Sentetik Veri Üretici (Mock Data Generator)
============================================================
Elektrikli araç bataryalarından gelen zaman serisi verilerini simüle eder.

Üretilen Öznitelikler (Features):
    1. Hücre Voltajı (V)    : 2.5 - 4.2V arasında degradasyon eğrisi
    2. Akım (A)             : Şarj/Deşarj döngüleri
    3. Sıcaklık (°C)        : Ortam + iç ısınma modeli
    4. Araç Hızı (km/h)     : Sürüş profili simülasyonu
    5. Yol Eğimi (%)        : Topoğrafik varyasyon

Hedef (Target):
    - RUL (Remaining Useful Life): %0 - %100 arası sürekli değer
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG


class BatteryDataset(Dataset):
    """
    PyTorch Dataset sınıfı.
    Her örnek: (sequence_length x num_features) boyutunda bir zaman serisi
    ve ilgili RUL (%) hedef değeri.
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: (N, seq_len, features) boyutlu numpy dizisi
            targets: (N,) boyutlu RUL hedef değerleri [0-100]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def generate_battery_degradation_data(
    num_batteries: int = None,
    sequence_length: int = None,
    noise_std: float = None,
    random_seed: int = None,
) -> tuple:
    """
    Sentetik batarya degradasyon verisi üretir.

    Her batarya için farklı bir yaşlanma profili oluşturulur.
    Gerçek dünya fiziksel modelleri temel alınarak:
      - Voltaj: Kapasite kaybına bağlı doğrusal olmayan düşüş
      - Akım: Şarj/Deşarj döngüsel varyasyon
      - Sıcaklık: Ortam + akıma bağlı ısınma
      - Hız/Eğim: Rastgele sürüş profili

    Returns:
        (sequences, targets) → numpy dizileri
    """
    num_batteries = num_batteries or DATA_CONFIG["num_batteries"]
    sequence_length = sequence_length or DATA_CONFIG["sequence_length"]
    noise_std = noise_std or DATA_CONFIG["noise_std"]
    random_seed = random_seed or DATA_CONFIG["random_seed"]

    np.random.seed(random_seed)

    all_sequences = []
    all_targets = []

    for i in range(num_batteries):
        # ── Her batarya için rastgele yaş (0.0 = yeni, 1.0 = ömür sonu)
        battery_age = np.random.uniform(0.0, 1.0)

        # ── RUL hedef değeri: yaş arttıkça RUL düşer
        rul = max(0.0, min(100.0, (1.0 - battery_age) * 100.0))

        # ── Birden fazla pencere oluştur (her batarya için)
        num_windows = np.random.randint(5, 15)

        for w in range(num_windows):
            t = np.linspace(0, 1, sequence_length)

            # --- Öznitelik 1: Hücre Voltajı (V) ---
            # Yaşlı batarya → daha düşük voltaj, daha fazla varyans
            base_voltage = 4.2 - battery_age * 1.2  # 4.2V → 3.0V
            voltage_cycle = 0.3 * np.sin(2 * np.pi * t * 3)  # Döngüsel
            degradation = -0.2 * battery_age * t  # Zamanla düşüş
            voltage = (base_voltage + voltage_cycle + degradation
                       + np.random.normal(0, noise_std * 0.5, sequence_length))
            voltage = np.clip(voltage, 2.5, 4.2)

            # --- Öznitelik 2: Akım (A) ---
            # Şarj (+) ve deşarj (-) döngüleri
            current_base = 30 * np.sin(2 * np.pi * t * 2)
            current_noise = np.random.normal(0, 5, sequence_length)
            current = current_base + current_noise
            current = np.clip(current, -150, 150)

            # --- Öznitelik 3: Sıcaklık (°C) ---
            ambient = np.random.uniform(15, 35)
            heat_from_current = 0.15 * np.abs(current)
            aging_heat = 5 * battery_age  # Yaşlı batarya daha çok ısınır
            temperature = (ambient + heat_from_current + aging_heat
                           + np.random.normal(0, noise_std * 10,
                                              sequence_length))
            temperature = np.clip(temperature, -10, 70)

            # --- Öznitelik 4: Araç Hızı (km/h) ---
            speed = (60 * np.abs(np.sin(2 * np.pi * t * 1.5))
                     + np.random.normal(0, 10, sequence_length))
            speed = np.clip(speed, 0, 180)

            # --- Öznitelik 5: Yol Eğimi (%) ---
            elevation = (5 * np.sin(2 * np.pi * t * 0.5)
                         + np.random.normal(0, 2, sequence_length))
            elevation = np.clip(elevation, -15, 15)

            # ── Öznitelikleri birleştir: (seq_len, 5)
            sequence = np.stack(
                [voltage, current, temperature, speed, elevation], axis=-1
            )
            all_sequences.append(sequence)

            # ── Pencere bazlı RUL varyasyonu (küçük dalgalanma)
            window_rul = rul + np.random.normal(0, 2)
            window_rul = np.clip(window_rul, 0, 100)
            all_targets.append(window_rul)

    sequences = np.array(all_sequences, dtype=np.float32)
    targets = np.array(all_targets, dtype=np.float32)

    print(f"  ⚡ Sentetik veri üretildi: {sequences.shape[0]} örnek, "
          f"sekans={sequences.shape[1]}, öznitelik={sequences.shape[2]}")
    print(f"  📊 RUL dağılımı: min={targets.min():.1f}%, "
          f"max={targets.max():.1f}%, ort={targets.mean():.1f}%")

    return sequences, targets


def create_dataloaders(
    batch_size: int = None,
    train_ratio: float = None,
) -> tuple:
    """
    Sentetik veri üretip PyTorch DataLoader'larına dönüştürür.

    Returns:
        (train_loader, val_loader, test_loader, feature_stats)
    """
    from config import TRAIN_CONFIG

    batch_size = batch_size or TRAIN_CONFIG["batch_size"]
    train_ratio = train_ratio or DATA_CONFIG["train_ratio"]

    # Veri üret
    sequences, targets = generate_battery_degradation_data()

    # Öznitelik normalizasyonu (Z-Score)
    feature_means = sequences.mean(axis=(0, 1))
    feature_stds = sequences.std(axis=(0, 1))
    feature_stds[feature_stds == 0] = 1.0  # Sıfıra bölmeyi engelle
    sequences = (sequences - feature_means) / feature_stds

    # Hedef normalizasyonu: [0-100] → [0-1]
    targets = targets / 100.0

    feature_stats = {
        "means": feature_means,
        "stds": feature_stds,
    }

    # Eğitim / Doğrulama / Test bölmesi
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, targets, train_size=train_ratio,
        random_state=DATA_CONFIG["random_seed"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=DATA_CONFIG["random_seed"]
    )

    print(f"  📂 Eğitim: {len(X_train)}, Doğrulama: {len(X_val)}, "
          f"Test: {len(X_test)}")

    train_loader = DataLoader(
        BatteryDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        BatteryDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        BatteryDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, feature_stats


def generate_single_realtime_sample(
    battery_age: float = 0.7,
    ambient_temp: float = 45.0,
    current_soc: float = 25.0,
) -> dict:
    """
    Tek bir anlık batarya durumu örneği üretir.
    Ajanlar tarafından gerçek zamanlı simülasyon için kullanılır.

    Args:
        battery_age: Batarya yaşı [0-1]
        ambient_temp: Ortam sıcaklığı (°C)
        current_soc: Anlık şarj durumu (%)

    Returns:
        Batarya durumu sözlüğü
    """
    np.random.seed(None)  # Gerçek rastgelelik

    voltage = 4.2 - battery_age * 1.0 - (1 - current_soc / 100) * 0.8
    voltage += np.random.normal(0, 0.02)
    voltage = np.clip(voltage, 2.5, 4.2)

    current = np.random.uniform(-50, 80)

    temperature = (ambient_temp + 0.1 * abs(current) + 3 * battery_age
                   + np.random.normal(0, 1))

    return {
        "voltage": round(float(voltage), 3),
        "current": round(float(current), 2),
        "temperature": round(float(temperature), 1),
        "soc": round(float(current_soc), 1),
        "battery_age": round(float(battery_age), 3),
        "speed": round(float(np.random.uniform(0, 120)), 1),
        "elevation": round(float(np.random.normal(0, 3)), 1),
    }
