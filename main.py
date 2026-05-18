"""
VoltOptimizer - Ana Senaryo Dosyası
=====================================
Ege Üniversitesi - Bilgisayar Mühendisliği
Computational Intelligence and Deep Learning Dersi Projesi

Bu dosya, VoltOptimizer sisteminin uçtan uca çalışmasını simüle eder:

1. MODÜL 1: Derin Öğrenme modelini eğitir ve değerlendirir
2. MODÜL 2: Çok Etmenli AI sistemini başlatır
3. SENARYO: "Düşük batarya ve yüksek sıcaklıkta uzun yola çıkmak
             isteyen bir araç" senaryosunu çalıştırır

Kullanım:
    python main.py                    # Tam senaryo (eğitim + ajanlar)
    python main.py --skip-training    # Sadece ajan senaryosu
    python main.py --experiments      # Hiper-parametre deneyleri dahil

Çıktılar:
    outputs/models/    - Eğitilmiş model ağırlıkları
    outputs/plots/     - Eğitim eğrileri ve tahmin grafikleri
    outputs/logs/      - Deney raporları
"""

import sys
import os
import argparse
import pickle
import time

# Proje kök dizinini Python yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_DIR, OUTPUT_DIR, PLOT_DIR, LOG_DIR,
    TRAIN_CONFIG, DATA_CONFIG,
)
from utils.logger import logger

FEATURE_STATS_PATH = os.path.join(OUTPUT_DIR, "feature_stats.pkl")


def save_feature_stats(feature_stats: dict):
    """Normalizasyon istatistiklerini outputs/ altına kaydeder."""
    with open(FEATURE_STATS_PATH, "wb") as f:
        pickle.dump(feature_stats, f)
    logger.log("dl_model",
               f"feature_stats kaydedildi → {FEATURE_STATS_PATH}")


def load_feature_stats():
    """Kayıtlı feature_stats dosyasını yükler; yoksa None döner."""
    if not os.path.exists(FEATURE_STATS_PATH):
        return None
    with open(FEATURE_STATS_PATH, "rb") as f:
        return pickle.load(f)


def run_deep_learning_module(run_experiments: bool = False):
    """
    MODÜL 1: Derin Öğrenme Modelinin Eğitimi ve Değerlendirmesi.

    Adımlar:
        1. Sentetik veri üretimi
        2. 1D-CNN + GRU hibrit modelin oluşturulması
        3. Modelin eğitilmesi (Early Stopping + LR Scheduling)
        4. Test seti üzerinde değerlendirme
        5. (Opsiyonel) Hiper-parametre deneyleri
    """
    logger.header("🧠 MODÜL 1: DERİN ÖĞRENME MODELİ")
    logger.log("dl_model", "Derin öğrenme pipeline'ı başlatılıyor...")
    logger.separator("─")

    # ── 1. Veri Üretimi ──
    logger.log("dl_model", "ADIM 1: Sentetik batarya verisi üretiliyor...")
    from data.mock_data_generator import create_dataloaders

    train_loader, val_loader, test_loader, feature_stats = (
        create_dataloaders()
    )

    # ── 2. Model Oluşturma ve Eğitim ──
    logger.log("dl_model", "ADIM 2: Hibrit 1D-CNN + GRU modeli eğitiliyor...")
    from models.cnn_gru_model import HybridCNNGRU
    from models.trainer import ModelTrainer

    model = HybridCNNGRU()
    trainer = ModelTrainer(model=model)

    # Eğitim
    train_results = trainer.train(
        train_loader, val_loader,
        experiment_name="Ana_Model"
    )

    # ── 3. Değerlendirme ──
    logger.log("dl_model", "ADIM 3: Test seti üzerinde değerlendirme...")
    test_results = trainer.evaluate(test_loader, experiment_name="Ana_Model")

    # ── 4. Model Kaydetme ──
    logger.log("dl_model", "ADIM 4: En iyi model kaydediliyor...")
    model_path = trainer.save_model("best_model")

    # ── 5. Epoch Tablosu ──
    logger.log("dl_model", "ADIM 5: Eğitim metrikleri tablosu:")
    trainer.logger.print_epoch_table(last_n=10)

    # ── 6. Hiper-parametre Deneyleri (Opsiyonel) ──
    if run_experiments:
        logger.separator("─")
        logger.log("dl_model",
                    "ADIM 6: Hiper-parametre deneyleri başlatılıyor...")

        exp_trainer = ModelTrainer()
        exp_results = exp_trainer.run_hyperparameter_experiments(
            train_loader, val_loader, test_loader
        )

    logger.separator("═")
    logger.log("dl_model",
               f"Derin Öğrenme Modülü Tamamlandı!")
    logger.log("dl_model",
               f"  Test MSE: {test_results['mse']:.4f}")
    logger.log("dl_model",
               f"  Test MAE: {test_results['mae']:.4f}")
    logger.log("dl_model",
               f"  Test R²:  {test_results['r2']:.4f}")

    save_feature_stats(feature_stats)

    return model_path, feature_stats


def run_multi_agent_scenario(model_path: str = None, feature_stats: dict = None):
    """
    MODÜL 2: Çok Etmenli Yapay Zeka Sistemi Senaryosu.

    Senaryo:
        "Düşük batarya (%25 SoC) ve yüksek sıcaklıkta (48°C)
         İzmir'den İstanbul'a uzun yola çıkmak isteyen bir EV"

    Ajanların görevi:
        1. Battery Guardian: Batarya güvenliğini değerlendirmek
        2. Grid Tariff: En ucuz şarj saatlerini bulmak
        3. Smart Trip: Güvenli ve ucuz seyahat planı oluşturmak
    """
    logger.header("🤖 MODÜL 2: ÇOK ETMENLİ YAPAY ZEKA SİSTEMİ")
    logger.log("orchestrator",
               "Agentic AI senaryosu başlatılıyor...")
    logger.separator("─")

    from orchestrator.crew_orchestrator import CrewOrchestrator

    # Orkestratörü başlat
    crew = CrewOrchestrator(model_path=model_path, feature_stats=feature_stats)

    # ══════════════════════════════════════════════════
    # SENARYO: Zorlayıcı koşullarda uzun yol seyahati
    # ══════════════════════════════════════════════════
    scenario = {
        "name": "Kritik Batarya ile Uzun Yol",
        "description": (
            "Düşük batarya (%25) ve yüksek sıcaklıkta (48°C) "
            "İzmir'den İstanbul'a 600 km'lik uzun yola çıkmak "
            "isteyen bir elektrikli araç. Batarya yaşlanmış (70%), "
            "sıcaklık kritik seviyeye yakın."
        ),
        "battery_age": 0.70,        # %70 yaşlanmış
        "ambient_temp": 48.0,       # 48°C (kritik sıcaklık)
        "current_soc": 25.0,        # %25 şarj
        "origin": "İzmir",
        "destination": "İstanbul",
        "total_distance_km": 600.0,
    }

    logger.log("orchestrator", f"Senaryo: {scenario['name']}")
    logger.log("orchestrator", f"Açıklama: {scenario['description']}")

    # Senaryoyu çalıştır
    result = crew.run_scenario(scenario)

    return result


def main():
    """Ana giriş noktası."""
    parser = argparse.ArgumentParser(
        description="VoltOptimizer - EV Batarya Optimizasyon Sistemi"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="DL model eğitimini atla, sadece ajan senaryosunu çalıştır"
    )
    parser.add_argument(
        "--experiments", action="store_true",
        help="Hiper-parametre deneylerini de çalıştır"
    )
    parser.add_argument(
        "--training-only", action="store_true",
        help="Sadece DL model eğitimini çalıştır"
    )

    args = parser.parse_args()

    # Başlık
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "⚡ V O L T O P T I M I Z E R ⚡".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "EV Batarya Optimizasyon ve Çok Etmenli AI Sistemi".center(68) + "║")
    print("║" + "Ege Üniversitesi - Bilgisayar Mühendisliği".center(68) + "║")
    print("║" + "CI & Deep Learning - 2025/2026 Bahar".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    total_start = time.time()
    model_path = None
    feature_stats = None

    # ── MODÜL 1: Derin Öğrenme ──
    if not args.skip_training:
        model_path, feature_stats = run_deep_learning_module(
            run_experiments=args.experiments
        )
    elif not args.training_only:
        model_path = os.path.join(MODEL_DIR, "best_model.pth")
        feature_stats = load_feature_stats()
        if feature_stats is None:
            logger.log("system",
                       "Uyarı: feature_stats bulunamadı "
                       f"({FEATURE_STATS_PATH}). "
                       "Önce eğitim çalıştırın veya --skip-training olmadan deneyin.")

    # ── MODÜL 2: Çok Etmenli AI ──
    if not args.training_only:
        result = run_multi_agent_scenario(
            model_path=model_path, feature_stats=feature_stats
        )

    # ── Bitiş ──
    total_elapsed = time.time() - total_start

    logger.separator("═")
    logger.log("system",
               f"VoltOptimizer tamamlandı! Toplam süre: "
               f"{total_elapsed:.1f} saniye")
    logger.log("system",
               f"Çıktılar: {os.path.abspath('outputs')}")
    logger.separator("═")


if __name__ == "__main__":
    main()
