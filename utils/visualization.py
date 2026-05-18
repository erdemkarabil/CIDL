"""
VoltOptimizer - Görselleştirme Modülü
======================================
Eğitim metrikleri, hiper-parametre karşılaştırmaları ve model performans
grafikleri için görselleştirme fonksiyonları.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # GUI olmayan ortamlar için

# Türkçe karakter desteği
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def plot_training_curves(train_losses: list, val_losses: list,
                         train_maes: list, val_maes: list,
                         save_path: str):
    """
    Eğitim ve doğrulama kayıp (loss) ve MAE eğrilerini çizer.

    Args:
        train_losses: Eğitim MSE kayıp değerleri
        val_losses: Doğrulama MSE kayıp değerleri
        train_maes: Eğitim MAE değerleri
        val_maes: Doğrulama MAE değerleri
        save_path: Grafik kayıt yolu
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss (MSE) Grafiği ---
    axes[0].plot(train_losses, label="Eğitim MSE", color="#FF6B6B",
                 linewidth=2)
    axes[0].plot(val_losses, label="Doğrulama MSE", color="#4ECDC4",
                 linewidth=2, linestyle="--")
    axes[0].set_title("Eğitim ve Doğrulama Kayıp Eğrisi (MSE)", fontsize=12,
                      fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- MAE Grafiği ---
    axes[1].plot(train_maes, label="Eğitim MAE", color="#FF6B6B",
                 linewidth=2)
    axes[1].plot(val_maes, label="Doğrulama MAE", color="#4ECDC4",
                 linewidth=2, linestyle="--")
    axes[1].set_title("Eğitim ve Doğrulama MAE Eğrisi", fontsize=12,
                      fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Eğitim eğrileri kaydedildi: {save_path}")


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: str):
    """
    Gerçek RUL değerleri ile tahmin edilen RUL değerlerini scatter plot
    olarak çizer.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, color="#6C5CE7", s=20,
               edgecolors="white", linewidth=0.3)
    ax.plot([0, 100], [0, 100], "r--", linewidth=2, label="İdeal Tahmin")

    ax.set_xlabel("Gerçek RUL (%)", fontsize=12)
    ax.set_ylabel("Tahmin Edilen RUL (%)", fontsize=12)
    ax.set_title("Gerçek vs Tahmin - RUL Değerleri", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Tahmin grafiği kaydedildi: {save_path}")


def plot_hyperparameter_comparison(results: list, save_path: str):
    """
    Hiper-parametre deney sonuçlarını bar chart olarak çizer.

    Args:
        results: [{"name": str, "mse": float, "mae": float, "r2": float}]
        save_path: Grafik kayıt yolu
    """
    names = [r["name"] for r in results]
    mses = [r["mse"] for r in results]
    maes = [r["mae"] for r in results]
    r2s = [r["r2"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    # MSE
    axes[0].bar(names, mses, color=colors, edgecolor="white")
    axes[0].set_title("MSE Karşılaştırması", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("MSE")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.3)

    # MAE
    axes[1].bar(names, maes, color=colors, edgecolor="white")
    axes[1].set_title("MAE Karşılaştırması", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.3)

    # R²
    axes[2].bar(names, r2s, color=colors, edgecolor="white")
    axes[2].set_title("R² Skoru Karşılaştırması", fontsize=12,
                      fontweight="bold")
    axes[2].set_ylabel("R² Score")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Hiper-parametre karşılaştırma grafiği kaydedildi: "
          f"{save_path}")
