"""
VoltOptimizer - Visualisation Module
=====================================
Visualisation functions for training metrics, hyperparameter comparisons,
and model performance plots.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-GUI backend

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def plot_training_curves(train_losses: list, val_losses: list,
                         train_maes: list, val_maes: list,
                         save_path: str):
    """
    Plots training and validation loss (MSE) and MAE curves.

    Args:
        train_losses: Training MSE loss values
        val_losses: Validation MSE loss values
        train_maes: Training MAE values
        val_maes: Validation MAE values
        save_path: File path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss (MSE) Plot ---
    axes[0].plot(train_losses, label="Train MSE", color="#FF6B6B", linewidth=2)
    axes[0].plot(val_losses, label="Val MSE", color="#4ECDC4",
                 linewidth=2, linestyle="--")
    axes[0].set_title("Training and Validation Loss (MSE)", fontsize=12,
                      fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- MAE Plot ---
    axes[1].plot(train_maes, label="Train MAE", color="#FF6B6B", linewidth=2)
    axes[1].plot(val_maes, label="Val MAE", color="#4ECDC4",
                 linewidth=2, linestyle="--")
    axes[1].set_title("Training and Validation MAE", fontsize=12,
                      fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Training curves saved: {save_path}")


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: str):
    """
    Scatter plot of actual vs predicted RUL values.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, color="#6C5CE7", s=20,
               edgecolors="white", linewidth=0.3)
    ax.plot([0, 100], [0, 100], "r--", linewidth=2, label="Ideal Prediction")

    ax.set_xlabel("Actual RUL (%)", fontsize=12)
    ax.set_ylabel("Predicted RUL (%)", fontsize=12)
    ax.set_title("Actual vs Predicted RUL", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Prediction plot saved: {save_path}")


def plot_hyperparameter_comparison(results: list, save_path: str):
    """
    Bar chart comparing hyperparameter experiment results.

    Args:
        results: [{"name": str, "mse": float, "mae": float, "r2": float}]
        save_path: File path to save the plot
    """
    names = [r["name"] for r in results]
    mses = [r["mse"] for r in results]
    maes = [r["mae"] for r in results]
    r2s = [r["r2"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    # MSE
    axes[0].bar(names, mses, color=colors, edgecolor="white")
    axes[0].set_title("MSE Comparison", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("MSE")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.3)

    # MAE
    axes[1].bar(names, maes, color=colors, edgecolor="white")
    axes[1].set_title("MAE Comparison", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.3)

    # R²
    axes[2].bar(names, r2s, color=colors, edgecolor="white")
    axes[2].set_title("R² Score Comparison", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("R² Score")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Hyperparameter comparison plot saved: {save_path}")
