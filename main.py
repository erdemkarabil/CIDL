"""
VoltOptimizer - Main Scenario Script
=====================================
Ege University - Computer Engineering
Computational Intelligence and Deep Learning Course Project

This file simulates the end-to-end execution of the VoltOptimizer system:

1. MODULE 1: Trains and evaluates the deep learning model
2. MODULE 2: Launches the multi-agent AI system
3. SCENARIO: Runs a scenario of "an EV wanting to take a long trip
             with a low battery and high temperature"

Usage:
    python main.py                    # Full scenario (training + agents)
    python main.py --skip-training    # Agent scenario only
    python main.py --experiments      # Include hyperparameter experiments

Outputs:
    outputs/models/    - Trained model weights
    outputs/plots/     - Training curves and prediction plots
    outputs/logs/      - Experiment reports
"""

import sys
import os
import argparse
import pickle
import time

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_DIR, OUTPUT_DIR, PLOT_DIR, LOG_DIR,
    TRAIN_CONFIG, DATA_CONFIG,
)
from utils.logger import logger

FEATURE_STATS_PATH = os.path.join(OUTPUT_DIR, "feature_stats.pkl")


def save_feature_stats(feature_stats: dict):
    """Saves normalisation statistics to the outputs/ directory."""
    with open(FEATURE_STATS_PATH, "wb") as f:
        pickle.dump(feature_stats, f)
    logger.log("dl_model",
               f"feature_stats saved → {FEATURE_STATS_PATH}")


def load_feature_stats():
    """Loads saved feature_stats file; returns None if not found."""
    if not os.path.exists(FEATURE_STATS_PATH):
        return None
    with open(FEATURE_STATS_PATH, "rb") as f:
        return pickle.load(f)


def run_deep_learning_module(run_experiments: bool = False):
    """
    MODULE 1: Deep Learning Model Training and Evaluation.

    Steps:
        1. Synthetic data generation
        2. Build the 1D-CNN + GRU hybrid model
        3. Train the model (Early Stopping + LR Scheduling)
        4. Evaluate on the test set
        5. (Optional) Hyperparameter experiments
    """
    logger.header("🧠 MODULE 1: DEEP LEARNING MODEL")
    logger.log("dl_model", "Starting deep learning pipeline...")
    logger.separator("─")

    # ── 1. Data Generation ──
    logger.log("dl_model", "STEP 1: Generating synthetic battery data...")
    from data.mock_data_generator import create_dataloaders

    train_loader, val_loader, test_loader, feature_stats = (
        create_dataloaders()
    )

    # ── 2. Model Construction and Training ──
    logger.log("dl_model", "STEP 2: Training hybrid 1D-CNN + GRU model...")
    from models.cnn_gru_model import HybridCNNGRU
    from models.trainer import ModelTrainer

    model = HybridCNNGRU()
    trainer = ModelTrainer(model=model)

    train_results = trainer.train(
        train_loader, val_loader,
        experiment_name="Main_Model"
    )

    # ── 3. Evaluation ──
    logger.log("dl_model", "STEP 3: Evaluating on test set...")
    test_results = trainer.evaluate(test_loader, experiment_name="Main_Model")

    # ── 4. Save Model ──
    logger.log("dl_model", "STEP 4: Saving best model...")
    model_path = trainer.save_model("best_model")

    # ── 5. Epoch Table ──
    logger.log("dl_model", "STEP 5: Training metrics table:")
    trainer.logger.print_epoch_table(last_n=10)

    # ── 6. Hyperparameter Experiments (Optional) ──
    if run_experiments:
        logger.separator("─")
        logger.log("dl_model",
                    "STEP 6: Starting hyperparameter experiments...")

        exp_trainer = ModelTrainer()
        exp_results = exp_trainer.run_hyperparameter_experiments(
            train_loader, val_loader, test_loader
        )

    logger.separator("═")
    logger.log("dl_model", "Deep Learning Module Completed!")
    logger.log("dl_model", f"  Test MSE: {test_results['mse']:.4f}")
    logger.log("dl_model", f"  Test MAE: {test_results['mae']:.4f}")
    logger.log("dl_model", f"  Test R²:  {test_results['r2']:.4f}")

    save_feature_stats(feature_stats)

    return model_path, feature_stats


def run_multi_agent_scenario(model_path: str = None, feature_stats: dict = None):
    """
    MODULE 2: Multi-Agent AI System Scenario.

    Scenario:
        "An EV with low battery (25% SoC) and high temperature (48°C)
         wanting to travel 600 km from Izmir to Istanbul"

    Agent tasks:
        1. Battery Guardian: Assess battery safety
        2. Grid Tariff: Find the cheapest charging hours
        3. Smart Trip: Create a safe and cost-efficient travel plan
    """
    logger.header("🤖 MODULE 2: MULTI-AGENT AI SYSTEM")
    logger.log("orchestrator", "Starting Agentic AI scenario...")
    logger.separator("─")

    from orchestrator.crew_orchestrator import CrewOrchestrator

    crew = CrewOrchestrator(model_path=model_path, feature_stats=feature_stats)

    # ══════════════════════════════════════════════════
    # SCENARIO: Long-distance trip under challenging conditions
    # ══════════════════════════════════════════════════
    scenario = {
        "name": "Long Trip with Critical Battery",
        "description": (
            "An electric vehicle with low battery (25%) and high ambient "
            "temperature (48°C) wanting to travel 600 km from Izmir to "
            "Istanbul. Battery is aged (70%), temperature near critical level."
        ),
        "battery_age": 0.70,        # 70% aged
        "ambient_temp": 48.0,       # 48°C (near critical temperature)
        "current_soc": 25.0,        # 25% state of charge
        "origin": "Izmir",
        "destination": "Istanbul",
        "total_distance_km": 600.0,
    }

    logger.log("orchestrator", f"Scenario: {scenario['name']}")
    logger.log("orchestrator", f"Description: {scenario['description']}")

    result = crew.run_scenario(scenario)

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VoltOptimizer - EV Battery Optimisation System"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip DL model training, run agent scenario only"
    )
    parser.add_argument(
        "--experiments", action="store_true",
        help="Also run hyperparameter experiments"
    )
    parser.add_argument(
        "--training-only", action="store_true",
        help="Run DL model training only"
    )

    args = parser.parse_args()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "⚡ V O L T O P T I M I Z E R ⚡".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "EV Battery Optimisation & Multi-Agent AI System".center(68) + "║")
    print("║" + "Ege University - Computer Engineering".center(68) + "║")
    print("║" + "CI & Deep Learning - 2025/2026 Spring".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    total_start = time.time()
    model_path = None
    feature_stats = None

    # ── MODULE 1: Deep Learning ──
    if not args.skip_training:
        model_path, feature_stats = run_deep_learning_module(
            run_experiments=args.experiments
        )
    elif not args.training_only:
        model_path = os.path.join(MODEL_DIR, "best_model.pth")
        feature_stats = load_feature_stats()
        if feature_stats is None:
            logger.log("system",
                       "Warning: feature_stats not found "
                       f"({FEATURE_STATS_PATH}). "
                       "Run training first or try without --skip-training.")

    # ── MODULE 2: Multi-Agent AI ──
    if not args.training_only:
        result = run_multi_agent_scenario(
            model_path=model_path, feature_stats=feature_stats
        )

    # ── Finish ──
    total_elapsed = time.time() - total_start

    logger.separator("═")
    logger.log("system",
               f"VoltOptimizer completed! Total time: "
               f"{total_elapsed:.1f} seconds")
    logger.log("system",
               f"Outputs: {os.path.abspath('outputs')}")
    logger.separator("═")


if __name__ == "__main__":
    main()
