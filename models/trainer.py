"""
VoltOptimizer - Model Training and Reporting Module
====================================================
Training, evaluation, and hyperparameter experiment logging for the
hybrid 1D-CNN + GRU model.

Features:
    - Early Stopping to prevent overfitting
    - Learning Rate Scheduling
    - Per-epoch MSE/MAE logging
    - Hyperparameter experiment table generation
    - Saving best model weights
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG, MODEL_DIR, PLOT_DIR, LOG_DIR
from models.cnn_gru_model import HybridCNNGRU
from utils.visualization import (
    plot_training_curves,
    plot_predictions_vs_actual,
    plot_hyperparameter_comparison,
)


class TrainingLogger:
    """Logs and reports the training process."""

    def __init__(self):
        self.epoch_logs = []
        self.experiment_results = []

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_mae: float, val_mae: float, lr: float):
        """Logs metrics at the end of each epoch."""
        entry = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "lr": lr,
        }
        self.epoch_logs.append(entry)

    def log_experiment(self, name: str, config: dict,
                       mse: float, mae: float, r2: float):
        """Logs hyperparameter experiment results."""
        self.experiment_results.append({
            "name": name,
            "config": config,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        })

    def print_epoch_table(self, last_n: int = 10):
        """Prints a metrics table for the last N epochs."""
        if not self.epoch_logs:
            return

        rows = []
        for log in self.epoch_logs[-last_n:]:
            rows.append([
                log["epoch"],
                f"{log['train_mse']:.6f}",
                f"{log['val_mse']:.6f}",
                f"{log['train_mae']:.6f}",
                f"{log['val_mae']:.6f}",
                f"{log['lr']:.6f}",
            ])

        headers = ["Epoch", "Train MSE", "Val MSE", "Train MAE", "Val MAE", "LR"]
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    def print_experiment_table(self):
        """Prints the hyperparameter experiment results table."""
        if not self.experiment_results:
            return

        rows = []
        for r in self.experiment_results:
            rows.append([
                r["name"],
                f"{r['mse']:.6f}",
                f"{r['mae']:.6f}",
                f"{r['r2']:.4f}",
            ])

        headers = ["Experiment", "Test MSE", "Test MAE", "R² Score"]
        print("\n  📊 Hyperparameter Experiment Results:")
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    def save_experiment_report(self, filepath: str):
        """Writes experiment results to a file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("VoltOptimizer - Hyperparameter Experiment Report\n")
            f.write("=" * 60 + "\n\n")

            for r in self.experiment_results:
                f.write(f"Experiment: {r['name']}\n")
                f.write(f"  MSE : {r['mse']:.6f}\n")
                f.write(f"  MAE : {r['mae']:.6f}\n")
                f.write(f"  R²  : {r['r2']:.4f}\n")
                f.write(f"  Config: {r['config']}\n\n")

            if self.epoch_logs:
                f.write("\nPer-Epoch Training Log:\n")
                f.write("-" * 80 + "\n")
                for log in self.epoch_logs:
                    f.write(
                        f"  Epoch {log['epoch']:3d} | "
                        f"Train MSE: {log['train_mse']:.6f} | "
                        f"Val MSE: {log['val_mse']:.6f} | "
                        f"Train MAE: {log['train_mae']:.6f} | "
                        f"Val MAE: {log['val_mae']:.6f} | "
                        f"LR: {log['lr']:.6f}\n"
                    )

        print(f"  📄 Experiment report saved: {filepath}")


class ModelTrainer:
    """
    Model training, evaluation, and hyperparameter search.

    Args:
        model: HybridCNNGRU model instance
        device: torch device (cpu/cuda)
    """

    def __init__(self, model: HybridCNNGRU = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.model = model or HybridCNNGRU()
        self.model = self.model.to(self.device)
        self.logger = TrainingLogger()
        self.best_model_state = None

        print(f"  🖥️  Device: {self.device}")
        self.model.summary()

    def train(self, train_loader, val_loader,
              epochs: int = None,
              learning_rate: float = None,
              weight_decay: float = None,
              patience: int = None,
              experiment_name: str = "default") -> dict:
        """
        Trains the model.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            experiment_name: Experiment name

        Returns:
            Training results dictionary
        """
        epochs = epochs or TRAIN_CONFIG["epochs"]
        learning_rate = learning_rate or TRAIN_CONFIG["learning_rate"]
        weight_decay = weight_decay or TRAIN_CONFIG["weight_decay"]
        patience = patience or TRAIN_CONFIG["patience"]

        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=TRAIN_CONFIG["lr_scheduler_step"],
            gamma=TRAIN_CONFIG["lr_scheduler_gamma"],
        )

        train_losses, val_losses = [], []
        train_maes, val_maes = [], []
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\n  🚀 Starting training: {experiment_name}")
        print(f"     Epochs={epochs}, LR={learning_rate}, "
              f"WD={weight_decay}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # ── Training ──
            self.model.train()
            epoch_train_loss = 0.0
            epoch_train_mae = 0.0
            num_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                mae = mae_criterion(predictions, batch_y)

                loss.backward()
                # Clip gradient norm to 1.0 to prevent exploding gradients,
                # which are a known issue with deep RNNs on long sequences
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_train_loss += loss.item()
                epoch_train_mae += mae.item()
                num_batches += 1

            avg_train_loss = epoch_train_loss / num_batches
            avg_train_mae = epoch_train_mae / num_batches

            # ── Validation ──
            self.model.eval()
            epoch_val_loss = 0.0
            epoch_val_mae = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    predictions = self.model(batch_x)
                    loss = criterion(predictions, batch_y)
                    mae = mae_criterion(predictions, batch_y)

                    epoch_val_loss += loss.item()
                    epoch_val_mae += mae.item()
                    val_batches += 1

            avg_val_loss = epoch_val_loss / val_batches
            avg_val_mae = epoch_val_mae / val_batches

            current_lr = optimizer.param_groups[0]["lr"]
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_maes.append(avg_train_mae)
            val_maes.append(avg_val_mae)
            self.logger.log_epoch(epoch, avg_train_loss, avg_val_loss,
                                  avg_train_mae, avg_val_mae, current_lr)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{epochs} │ "
                    f"Train MSE: {avg_train_loss:.6f} │ "
                    f"Val MSE: {avg_val_loss:.6f} │ "
                    f"Val MAE: {avg_val_mae:.6f} │ "
                    f"LR: {current_lr:.6f}"
                )

            # Early Stopping: snapshot the best weights only when validation
            # loss improves — avoids storing a full copy every epoch while
            # still guaranteeing we recover the best checkpoint at the end
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\n  ⏸️  Early stopping (epoch {epoch}), "
                      f"best val MSE: {best_val_loss:.6f}")
                break

            scheduler.step()

        elapsed = time.time() - start_time
        print(f"\n  ⏱️  Training time: {elapsed:.1f} seconds")

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        plot_training_curves(
            train_losses, val_losses, train_maes, val_maes,
            os.path.join(PLOT_DIR, f"training_curves_{experiment_name}.png")
        )

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_maes": train_maes,
            "val_maes": val_maes,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses),
            "elapsed_seconds": elapsed,
        }

    def evaluate(self, test_loader, experiment_name: str = "default") -> dict:
        """
        Evaluates the model on the test set.

        Returns:
            {mse, mae, r2, y_true, y_pred} dictionary
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                predictions = self.model(batch_x)
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.numpy())

        # Rescale back to percentage points for human-readable metrics;
        # training used normalised [0, 1] targets to match the Sigmoid output
        y_true = np.array(all_targets) * 100
        y_pred = np.array(all_preds) * 100

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"\n  📊 Test Results ({experiment_name}):")
        print(f"     MSE  : {mse:.4f}")
        print(f"     MAE  : {mae:.4f}")
        print(f"     R²   : {r2:.4f}")
        print(f"     RMSE : {np.sqrt(mse):.4f}")

        plot_predictions_vs_actual(
            y_true, y_pred,
            os.path.join(PLOT_DIR,
                         f"predictions_{experiment_name}.png")
        )

        self.logger.log_experiment(
            name=experiment_name,
            config=self.model.get_config(),
            mse=mse, mae=mae, r2=r2,
        )

        return {
            "mse": mse, "mae": mae, "r2": r2,
            "y_true": y_true, "y_pred": y_pred,
        }

    def save_model(self, name: str = "best_model"):
        """Saves model weights to disk."""
        path = os.path.join(MODEL_DIR, f"{name}.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model.get_config(),
        }, path)
        print(f"  💾 Model saved: {path}")
        return path

    def load_model(self, path: str):
        """Loads saved model weights."""
        checkpoint = torch.load(path, map_location=self.device,
                                weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  📂 Model loaded: {path}")

    def run_hyperparameter_experiments(self, train_loader, val_loader,
                                       test_loader):
        """
        Runs experiments with different hyperparameter configurations.
        Produces table/plot data for the paper.
        """
        experiments = [
            {
                "name": "Base_Model",
                "config": {},  # Default config
                "train_args": {"epochs": 30},
            },
            {
                "name": "High_LR",
                "config": {},
                "train_args": {"epochs": 30, "learning_rate": 0.005},
            },
            {
                "name": "Low_LR",
                "config": {},
                "train_args": {"epochs": 30, "learning_rate": 0.0001},
            },
            {
                "name": "Large_GRU",
                "config": {"gru_hidden_size": 256, "gru_num_layers": 3},
                "train_args": {"epochs": 30},
            },
            {
                "name": "Deep_CNN",
                "config": {"cnn_filters": [32, 64, 128]},
                "train_args": {"epochs": 30},
            },
        ]

        print("\n" + "=" * 60)
        print("  🔬 STARTING HYPERPARAMETER EXPERIMENTS")
        print("=" * 60)

        all_results = []

        for exp in experiments:
            print(f"\n  🧪 Experiment: {exp['name']}")
            print("-" * 40)

            model = HybridCNNGRU(**exp["config"])
            self.model = model.to(self.device)
            self.best_model_state = None

            self.train(train_loader, val_loader,
                       experiment_name=exp["name"], **exp["train_args"])

            result = self.evaluate(test_loader,
                                   experiment_name=exp["name"])
            all_results.append({
                "name": exp["name"],
                "mse": result["mse"],
                "mae": result["mae"],
                "r2": result["r2"],
            })

        self.logger.print_experiment_table()

        plot_hyperparameter_comparison(
            all_results,
            os.path.join(PLOT_DIR, "hyperparameter_comparison.png")
        )

        self.logger.save_experiment_report(
            os.path.join(LOG_DIR, "experiment_report.txt")
        )

        return all_results
