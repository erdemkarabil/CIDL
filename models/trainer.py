"""
VoltOptimizer - Model Eğitim ve Raporlama Modülü
==================================================
Hibrit 1D-CNN + GRU modelinin eğitimi, değerlendirmesi ve
hiper-parametre deneylerinin loglanması.

Özellikler:
    - Early Stopping ile aşırı öğrenmeyi engelleme
    - Learning Rate Scheduling
    - Epoch bazlı MSE/MAE loglama
    - Hiper-parametre deney tablosu oluşturma
    - En iyi model ağırlıklarını kaydetme
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
    """Eğitim sürecini loglar ve raporlar."""

    def __init__(self):
        self.epoch_logs = []
        self.experiment_results = []

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_mae: float, val_mae: float, lr: float):
        """Her epoch sonunda metrikleri loglar."""
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
        """Hiper-parametre deney sonuçlarını loglar."""
        self.experiment_results.append({
            "name": name,
            "config": config,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        })

    def print_epoch_table(self, last_n: int = 10):
        """Son N epoch'un metrik tablosunu yazdırır."""
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

        headers = ["Epoch", "Eğitim MSE", "Doğ. MSE",
                    "Eğitim MAE", "Doğ. MAE", "LR"]
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    def print_experiment_table(self):
        """Hiper-parametre deney sonuçları tablosunu yazdırır."""
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

        headers = ["Deney Adı", "Test MSE", "Test MAE", "R² Skoru"]
        print("\n  📊 Hiper-Parametre Deney Sonuçları:")
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    def save_experiment_report(self, filepath: str):
        """Deney sonuçlarını dosyaya yazar."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("VoltOptimizer - Hiper-Parametre Deney Raporu\n")
            f.write("=" * 60 + "\n\n")

            for r in self.experiment_results:
                f.write(f"Deney: {r['name']}\n")
                f.write(f"  MSE : {r['mse']:.6f}\n")
                f.write(f"  MAE : {r['mae']:.6f}\n")
                f.write(f"  R²  : {r['r2']:.4f}\n")
                f.write(f"  Konfig: {r['config']}\n\n")

            # Epoch log tablosu
            if self.epoch_logs:
                f.write("\nEpoch Bazlı Eğitim Logu:\n")
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

        print(f"  📄 Deney raporu kaydedildi: {filepath}")


class ModelTrainer:
    """
    Model eğitimi, değerlendirmesi ve hiper-parametre araması.

    Args:
        model: HybridCNNGRU model örneği
        device: torch device (cpu/cuda)
    """

    def __init__(self, model: HybridCNNGRU = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.model = model or HybridCNNGRU()
        self.model = self.model.to(self.device)
        self.logger = TrainingLogger()
        self.best_model_state = None

        print(f"  🖥️  Cihaz: {self.device}")
        self.model.summary()

    def train(self, train_loader, val_loader,
              epochs: int = None,
              learning_rate: float = None,
              weight_decay: float = None,
              patience: int = None,
              experiment_name: str = "default") -> dict:
        """
        Modeli eğitir.

        Args:
            train_loader: Eğitim DataLoader
            val_loader: Doğrulama DataLoader
            epochs: Epoch sayısı
            learning_rate: Öğrenme oranı
            weight_decay: Ağırlık çürümesi
            patience: Early stopping sabır değeri
            experiment_name: Deney adı

        Returns:
            Eğitim sonuçları sözlüğü
        """
        epochs = epochs or TRAIN_CONFIG["epochs"]
        learning_rate = learning_rate or TRAIN_CONFIG["learning_rate"]
        weight_decay = weight_decay or TRAIN_CONFIG["weight_decay"]
        patience = patience or TRAIN_CONFIG["patience"]

        # Kayıp fonksiyonu ve optimizer
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

        # Loglar
        train_losses, val_losses = [], []
        train_maes, val_maes = [], []
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\n  🚀 Eğitim başlatılıyor: {experiment_name}")
        print(f"     Epochs={epochs}, LR={learning_rate}, "
              f"WD={weight_decay}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # ── Eğitim ──
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_train_loss += loss.item()
                epoch_train_mae += mae.item()
                num_batches += 1

            avg_train_loss = epoch_train_loss / num_batches
            avg_train_mae = epoch_train_mae / num_batches

            # ── Doğrulama ──
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

            # Metrikleri kaydet
            current_lr = optimizer.param_groups[0]["lr"]
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_maes.append(avg_train_mae)
            val_maes.append(avg_val_mae)
            self.logger.log_epoch(epoch, avg_train_loss, avg_val_loss,
                                  avg_train_mae, avg_val_mae, current_lr)

            # Epoch çıktısı
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{epochs} │ "
                    f"Train MSE: {avg_train_loss:.6f} │ "
                    f"Val MSE: {avg_val_loss:.6f} │ "
                    f"Val MAE: {avg_val_mae:.6f} │ "
                    f"LR: {current_lr:.6f}"
                )

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\n  ⏸️  Early stopping (epoch {epoch}), "
                      f"en iyi val MSE: {best_val_loss:.6f}")
                break

            scheduler.step()

        elapsed = time.time() - start_time
        print(f"\n  ⏱️  Eğitim süresi: {elapsed:.1f} saniye")

        # En iyi modeli yükle
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        # Grafikleri kaydet
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
        Test seti üzerinde modeli değerlendirir.

        Returns:
            {mse, mae, r2, y_true, y_pred} sözlüğü
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

        y_true = np.array(all_targets) * 100  # [0,1] → [0,100]
        y_pred = np.array(all_preds) * 100

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"\n  📊 Test Sonuçları ({experiment_name}):")
        print(f"     MSE  : {mse:.4f}")
        print(f"     MAE  : {mae:.4f}")
        print(f"     R²   : {r2:.4f}")
        print(f"     RMSE : {np.sqrt(mse):.4f}")

        # Tahmin grafiğini kaydet
        plot_predictions_vs_actual(
            y_true, y_pred,
            os.path.join(PLOT_DIR,
                         f"predictions_{experiment_name}.png")
        )

        # Deney loguna ekle
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
        """Model ağırlıklarını kaydeder."""
        path = os.path.join(MODEL_DIR, f"{name}.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model.get_config(),
        }, path)
        print(f"  💾 Model kaydedildi: {path}")
        return path

    def load_model(self, path: str):
        """Kaydedilmiş model ağırlıklarını yükler."""
        checkpoint = torch.load(path, map_location=self.device,
                                weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  📂 Model yüklendi: {path}")

    def run_hyperparameter_experiments(self, train_loader, val_loader,
                                       test_loader):
        """
        Farklı hiper-parametre kombinasyonlarıyla deneyler çalıştırır.
        Makale Tablo/Grafik verisi üretir.
        """
        experiments = [
            {
                "name": "Temel_Model",
                "config": {},  # Varsayılan config
                "train_args": {"epochs": 30},
            },
            {
                "name": "Yüksek_LR",
                "config": {},
                "train_args": {"epochs": 30, "learning_rate": 0.005},
            },
            {
                "name": "Düşük_LR",
                "config": {},
                "train_args": {"epochs": 30, "learning_rate": 0.0001},
            },
            {
                "name": "Büyük_GRU",
                "config": {"gru_hidden_size": 256, "gru_num_layers": 3},
                "train_args": {"epochs": 30},
            },
            {
                "name": "Derin_CNN",
                "config": {"cnn_filters": [32, 64, 128]},
                "train_args": {"epochs": 30},
            },
        ]

        print("\n" + "=" * 60)
        print("  🔬 HİPER-PARAMETRE DENEYLERİ BAŞLATILIYOR")
        print("=" * 60)

        all_results = []

        for exp in experiments:
            print(f"\n  🧪 Deney: {exp['name']}")
            print("-" * 40)

            # Yeni model oluştur
            model = HybridCNNGRU(**exp["config"])
            self.model = model.to(self.device)
            self.best_model_state = None

            # Eğit
            self.train(train_loader, val_loader,
                       experiment_name=exp["name"], **exp["train_args"])

            # Değerlendir
            result = self.evaluate(test_loader,
                                   experiment_name=exp["name"])
            all_results.append({
                "name": exp["name"],
                "mse": result["mse"],
                "mae": result["mae"],
                "r2": result["r2"],
            })

        # Sonuç tablosu
        self.logger.print_experiment_table()

        # Karşılaştırma grafiği
        plot_hyperparameter_comparison(
            all_results,
            os.path.join(PLOT_DIR, "hyperparameter_comparison.png")
        )

        # Rapor dosyası
        self.logger.save_experiment_report(
            os.path.join(LOG_DIR, "experiment_report.txt")
        )

        return all_results
