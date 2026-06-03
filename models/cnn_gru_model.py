"""
VoltOptimizer - Hibrit 1D-CNN + GRU Derin Öğrenme Modeli
=========================================================
Elektrikli araç bataryalarının Kalan Kullanım Ömrünü (RUL) tahmin eden
hibrit bir mimari.

Mimari Akışı:
    Input(batch, seq_len, features)
        → 1D-CNN Blokları (zamansal yerel öznitelik çıkarımı)
        → GRU Katmanları (uzun vadeli bağımlılık öğrenimi)
        → Fully Connected Katmanlar (regresyon çıktısı)
        → Output: RUL tahmini [0, 1]

Referans:
    Bu hibrit yaklaşım, CNN'in kısa vadeli zamansal desenleri yakalama
    yeteneğini, GRU'nun uzun vadeli bağımlılıkları modelleme kapasitesi
    ile birleştirir.
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, DATA_CONFIG


class CNNBlock(nn.Module):
    """
    1D Konvolüsyon Bloğu.
    Conv1D → BatchNorm → ReLU → Dropout
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                # same-padding: keeps sequence length unchanged so GRU
                # receives a tensor with the exact same time dimension
                padding=kernel_size // 2
            ),
            # BatchNorm before activation stabilises training;
            # especially helpful here because battery sensor scales differ
            # by orders of magnitude (volts vs. amps vs. °C)
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class HybridCNNGRU(nn.Module):
    """
    1D-CNN + GRU Hibrit Model.

    Katman yapısı:
        1. 1D-CNN Katmanları: Zamansal yerel öznitelik çıkarımı
        2. GRU Katmanları: Sıralı bağımlılık modelleme
        3. Attention Mekanizması: Önemli zaman adımlarına odaklanma
        4. Fully Connected: Regresyon çıktısı

    Args:
        num_features: Girdi öznitelik sayısı
        cnn_filters: CNN filtre sayıları listesi
        cnn_kernel_size: CNN çekirdek boyutu
        gru_hidden_size: GRU gizli katman boyutu
        gru_num_layers: GRU katman sayısı
        fc_hidden: Tam bağlantılı katman boyutu
        dropout: Dropout oranı
    """

    def __init__(
        self,
        num_features: int = None,
        cnn_filters: list = None,
        cnn_kernel_size: int = None,
        gru_hidden_size: int = None,
        gru_num_layers: int = None,
        fc_hidden: int = None,
        dropout: float = None,
    ):
        super().__init__()

        # Varsayılan değerleri config'den al
        num_features = num_features or DATA_CONFIG["num_features"]
        cnn_filters = cnn_filters or MODEL_CONFIG["cnn_filters"]
        cnn_kernel_size = cnn_kernel_size or MODEL_CONFIG["cnn_kernel_size"]
        gru_hidden_size = gru_hidden_size or MODEL_CONFIG["gru_hidden_size"]
        gru_num_layers = gru_num_layers or MODEL_CONFIG["gru_num_layers"]
        fc_hidden = fc_hidden or MODEL_CONFIG["fc_hidden"]
        dropout = dropout if dropout is not None else MODEL_CONFIG["dropout"]

        # ── 1D-CNN Katmanları ──
        cnn_layers = []
        in_ch = num_features
        for out_ch in cnn_filters:
            cnn_layers.append(CNNBlock(in_ch, out_ch, cnn_kernel_size,
                                       dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # ── GRU Katmanları ──
        self.gru = nn.GRU(
            input_size=cnn_filters[-1],
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            # PyTorch raises an error if dropout > 0 with a single GRU layer,
            # because inter-layer dropout requires at least two layers
            dropout=dropout if gru_num_layers > 1 else 0.0,
            # Unidirectional: in real-time BMS inference we can only see
            # past measurements, not future ones
            bidirectional=False,
        )

        # ── Attention Mekanizması (Basit) ──
        # Additive (Bahdanau-style) scoring: maps each GRU hidden state to a
        # scalar score; tanh keeps gradients healthy across long sequences
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(gru_hidden_size // 2, 1),
        )

        # ── Fully Connected Katmanlar (Regresyon Çıktısı) ──
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden // 2, 1),
            # Sigmoid enforces the physical constraint RUL ∈ [0, 1].
            # Without it the model can predict negative or >100% RUL,
            # which is physically meaningless for a battery health metric.
            nn.Sigmoid(),
        )

        # Model bilgisi
        self._config = {
            "num_features": num_features,
            "cnn_filters": cnn_filters,
            "cnn_kernel_size": cnn_kernel_size,
            "gru_hidden_size": gru_hidden_size,
            "gru_num_layers": gru_num_layers,
            "fc_hidden": fc_hidden,
            "dropout": dropout,
        }

    def forward(self, x):
        """
        İleri yayılım.

        Args:
            x: (batch_size, seq_len, num_features) boyutlu girdi tensörü

        Returns:
            (batch_size, 1) boyutlu RUL tahmin tensörü [0, 1]
        """
        # Conv1d expects (batch, channels, length); input is (batch, length, features)
        x = x.permute(0, 2, 1)

        # Local temporal pattern extraction (short-range dependencies)
        x = self.cnn(x)

        # Restore to (batch, seq_len, cnn_out) for GRU's batch_first mode
        x = x.permute(0, 2, 1)

        # GRU ile sıralı modelleme
        gru_out, _ = self.gru(x)
        # gru_out: (batch, seq_len, hidden_size)

        # Score each time step, then normalise across the time axis so weights
        # sum to 1 — this lets the model focus on degradation events (e.g. a
        # sudden temperature spike) rather than treating all steps equally
        attn_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum collapses the sequence into a single context vector
        context = torch.sum(attn_weights * gru_out, dim=1)
        # context: (batch, hidden_size)

        # Regresyon çıktısı
        output = self.fc(context)
        return output.squeeze(-1)

    def get_config(self) -> dict:
        """Model konfigürasyonunu döndürür."""
        return self._config.copy()

    def count_parameters(self) -> int:
        """Toplam eğitilebilir parametre sayısını döndürür."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Model özetini yazdırır."""
        total_params = self.count_parameters()
        print("\n" + "=" * 60)
        print(f"  🧠 VoltOptimizer - Hibrit 1D-CNN + GRU Modeli")
        print("=" * 60)
        print(f"  CNN Filtreleri    : {self._config['cnn_filters']}")
        print(f"  CNN Kernel Boyutu : {self._config['cnn_kernel_size']}")
        print(f"  GRU Gizli Boyut   : {self._config['gru_hidden_size']}")
        print(f"  GRU Katman Sayısı : {self._config['gru_num_layers']}")
        print(f"  FC Gizli Boyut    : {self._config['fc_hidden']}")
        print(f"  Dropout           : {self._config['dropout']}")
        print(f"  Toplam Parametre  : {total_params:,}")
        print("=" * 60)
        return total_params
