"""
VoltOptimizer - Hybrid 1D-CNN + GRU Deep Learning Model
=========================================================
A hybrid architecture for predicting the Remaining Useful Life (RUL)
of electric vehicle batteries.

Architecture Flow:
    Input(batch, seq_len, features)
        → 1D-CNN Blocks (local temporal feature extraction)
        → GRU Layers (long-range dependency learning)
        → Fully Connected Layers (regression output)
        → Output: RUL prediction [0, 1]

Reference:
    This hybrid approach combines CNN's ability to capture short-term
    temporal patterns with GRU's capacity to model long-term dependencies.
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, DATA_CONFIG


class CNNBlock(nn.Module):
    """
    1D Convolution Block.
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
    1D-CNN + GRU Hybrid Model.

    Layer structure:
        1. 1D-CNN Layers: Local temporal feature extraction
        2. GRU Layers: Sequential dependency modelling
        3. Attention Mechanism: Focus on important time steps
        4. Fully Connected: Regression output

    Args:
        num_features: Number of input features
        cnn_filters: List of CNN filter counts
        cnn_kernel_size: CNN kernel size
        gru_hidden_size: GRU hidden layer size
        gru_num_layers: Number of GRU layers
        fc_hidden: Fully connected layer size
        dropout: Dropout rate
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

        # Load defaults from config
        num_features = num_features or DATA_CONFIG["num_features"]
        cnn_filters = cnn_filters or MODEL_CONFIG["cnn_filters"]
        cnn_kernel_size = cnn_kernel_size or MODEL_CONFIG["cnn_kernel_size"]
        gru_hidden_size = gru_hidden_size or MODEL_CONFIG["gru_hidden_size"]
        gru_num_layers = gru_num_layers or MODEL_CONFIG["gru_num_layers"]
        fc_hidden = fc_hidden or MODEL_CONFIG["fc_hidden"]
        dropout = dropout if dropout is not None else MODEL_CONFIG["dropout"]

        # ── 1D-CNN Layers ──
        cnn_layers = []
        in_ch = num_features
        for out_ch in cnn_filters:
            cnn_layers.append(CNNBlock(in_ch, out_ch, cnn_kernel_size,
                                       dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # ── GRU Layers ──
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

        # ── Attention Mechanism ──
        # Additive (Bahdanau-style) scoring: maps each GRU hidden state to a
        # scalar score; tanh keeps gradients healthy across long sequences
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(gru_hidden_size // 2, 1),
        )

        # ── Fully Connected Layers (Regression Output) ──
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
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            RUL prediction tensor of shape (batch_size,), values in [0, 1]
        """
        # Conv1d expects (batch, channels, length); input is (batch, length, features)
        x = x.permute(0, 2, 1)

        # Local temporal pattern extraction (short-range dependencies)
        x = self.cnn(x)

        # Restore to (batch, seq_len, cnn_out) for GRU's batch_first mode
        x = x.permute(0, 2, 1)

        # Sequential modelling with GRU
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

        output = self.fc(context)
        return output.squeeze(-1)

    def get_config(self) -> dict:
        """Returns model configuration."""
        return self._config.copy()

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Prints a model summary."""
        total_params = self.count_parameters()
        print("\n" + "=" * 60)
        print(f"  🧠 VoltOptimizer - Hybrid 1D-CNN + GRU Model")
        print("=" * 60)
        print(f"  CNN Filters       : {self._config['cnn_filters']}")
        print(f"  CNN Kernel Size   : {self._config['cnn_kernel_size']}")
        print(f"  GRU Hidden Size   : {self._config['gru_hidden_size']}")
        print(f"  GRU Layers        : {self._config['gru_num_layers']}")
        print(f"  FC Hidden Size    : {self._config['fc_hidden']}")
        print(f"  Dropout           : {self._config['dropout']}")
        print(f"  Total Parameters  : {total_params:,}")
        print("=" * 60)
        return total_params
