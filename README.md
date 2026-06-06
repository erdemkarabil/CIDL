# VoltOptimizer

A **hybrid 1D-CNN + GRU** deep learning model for electric vehicle battery Remaining Useful Life (RUL) prediction, combined with a multi-agent optimisation system powered by three specialist agents: **Battery Guardian**, **Grid Tariff**, and **Smart Trip**; coordinated through an orchestrator.

**GitHub:** [github.com/erdemkarabil/CIDL](https://github.com/erdemkarabil/CIDL)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py                      # Full scenario (training + agents)
python main.py --skip-training      # Skip training, agent scenario only
python main.py --experiments        # Include hyperparameter experiments
python main.py --training-only      # DL model training only
```

## Folder Structure

| Folder | Description |
|--------|-------------|
| `models/` | 1D-CNN+GRU model, trainer, and evaluation |
| `agents/` | Three specialist agents (battery, tariff, route) |
| `tools/` | Tools used by the agents |
| `orchestrator/` | Agent coordination and scenario flow |
| `data/` | Synthetic data generator |
| `outputs/` | Trained models (`models/`), plots (`plots/`), logs (`logs/`) |

## Report

Detailed project report: [Report.pdf](Report.pdf) · [Report.docx](Report.docx)

## Demo

`demo.mp4` — *to be added*
