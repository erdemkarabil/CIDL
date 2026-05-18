# VoltOptimizer

Elektrikli araç batarya ömrü (RUL) tahmini için **1D-CNN + GRU** hibrit derin öğrenme modeli ve üç uzman ajanlı çok etmenli optimizasyon sistemi. Ajanlar: **Battery Guardian**, **Grid Tariff** ve **Smart Trip**; orchestrator üzerinden koordine edilir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python main.py                      # Tam senaryo (eğitim + ajanlar)
python main.py --skip-training      # Eğitimi atla, sadece ajan senaryosu
python main.py --experiments        # Hiper-parametre deneyleri dahil
python main.py --training-only      # Sadece DL model eğitimi
```

## Klasör yapısı

| Klasör | Açıklama |
|--------|----------|
| `models/` | 1D-CNN+GRU modeli, eğitici ve değerlendirme |
| `agents/` | Üç uzman ajan (batarya, tarife, rota) |
| `tools/` | Ajanların kullandığı araçlar |
| `orchestrator/` | Ajan koordinasyonu ve senaryo akışı |
| `data/` | Veri setleri ve ön işleme |
| `outputs/` | Eğitilmiş modeller (`models/`), grafikler (`plots/`), loglar (`logs/`) |

## Rapor

Detaylı proje raporu: [Report.pdf](Report.pdf)

## Demo

`demo.mp4` — *eklenecek*
