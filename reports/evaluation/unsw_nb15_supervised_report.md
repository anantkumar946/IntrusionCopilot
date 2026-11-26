# UNSW_NB15 — Supervised ML Evaluation Report (Random Forest vs. XGBoost)

- **Random Seed:** `42`
- **Split Sizes:** Train `84,377` | Val `14,891` | Test `52,418`
- **Tree Input Features:** `46`

## Test-Set Model Comparison

| Task | Model | Accuracy | Macro-F1 | Weighted-F1 | Op. Benign FPR | ROC-AUC | Train (s) | Single Flow (ms) | Batch (ms/1k) | Artifact (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `binary` | `RF` | 0.8229 | **0.8199** | 0.8266 | **26.12%** | 0.9645 | 3.8 | 30.54 | 6.12 | 19.69 |
| `binary` | `XGB` | 0.8221 | **0.8187** | 0.8258 | **25.60%** | 0.9650 | 2.4 | 0.51 | 1.28 | 1.83 |
| `multiclass` | `RF` | 0.7100 | **0.5978** | 0.7552 | **35.55%** | — | 6.4 | 31.54 | 9.19 | 46.33 |
| `multiclass` | `XGB` | 0.7450 | **0.6092** | 0.7811 | **29.94%** | — | 15.8 | 2.99 | 15.27 | 14.87 |

## Model Selection Summary

- **`binary` Winner:** `RF` (XGB Macro-F1=0.8187, FPR=25.60% vs. RF Macro-F1=0.8199, FPR=26.12%)
- **`multiclass` Winner:** `XGB` (XGB Macro-F1=0.6092, FPR=29.94% vs. RF Macro-F1=0.5978, FPR=35.55%)

## Multiclass Per-Class Test Performance (RF vs. XGBoost)

| Attack Class | Support | RF Prec | RF Rec | RF F1 | XGB Prec | XGB Rec | XGB F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Normal` | 33,756 | 0.9848 | 0.6445 | 0.7791 | 0.9752 | 0.7006 | 0.8154 |
| `Analysis` | 58 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `Backdoor` | 57 | 0.5070 | 0.6316 | 0.5625 | 0.5147 | 0.6140 | 0.5600 |
| `DoS` | 1,163 | 0.6603 | 0.3861 | 0.4872 | 0.6306 | 0.4110 | 0.4977 |
| `Exploits` | 7,041 | 0.7441 | 0.9034 | 0.8160 | 0.7507 | 0.9067 | 0.8214 |
| `Fuzzers` | 4,321 | 0.2622 | 0.7945 | 0.3943 | 0.2780 | 0.7605 | 0.4071 |
| `Generic` | 3,399 | 0.9861 | 0.8579 | 0.9176 | 0.9777 | 0.8653 | 0.9181 |
| `Reconnaissance` | 2,215 | 0.8772 | 0.8709 | 0.8740 | 0.8829 | 0.8682 | 0.8755 |
| `Shellcode` | 365 | 0.2761 | 0.8411 | 0.4157 | 0.3223 | 0.8822 | 0.4721 |
| `Worms` | 43 | 0.7692 | 0.6977 | 0.7317 | 0.7838 | 0.6744 | 0.7250 |

## False-Positive Error Analysis (`multiclass` `XGB` on Held-Out Test)

- **Total Benign Test Flows:** `33,756`
- **False Positives:** `10,105` (`29.94%` FPR)
- **FP Confidence Quantiles:** `{'min': 0.2721, 'median': 0.7214, 'p90': 0.9246, 'max': 0.9983, 'high_confidence_ge_0_80': 3510}`
- **Benign Traffic Misclassified As:** `{'Fuzzers': 8364, 'Analysis': 781, 'Exploits': 707, 'Shellcode': 171, 'DoS': 67, 'Generic': 9, 'Reconnaissance': 6}`
- **Top Network Context Attributes Among FPs:** `{'service': {'unknown': 8532, 'http': 1239, 'ftp': 331, 'radius': 2, 'dns': 1}, 'proto': {'tcp': 7791, 'udp': 2312, 'igmp': 2}, 'state': {'FIN': 7786, 'INT': 2305, 'REQ': 9, 'CON': 4, 'RST': 1}}`
