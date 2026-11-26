# CICIDS2017 — Supervised ML Evaluation Report (Random Forest vs. XGBoost)

- **Random Seed:** `42`
- **Split Sizes:** Train `1,748,673` | Val `374,716` | Test `374,716`
- **Tree Input Features:** `72`

## Test-Set Model Comparison

| Task | Model | Accuracy | Macro-F1 | Weighted-F1 | Op. Benign FPR | ROC-AUC | Train (s) | Single Flow (ms) | Batch (ms/1k) | Artifact (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `binary` | `RF` | 0.9992 | **0.9985** | 0.9992 | **0.06%** | 1.0000 | 103.9 | 31.27 | 5.96 | 5.66 |
| `binary` | `XGB` | 0.9994 | **0.9989** | 0.9994 | **0.07%** | 1.0000 | 11.6 | 0.59 | 2.75 | 1.06 |
| `multiclass` | `RF` | 0.9974 | **0.7752** | 0.9979 | **0.26%** | — | 88.8 | 31.35 | 9.13 | 8.67 |
| `multiclass` | `XGB` | 0.9990 | **0.8921** | 0.9990 | **0.07%** | — | 89.3 | 1.36 | 7.54 | 7.77 |

## Model Selection Summary

- **`binary` Winner:** `XGB` (XGB Macro-F1=0.9989, FPR=0.07% vs. RF Macro-F1=0.9985, FPR=0.06%)
- **`multiclass` Winner:** `XGB` (XGB Macro-F1=0.8921, FPR=0.07% vs. RF Macro-F1=0.7752, FPR=0.26%)

## Multiclass Per-Class Test Performance (RF vs. XGBoost)

| Attack Class | Support | RF Prec | RF Rec | RF F1 | XGB Prec | XGB Rec | XGB F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BENIGN` | 310,945 | 0.9998 | 0.9974 | 0.9986 | 0.9999 | 0.9993 | 0.9996 |
| `Bot` | 293 | 0.3073 | 0.9795 | 0.4678 | 0.6817 | 0.9795 | 0.8039 |
| `DDoS` | 19,202 | 0.9999 | 0.9995 | 0.9997 | 0.9997 | 0.9998 | 0.9998 |
| `DoS GoldenEye` | 1,542 | 0.9961 | 0.9961 | 0.9961 | 0.9955 | 0.9987 | 0.9971 |
| `DoS Hulk` | 25,905 | 0.9969 | 0.9993 | 0.9981 | 0.9993 | 0.9998 | 0.9996 |
| `DoS Slowhttptest` | 784 | 0.9898 | 0.9911 | 0.9904 | 0.9886 | 0.9923 | 0.9905 |
| `DoS slowloris` | 806 | 0.9938 | 0.9926 | 0.9932 | 0.9914 | 0.9963 | 0.9938 |
| `FTP-Patator` | 890 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `Heartbleed` | 2 | 1.0000 | 0.5000 | 0.6667 | 1.0000 | 1.0000 | 1.0000 |
| `Infiltration` | 5 | 1.0000 | 0.2000 | 0.3333 | 0.5000 | 0.4000 | 0.4444 |
| `PortScan` | 13,538 | 0.9957 | 0.9994 | 0.9975 | 0.9956 | 0.9994 | 0.9975 |
| `SSH-Patator` | 483 | 1.0000 | 0.9959 | 0.9979 | 1.0000 | 1.0000 | 1.0000 |
| `Web Attack - Brute Force` | 221 | 0.7489 | 0.7421 | 0.7455 | 0.7414 | 0.7783 | 0.7594 |
| `Web Attack - Sql Injection` | 3 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| `Web Attack - XSS` | 97 | 0.4245 | 0.4639 | 0.4433 | 0.4235 | 0.3711 | 0.3956 |

## False-Positive Error Analysis (`multiclass` `XGB` on Held-Out Test)

- **Total Benign Test Flows:** `310,945`
- **False Positives:** `225` (`0.07%` FPR)
- **FP Confidence Quantiles:** `{'min': 0.4765, 'median': 0.9634, 'p90': 0.9982, 'max': 0.9995, 'high_confidence_ge_0_80': 185}`
- **Benign Traffic Misclassified As:** `{'Bot': 134, 'PortScan': 60, 'DoS Hulk': 13, 'DDoS': 6, 'DoS Slowhttptest': 5, 'DoS GoldenEye': 3, 'Infiltration': 2, 'DoS slowloris': 1, 'Web Attack - XSS': 1}`
- **Top Network Context Attributes Among FPs:** `{'Destination Port': {'8080': 120, '80': 29, '2525': 1, '21': 1, '32779': 1, '5357': 1, '3000': 1, '987': 1}}`
