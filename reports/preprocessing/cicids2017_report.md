# CICIDS2017 Preprocessing Validation Report

- **Random Seed:** `42`
- **Dropped Columns (9):** `['Bwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bulk Rate', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Header Length.1']`
- **Sentinel Negative Columns Preserved (`-1`):** `['Init_Win_bytes_backward', 'Init_Win_bytes_forward']`
- **Inf Values Replaced with NaN (train-imputed):** `4,376`
- **Invalid Negative Values Replaced with NaN (train-imputed):** `3,545`
- **Engineered Features (3):** `['fwd_bwd_pkt_ratio', 'fwd_bwd_byte_ratio', 'total_packets']`
- **Total Model Input Features:** `72` (Tree dim: `72` | AE dim: `72`)

## Row Reduction & Deduplication Summary

| Stage | Row Count |
|---|---:|
| Raw rows across 8 CSVs | 2,830,743 |
| In-file exact duplicates removed during streaming | 256,684 |
| Cross-file exact duplicates removed | 74,516 |
| Conflicting-label rows removed | 1,438 |
| Final clean unique rows | 2,498,105 |

## Final Split Sizes & Leakage Audit

| Split | Rows | Benign (0) | Malicious (1) | Parquet Size (MB) |
|---|---:|---:|---:|---:|
| `train` | 1,748,673 | 1,451,072 | 297,601 | 203.96 |
| `val` | 374,716 | 310,944 | 63,772 | 44.57 |
| `test` | 374,716 | 310,945 | 63,771 | 44.44 |

- **Zero-Leakage Verification:** train∩val = `0`, train∩test = `0`, val∩test = `0`
- **Tree preprocessor fit strictly on:** `train` (1,748,673 rows)
- **Autoencoder preprocessor fit strictly on:** `train` Benign rows only (1,451,072 rows)

## Multiclass Distribution (`attack_cat`)

| Class | ID | Train | Val | Test |
|---|---:|---:|---:|---:|
| `BENIGN` | 0 | 1,451,072 | 310,944 | 310,945 |
| `Bot` | 1 | 1,367 | 293 | 293 |
| `DDoS` | 2 | 89,609 | 19,202 | 19,202 |
| `DoS GoldenEye` | 3 | 7,197 | 1,543 | 1,542 |
| `DoS Hulk` | 4 | 120,889 | 25,905 | 25,905 |
| `DoS Slowhttptest` | 5 | 3,660 | 784 | 784 |
| `DoS slowloris` | 6 | 3,760 | 806 | 806 |
| `FTP-Patator` | 7 | 4,153 | 890 | 890 |
| `Heartbleed` | 8 | 8 | 1 | 2 |
| `Infiltration` | 9 | 25 | 6 | 5 |
| `PortScan` | 10 | 63,179 | 13,538 | 13,538 |
| `SSH-Patator` | 11 | 2,253 | 483 | 483 |
| `Web Attack - Brute Force` | 12 | 1,029 | 220 | 221 |
| `Web Attack - Sql Injection` | 13 | 15 | 3 | 3 |
| `Web Attack - XSS` | 14 | 457 | 98 | 97 |
