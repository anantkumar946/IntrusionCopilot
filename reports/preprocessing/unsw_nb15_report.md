# UNSW-NB15 Preprocessing Validation Report

- **Random Seed:** `42`
- **Dropped Columns:** `['id']`
- **Engineered Features (4):** `['bytes_ratio', 'pkts_ratio', 'total_bytes', 'total_pkts']`
- **Total Model Input Features:** `46` (Tree matrix: `46` dims | Autoencoder matrix: `60` dims)

## Row Reduction & Deduplication Summary

| Stage | Official Train CSV | Official Test CSV |
|---|---:|---:|
| Raw rows | 175,341 | 82,332 |
| Exact duplicates removed | 67,601 | 26,387 |
| Conflicting-label rows removed | 8,472 | 2,574 |
| Cross-split train overlap removed from test | 0 | 953 |
| Final clean rows | 99,268 | 52,418 |

## Final Split Sizes & Leakage Audit

| Split | Rows | Benign (0) | Malicious (1) | Parquet Size (MB) |
|---|---:|---:|---:|---:|
| `train` | 84,377 | 43,911 | 40,466 | 7.41 |
| `val` | 14,891 | 7,750 | 7,141 | 1.34 |
| `test` | 52,418 | 33,756 | 18,662 | 4.27 |

- **Zero-Leakage Verification:** train∩val = `0`, train∩test = `0`, val∩test = `0`
- **Tree preprocessor fit strictly on:** `train` (84,377 rows)
- **Autoencoder preprocessor fit strictly on:** `train` Benign rows only (43,911 rows)

## Multiclass Distribution (`attack_cat`)

| Class | ID | Train | Val | Test |
|---|---:|---:|---:|---:|
| `Normal` | 0 | 43,911 | 7,750 | 33,756 |
| `Analysis` | 1 | 334 | 59 | 58 |
| `Backdoor` | 2 | 343 | 60 | 57 |
| `DoS` | 3 | 2,105 | 372 | 1,163 |
| `Exploits` | 4 | 15,654 | 2,763 | 7,041 |
| `Fuzzers` | 5 | 12,412 | 2,190 | 4,321 |
| `Generic` | 6 | 3,318 | 586 | 3,399 |
| `Reconnaissance` | 7 | 5,278 | 931 | 2,215 |
| `Shellcode` | 8 | 922 | 163 | 365 |
| `Worms` | 9 | 100 | 17 | 43 |
