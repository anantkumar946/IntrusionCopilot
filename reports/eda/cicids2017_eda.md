# CICIDS2017 Raw Dataset EDA Report

- **Files:** 8 | **Columns:** 79 | **Schema mismatches:** None
- **Total rows:** 2,830,743
- **Unique rows (hash audit):** 2,522,362
- **Exact duplicates:** 308,381 (10.89%)
- **Non-numeric columns seen:** ['Label']
- **Zero-variance columns (8):** ['Bwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bulk Rate', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk']
- **Columns with NaN:** {'Flow Bytes/s': 1358}
- **Columns with Inf:** {'Flow Packets/s': 2867, 'Flow Bytes/s': 1509}
- **Columns with Negative values:** {'Init_Win_bytes_backward': 1441552, 'Init_Win_bytes_forward': 1001189, 'Flow IAT Min': 2891, 'Flow Duration': 115, 'Flow Packets/s': 115, 'Flow IAT Mean': 115, 'Flow IAT Max': 115, 'Flow Bytes/s': 85, 'Fwd Header Length': 35, 'Fwd Header Length.1': 35, 'min_seg_size_forward': 35, 'Bwd Header Length': 22, 'Fwd IAT Min': 17}

## Class Distribution & Deduplication Impact

| Raw Label (`repr`) | Raw Count | After Dedup | Removed | % Removed |
|---|---:|---:|---:|---:|
| `'BENIGN'` | 2,273,097 | 2,096,484 | 176,613 | 7.77% |
| `'DoS Hulk'` | 231,073 | 172,849 | 58,224 | 25.2% |
| `'PortScan'` | 158,930 | 90,819 | 68,111 | 42.86% |
| `'DDoS'` | 128,027 | 128,016 | 11 | 0.01% |
| `'DoS GoldenEye'` | 10,293 | 10,286 | 7 | 0.07% |
| `'FTP-Patator'` | 7,938 | 5,933 | 2,005 | 25.26% |
| `'SSH-Patator'` | 5,897 | 3,219 | 2,678 | 45.41% |
| `'DoS slowloris'` | 5,796 | 5,385 | 411 | 7.09% |
| `'DoS Slowhttptest'` | 5,499 | 5,228 | 271 | 4.93% |
| `'Bot'` | 1,966 | 1,953 | 13 | 0.66% |
| `'Web Attack � Brute Force'` | 1,507 | 1,470 | 37 | 2.46% |
| `'Web Attack � XSS'` | 652 | 652 | 0 | 0.0% |
| `'Infiltration'` | 36 | 36 | 0 | 0.0% |
| `'Web Attack � Sql Injection'` | 21 | 21 | 0 | 0.0% |
| `'Heartbleed'` | 11 | 11 | 0 | 0.0% |
