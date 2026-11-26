# UNSW-NB15 Raw Dataset EDA Report

## File: `UNSW_NB15_testing-set.csv`
- **Shape:** 82,332 rows × 45 cols (14.67 MB, encoding=`utf-8-sig`)
- **Categorical columns:** ['proto', 'service', 'state', 'attack_cat']
- **Zero-variance columns:** []
- **Duplicates:** 0 (all cols) | 26,387 (excluding `id`)
- **NaN cols:** None
- **Inf cols:** None
- **Negative cols:** None
- **Binary `label`:** {'1': 45332, '0': 37000}
- **Multiclass `attack_cat`:** {"'Normal'": 37000, "'Generic'": 18871, "'Exploits'": 11132, "'Fuzzers'": 6062, "'DoS'": 4089, "'Reconnaissance'": 3496, "'Analysis'": 677, "'Backdoor'": 583, "'Shellcode'": 378, "'Worms'": 44}

## File: `UNSW_NB15_training-set.csv`
- **Shape:** 175,341 rows × 45 cols (30.8 MB, encoding=`utf-8-sig`)
- **Categorical columns:** ['proto', 'service', 'state', 'attack_cat']
- **Zero-variance columns:** []
- **Duplicates:** 0 (all cols) | 67,601 (excluding `id`)
- **NaN cols:** None
- **Inf cols:** None
- **Negative cols:** None
- **Binary `label`:** {'1': 119341, '0': 56000}
- **Multiclass `attack_cat`:** {"'Normal'": 56000, "'Generic'": 40000, "'Exploits'": 33393, "'Fuzzers'": 18184, "'DoS'": 12264, "'Reconnaissance'": 10491, "'Analysis'": 2000, "'Backdoor'": 1746, "'Shellcode'": 1133, "'Worms'": 130}

## Cross-File Audit
```json
{
  "identical_columns": true,
  "cross_file_overlap_excluding_id": {
    "files": [
      "UNSW_NB15_testing-set.csv",
      "UNSW_NB15_training-set.csv"
    ],
    "unique_shared_row_hashes": 940,
    "rows_in_UNSW_NB15_testing-set.csv_also_in_UNSW_NB15_training-set.csv": 4247,
    "rows_in_UNSW_NB15_training-set.csv_also_in_UNSW_NB15_testing-set.csv": 8421
  },
  "categorical_value_differences": {
    "proto": {
      "only_in_UNSW_NB15_testing-set.csv": [],
      "only_in_UNSW_NB15_training-set.csv": [
        "icmp",
        "rtp"
      ]
    },
    "service": {
      "only_in_UNSW_NB15_testing-set.csv": [],
      "only_in_UNSW_NB15_training-set.csv": []
    },
    "state": {
      "only_in_UNSW_NB15_testing-set.csv": [
        "ACC",
        "CLO"
      ],
      "only_in_UNSW_NB15_training-set.csv": [
        "ECO",
        "PAR",
        "URN",
        "no"
      ]
    },
    "attack_cat": {
      "only_in_UNSW_NB15_testing-set.csv": [],
      "only_in_UNSW_NB15_training-set.csv": []
    }
  }
}
```
