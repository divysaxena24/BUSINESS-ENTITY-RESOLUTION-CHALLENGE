# Baseline Results

*Results measured on a 5,000 S1 entity development subset.*

| Experiment | Blocking | Features | Model | Threshold | Candidate Recall | Macro F0.5 | Precision | Recall | Runtime | Notes |
| ---------- | -------- | -------- | ----- | --------: | ---------------: | ---------: | --------: | -----: | ------: | ----- |
| 1. Exact Name + 2-token | exact_name U 2_shared_tokens | 14 baseline features (name, addr, country) | GBDT (sklearn) | 0.65 | 54.21% | 0.6722 | 0.7889 | 0.5012 | ~5 min | Capped at 200 candidates per S1. Recall ceiling is artificially constrained by this safety cap. |

*Note: Model metrics reflect performance ON the capped candidate set. The true recall ceiling is 54.21%.*
