# Experiment Plan

## Experiment 0 — Data Audit

Goal:

Understand dataset size, missingness, duplicates, countries, and ground truth.

Output:

reports/data_profile.md

---

## Experiment 1 — Rule-Based Baseline

Features:

* name similarity
* address similarity
* country match

Measure macro F_0.5.

---

## Experiment 2 — Blocking A

Blocking:

exact normalized name.

Measure:

* candidate recall
* candidates per S1
* reduction ratio

---

## Experiment 3 — Blocking B

Add:

name token blocking.

Compare with Experiment 2.

---

## Experiment 4 — Blocking C

Add:

address token blocking.

Compare candidate recall.

---

## Experiment 5 — TF-IDF Blocking

Add:

TF-IDF nearest-neighbor candidates.

Measure whether recall improves.

---

## Experiment 6 — Feature Model

Train Logistic Regression.

Measure:

macro F_0.5
precision
recall

---

## Experiment 7 — Tree Model

Train a tree-based model using the same features.

Compare against Experiment 6.

---

## Experiment 8 — Threshold Search

Search:

0.30 → 0.90

Select threshold using validation macro F_0.5.

---

## Experiment 9 — Hard Negatives

Mine difficult false positives.

Retrain.

Measure improvement.

---

## Experiment 10 — Singleton Optimization

Analyze true singleton score distributions.

Test conservative decision rules.

Measure macro F_0.5.

---

## Experiment 11 — Final Candidate Pipeline

Freeze candidate generation.

Verify candidate recall.

---

## Experiment 12 — Final Model

Freeze:

* preprocessing
* blocking
* features
* model
* threshold

Generate final test outputs.

---

# Experiment Recording Format

Every experiment must contain:

```text
experiment_id
description
dataset_split
blocking_strategy
feature_set
model
hyperparameters
threshold
candidate_count
candidate_recall
precision
recall
macro_f0.5
singleton_accuracy
runtime
conclusion
```

Never delete previous experiments.
