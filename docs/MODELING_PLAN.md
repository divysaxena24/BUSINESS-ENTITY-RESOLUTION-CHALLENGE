# Modeling Plan

## Objective

Build a high-precision pairwise classifier for business entity matching.

Primary metric:

Macro F_0.5.

---

# Phase 1 — Baseline

Implement:

```text
normalized name similarity
+
normalized address similarity
+
country equality
```

Create a simple rule-based score.

This establishes a sanity-check baseline.

---

# Phase 2 — Candidate Generation

Implement multiple candidate generators:

1. exact normalized name
2. shared name tokens
3. shared address tokens
4. TF-IDF name retrieval
5. TF-IDF address retrieval

Measure candidate recall.

---

# Phase 3 — Pair Features

Build:

```text
name_exact
name_edit_similarity
name_jaccard
name_tfidf
name_ngram

address_exact
address_edit_similarity
address_jaccard
address_tfidf
address_ngram

country_exact

digit_overlap
postal_match
length_difference
```

---

# Phase 4 — Supervised Model

Train:

### Model A

Logistic Regression

### Model B

Tree-based classifier

Compare using identical validation data.

---

# Phase 5 — Threshold Search

Evaluate thresholds from 0.30 to 0.90.

Record:

```text
threshold
precision
recall
macro_F0.5
singleton_accuracy
predicted_match_count
```

Select based on validation macro F_0.5.

---

# Phase 6 — Hard Negative Mining

Find high-scoring false positives.

Add them to training data.

Retrain.

Evaluate again.

Do not add validation examples to training.

---

# Phase 7 — Error Analysis

Analyze:

### False positives

* common names
* franchises
* same address
* same city
* legal suffix confusion

### False negatives

* severe typos
* transliteration
* abbreviations
* incomplete addresses
* reordered tokens

---

# Phase 8 — Optional Advanced Models

Only after strong classical baselines.

Potential additions:

* character n-gram models
* calibrated gradient boosting
* learned pairwise embeddings
* compact transformer embeddings if license and parameter limits permit

Any advanced model must demonstrate measurable validation improvement.

---

# Phase 9 — Finalization

Freeze:

* normalization
* blocking
* features
* model
* threshold
* random seed

Then run test inference exactly once for the final candidate submission.
