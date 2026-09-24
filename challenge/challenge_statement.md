# Business Entity Resolution Challenge

## 1. Challenge Overview

Large-scale commercial platforms receive business identity data from multiple independent sources. Each source may contain partial, noisy, inconsistent, duplicated, or differently formatted information about the same real-world businesses.

The task is to perform **Entity Resolution (ER)**: determine which records across independent data sources refer to the same real-world business entity.

There are three data sources:

* Source 1
* Source 2
* Source 3

Source 1 is the deduplicated reference source.

For every Source 1 entity, identify **all matching records from Source 2 and Source 3**.

A Source 1 entity may have:

* zero matches
* one match
* multiple matches

The task is therefore not a simple one-to-one classification problem.

---

# 2. Objective

Given:

* Source 1 business records
* Source 2 business records
* Source 3 business records
* Ground-truth matches for the training data

build an ML-based entity resolution pipeline that predicts, for every Source 1 entity in the test set, the matching Source 2 and Source 3 entity IDs.

The system must optimize for the challenge's primary evaluation metric:

**Macro-averaged F_0.5**

F_0.5 is precision-heavy and therefore penalizes false merges more strongly than missed matches.

---

# 3. Important Entity Resolution Characteristics

The data contains noisy and inconsistent business information.

Expected sources of variation include:

### Business Name Variations

Examples of expected variation:

* abbreviations
* legal suffix differences
* punctuation differences
* spelling mistakes
* token reordering
* transliteration
* DBA/trade names
* `Corp` vs `Corporation`
* `Pvt` vs `Private`
* `Ltd` vs `Limited`
* `&` vs `and`

### Address Variations

Examples include:

* `Rd` vs `Road`
* `St` vs `Street`
* missing address components
* missing PIN/postal code
* missing state
* component reordering
* different numbering formats
* landmark-based descriptions
* transliteration
* partial addresses
* formatting differences

The matching system must therefore use more than exact string equality.

---

# 4. Data Format

All files are **tab-separated values (TSV)** files.

Always load TSV files using an explicit tab separator:

```python
import pandas as pd

df = pd.read_csv("file.tsv", sep="\t")
```

Do not rely on the default comma separator.

This is important because:

* business addresses may contain commas
* ground-truth ID lists contain commas
* output ID lists contain commas

Tabs separate columns.

Commas separate IDs inside list fields.

---

# 5. Input Files

## Training Files

Located in:

```text
dataset/train/
```

Files:

```text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv
```

---

## Test Files

Located in:

```text
dataset/test/
```

Files:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

The test set does not contain ground-truth matching labels.

---

# 6. Source Record Schema

Each source file contains the following columns:

```text
entity_id
business_name
business_address
country
```

## 6.1 entity_id

Unique identifier for a record.

The prefix identifies the source:

```text
S1-    Source 1
S2-    Source 2
S3-    Source 3
```

Examples:

```text
S1-00001
S2-00047
S3-00812
```

Do not assume that IDs are sequential beyond what is actually observed in the data.

---

## 6.2 business_name

Name of the business.

May contain:

* abbreviations
* legal suffixes
* typos
* transliterations
* punctuation variations
* word-order variations
* trade names

---

## 6.3 business_address

Business address.

May contain:

* partial addresses
* missing components
* formatting variations
* abbreviations
* transliterations
* landmarks
* municipal numbering variations
* missing postal/PIN information

Do not assume that city, state, postal code, or other address components exist as separate columns.

If they exist only inside `business_address`, they may be extracted as part of feature engineering.

---

## 6.4 country

Country label associated with the record.

The training data covers:

```text
US
India
```

The test data additionally contains:

```text
France
```

However, **country must be treated as an open-set string field**.

The implementation must NOT:

* hard-code `{US, India}`
* discard France
* filter to training countries
* assume only two countries exist
* use a categorical encoding that fails on unseen countries

Every test Source 1 entity must appear in the final submission, including entities associated with unseen country labels.

---

# 7. Ground Truth Format

The training ground-truth file is:

```text
dataset/train/train_ground_truth.tsv
```

It contains:

```text
source1_entity_id
matched_entity_ids
```

`matched_entity_ids` is a comma-separated list of matching Source 2 and/or Source 3 IDs.

Example:

```text
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

An empty `matched_entity_ids` field means that the Source 1 entity has no matching Source 2 or Source 3 records.

This is a valid ground-truth singleton.

It must NOT be interpreted as missing training data.

---

# 8. Matching Objective

For every Source 1 entity:

```text
S1 entity
    ↓
zero or more S2 matches
+
zero or more S3 matches
```

The system must not assume:

```text
one S1 → exactly one S2/S3
```

Instead:

```text
one S1 → zero, one, or many records
```

A Source 1 entity may match records in both Source 2 and Source 3.

---

# 9. Expected ML Pipeline

A recommended high-level architecture is:

```text
Input Data
    ↓
Data Validation
    ↓
Text Normalization
    ↓
Candidate Generation / Blocking
    ↓
Pair Feature Engineering
    ↓
Supervised Matching Model
    ↓
Score / Probability
    ↓
Threshold / Decision Layer
    ↓
Final Matches
    ↓
Submission Validation
```

Candidate generation and final matching are separate stages.

---

# 10. Candidate Generation / Blocking

Candidate generation is critical because the matching model can only identify matches among candidates it receives.

If a true match is excluded during blocking, the model cannot recover it.

The candidate-generation stage should therefore prioritize **high recall** while keeping the number of candidate pairs computationally manageable.

Potential blocking strategies include:

* exact normalized business name
* normalized name token overlap
* shared informative name tokens
* address token overlap
* postal/PIN similarity where extractable
* character n-gram retrieval
* TF-IDF similarity
* nearest-neighbor retrieval

Multiple blocking strategies may be combined.

The final candidate set passed to the matching model must be recorded in:

```text
output/candidate_pairs.tsv
```

---

# 11. Matching Model

The candidate-generation stage produces candidate pairs.

The matching model then determines whether each candidate pair represents the same real-world business.

Potential pairwise features include:

## Business Name Features

* exact normalized equality
* edit similarity
* Levenshtein-style similarity
* Jaro/Jaro-Winkler similarity
* token Jaccard similarity
* token overlap
* character n-gram similarity
* TF-IDF cosine similarity
* shared token count
* containment features

## Address Features

* exact normalized equality
* edit similarity
* token Jaccard
* token overlap
* character n-gram similarity
* TF-IDF cosine similarity
* digit overlap
* postal/PIN similarity where extractable
* address component similarity where reliably extractable

## Cross-Field Features

* country equality
* combined name/address similarity
* name × address interactions
* source-pair information

Possible supervised models include:

* logistic regression
* tree-based classifiers
* gradient boosting models
* other appropriately licensed ML models

The final selected model must comply with the challenge's licensing and parameter constraints.

---

# 12. Validation

Because the test set has no labels, local validation must be performed using the training data.

Validation must be performed at the **Source 1 entity level**.

Do not randomly split individual candidate pairs.

For example:

```text
Training Source 1 entities
        ↓
     80%

Validation Source 1 entities
        ↓
     20%
```

All candidate pairs associated with a validation Source 1 entity must remain in validation.

This prevents leakage between training and validation.

---

# 13. Primary Evaluation Metric

The official metric is:

```text
F_0.5
```

Formula:

```text
F_0.5 =
(1.25 × Precision × Recall)
/
(0.25 × Precision + Recall)
```

The metric is calculated per Source 1 entity and then macro-averaged across Source 1 entities.

---

# 14. Precision-Heavy Evaluation

F_0.5 weights precision more strongly than recall.

False merges are therefore particularly harmful.

The system should not blindly maximize the number of predicted matches.

It should balance:

* identifying true matches
* avoiding incorrect merges
* correctly identifying singleton entities

---

# 15. Singleton Evaluation

Singletons are Source 1 entities with no true matching Source 2 or Source 3 records.

If:

```text
true matches = {}
predicted matches = {}
```

the entity receives:

```text
F_0.5 = 1.0
```

If:

```text
true matches = {}
predicted matches = {some record}
```

the prediction is incorrect and receives:

```text
F_0.5 = 0.0
```

Correctly identifying singletons is therefore important.

The matching system must be allowed to output an empty match list.

It must never be forced to select the highest-scoring candidate.

---

# 16. Threshold Selection

The matching model may produce a probability or confidence score.

Do not assume a threshold of `0.5` is optimal.

The threshold should be selected using the training validation set.

Possible thresholds can be evaluated, for example:

```text
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
```

The threshold must be selected using validation data only.

Do not use test-set labels because they are unavailable.

Do not tune the final threshold against hidden leaderboard results as part of the reproducible pipeline.

---

# 17. One-to-Many Matching

The final system must not simply select the single highest-scoring candidate for each Source 1 entity.

Multiple candidates may be correct.

The decision system should evaluate candidates independently or using an appropriately designed multi-match strategy.

The final predicted set can contain:

```text
S2 records
+
S3 records
```

for the same Source 1 entity.

---

# 18. Output Requirements

The final submission must contain two files.

---

## 18.1 matching_results.tsv

Location:

```text
output/matching_results.tsv
```

Columns:

```text
source1_entity_id
matched_entity_ids
```

Example:

```text
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

Requirements:

1. Every Source 1 test entity must have exactly one row.
2. `matched_entity_ids` may be empty.
3. Only S2 and S3 IDs may appear.
4. IDs must exist in the test Source 2/Source 3 files.
5. No duplicate IDs within one list.
6. No Source 1 IDs may appear as matches.
7. The file must be tab-separated.
8. Commas separate IDs inside `matched_entity_ids`.

---

# 19. candidate_pairs.tsv

Location:

```text
output/candidate_pairs.tsv
```

Columns:

```text
source1_entity_id
candidate_entity_ids
```

Example:

```text
source1_entity_id    candidate_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812,S3-00999
S1-00002             S3-00004
S1-00003
```

This file represents the **final candidate set passed to the matching model**.

It is not merely an early blocking output.

If the pipeline has:

```text
blocking stage 1
    ↓
filter
    ↓
blocking stage 2
    ↓
filter
    ↓
matching model
```

then
