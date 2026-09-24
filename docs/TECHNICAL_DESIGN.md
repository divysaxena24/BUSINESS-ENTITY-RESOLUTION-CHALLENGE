# Technical Design

## Business Entity Resolution Pipeline

## 1. Architecture

```text
                 ┌────────────────────┐
                 │   TSV Data Loader  │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Schema Validation  │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Normalization      │
                 │ Name + Address     │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Candidate Blocking │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Pair Features      │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ ML Matcher         │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Decision Layer     │
                 └─────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
       matching_results.tsv   candidate_pairs.tsv
```

---

# 2. Data Layer

Use pandas for initial data loading.

All TSV files must use:

sep="\t"

Validate:

* columns
* nulls
* duplicate IDs
* source prefixes
* ground-truth references

---

# 3. Normalization Layer

Maintain:

original_business_name
normalized_business_name

original_business_address
normalized_business_address

original_country
normalized_country

Normalization should be deterministic.

Recommended operations:

* Unicode normalization
* lowercase
* whitespace cleanup
* punctuation normalization
* ampersand normalization
* safe abbreviation handling
* legal suffix normalization

Do not remove meaningful numeric/address information.

---

# 4. Blocking Layer

Candidate generation should be a union of multiple blocking methods.

Potential blocks:

## Block A — Exact normalized name

Records sharing the same normalized name.

## Block B — Strong name token

Records sharing informative name tokens.

## Block C — Address token

Records sharing informative address tokens.

## Block D — Character n-gram retrieval

Retrieve records with high character n-gram similarity.

## Block E — TF-IDF retrieval

Retrieve top-k approximate nearest records.

Candidates from all blocks are deduplicated.

---

# 5. Pair Feature Layer

Generate numerical features.

### Name features

* exact_match
* edit_similarity
* token_jaccard
* token_overlap
* tfidf_cosine
* shared_token_count
* character_ngram_similarity

### Address features

* exact_match
* edit_similarity
* token_jaccard
* token_overlap
* tfidf_cosine
* shared_digit_count
* postal_similarity

### Cross-field features

* country_match
* combined_similarity
* name_address_interaction

### Metadata

* source_pair

---

# 6. Modeling Layer

Start with:

1. Logistic Regression
2. Tree-based classifier

Compare models on the same validation split.

Use the same candidate generation strategy when comparing models.

Do not change blocking and model simultaneously without recording the experiment.

---

# 7. Validation Layer

Split Source 1 entities.

Example:

```text
80% Source 1 → training
20% Source 1 → validation
```

Generate candidate pairs separately for each partition.

Calculate predictions.

Aggregate predictions by Source 1 entity.

Calculate macro F_0.5.

---

# 8. Decision Layer

Candidate score:

```text
P(match | pair)
```

Prediction:

```text
match if probability >= selected_threshold
```

Threshold is selected using validation data.

Optionally evaluate:

```text
threshold + confidence margin
```

but only retain this if it improves validation F_0.5.

---

# 9. Output Layer

Create:

matching_results.tsv

with:

```text
source1_entity_id
matched_entity_ids
```

Create:

candidate_pairs.tsv

with:

```text
source1_entity_id
candidate_entity_ids
```

IDs inside lists are comma-separated.

Files themselves are tab-separated.

---

# 10. Reproducibility

Set deterministic seeds where supported.

Save:

* model parameters
* normalization configuration
* blocking configuration
* threshold
* feature configuration

Document all versions.

---

# 11. Computational Considerations

Avoid constructing the full Cartesian product if datasets are large.

Use:

* inverted token indexes
* TF-IDF sparse matrices
* nearest-neighbor retrieval
* batched pair feature computation

Do not materialize billions of pairs.

---

# 12. Error Handling

The pipeline should fail clearly for:

* missing files
* missing columns
* malformed TSV
* duplicate IDs
* invalid ground-truth references

Never silently continue with corrupted data.

---

# 13. Logging

Log:

* number of records
* candidate count
* average candidates per S1
* candidate recall on validation
* feature dimensions
* model training size
* validation metrics
* selected threshold
* final prediction count

---

# 14. Security / Fair Play

No network calls should be required by the production pipeline.

External lookup is forbidden by the competition.

The production pipeline should be able to run offline.
