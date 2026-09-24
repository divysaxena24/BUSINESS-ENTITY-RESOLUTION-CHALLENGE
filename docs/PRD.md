# Product Requirements Document

## Business Entity Resolution ML System

## 1. Product Objective

Build a machine-learning-based Entity Resolution system that identifies which Source 2 and Source 3 business records correspond to each Source 1 reference business.

The system must handle noisy, incomplete, inconsistent, abbreviated, misspelled, transliterated, and partially matching business names and addresses.

The system must optimize for the competition's macro F_0.5 metric, which emphasizes precision more heavily than recall.

---

## 2. Users

The primary user is the competition participant who needs to:

* train an Entity Resolution model
* evaluate it locally
* generate test predictions
* validate submission files
* package the complete reproducible solution

---

## 3. Inputs

### Training

* train_source1.tsv
* train_source2.tsv
* train_source3.tsv
* train_ground_truth.tsv

### Test

* test_source1.tsv
* test_source2.tsv
* test_source3.tsv

---

## 4. Input Schema

Business source files:

* entity_id
* business_name
* business_address
* country

Ground truth:

* source1_entity_id
* matched_entity_ids

---

## 5. Core Functional Requirements

### FR-01 — Data Loading

System shall load all TSV files using an explicit tab separator.

### FR-02 — Data Validation

System shall verify expected columns, unique IDs, source prefixes, and ground-truth references.

### FR-03 — Normalization

System shall create normalized versions of business names, addresses, and country values without destroying original fields.

### FR-04 — Candidate Generation

System shall generate plausible S1-S2 and S1-S3 candidate pairs.

### FR-05 — Pair Scoring

System shall calculate similarity features for every candidate pair.

### FR-06 — ML Classification

System shall train a model to classify candidate pairs as match/non-match.

### FR-07 — Singleton Detection

System shall permit an S1 entity to have zero matches.

### FR-08 — One-to-Many Matching

System shall permit multiple S2/S3 matches for one S1 entity.

### FR-09 — Validation

System shall calculate macro F_0.5 on a held-out Source 1 validation set.

### FR-10 — Test Inference

System shall generate predictions for every Source 1 test entity.

### FR-11 — Submission Generation

System shall generate:

output/matching_results.tsv
output/candidate_pairs.tsv

### FR-12 — Submission Validation

System shall run the official validator before declaring the submission complete.

---

## 6. Non-Functional Requirements

### Reproducibility

The complete pipeline must be reproducible.

### Performance

Candidate generation must substantially reduce pair count while maintaining high recall.

### Auditability

Model decisions must be explainable through pairwise features and scores.

### Portability

The pipeline should run from a clean Python environment using the documented requirements.

### Fair Play

No external business data may be used.

---

## 7. Success Metrics

Primary:

Macro F_0.5

Secondary:

* macro precision
* macro recall
* singleton accuracy
* candidate recall
* reduction ratio
* runtime
* candidate count per S1

---

## 8. Critical Business Rules

1. Every Source 1 test ID must appear exactly once.
2. Matching IDs may only come from Source 2 and Source 3 test files.
3. No duplicate matching IDs.
4. Empty match lists are valid.
5. Final matches must be candidates.
6. France and other unseen countries must not be rejected.
7. No external lookup is allowed.

---

## 9. Out of Scope

The system will not:

* perform web searches
* query business registries
* geocode addresses
* retrieve external company information
* use external identity databases
* manually resolve individual businesses

---

## 10. Expected User Flow

1. Place datasets in expected folders.
2. Run data audit.
3. Train baseline.
4. Evaluate validation.
5. Review errors.
6. Improve pipeline.
7. Freeze configuration.
8. Run test inference.
9. Validate submission.
10. Package repository.
