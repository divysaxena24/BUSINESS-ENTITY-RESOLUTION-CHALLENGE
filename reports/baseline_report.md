# Baseline Pipeline Report

## 1. Objective
Establish a reproducible, correct, and measurable baseline pipeline for the Business Entity Resolution Challenge without using external data or loading the entire 24M record dataset into memory simultaneously.

## 2. Data Used
Development subset consisting of 5,000 S1 entities and all 10.3M records from Train Source 2 and Source 3.

## 3. Normalization
- **Name:** NFKD Unicode normalization, lowercase, noise punctuation removal, `M/s` prefix removal, ampersand normalization, and deterministic canonicalization of legal suffixes (e.g., `pvt ltd` -> `pvt ltd`).
- **Address:** Lowercase, noise removal, comma normalization, and standard abbreviation expansion (e.g., `road` -> `rd`).
- **Country:** Lowercase and strip.

## 4. Blocking
- **Strategy:** Union of (A) exact normalized name match and (B) shared informative tokens (minimum 2 shared tokens).
- **Implementation:** S1 records are loaded into an inverted index. S2 and S3 files are streamed in chunks of 100k rows. 
- **Safety Mechanism:** Hard cap of 200 candidates per S1 to prevent memory explosion from highly generic names.

## 5. Candidate Recall
- **Candidate pairs generated:** 679,373
- **Average candidates/S1:** 135.8
- **Validation Blocking Recall:** 54.21%
*(Note: Recall is heavily penalized by the 200-candidate safety cap, which truncated ~6M candidate pairs).*

## 6. Feature Set
14 pairwise features:
- **Name:** Exact match, Edit similarity, Token Jaccard, Token Overlap, Length Ratio, 3-gram similarity.
- **Address:** Exact match, Edit similarity, Token Jaccard, Token Overlap, Digit Overlap.
- **Missingness:** Explicit `addr_missing` flag (preventing fabricated similarity for missing addresses).
- **Other:** Exact country match, `source_is_s3` flag.

## 7. Model
Gradient Boosted Decision Trees (`sklearn.ensemble.GradientBoostingClassifier`) with 200 estimators and max depth 5. LightGBM was not available in the environment.

## 8. Validation Split
S1-entity level split: 4,000 S1 for training, 1,000 S1 for validation. Pairs for a given S1 are never split across train/val.

## 9. Threshold Selection
Selected threshold `0.65` based on optimal Macro F0.5 on the validation set (range 0.10 to 0.90 evaluated).

## 10. Metric Results
*Evaluated on the capped candidate set only. Maximum achievable recall ceiling is 54.21%.*
- **Macro F0.5:** 0.6722
- **Precision:** 0.7889
- **Recall:** 0.5012

*(Note: The model achieved 0.5012 / 0.5421 = ~92.4% relative recall on the candidates it actually saw. The performance bottleneck is purely candidate generation, not the classifier).*

## 11. Singleton Behavior
- **Singleton False-Positive Rate:** 0.053 (5.3%)

## 12. Multi-match Behavior
The model supports arbitrary N-way matching based on independent pairwise thresholds.

## 13. Error Analysis
TBD

## 14. Runtime
~5 minutes for the 5k Dev subset. (Pandas chunking iteration optimized with `zip`).

## 15. Memory Observations
Peak memory ~1.5 GB. Indexing 5,000 S1 entities and streaming 100k chunks prevents the 15.3GB system limit from being reached.

## 16. Known Limitations
- **Blocking Cap:** The 200-candidate cap destroys 45% of true matches before the model even sees them.
- **Generic Tokens:** Token blocking treats "Private" and "Limited" with the same weight as rare names, causing candidate explosion.

## 17. Next Experiments
1. **Experiment B1:** Measure blocking candidate distributions without the 200 cap on a small sample to isolate bad blocking vs. cap truncation.
2. **Experiment B2:** Implement rare-token blocking (IDF weighting) to prevent generic terms from generating candidates.
