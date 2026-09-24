# BUSINESS ENTITY RESOLUTION CHALLENGE

## Master Engineering Prompt for Antigravity

You are the primary ML engineer responsible for designing, implementing, validating, documenting, and packaging a complete Entity Resolution solution for the Business Entity Resolution Challenge.

Your job is NOT to merely write a quick matching script.

You must build a reproducible, leakage-safe, precision-oriented ML pipeline that:

1. Reads the provided TSV datasets correctly.
2. Understands the actual dataset before making assumptions.
3. Normalizes noisy business names and addresses.
4. Generates high-recall candidate pairs using blocking.
5. Creates informative pairwise similarity features.
6. Trains a supervised matching model using the provided training ground truth.
7. Validates the complete pipeline using a source1-level validation split.
8. Optimizes for the competition's macro F_0.5 metric.
9. Handles zero-match Source 1 entities correctly.
10. Supports one-to-many relationships.
11. Supports unseen country values such as France.
12. Produces exactly:

    * output/matching_results.tsv
    * output/candidate_pairs.tsv
13. Runs the official submission validator.
14. Produces a reproducible submission package.
15. Documents all methodology and experiments.

---

# 1. AUTHORITATIVE PROJECT INFORMATION

The following files are authoritative:

challenge/challenge_statement.md
docs/PRD.md
docs/TECHNICAL_DESIGN.md
docs/MODELING_PLAN.md
docs/EXPERIMENT_PLAN.md
docs/AGENT_RULES.md

The actual TSV files are authoritative for the data schema and observed data distributions.

If there is any disagreement between your assumptions and the actual dataset, inspect the actual dataset and follow the actual dataset/schema unless the challenge specification explicitly says otherwise.

DO NOT invent columns.

DO NOT invent ground truth.

DO NOT invent business information.

DO NOT use external business databases.

DO NOT perform web searches for business identity resolution.

DO NOT geocode addresses.

DO NOT use Google Maps, OpenStreetMap, government registries, business directories, commercial ER APIs, or external enrichment.

The solution must operate only on the provided challenge data and locally installed/open-source libraries.

---

# 2. FIRST TASK: DATA AUDIT

Before writing the final ML pipeline, inspect all datasets.

Do not immediately train a model.

Create a data profiling report containing:

* number of rows per file
* column names
* dtypes
* missing-value counts
* duplicate counts
* unique-value counts
* country distribution
* business-name length statistics
* address length statistics
* examples of noisy names
* examples of noisy addresses
* source distributions
* ground-truth match-count distribution
* percentage of Source 1 entities with zero matches
* number of S1→S2 matches
* number of S1→S3 matches
* number of one-to-many entities
* maximum number of matches for a Source 1 entity
* whether IDs are unique
* whether all ground-truth IDs exist in their corresponding source files

Do not expose potentially huge raw datasets in logs.

Save the report under:

reports/data_profile.md

Also save useful machine-readable statistics if appropriate.

---

# 3. IMPORTANT TSV REQUIREMENT

All input and output files are tab-separated.

Always explicitly use:

sep="\t"

Never rely on pandas' default CSV separator.

Preserve commas inside the matched_entity_ids and candidate_entity_ids fields.

---

# 4. DATA MODEL

Each source contains:

entity_id
business_name
business_address
country

Source is determined from:

S1-
S2-
S3-

Source 1 is the reference/master source.

For every Source 1 entity, the system must find:

* zero matching Source 2 entities
* one matching Source 2 entity
* many matching Source 2 entities
* zero matching Source 3 entities
* one matching Source 3 entity
* many matching Source 3 entities

The final system must NOT force a Source 1 entity to have a match.

---

# 5. GROUND TRUTH

train_ground_truth.tsv contains:

source1_entity_id
matched_entity_ids

matched_entity_ids is a comma-separated list.

An empty list means the Source 1 entity has no known matching records.

Parse this carefully.

Never interpret an empty string as a missing training example.

It means a genuine singleton/no-match entity.

---

# 6. REQUIRED ARCHITECTURE

Implement the system as a pipeline:

DATA LOADING
↓
DATA VALIDATION
↓
TEXT NORMALIZATION
↓
BLOCKING / CANDIDATE GENERATION
↓
PAIR FEATURE ENGINEERING
↓
SUPERVISED MATCHING MODEL
↓
CALIBRATION / SCORE ANALYSIS
↓
DECISION / THRESHOLDING
↓
POST-PROCESSING
↓
OUTPUT GENERATION
↓
SUBMISSION VALIDATION

The architecture must keep candidate generation separate from final matching.

---

# 7. NORMALIZATION

Implement conservative normalization functions for business names and addresses.

Maintain both:

Original field
Normalized field

Do not overwrite original data.

At minimum implement:

normalize_name()
normalize_address()
normalize_country()

Potential normalization operations may include:

* lowercase
* Unicode normalization
* whitespace normalization
* punctuation normalization
* ampersand normalization
* repeated whitespace removal
* safe abbreviation normalization
* legal suffix normalization where justified
* transliteration-aware normalization where safely possible

Do not aggressively delete information.

Do not assume every abbreviation means the same thing.

Every normalization rule should be documented.

If you introduce a normalization dictionary, keep it inside the project and document its purpose.

---

# 8. COUNTRY HANDLING

Country is an open-set string field.

Training may contain US and India while test may contain France.

NEVER hard-code:

["US", "India"]

NEVER discard France or any unseen country.

Use generic string comparison/features.

For example:

country_exact_match

country_normalized_match

Unseen countries must work without retraining a categorical encoder that rejects unseen labels.

---

# 9. CANDIDATE GENERATION

Candidate generation must prioritize recall.

A true match that is removed during blocking can never be recovered by the matching model.

Therefore implement multiple blocking strategies where useful.

Possible blocking mechanisms include:

1. exact normalized name
2. name token overlap
3. shared rare name token
4. address token overlap
5. postal/PIN code where safely extracted
6. city/state/address component overlap
7. character n-gram retrieval
8. TF-IDF nearest-neighbor retrieval

Do not assume all of these will improve performance.

Experiment with them.

The final candidate set must be the exact candidate set sent to the final matching model.

That exact set must be written to:

output/candidate_pairs.tsv

---

# 10. BLOCKING SAFETY

Blocking should not be unnecessarily restrictive.

Track:

* candidate count per S1
* average candidates per S1
* median candidates
* p95 candidates
* maximum candidates
* candidate generation time
* validation true-match recall after blocking
* reduction ratio

For validation, calculate:

blocking_recall =
number of ground-truth match pairs present in candidates
/
number of ground-truth match pairs

The candidate stage should aim for extremely high recall while remaining computationally manageable.

---

# 11. TRAINING PAIR GENERATION

Generate positive and negative pair examples.

Positive pair:

A candidate pair that appears in train_ground_truth.

Negative pair:

A candidate pair that does not appear in train_ground_truth.

Avoid random negative sampling only.

Where practical, include hard negatives:

* similar business names
* similar addresses
* same country
* shared tokens
* similar names but different addresses
* similar addresses but different names

Hard negatives are important because the main challenge is distinguishing similar but different businesses.

---

# 12. DATA LEAKAGE PREVENTION

This is critical.

Validation must be split by Source 1 entity, not by individual candidate pair.

Never allow the same Source 1 entity to contribute pairs to both training and validation.

Any learned transformation that uses the training labels must be fit only on the training portion.

Do not fit a supervised model on the validation data.

Do not tune thresholds using the final test set.

Do not inspect or infer hidden test ground truth.

---

# 13. FEATURE ENGINEERING

For every S1-candidate pair, calculate features for:

## Name

* exact normalized equality
* character length difference
* normalized edit similarity
* Jaro/Jaro-Winkler if available
* token Jaccard
* token overlap
* character n-gram similarity
* TF-IDF cosine similarity
* containment features
* shared-token count
* rare-token overlap where available

## Address

* exact normalized equality
* character length difference
* edit similarity
* token Jaccard
* token overlap
* character n-gram similarity
* TF-IDF cosine similarity
* digit overlap
* extracted postal/PIN similarity
* number overlap
* city/state component similarity where reliably extracted

## Cross-field

* country equality
* name/address combined similarity
* name similarity × address similarity
* high-name/low-address indicators
* low-name/high-address indicators

## Source

Include source-pair information if useful:

S1-S2
S1-S3

Do not blindly assume S2 and S3 have identical noise patterns.

Validate whether source-specific behavior exists.

---

# 14. MODELING

Start with simple interpretable baselines.

At minimum evaluate:

1. rule-based similarity baseline
2. logistic regression
3. tree-based model

If using gradient boosting, ensure the selected implementation and model/license satisfy the challenge's license requirements.

Do not use an LLM merely because the challenge says ML.

The baseline must be strong, reproducible, and explainable.

Model output should be a probability or calibrated confidence score where possible.

---

# 15. F_0.5 EVALUATION

The official metric is macro-averaged F_0.5 over Source 1 entities.

Formula:

F0.5 =
1.25 × Precision × Recall
/
(0.25 × Precision + Recall)

Evaluate per Source 1 entity.

Singleton behavior is critical.

For each Source 1 entity:

predicted set = predicted matched IDs
true set = ground-truth matched IDs

Calculate entity-level precision/recall/F0.5 according to the challenge specification.

Correct empty prediction for a true singleton earns 1.0.

Incorrectly predicting any match for a true singleton earns 0.0.

Do not optimize plain accuracy.

Do not optimize only pair-level F1.

Report:

* macro F0.5
* macro precision
* macro recall
* singleton accuracy
* pair precision
* pair recall
* number of predicted matches
* number of true matches
* number of predicted singleton entities

---

# 16. THRESHOLD TUNING

Do not assume 0.5 is optimal.

Evaluate multiple thresholds on validation data.

Example:

0.30
0.35
0.40
...
0.90

Record:

threshold
precision
recall
macro F0.5
singleton performance
number of predicted matches

Select the threshold using validation macro F0.5 while respecting the precision-heavy nature of the challenge.

Do not tune using test data.

---

# 17. MULTIPLE MATCHES

Do NOT select only the highest-scoring candidate.

A Source 1 entity can legitimately match multiple records.

The model should independently score candidate pairs.

All candidates above the final decision criteria can be included.

However, avoid indiscriminately accepting weak candidates.

Explore whether a combination of:

score threshold
+
confidence margin
+
evidence requirements

improves validation F0.5.

Only keep such mechanisms if validation demonstrates improvement.

---

# 18. SINGLETON HANDLING

Explicitly support no-match predictions.

If no candidate meets the final decision rule:

matched_entity_ids = empty string

Do not force the best candidate.

Analyze the score distribution for known singleton entities during validation.

Use this analysis to reduce false positives.

---

# 19. TEST INFERENCE

Once the pipeline and hyperparameters are finalized:

1. Load all test sources.
2. Normalize.
3. Generate candidates.
4. Save exact final candidate set.
5. Run the trained model.
6. Apply frozen validation-derived decision rules.
7. Generate matching_results.tsv.
8. Ensure every test Source 1 ID appears exactly once.
9. Ensure all matched IDs belong to S2 or S3 test data.
10. Ensure every final match exists in candidate_pairs.tsv.
11. Remove duplicate IDs.
12. Preserve empty lists for singleton predictions.

Never alter the model based on test predictions after seeing leaderboard results unless explicitly conducting a new experiment and documenting it.

---

# 20. OUTPUT REQUIREMENTS

Generate:

output/matching_results.tsv

Columns exactly:

source1_entity_id
matched_entity_ids

Generate:

output/candidate_pairs.tsv

Columns exactly:

source1_entity_id
candidate_entity_ids

Both must be TSV.

No extra columns.

Every Source 1 test entity must have exactly one row.

IDs within comma-separated lists must be unique.

Final matches must be a subset of candidate IDs.

---

# 21. VALIDATION

Always execute:

python3 utils/validate_submission.py 
--matching output/matching_results.tsv 
--candidate output/candidate_pairs.tsv 
--test-dir dataset/test

Do not claim completion until the validator reports PASS.

If validation fails:

1. inspect the exact issue
2. fix the pipeline
3. regenerate outputs
4. rerun validator

Never manually patch the output files as a substitute for fixing the pipeline.

---

# 22. REPRODUCIBILITY

The entire solution must be reproducible from the repository.

Create:

code/business_entity_resolution/

or, if the current project root already represents that package, ensure all implementation lives under:

src/

Provide:

README.md
requirements.txt

The README must explain:

* environment setup
* dependencies
* training command
* validation command
* test inference command
* output locations
* expected runtime
* random seeds
* model configuration
* reproduction procedure

Pin dependency versions where practical.

---

# 23. EXPERIMENT TRACKING

Every meaningful experiment must record:

experiment ID
date
blocking strategy
feature set
model
hyperparameters
threshold
validation macro F0.5
precision
recall
singleton performance
candidate recall
candidate count
notes

Do not overwrite experiment results.

Create:

reports/experiments.csv

and/or markdown experiment reports.

---

# 24. ERROR ANALYSIS

After establishing a baseline, inspect validation errors.

Categorize false positives:

* similar business names
* shared address
* common franchise names
* legal suffix confusion
* transliteration
* landmark address
* partial address
* same city but different business
* same country but unrelated business

Categorize false negatives:

* severe typo
* abbreviation
* missing address
* missing name information
* transliteration
* reordered tokens
* partial address
* source-specific formatting

Use these findings to improve normalization, blocking, or features.

Do not blindly add rules for individual examples.

---

# 25. FAIR-PLAY CONSTRAINTS

Strictly prohibited:

* external business lookup
* web search for business identity
* commercial ER APIs
* geocoding APIs
* Google Maps
* OpenStreetMap lookup
* government registries
* external company databases
* external data augmentation

Allowed:

* local computation
* standard Python libraries
* open-source ML libraries
* string similarity algorithms
* TF-IDF
* locally trained models
* models trained only from provided challenge data
* local validation and experimentation

The pipeline must be auditable.

---

# 26. DO NOT HALLUCINATE

Never claim that a field exists unless you inspect the actual TSV.

Never assume:

* city is a separate field
* state is a separate field
* postal code is a separate field
* latitude/longitude exists
* industry exists
* phone number exists
* website exists

If these components are embedded in business_address, extraction may be attempted.

If they do not exist, do not invent them.

---

# 27. DO NOT OVERENGINEER BEFORE BASELINE

Work incrementally.

Phase 1:
Data audit.

Phase 2:
Simple normalization.

Phase 3:
Simple blocking.

Phase 4:
Simple similarity features.

Phase 5:
Baseline classifier.

Phase 6:
Validation and F0.5.

Phase 7:
Error analysis.

Phase 8:
Improve blocking.

Phase 9:
Improve features/model.

Phase 10:
Finalize and package.

After each phase, report what changed and why.

---

# 28. REQUIRED FINAL DELIVERABLES

At completion, the project must contain:

output/
matching_results.tsv
candidate_pairs.tsv

code/business_entity_resolution/
src/
README.md
requirements.txt

Documentation_template.md

reports/
data_profile.md
experiments.csv
validation_report.md
error_analysis.md

The final methodology document must explain:

* methodology
* candidate generation
* feature engineering
* model architecture
* training
* validation
* threshold selection
* singleton handling
* limitations
* reproducibility

---

# 29. HOW YOU SHOULD WORK

Do not silently make major architectural decisions.

Before implementing major components, inspect the repository and data.

For every major decision, state:

DECISION
WHY
EVIDENCE
ALTERNATIVES CONSIDERED
VALIDATION PLAN

Do not ask unnecessary questions when the repository contains enough information to answer them.

If something genuinely cannot be determined from the provided files, explicitly flag it rather than inventing an answer.

Start now with:

STEP 1 — inspect repository and files
STEP 2 — validate schema
STEP 3 — produce data profile
STEP 4 — inspect ground truth
STEP 5 — design baseline
STEP 6 — implement baseline
STEP 7 — run source1-level validation
STEP 8 — report baseline F0.5
STEP 9 — improve systematically
STEP 10 — generate and validate final test submission

Do not jump directly to final submission generation.

The goal is a scientifically evaluated entity-resolution system, not merely a script that produces two TSV files.
