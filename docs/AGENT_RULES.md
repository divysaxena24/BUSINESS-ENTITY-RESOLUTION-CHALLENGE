# Antigravity Agent Rules

## Rule 1 — Inspect Before Implementing

Never assume dataset characteristics.

Inspect actual files first.

---

## Rule 2 — Never Invent Columns

The documented source schema is:

```text
entity_id
business_name
business_address
country
```

Do not create assumed fields such as:

```text
city
state
zip
phone
email
website
latitude
longitude
industry
```

unless they are actually present or explicitly extracted from existing fields.

---

## Rule 3 — Never Use External Business Data

No:

* Google
* Google Maps
* OpenStreetMap
* government registry
* business directory
* company database
* geocoding API
* entity-resolution API

The pipeline must be offline.

---

## Rule 4 — Do Not Optimize Against Test Labels

Test ground truth is unavailable.

Never infer hidden labels.

Never use leaderboard results as training labels.

---

## Rule 5 — Preserve Validation Integrity

Validation Source 1 entities must not leak into training.

Do not randomly split candidate pairs.

---

## Rule 6 — Never Force Matches

A Source 1 entity can legitimately have zero matches.

Empty prediction is valid.

---

## Rule 7 — Support Multiple Matches

Do not assume one-to-one matching.

One Source 1 entity can match multiple Source 2/3 records.

---

## Rule 8 — Country Is Open Set

Never hard-code US and India.

France must work.

---

## Rule 9 — Candidate Set Must Be Auditable

Every final match must occur in candidate_pairs.tsv.

---

## Rule 10 — Do Not Manually Patch Final Output

If output validation fails, fix the pipeline.

Do not hand-edit prediction files.

---

## Rule 11 — Every Change Requires Validation

After changing:

* normalization
* blocking
* features
* model
* threshold

rerun validation.

---

## Rule 12 — Do Not Claim Improvement Without Measurement

Never say:

"this improves performance"

unless validation metrics demonstrate it.

Use:

"this is hypothesized to improve performance"

until measured.

---

## Rule 13 — Prefer Evidence Over Complexity

A simpler model with higher validation F_0.5 is preferable to a complex model with lower validation F_0.5.

---

## Rule 14 — Record Experiments

Every meaningful experiment must be logged.

---

## Rule 15 — Respect Competition Licensing

Any pretrained model must satisfy:

* maximum 8B parameters
* MIT or Apache 2.0 license requirement

When license status cannot be verified locally, do not use the model.

---

## Rule 16 — No Hidden Network Dependencies

The final pipeline must work without network access.

---

## Rule 17 — Final Validator Is Mandatory

Never declare the project complete until:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

returns PASS.

---

## Rule 18 — Reproducibility

Another person should be able to clone/copy the project, install requirements, run the documented command, and regenerate the outputs.

---

## Rule 19 — Explain Major Decisions

For every major architectural choice, document:

* what was chosen
* why
* alternatives
* validation evidence

---

## Rule 20 — Stop and Report Unknowns

If something cannot be determined from:

* challenge specification
* project documents
* actual dataset
* official validator

do not hallucinate.

Explicitly report the uncertainty.
