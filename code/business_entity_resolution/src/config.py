"""
Centralized configuration for the Entity Resolution baseline pipeline.
All paths, constants, and hyperparameters live here.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (business-entity-resolution/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

MATCHING_OUTPUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"

VALIDATOR_SCRIPT = PROJECT_ROOT / "utils" / "validate_submission.py"

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
CHUNK_SIZE = 100_000        # rows per chunk when reading large TSVs
RANDOM_SEED = 42
VAL_FRACTION = 0.20         # fraction of S1 entities held out for validation

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------
BLOCKING_TOP_K_TOKENS = 3   # min shared informative tokens for token blocking
MIN_TOKEN_LENGTH = 2        # ignore single-char tokens in token blocking
MAX_CANDIDATES_PER_S1 = 200 # safety cap per S1 entity

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_N_ESTIMATORS = 200
MODEL_MAX_DEPTH = 5
MODEL_LEARNING_RATE = 0.1
MODEL_MIN_SAMPLES_LEAF = 50

# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.05

# ---------------------------------------------------------------------------
# Development subset (set to None for full dataset)
# ---------------------------------------------------------------------------
DEV_S1_COUNT = 5_000       # number of S1 entities for dev runs; None = full

# ---------------------------------------------------------------------------
# Feature names (for consistent ordering)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "name_exact",
    "name_edit_sim",
    "name_token_jaccard",
    "name_token_overlap",
    "name_len_ratio",
    "name_char_ngram_sim",
    "addr_exact",
    "addr_edit_sim",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_missing",
    "addr_digit_overlap",
    "country_match",
    "source_is_s3",
]
