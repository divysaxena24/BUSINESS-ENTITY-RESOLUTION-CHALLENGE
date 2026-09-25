"""
Pairwise feature engineering for entity resolution candidate pairs.

All features are computed on already-blocked candidate pairs.
Missing addresses produce explicit missingness indicators, never fabricated similarity.
"""
from difflib import SequenceMatcher
from .normalization import name_tokens, address_tokens, extract_digits, char_ngrams


def _edit_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio (0-1). Handles empty strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_jaccard(tokens_a: set, tokens_b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _token_overlap_count(tokens_a: set, tokens_b: set) -> int:
    """Number of shared tokens."""
    return len(tokens_a & tokens_b)


def _len_ratio(a: str, b: str) -> float:
    """Length ratio (shorter / longer). 1.0 for equal lengths."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _ngram_similarity(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    ng_a = char_ngrams(a, n)
    ng_b = char_ngrams(b, n)
    if not ng_a and not ng_b:
        return 1.0
    if not ng_a or not ng_b:
        return 0.0
    return len(ng_a & ng_b) / len(ng_a | ng_b)


def _digit_overlap(a: str, b: str) -> float:
    """Overlap of digit sequences between two strings."""
    digits_a = set(extract_digits(a).split())
    digits_b = set(extract_digits(b).split())
    if not digits_a and not digits_b:
        return 1.0
    if not digits_a or not digits_b:
        return 0.0
    return len(digits_a & digits_b) / len(digits_a | digits_b)


def compute_pair_features(
    s1_norm_name: str,
    s1_norm_addr: str,
    s1_norm_country: str,
    sx_norm_name: str,
    sx_norm_addr: str,
    sx_norm_country: str,
    source_is_s3: bool,
) -> list:
    """
    Compute feature vector for a single (S1, SX) candidate pair.

    Returns a list of floats in the order defined by config.FEATURE_NAMES:
        name_exact, name_edit_sim, name_token_jaccard, name_token_overlap,
        name_len_ratio, name_char_ngram_sim,
        addr_exact, addr_edit_sim, addr_token_jaccard, addr_token_overlap,
        addr_missing, addr_digit_overlap,
        country_match, source_is_s3
    """
    # --- Name features ---
    name_exact = 1.0 if (s1_norm_name == sx_norm_name and s1_norm_name) else 0.0
    name_edit_sim = _edit_similarity(s1_norm_name, sx_norm_name)

    s1_name_tok = name_tokens(s1_norm_name)
    sx_name_tok = name_tokens(sx_norm_name)
    name_tok_jaccard = _token_jaccard(s1_name_tok, sx_name_tok)
    name_tok_overlap = float(_token_overlap_count(s1_name_tok, sx_name_tok))
    name_len_rat = _len_ratio(s1_norm_name, sx_norm_name)
    name_ngram = _ngram_similarity(s1_norm_name, sx_norm_name, 3)

    # --- Address features ---
    addr_missing = 1.0 if (not sx_norm_addr) else 0.0

    if sx_norm_addr and s1_norm_addr:
        addr_exact = 1.0 if s1_norm_addr == sx_norm_addr else 0.0
        addr_edit_sim = _edit_similarity(s1_norm_addr, sx_norm_addr)

        s1_addr_tok = address_tokens(s1_norm_addr)
        sx_addr_tok = address_tokens(sx_norm_addr)
        addr_tok_jaccard = _token_jaccard(s1_addr_tok, sx_addr_tok)
        addr_tok_overlap = float(_token_overlap_count(s1_addr_tok, sx_addr_tok))
        addr_digit_ovlp = _digit_overlap(s1_norm_addr, sx_norm_addr)
    else:
        # Missing address: explicit 0 / indicator
        addr_exact = 0.0
        addr_edit_sim = 0.0
        addr_tok_jaccard = 0.0
        addr_tok_overlap = 0.0
        addr_digit_ovlp = 0.0

    # --- Country ---
    country_match = 1.0 if (s1_norm_country == sx_norm_country and s1_norm_country) else 0.0

    # --- Source ---
    src_s3 = 1.0 if source_is_s3 else 0.0

    return [
        name_exact,
        name_edit_sim,
        name_tok_jaccard,
        name_tok_overlap,
        name_len_rat,
        name_ngram,
        addr_exact,
        addr_edit_sim,
        addr_tok_jaccard,
        addr_tok_overlap,
        addr_missing,
        addr_digit_ovlp,
        country_match,
        src_s3,
    ]
