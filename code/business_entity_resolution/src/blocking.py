"""
Blocking / candidate generation for entity resolution.

Strategy:
  Block A — exact normalized name match
  Block B — shared informative name tokens (via inverted index)

Both blocks are combined (union) to maximize recall.
Candidates are produced by processing S2/S3 in chunks against an S1 index.
"""
import gc
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional

import pandas as pd

from . import config
from .normalization import normalize_name, normalize_address, normalize_country, name_tokens


def p(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# S1 record storage (compact)
# ---------------------------------------------------------------------------
# s1_data[entity_id] = (norm_name, norm_addr, norm_country)
S1Record = Tuple[str, str, str]


def load_s1_index(
    s1_path,
    dev_ids: Optional[Set[str]] = None,
) -> Tuple[Dict[str, S1Record], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Load Source 1, normalize, and build inverted indexes for blocking.

    Returns:
        s1_data:       {entity_id -> (norm_name, norm_addr, norm_country)}
        name_index:    {normalized_name -> set of S1 entity_ids}
        token_index:   {token -> set of S1 entity_ids}
    """
    p(f"  Loading S1 from {s1_path.name} ...")
    s1_data: Dict[str, S1Record] = {}
    name_index: Dict[str, Set[str]] = defaultdict(set)
    token_index: Dict[str, Set[str]] = defaultdict(set)

    total = 0
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str,
                             keep_default_na=False, chunksize=config.CHUNK_SIZE):
        for eid, b_name, b_addr, ctry in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]):
            if dev_ids is not None and eid not in dev_ids:
                continue

            norm_n = normalize_name(b_name)
            norm_a = normalize_address(b_addr)
            norm_c = normalize_country(ctry)

            s1_data[eid] = (norm_n, norm_a, norm_c)

            # Block A index: exact normalized name
            if norm_n:
                name_index[norm_n].add(eid)

            # Block B index: informative name tokens
            for tok in name_tokens(norm_n):
                token_index[tok].add(eid)

            total += 1

        del chunk
        gc.collect()

    p(f"    Loaded {total:,} S1 records.")
    p(f"    Name index keys: {len(name_index):,}")
    p(f"    Token index keys: {len(token_index):,}")
    return s1_data, dict(name_index), dict(token_index)


def _find_candidates_for_record(
    norm_name: str,
    tokens: set,
    name_index: Dict[str, Set[str]],
    token_index: Dict[str, Set[str]],
    min_shared_tokens: int = 2,
) -> Set[str]:
    """Find S1 candidates for a single S2/S3 record."""
    candidates = set()

    # Block A: exact normalized name
    if norm_name in name_index:
        candidates.update(name_index[norm_name])

    # Block B: shared informative tokens
    if len(tokens) >= min_shared_tokens:
        # Count how many tokens each S1 shares
        s1_token_counts: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            if tok in token_index:
                for s1_id in token_index[tok]:
                    s1_token_counts[s1_id] += 1
        # Accept S1s sharing >= min_shared_tokens
        for s1_id, count in s1_token_counts.items():
            if count >= min_shared_tokens:
                candidates.add(s1_id)
    elif len(tokens) == 1:
        # For single-token names, only match via exact name
        pass

    return candidates


def generate_candidates_chunked(
    source_path,
    s1_data: Dict[str, S1Record],
    name_index: Dict[str, Set[str]],
    token_index: Dict[str, Set[str]],
    source_label: str,
    min_shared_tokens: int = 2,
    max_per_s1: int = config.MAX_CANDIDATES_PER_S1,
) -> Tuple[list, Dict[str, Set[str]]]:
    """
    Process a Source 2 or Source 3 file in chunks.
    For each record, find matching S1 candidates via blocking.

    Returns:
        candidate_pairs: list of (s1_id, sx_id, s1_record, sx_norm_name, sx_norm_addr, sx_norm_country)
        s1_candidates:   {s1_id -> set of candidate sx_ids}
    """
    p(f"  Generating candidates from {source_path.name} ...")

    candidate_pairs = []
    s1_candidates: Dict[str, Set[str]] = defaultdict(set)
    total_records = 0
    total_pairs = 0
    skipped_cap = 0

    for chunk in pd.read_csv(source_path, sep="\t", dtype=str,
                             keep_default_na=False, chunksize=config.CHUNK_SIZE):
        for sx_id, b_name, b_addr, ctry in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]):
            norm_n = normalize_name(b_name)
            norm_a = normalize_address(b_addr)
            norm_c = normalize_country(ctry)
            tokens = name_tokens(norm_n)

            total_records += 1

            matched_s1 = _find_candidates_for_record(
                norm_n, tokens, name_index, token_index, min_shared_tokens
            )

            for s1_id in matched_s1:
                # Safety cap: don't let one S1 accumulate too many candidates
                if len(s1_candidates[s1_id]) >= max_per_s1:
                    skipped_cap += 1
                    continue
                s1_candidates[s1_id].add(sx_id)
                candidate_pairs.append((
                    s1_id, sx_id,
                    norm_n, norm_a, norm_c,
                ))
                total_pairs += 1

        if total_records % 500_000 == 0:
            p(f"    Processed {total_records:,} {source_label} records, "
              f"{total_pairs:,} candidate pairs so far ...")

        del chunk
        gc.collect()

    p(f"    Done: {total_records:,} {source_label} records -> "
      f"{total_pairs:,} candidate pairs")
    if skipped_cap:
        p(f"    Capped {skipped_cap:,} pairs due to max_per_s1={max_per_s1}")

    return candidate_pairs, dict(s1_candidates)
