#!/usr/bin/env python3
"""
Experiment B4 — Phase 1: Diagnose C_500 Missed Pairs.
Identifies what kinds of true matches are missed by B2 C_500 blocking,
and whether address evidence could recover them.
"""
import sys
import gc
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))
from src import config
from src.normalization import normalize_name, name_tokens, normalize_address, address_tokens, extract_digits
from src.pipeline import load_ground_truth

def p(msg):
    print(msg, flush=True)

def main():
    p("="*60)
    p(" B4 Phase 1: Diagnose C_500 Missed Pairs")
    p("="*60)

    # Load token DF
    df_path = config.PROJECT_ROOT / "dataset" / "token_df.json"
    p("Loading Token DF map...")
    with open(df_path, "r") as f:
        token_df = json.load(f)
    p(f"  Loaded {len(token_df):,} tokens.")

    # Dev split (identical to B1/B2)
    all_gt = load_ground_truth(config.TRAIN_GT)
    all_s1 = list(all_gt.keys())
    random.seed(config.RANDOM_SEED)
    dev_s1_ids = set(random.sample(all_s1, min(5000, len(all_s1))))
    dev_gt = {k: v for k, v in all_gt.items() if k in dev_s1_ids}
    del all_gt, all_s1; gc.collect()
    p(f"Dev S1 IDs: {len(dev_s1_ids):,}")

    total_gt_pairs = sum(len(v) for v in dev_gt.values())
    p(f"Total GT pairs: {total_gt_pairs:,}")

    # Build set of all GT sx_ids we need to look up
    gt_sx_ids = set()
    for matches in dev_gt.values():
        gt_sx_ids.update(matches)
    p(f"Unique GT S2/S3 IDs to look up: {len(gt_sx_ids):,}")

    # --- Pass 1: Load S1 data ---
    p("\nLoading S1 data...")
    s1_data = {}  # eid -> {name, addr, country, norm_name, norm_addr, name_toks, addr_toks}
    name_index = defaultdict(set)
    token_index = defaultdict(set)

    for chunk in pd.read_csv(config.TRAIN_S1, sep="\t", dtype=str, keep_default_na=False, chunksize=config.CHUNK_SIZE):
        for eid, b_name, b_addr, country in zip(
            chunk["entity_id"], chunk["business_name"],
            chunk["business_address"], chunk["country"]
        ):
            if eid not in dev_s1_ids:
                continue
            nn = normalize_name(b_name)
            na = normalize_address(b_addr)
            nt = name_tokens(nn)
            at = address_tokens(na)
            s1_data[eid] = {
                "name": b_name, "addr": b_addr, "country": country,
                "norm_name": nn, "norm_addr": na,
                "name_toks": nt, "addr_toks": at,
                "digits": extract_digits(b_addr),
            }
            if nn:
                name_index[nn].add(eid)
            for t in nt:
                token_index[t].add(eid)
    p(f"  Loaded {len(s1_data):,} S1 records.")

    # --- Pass 2: Simulate C_500 blocking AND collect SX data for GT pairs ---
    RARE_THRESH = 500
    MODERATE_THRESH = 50000

    # We need to track:
    # 1. Which GT pairs are recovered by C_500 (to find the misses)
    # 2. The SX record data for missed GT pairs (for diagnosis)
    c500_recovered = set()  # (s1_id, sx_id) tuples
    sx_data = {}  # sx_id -> {norm_name, norm_addr, ...} — only for GT sx_ids

    def process_source(source_path):
        p(f"\nProcessing {source_path.name}...")
        total = 0
        for chunk in pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False, chunksize=config.CHUNK_SIZE):
            for sx_id, b_name, b_addr, country in zip(
                chunk["entity_id"], chunk["business_name"],
                chunk["business_address"], chunk["country"]
            ):
                # Always collect data for GT sx_ids
                if sx_id in gt_sx_ids and sx_id not in sx_data:
                    nn = normalize_name(b_name)
                    na = normalize_address(b_addr)
                    sx_data[sx_id] = {
                        "name": b_name, "addr": b_addr, "country": country,
                        "norm_name": nn, "norm_addr": na,
                        "name_toks": name_tokens(nn),
                        "addr_toks": address_tokens(na),
                        "digits": extract_digits(b_addr),
                    }

                # Simulate C_500 blocking
                nn = normalize_name(b_name)
                toks = name_tokens(nn)

                # Exact name match
                if nn in name_index:
                    for s1_id in name_index[nn]:
                        if sx_id in dev_gt.get(s1_id, set()):
                            c500_recovered.add((s1_id, sx_id))

                # Token-based: 1 rare (DF<=500) OR 2 moderate (DF<=50000)
                s1_counts = defaultdict(list)
                for t in toks:
                    if t in token_index:
                        for s1_id in token_index[t]:
                            s1_counts[s1_id].append(t)

                for s1_id, shared in s1_counts.items():
                    if sx_id not in dev_gt.get(s1_id, set()):
                        continue  # only care about GT pairs for recovery tracking
                    df_vals = [token_df.get(t, 0) for t in shared]
                    min_df = min(df_vals)
                    second_min_df = sorted(df_vals)[1] if len(df_vals) > 1 else float('inf')
                    if min_df <= RARE_THRESH or second_min_df <= MODERATE_THRESH:
                        c500_recovered.add((s1_id, sx_id))

            total += len(chunk)
            if total % 1_000_000 == 0:
                p(f"  {total:,} rows...")
            del chunk; gc.collect()

    process_source(config.TRAIN_S2)
    process_source(config.TRAIN_S3)

    p(f"\nC_500 recovered: {len(c500_recovered):,}")
    recall = len(c500_recovered) / total_gt_pairs
    p(f"C_500 recall: {recall:.4f}")

    # --- Identify missed pairs ---
    all_gt_pairs = set()
    for s1_id, matches in dev_gt.items():
        for sx_id in matches:
            all_gt_pairs.add((s1_id, sx_id))

    missed_pairs = all_gt_pairs - c500_recovered
    p(f"Missed pairs: {len(missed_pairs):,}")

    # --- Diagnose missed pairs ---
    categories = Counter()
    detail_rows = []

    for s1_id, sx_id in missed_pairs:
        s1 = s1_data.get(s1_id)
        sx = sx_data.get(sx_id)
        if not s1 or not sx:
            categories["sx_data_not_found"] += 1
            continue

        s1_nn = s1["norm_name"]
        sx_nn = sx["norm_name"]
        s1_na = s1["norm_addr"]
        sx_na = sx["norm_addr"]
        s1_at = s1["addr_toks"]
        sx_at = sx["addr_toks"]
        s1_digits = s1["digits"]
        sx_digits = sx["digits"]

        # Name analysis
        name_exact = (s1_nn == sx_nn) if s1_nn and sx_nn else False
        s1_nt = s1["name_toks"]
        sx_nt = sx["name_toks"]
        name_token_overlap = len(s1_nt & sx_nt) if s1_nt and sx_nt else 0
        name_token_jaccard = (len(s1_nt & sx_nt) / len(s1_nt | sx_nt)) if (s1_nt and sx_nt) else 0.0

        # Address analysis
        s1_addr_empty = (s1_na == "")
        sx_addr_empty = (sx_na == "")
        addr_exact = (s1_na == sx_na) if s1_na and sx_na else False
        addr_token_overlap = len(s1_at & sx_at) if s1_at and sx_at else 0
        addr_token_jaccard = (len(s1_at & sx_at) / len(s1_at | sx_at)) if (s1_at and sx_at) else 0.0

        # Digit overlap
        s1_dig_set = set(s1_digits.split()) if s1_digits else set()
        sx_dig_set = set(sx_digits.split()) if sx_digits else set()
        digit_overlap = len(s1_dig_set & sx_dig_set)

        # Country
        same_country = s1["country"].strip().lower() == sx["country"].strip().lower()

        # Classify
        if s1_addr_empty or sx_addr_empty:
            categories["missing_address"] += 1
        elif addr_exact:
            categories["addr_exact_name_mismatch"] += 1
        elif addr_token_jaccard > 0.5:
            categories["addr_high_overlap_name_mismatch"] += 1
        elif addr_token_overlap >= 3:
            categories["addr_moderate_overlap"] += 1
        elif digit_overlap >= 2:
            categories["digit_overlap_name_mismatch"] += 1
        elif name_token_overlap >= 1:
            categories["partial_name_overlap_weak_addr"] += 1
        elif name_token_jaccard == 0 and addr_token_jaccard < 0.1:
            categories["both_name_addr_mismatch"] += 1
        else:
            categories["other"] += 1

        # Collect a sample for detailed inspection
        if len(detail_rows) < 200:
            detail_rows.append({
                "s1_id": s1_id, "sx_id": sx_id,
                "s1_name": s1_nn, "sx_name": sx_nn,
                "s1_addr": s1_na[:80], "sx_addr": sx_na[:80],
                "name_jaccard": round(name_token_jaccard, 3),
                "addr_jaccard": round(addr_token_jaccard, 3),
                "addr_tok_overlap": addr_token_overlap,
                "digit_overlap": digit_overlap,
                "same_country": same_country,
            })

    # --- Output ---
    p("\n" + "="*60)
    p(" Missed-Pair Diagnosis")
    p("="*60)

    total_missed = len(missed_pairs)
    p(f"\n| Category | Count | % of Missed |")
    p(f"| -------- | ----: | ----------: |")
    for cat, cnt in categories.most_common():
        pct = cnt / total_missed * 100 if total_missed else 0
        p(f"| {cat} | {cnt:,} | {pct:.1f}% |")

    p(f"\nTotal missed: {total_missed:,}")
    p(f"Total GT: {total_gt_pairs:,}")

    # Save diagnosis
    out = {
        "total_gt_pairs": total_gt_pairs,
        "c500_recovered": len(c500_recovered),
        "c500_recall": recall,
        "missed_count": total_missed,
        "categories": dict(categories),
        "sample_missed": detail_rows[:50],
    }
    out_path = config.PROJECT_ROOT / "reports" / "b4_miss_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    p(f"\nDiagnosis saved to {out_path}")

if __name__ == "__main__":
    main()
