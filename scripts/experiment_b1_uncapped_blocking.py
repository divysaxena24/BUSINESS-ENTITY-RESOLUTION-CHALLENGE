#!/usr/bin/env python3
"""
Experiment B1 — Uncapped Blocking Analysis
Measures the true recall of the baseline blocker and the candidate explosion
without the artificial 200-candidate safety cap.
"""
import sys
import gc
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

# Adjust path to import src
sys.path.append(str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))
from src import config
from src.blocking import load_s1_index, _find_candidates_for_record
from src.normalization import normalize_name, name_tokens
from src.pipeline import load_ground_truth

def p(msg):
    print(msg, flush=True)

def main():
    p("="*60)
    p(" Experiment B1: Uncapped Blocking Distribution")
    p("="*60)
    
    # 1. Dev Split
    all_gt = load_ground_truth(config.TRAIN_GT)
    all_s1 = list(all_gt.keys())
    random.seed(config.RANDOM_SEED)
    dev_s1_ids = set(random.sample(all_s1, min(5000, len(all_s1))))
    p(f"Subsampled {len(dev_s1_ids):,} S1 IDs.")
    
    dev_gt = {k: v for k, v in all_gt.items() if k in dev_s1_ids}
    del all_gt, all_s1; gc.collect()

    # 2. Index S1
    s1_data, name_index, token_index = load_s1_index(config.TRAIN_S1, dev_s1_ids)
    
    # 3. Stream S2 and S3 without capping
    # We only need to count candidates per S1, we don't need to store the actual pairs
    # to save memory. We just store {s1_id: set(sx_ids)}
    s1_candidates = defaultdict(set)
    
    def process_source(source_path):
        p(f"\nProcessing {source_path.name}...")
        total_records = 0
        for chunk in pd.read_csv(source_path, sep="\t", dtype=str,
                                 keep_default_na=False, chunksize=config.CHUNK_SIZE):
            for sx_id, b_name in zip(chunk["entity_id"], chunk["business_name"]):
                norm_n = normalize_name(b_name)
                tokens = name_tokens(norm_n)
                
                matched_s1 = _find_candidates_for_record(
                    norm_n, tokens, name_index, token_index, min_shared_tokens=2
                )
                
                for s1_id in matched_s1:
                    s1_candidates[s1_id].add(sx_id)
                    
            total_records += config.CHUNK_SIZE
            if total_records % 1_000_000 == 0:
                p(f"  Processed {total_records:,} rows...")
            del chunk
            gc.collect()

    process_source(config.TRAIN_S2)
    process_source(config.TRAIN_S3)
    
    # 4. Analysis
    p("\n" + "="*40)
    p(" Analysis Results")
    p("="*40)
    
    counts = np.array([len(s1_candidates.get(s1, set())) for s1 in dev_s1_ids])
    total_cands = counts.sum()
    
    p(f"Total Candidate Pairs (Uncapped): {total_cands:,}")
    p(f"Average candidates / S1:          {counts.mean():.1f}")
    p(f"Median candidates / S1:           {np.median(counts):.1f}")
    p(f"P95 candidates / S1:              {np.percentile(counts, 95):.1f}")
    p(f"Max candidates / S1:              {counts.max():,}")
    
    # Exceedance counts
    p("\nS1 Entities exceeding candidate thresholds:")
    for thresh in [200, 500, 1000, 5000, 10000]:
        exceed = (counts > thresh).sum()
        p(f"  > {thresh:<5} : {exceed:,} S1s ({(exceed/len(dev_s1_ids))*100:.1f}%)")
        
    # Recall Analysis
    total_true = 0
    total_found = 0
    total_found_capped = 0
    
    for s1_id in dev_s1_ids:
        true_matches = dev_gt[s1_id]
        cands = s1_candidates.get(s1_id, set())
        
        total_true += len(true_matches)
        total_found += len(true_matches & cands)
        
        # Simulate the cap: if we randomly dropped candidates after 200...
        # Actually in the pipeline it drops chronologically, which is effectively random
        # for true matches distributed throughout the file.
        # Expected retained true matches = true_in_cands * min(1, 200/len(cands))
        true_in_cands = len(true_matches & cands)
        if len(cands) > 200:
            total_found_capped += true_in_cands * (200 / len(cands))
        else:
            total_found_capped += true_in_cands

    uncapped_recall = total_found / total_true if total_true else 0.0
    capped_expected_recall = total_found_capped / total_true if total_true else 0.0
    
    p(f"\nUncapped Blocking Recall: {uncapped_recall:.4f} ({total_found:,} / {total_true:,})")
    p(f"Expected Recall w/ 200 Cap: {capped_expected_recall:.4f} (~{int(total_found_capped):,} / {total_true:,})")
    p(f"True Matches Lost to Cap:   ~{total_found - int(total_found_capped):,}")
    p(f"True Matches Missed by Blocker: {total_true - total_found:,}")
    
    p("\nDone.")

if __name__ == "__main__":
    main()
