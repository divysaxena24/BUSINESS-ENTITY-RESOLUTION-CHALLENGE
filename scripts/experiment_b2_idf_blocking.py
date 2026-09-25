#!/usr/bin/env python3
"""
Experiment B2 — Rare-Token / IDF Blocking Analysis
Measures the tradeoff between candidate volume and blocking recall across various
document-frequency (DF) thresholds and token requirements.
"""
import sys
import gc
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))
from src import config
from src.normalization import normalize_name, name_tokens
from src.pipeline import load_ground_truth

def p(msg):
    print(msg, flush=True)

def main():
    p("="*60)
    p(" Experiment B2: Rare-Token Blocking")
    p("="*60)
    
    df_path = config.PROJECT_ROOT / "dataset" / "token_df.json"
    if not df_path.exists():
        p(f"ERROR: {df_path} not found. Run build_token_df.py first.")
        sys.exit(1)
        
    p("Loading Token DF map...")
    with open(df_path, "r") as f:
        token_df = json.load(f)
    p(f"Loaded {len(token_df):,} tokens.")
    
    # 1. Dev Split
    all_gt = load_ground_truth(config.TRAIN_GT)
    all_s1 = list(all_gt.keys())
    random.seed(config.RANDOM_SEED)
    dev_s1_ids = set(random.sample(all_s1, min(5000, len(all_s1))))
    p(f"Subsampled {len(dev_s1_ids):,} S1 IDs.")
    
    dev_gt = {k: v for k, v in all_gt.items() if k in dev_s1_ids}
    del all_gt, all_s1; gc.collect()

    # 2. Configurations to Test
    THRESHOLDS = [100, 500, 1000, 5000, 10000, 50000]
    MODERATE_THRESH = 50000
    
    cfg_names = []
    for cfg in ["A", "B", "C"]:
        for th in THRESHOLDS:
            cfg_names.append(f"{cfg}_{th}")
            
    # Track metrics via counters to prevent OOM
    candidate_counts = {c: {s1: 0 for s1 in dev_s1_ids} for c in cfg_names}
    true_found = {c: 0 for c in cfg_names}
    
    # Pre-index S1
    p("Loading S1 index...")
    name_index = defaultdict(set)
    token_index = defaultdict(set)
    
    for chunk in pd.read_csv(config.TRAIN_S1, sep="\t", dtype=str, keep_default_na=False, chunksize=config.CHUNK_SIZE):
        for eid, b_name in zip(chunk["entity_id"], chunk["business_name"]):
            if eid not in dev_s1_ids:
                continue
            norm_n = normalize_name(b_name)
            tokens = name_tokens(norm_n)
            
            if norm_n:
                name_index[norm_n].add(eid)
            for t in tokens:
                token_index[t].add(eid)
                
    def process_source(source_path):
        p(f"\nProcessing {source_path.name}...")
        total_records = 0
        for chunk in pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False, chunksize=config.CHUNK_SIZE):
            for sx_id, b_name in zip(chunk["entity_id"], chunk["business_name"]):
                norm_n = normalize_name(b_name)
                tokens = name_tokens(norm_n)
                
                # 1. Exact name matches (fallback)
                if norm_n in name_index:
                    for s1_id in name_index[norm_n]:
                        is_match = (sx_id in dev_gt.get(s1_id, set()))
                        for cname in cfg_names:
                            candidate_counts[cname][s1_id] += 1
                            if is_match: true_found[cname] += 1
                
                # 2. Shared tokens
                s1_counts = defaultdict(list)
                for t in tokens:
                    if t in token_index:
                        for s1_id in token_index[t]:
                            s1_counts[s1_id].append(t)
                
                # Evaluate token rules
                for s1_id, shared_toks in s1_counts.items():
                    df_vals = [token_df.get(t, 0) for t in shared_toks]
                    
                    min_df = min(df_vals) if df_vals else float('inf')
                    second_min_df = sorted(df_vals)[1] if len(df_vals) > 1 else float('inf')
                    
                    is_match = (sx_id in dev_gt.get(s1_id, set()))
                    
                    for thresh in THRESHOLDS:
                        if min_df <= thresh:
                            cname = f"A_{thresh}"
                            candidate_counts[cname][s1_id] += 1
                            if is_match: true_found[cname] += 1
                            
                        if second_min_df <= thresh:
                            cname = f"B_{thresh}"
                            candidate_counts[cname][s1_id] += 1
                            if is_match: true_found[cname] += 1
                            
                        if min_df <= thresh or second_min_df <= MODERATE_THRESH:
                            cname = f"C_{thresh}"
                            candidate_counts[cname][s1_id] += 1
                            if is_match: true_found[cname] += 1
                            
            total_records += config.CHUNK_SIZE
            if total_records % 1_000_000 == 0:
                p(f"  Processed {total_records:,} rows...")
            del chunk
            gc.collect()

    process_source(config.TRAIN_S2)
    process_source(config.TRAIN_S3)
    
    # 4. Analysis and Report Generation
    p("\n" + "="*40)
    p(" Analysis Results")
    p("="*40)
    
    total_true_overall = sum(len(matches) for matches in dev_gt.values())
    
    report_lines = [
        "# Experiment B2 Results",
        "",
        "| Configuration | DF Threshold | Candidates | Mean/S1 | Median | P95 | Max | Recall |",
        "| ------------- | -----------: | ---------: | ------: | -----: | --: | --: | -----: |"
    ]
    
    for cfg_type in ["A", "B", "C"]:
        for thresh in THRESHOLDS:
            cfg_name = f"{cfg_type}_{thresh}"
            
            counts = np.array(list(candidate_counts[cfg_name].values()))
            total_cands = counts.sum()
            
            recall = true_found[cfg_name] / total_true_overall if total_true_overall else 0.0
            
            mean_c = counts.mean()
            med_c = np.median(counts)
            p95_c = np.percentile(counts, 95)
            max_c = counts.max()
            
            rule_map = {"A": "1 Rare", "B": "2 Rare", "C": "1 Rare OR 2 Mod"}
            rule = rule_map[cfg_type]
            
            line = f"| {cfg_name} ({rule}) | {thresh} | {total_cands:,} | {mean_c:.1f} | {med_c:.0f} | {p95_c:.0f} | {max_c:,} | {recall:.4f} |"
            report_lines.append(line)
            p(line)
            
    out_path = config.PROJECT_ROOT / "reports" / "experiment_b2_results.md"
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))
        
    p(f"\nReport written to {out_path}")

if __name__ == "__main__":
    main()
