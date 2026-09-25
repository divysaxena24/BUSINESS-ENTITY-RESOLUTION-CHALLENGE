#!/usr/bin/env python3
"""
Experiment B4 — Address-Assisted Blocking on top of C_500.
Memory-safe: uses integer-indexed sets for GT recall dedup, counters for volume.

Optimized: evaluates blocking rules per-record and records hits efficiently.
"""
import sys
import gc
import json
import re
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))
from src import config
from src.normalization import normalize_name, name_tokens, normalize_address, address_tokens
from src.pipeline import load_ground_truth

def p(msg):
    print(msg, flush=True)

_RE_PIN = re.compile(r"\b(\d{5,6})\b")

def extract_pins(text):
    if not text:
        return set()
    return set(_RE_PIN.findall(text))

def main():
    p("="*60)
    p(" Experiment B4: Address-Assisted Blocking")
    p("="*60)

    # Load DFs
    p("Loading name token DF...")
    with open(config.PROJECT_ROOT / "dataset" / "token_df.json", "r") as f:
        name_token_df = json.load(f)
    p(f"  {len(name_token_df):,} name tokens.")

    p("Loading address token DF...")
    with open(config.PROJECT_ROOT / "dataset" / "addr_token_df.json", "r") as f:
        addr_token_df = json.load(f)
    p(f"  {len(addr_token_df):,} address tokens.")

    # Dev split (identical to B1/B2)
    all_gt = load_ground_truth(config.TRAIN_GT)
    all_s1 = list(all_gt.keys())
    random.seed(config.RANDOM_SEED)
    dev_s1_ids = set(random.sample(all_s1, min(5000, len(all_s1))))
    dev_gt = {k: v for k, v in all_gt.items() if k in dev_s1_ids}
    del all_gt, all_s1; gc.collect()

    total_gt_pairs = sum(len(v) for v in dev_gt.values())
    p(f"Dev S1 IDs: {len(dev_s1_ids):,}")
    p(f"Total GT pairs: {total_gt_pairs:,}")

    # Constants
    RARE_NAME = 500
    MOD_NAME = 50000
    ADDR_TH = [50, 100, 500, 1000, 5000]

    # GT pair index
    gt_pair_idx = {}
    for s1_id, matches in dev_gt.items():
        for sx_id in matches:
            gt_pair_idx[(s1_id, sx_id)] = len(gt_pair_idx)
    p(f"GT pair index size: {len(gt_pair_idx):,}")

    # Config list (we track independently, then compute unions in analysis)
    # Standalone configs only during streaming; unions computed post-hoc
    STANDALONE = ["C500", "D1"]
    for th in ADDR_TH:
        STANDALONE.append(f"D2_{th}")
        STANDALONE.append(f"D3_{th}")
    STANDALONE.append("D4")

    cand_counts = {c: defaultdict(int) for c in STANDALONE}
    recovered = {c: set() for c in STANDALONE}

    # Build S1 indexes
    p("\nBuilding S1 indexes...")
    s1_name_idx = defaultdict(set)
    s1_ntok_idx = defaultdict(set)
    s1_addr_idx = defaultdict(set)
    s1_atok_idx = defaultdict(set)
    s1_pin_idx = defaultdict(set)

    for chunk in pd.read_csv(config.TRAIN_S1, sep="\t", dtype=str, keep_default_na=False, chunksize=config.CHUNK_SIZE):
        for eid, bname, baddr in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"]):
            if eid not in dev_s1_ids:
                continue
            nn = normalize_name(bname)
            na = normalize_address(baddr)
            for t in name_tokens(nn):
                s1_ntok_idx[t].add(eid)
            if nn:
                s1_name_idx[nn].add(eid)
            if na:
                s1_addr_idx[na].add(eid)
            for t in address_tokens(na):
                s1_atok_idx[t].add(eid)
            for pin in extract_pins(baddr):
                s1_pin_idx[pin].add(eid)

    p(f"  Name keys: {len(s1_name_idx):,}, Token keys: {len(s1_ntok_idx):,}")
    p(f"  Addr keys: {len(s1_addr_idx):,}, Addr tok keys: {len(s1_atok_idx):,}")
    p(f"  PIN keys: {len(s1_pin_idx):,}")

    def process_source(source_path):
        p(f"\nProcessing {source_path.name}...")
        total = 0
        for chunk in pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False, chunksize=config.CHUNK_SIZE):
            for sx_id, bname, baddr in zip(
                chunk["entity_id"], chunk["business_name"], chunk["business_address"]
            ):
                nn = normalize_name(bname)
                na = normalize_address(baddr)
                # Only lookup tokens that could possibly meet the threshold to save massive amounts of time
                n_toks = [t for t in name_tokens(nn) if name_token_df.get(t, 0) <= MOD_NAME]
                a_toks = [t for t in address_tokens(na) if addr_token_df.get(t, 0) <= max(ADDR_TH)]
                pins = extract_pins(baddr)

                # --- C500: name blocking ---
                c500_s1s = set()
                if nn in s1_name_idx:
                    c500_s1s.update(s1_name_idx[nn])

                shared_ntoks = defaultdict(list)
                for t in n_toks:
                    if t in s1_ntok_idx:
                        for s1 in s1_ntok_idx[t]:
                            shared_ntoks[s1].append(t)

                for s1, toks in shared_ntoks.items():
                    dfs = [name_token_df.get(t, 0) for t in toks]
                    mn = min(dfs)
                    mn2 = sorted(dfs)[1] if len(dfs) > 1 else float('inf')
                    if mn <= RARE_NAME or mn2 <= MOD_NAME:
                        c500_s1s.add(s1)

                for s1 in c500_s1s:
                    cand_counts["C500"][s1] += 1
                    key = (s1, sx_id)
                    if key in gt_pair_idx:
                        recovered["C500"].add(gt_pair_idx[key])

                # --- D1: exact addr ---
                if na and na in s1_addr_idx:
                    for s1 in s1_addr_idx[na]:
                        cand_counts["D1"][s1] += 1
                        key = (s1, sx_id)
                        if key in gt_pair_idx:
                            recovered["D1"].add(gt_pair_idx[key])

                # --- D2/D3: addr token ---
                shared_atoks = defaultdict(list)
                for t in a_toks:
                    if t in s1_atok_idx:
                        for s1 in s1_atok_idx[t]:
                            shared_atoks[s1].append(t)

                for s1, atoks in shared_atoks.items():
                    adfs = [addr_token_df.get(t, 0) for t in atoks]
                    amn = min(adfs) if adfs else float('inf')
                    amn2 = sorted(adfs)[1] if len(adfs) > 1 else float('inf')
                    key = (s1, sx_id)
                    pidx = gt_pair_idx.get(key)

                    for th in ADDR_TH:
                        if amn <= th:
                            cand_counts[f"D2_{th}"][s1] += 1
                            if pidx is not None:
                                recovered[f"D2_{th}"].add(pidx)
                        if amn2 <= th:
                            cand_counts[f"D3_{th}"][s1] += 1
                            if pidx is not None:
                                recovered[f"D3_{th}"].add(pidx)

                # --- D4: PIN + 1 addr token DF<=5000 ---
                for pin in pins:
                    if pin in s1_pin_idx:
                        for s1 in s1_pin_idx[pin]:
                            if s1 in shared_atoks:
                                adfs = [addr_token_df.get(t, 0) for t in shared_atoks[s1]]
                                if any(d <= 5000 for d in adfs):
                                    cand_counts["D4"][s1] += 1
                                    key = (s1, sx_id)
                                    if key in gt_pair_idx:
                                        recovered["D4"].add(gt_pair_idx[key])

            total += len(chunk)
            if total % 1_000_000 == 0:
                p(f"  {total:,} rows...")
            del chunk; gc.collect()

    process_source(config.TRAIN_S2)
    process_source(config.TRAIN_S3)

    # --- Analysis ---
    p("\n" + "="*60)
    p(" Analysis Results")
    p("="*60)

    c500_rec = recovered["C500"]

    # Compute union configs post-hoc
    UNION_CFGS = {}
    for cfg in STANDALONE:
        if cfg != "C500":
            UNION_CFGS[f"C500+{cfg}"] = cfg

    all_cfgs = list(STANDALONE) + list(UNION_CFGS.keys())

    lines = [
        "# Experiment B4 Results: Address-Assisted Blocking",
        "",
        "## Missed-Pair Diagnosis (from b4_miss_diagnosis.json)",
        "",
        "| Category | Count | % of Missed |",
        "| -------- | ----: | ----------: |",
        "| addr_high_overlap_name_mismatch | 2,679 | 45.9% |",
        "| addr_moderate_overlap | 1,828 | 31.3% |",
        "| addr_exact_name_mismatch | 541 | 9.3% |",
        "| partial_name_overlap_weak_addr | 278 | 4.8% |",
        "| other | 204 | 3.5% |",
        "| missing_address | 161 | 2.8% |",
        "| digit_overlap_name_mismatch | 120 | 2.1% |",
        "| both_name_addr_mismatch | 30 | 0.5% |",
        "",
        "## Configuration Results",
        "",
        "| Config | Candidates | Mean/S1 | Median | P95 | P99 | Max | Recall | Incr vs C500 | New Pairs |",
        "| ------ | ---------: | ------: | -----: | --: | --: | --: | -----: | -----------: | --------: |",
    ]

    for cfg in all_cfgs:
        if cfg in STANDALONE:
            counts_dict = cand_counts[cfg]
            rec_set = recovered[cfg]
        else:
            # Union
            base_cfg = UNION_CFGS[cfg]
            # Candidate counts: per S1, take max of (C500 count + base count)
            # Actually for union, a candidate appears if it appears in EITHER
            # We can't exactly compute union volume from independent counters
            # because the same (s1, sx) pair may appear in both.
            # For volume, we sum as upper bound and note this.
            counts_dict = {}
            for s1 in dev_s1_ids:
                counts_dict[s1] = cand_counts["C500"].get(s1, 0) + cand_counts[base_cfg].get(s1, 0)
            rec_set = recovered["C500"] | recovered[base_cfg]

        counts = np.array([counts_dict.get(s1, 0) for s1 in dev_s1_ids])
        total_cands = int(counts.sum())
        mean_c = counts.mean()
        med_c = np.median(counts)
        p95 = np.percentile(counts, 95)
        p99 = np.percentile(counts, 99)
        max_c = int(counts.max())

        recall = len(rec_set) / total_gt_pairs
        new_pairs = len(rec_set - c500_rec)
        incr = new_pairs / total_gt_pairs

        line = (
            f"| {cfg} | {total_cands:,} | {mean_c:.1f} | {med_c:.0f} | "
            f"{p95:.0f} | {p99:.0f} | {max_c:,} | {recall:.4f} | "
            f"+{incr:.4f} | {new_pairs:,} |"
        )
        lines.append(line)
        p(line)

    # Summary
    c500_recall = len(c500_rec) / total_gt_pairs
    c500_total = int(np.array([cand_counts["C500"].get(s1, 0) for s1 in dev_s1_ids]).sum())
    c500_p95 = np.percentile([cand_counts["C500"].get(s1, 0) for s1 in dev_s1_ids], 95)

    best_cfg = None
    best_recall = 0
    for cfg in UNION_CFGS:
        base = UNION_CFGS[cfg]
        r = len(recovered["C500"] | recovered[base]) / total_gt_pairs
        if r > best_recall:
            best_recall = r
            best_cfg = cfg

    lines.extend([
        "",
        "## Summary Comparison",
        "",
        f"B1 Uncapped Recall: 0.6784",
        f"C_500 Recall (deduplicated): {c500_recall:.4f}",
        f"C_500 Candidates: {c500_total:,}",
        f"C_500 P95: {c500_p95:.0f}",
    ])

    if best_cfg:
        base = UNION_CFGS[best_cfg]
        best_rec = recovered["C500"] | recovered[base]
        best_new = len(best_rec - c500_rec)
        best_cands = int(np.array([
            cand_counts["C500"].get(s1, 0) + cand_counts[base].get(s1, 0)
            for s1 in dev_s1_ids
        ]).sum())
        best_p95 = np.percentile([
            cand_counts["C500"].get(s1, 0) + cand_counts[base].get(s1, 0)
            for s1 in dev_s1_ids
        ], 95)
        remaining = total_gt_pairs - len(best_rec)

        lines.extend([
            f"",
            f"Best union: {best_cfg}",
            f"  Recall: {best_recall:.4f}",
            f"  Candidates (upper bound): {best_cands:,}",
            f"  P95: {best_p95:.0f}",
            f"  New true pairs vs C500: {best_new:,}",
            f"  Additional candidates vs C500: {best_cands - c500_total:,}",
            f"",
            f"Remaining missed pairs: {remaining:,} ({remaining/total_gt_pairs*100:.1f}%)",
        ])

    out_path = config.PROJECT_ROOT / "reports" / "experiment_b4_results.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    p(f"\nReport saved to {out_path}")

if __name__ == "__main__":
    main()
