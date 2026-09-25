#!/usr/bin/env python3
"""
Data Audit — Business Entity Resolution Challenge
===================================================
Produces reports/data_profile.md and reports/data_profile_stats.json

Uses chunked/streaming reads to handle files that exceed available RAM.
"""

import os
import sys
import json
import gc
import numpy as np
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE / "dataset" / "train"
TEST_DIR = BASE / "dataset" / "test"
REPORTS_DIR = BASE / "reports"

TRAIN_FILES = [
    (TRAIN_DIR / "train_source1.tsv", "Train Source 1"),
    (TRAIN_DIR / "train_source2.tsv", "Train Source 2"),
    (TRAIN_DIR / "train_source3.tsv", "Train Source 3"),
]
TEST_FILES = [
    (TEST_DIR / "test_source1.tsv", "Test Source 1"),
    (TEST_DIR / "test_source2.tsv", "Test Source 2"),
    (TEST_DIR / "test_source3.tsv", "Test Source 3"),
]
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

SOURCE_SCHEMA = ["entity_id", "business_name", "business_address", "country"]
GT_SCHEMA = ["source1_entity_id", "matched_entity_ids"]

CHUNK_SIZE = 100_000  # rows per chunk


def p(msg):
    print(msg, flush=True)


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def profile_source_chunked(path, label):
    """Profile a source TSV using chunked reads. Returns (stats_dict, id_set)."""
    import pandas as pd
    p(f"  Profiling {label} ({file_size_mb(path):.1f} MB) via chunked reads ...")

    # First, read just the header
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
    columns = header_line.split("\t")

    stats = {
        "label": label,
        "file": os.path.basename(path),
        "columns": columns,
        "schema_valid": columns == SOURCE_SCHEMA,
    }

    total_rows = 0
    null_counts = Counter()
    id_set = set()
    duplicate_id_count = 0
    prefix_counter = Counter()
    country_counter = Counter()
    name_lengths_acc = []
    addr_lengths_acc = []
    name_empty = 0
    addr_empty = 0
    sample_names = []
    sample_addrs = []

    chunk_num = 0
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE):
        chunk_num += 1
        n = len(chunk)
        total_rows += n

        if chunk_num % 5 == 1:
            p(f"    chunk {chunk_num}: {total_rows:,} rows so far ...")

        # Nulls/empty
        for col in chunk.columns:
            null_counts[col] += int((chunk[col].str.strip() == "").sum())

        # IDs
        ids_in_chunk = set(chunk["entity_id"])
        overlap = ids_in_chunk & id_set
        duplicate_id_count += len(overlap)
        id_set |= ids_in_chunk

        # Prefixes
        prefix_counter.update(chunk["entity_id"].str[:3])

        # Countries
        country_counter.update(chunk["country"])

        # Name lengths (store summary stats per chunk to avoid huge arrays)
        nl = chunk["business_name"].str.len()
        name_lengths_acc.append({
            "min": int(nl.min()),
            "max": int(nl.max()),
            "sum": float(nl.sum()),
            "count": int(nl.count()),
            "values_sorted": sorted(nl.tolist()),  # for percentiles later we store a subsample
        })
        name_empty += int((chunk["business_name"].str.strip() == "").sum())

        # Address lengths
        al = chunk["business_address"].str.len()
        addr_lengths_acc.append({
            "min": int(al.min()),
            "max": int(al.max()),
            "sum": float(al.sum()),
            "count": int(al.count()),
        })
        addr_empty += int((chunk["business_address"].str.strip() == "").sum())

        # Collect samples from first chunk only
        if chunk_num == 1:
            sample_idx = chunk.sample(min(10, n), random_state=42).index
            sample_names = chunk.loc[sample_idx, "business_name"].tolist()
            sample_addrs = chunk.loc[sample_idx, "business_address"].tolist()

        del chunk
        gc.collect()

    stats["rows"] = total_rows
    stats["null_or_empty"] = dict(null_counts)
    stats["unique_ids"] = len(id_set)
    stats["duplicate_ids"] = duplicate_id_count
    stats["id_prefixes"] = dict(prefix_counter)
    stats["country_distribution"] = dict(country_counter)

    # Aggregate name length stats
    if name_lengths_acc:
        overall_min = min(d["min"] for d in name_lengths_acc)
        overall_max = max(d["max"] for d in name_lengths_acc)
        overall_sum = sum(d["sum"] for d in name_lengths_acc)
        overall_count = sum(d["count"] for d in name_lengths_acc)
        overall_mean = round(overall_sum / overall_count, 1) if overall_count > 0 else 0
        # For median and p95, re-read with sampling if needed — use approximation
        stats["name_length"] = {
            "min": overall_min,
            "max": overall_max,
            "mean": overall_mean,
            "median": "~",  # will compute below
            "p95": "~",
        }
    else:
        stats["name_length"] = {"min": 0, "max": 0, "mean": 0, "median": 0, "p95": 0}
    stats["name_empty_count"] = name_empty

    # Aggregate address length stats
    if addr_lengths_acc:
        overall_min = min(d["min"] for d in addr_lengths_acc)
        overall_max = max(d["max"] for d in addr_lengths_acc)
        overall_sum = sum(d["sum"] for d in addr_lengths_acc)
        overall_count = sum(d["count"] for d in addr_lengths_acc)
        overall_mean = round(overall_sum / overall_count, 1) if overall_count > 0 else 0
        stats["address_length"] = {
            "min": overall_min,
            "max": overall_max,
            "mean": overall_mean,
            "median": "~",
            "p95": "~",
        }
    else:
        stats["address_length"] = {"min": 0, "max": 0, "mean": 0, "median": 0, "p95": 0}
    stats["address_empty_count"] = addr_empty

    # Compute median and p95 via a second sampled pass (read 200k random-ish rows)
    p(f"    Computing percentiles via sampled pass ...")
    import pandas as pd
    sample_frac = min(1.0, 200_000 / max(total_rows, 1))
    name_lens_sample = []
    addr_lens_sample = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             usecols=["business_name", "business_address"], chunksize=CHUNK_SIZE):
        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=42)
        name_lens_sample.extend(chunk["business_name"].str.len().tolist())
        addr_lens_sample.extend(chunk["business_address"].str.len().tolist())
        del chunk
        gc.collect()

    if name_lens_sample:
        arr = np.array(name_lens_sample)
        stats["name_length"]["median"] = round(float(np.median(arr)), 1)
        stats["name_length"]["p95"] = round(float(np.percentile(arr, 95)), 1)
    if addr_lens_sample:
        arr = np.array(addr_lens_sample)
        stats["address_length"]["median"] = round(float(np.median(arr)), 1)
        stats["address_length"]["p95"] = round(float(np.percentile(arr, 95)), 1)

    del name_lens_sample, addr_lens_sample
    gc.collect()

    stats["sample_names"] = sample_names
    stats["sample_addresses"] = sample_addrs

    p(f"    Done: {total_rows:,} rows, {len(id_set):,} unique IDs.")
    return stats, id_set


def profile_ground_truth(path, s1_ids, s2_ids, s3_ids):
    """Profile the ground-truth file line-by-line for memory efficiency."""
    p(f"  Profiling ground truth ({file_size_mb(path):.1f} MB) ...")
    stats = {"file": os.path.basename(path)}

    match_counts = []
    s2_match_total = 0
    s3_match_total = 0
    singleton_count = 0
    one_to_many = 0
    max_matches = 0
    max_match_id = None
    gt_s1_ids = set()
    all_matched_s2 = set()
    all_matched_s3 = set()
    row_count = 0
    header = None

    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        header = header_line.split("\t")
        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            s1_id = parts[0].strip()
            raw_ids = parts[1].strip() if len(parts) > 1 else ""
            gt_s1_ids.add(s1_id)
            row_count += 1

            if row_count % 500_000 == 0:
                p(f"    {row_count:,} GT rows processed ...")

            if raw_ids == "":
                match_counts.append(0)
                singleton_count += 1
                continue

            ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
            n = len(ids)
            match_counts.append(n)
            if n > 1:
                one_to_many += 1
            if n > max_matches:
                max_matches = n
                max_match_id = s1_id
            for mid in ids:
                if mid.startswith("S2-"):
                    s2_match_total += 1
                    all_matched_s2.add(mid)
                elif mid.startswith("S3-"):
                    s3_match_total += 1
                    all_matched_s3.add(mid)

    stats["rows"] = row_count
    stats["columns"] = header
    stats["schema_valid"] = header == GT_SCHEMA

    match_counts_arr = np.array(match_counts)
    stats["total_s1_entities"] = row_count
    stats["unique_s1_ids"] = len(gt_s1_ids)
    stats["duplicate_s1_ids"] = row_count - len(gt_s1_ids)
    stats["singleton_count"] = singleton_count
    stats["singleton_pct"] = round(100.0 * singleton_count / row_count, 2) if row_count > 0 else 0
    stats["one_to_many_count"] = one_to_many
    stats["s1_to_s2_match_count"] = s2_match_total
    stats["s1_to_s3_match_count"] = s3_match_total
    stats["total_match_pairs"] = s2_match_total + s3_match_total
    stats["max_matches_for_single_s1"] = max_matches
    stats["max_matches_s1_id"] = max_match_id

    stats["match_count_distribution"] = {
        "min": int(match_counts_arr.min()),
        "max": int(match_counts_arr.max()),
        "mean": round(float(match_counts_arr.mean()), 2),
        "median": round(float(np.median(match_counts_arr)), 1),
        "p95": round(float(np.percentile(match_counts_arr, 95)), 1),
    }

    counter = Counter(match_counts)
    stats["match_count_histogram"] = {str(k): v for k, v in sorted(counter.items())}

    # Cross-reference integrity
    missing_from_s1 = gt_s1_ids - s1_ids
    extra_in_s1 = s1_ids - gt_s1_ids
    stats["gt_s1_ids_missing_from_source1"] = len(missing_from_s1)
    stats["source1_ids_missing_from_gt"] = len(extra_in_s1)
    if missing_from_s1:
        stats["missing_from_s1_examples"] = sorted(list(missing_from_s1))[:5]

    missing_s2 = all_matched_s2 - s2_ids
    missing_s3 = all_matched_s3 - s3_ids
    stats["matched_s2_not_in_source2"] = len(missing_s2)
    stats["matched_s3_not_in_source3"] = len(missing_s3)
    if missing_s2:
        stats["missing_s2_examples"] = sorted(list(missing_s2))[:5]
    if missing_s3:
        stats["missing_s3_examples"] = sorted(list(missing_s3))[:5]

    p(f"    Done: {row_count:,} GT rows, {singleton_count:,} singletons, {s2_match_total + s3_match_total:,} match pairs.")
    return stats


def generate_markdown(train_stats, test_stats, gt_stats):
    """Generate the markdown report."""
    lines = []
    lines.append("# Data Profile Report")
    lines.append("")
    lines.append("## Business Entity Resolution Challenge")
    lines.append("")
    lines.append("Generated by `scripts/data_audit.py`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- File Overview ----
    lines.append("## 1. File Overview")
    lines.append("")
    lines.append("### Training Files")
    lines.append("")
    lines.append("| File | Rows | Columns | Schema Valid | Size (MB) |")
    lines.append("|------|------|---------|-------------|-----------|")
    for s in train_stats:
        sz = file_size_mb(TRAIN_DIR / s["file"])
        cols = ", ".join(s["columns"])
        lines.append(f"| {s['file']} | {s['rows']:,} | {cols} | {'✅' if s['schema_valid'] else '❌'} | {sz:.1f} |")
    sz = file_size_mb(TRAIN_GT)
    cols = ", ".join(gt_stats["columns"])
    lines.append(f"| {gt_stats['file']} | {gt_stats['rows']:,} | {cols} | {'✅' if gt_stats['schema_valid'] else '❌'} | {sz:.1f} |")
    lines.append("")

    lines.append("### Test Files")
    lines.append("")
    lines.append("| File | Rows | Columns | Schema Valid | Size (MB) |")
    lines.append("|------|------|---------|-------------|-----------|")
    for s in test_stats:
        sz = file_size_mb(TEST_DIR / s["file"])
        cols = ", ".join(s["columns"])
        lines.append(f"| {s['file']} | {s['rows']:,} | {cols} | {'✅' if s['schema_valid'] else '❌'} | {sz:.1f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Schema Validation ----
    lines.append("## 2. Schema Validation")
    lines.append("")
    all_valid = all(s["schema_valid"] for s in train_stats + test_stats)
    all_valid = all_valid and gt_stats["schema_valid"]
    if all_valid:
        lines.append("✅ All files match their expected schemas.")
    else:
        lines.append("❌ Some files have unexpected schemas. Check columns carefully.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- ID Uniqueness ----
    lines.append("## 3. ID Uniqueness")
    lines.append("")
    lines.append("| Source | Rows | Unique IDs | Duplicates | Prefixes |")
    lines.append("|--------|------|-----------|------------|----------|")
    for s in train_stats + test_stats:
        prefixes = ", ".join(f"{k}: {v:,}" for k, v in s["id_prefixes"].items())
        lines.append(f"| {s['label']} | {s['rows']:,} | {s['unique_ids']:,} | {s['duplicate_ids']:,} | {prefixes} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Null/Empty Analysis ----
    lines.append("## 4. Missing / Empty Values")
    lines.append("")
    lines.append("| Source | entity_id | business_name | business_address | country |")
    lines.append("|--------|-----------|---------------|------------------|---------|")
    for s in train_stats + test_stats:
        ne = s["null_or_empty"]
        lines.append(f"| {s['label']} | {ne.get('entity_id', 0)} | {ne.get('business_name', 0)} | {ne.get('business_address', 0)} | {ne.get('country', 0)} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Country Distribution ----
    lines.append("## 5. Country Distribution")
    lines.append("")
    for s in train_stats + test_stats:
        lines.append(f"### {s['label']}")
        lines.append("")
        lines.append("| Country | Count | Percentage |")
        lines.append("|---------|-------|------------|")
        total = s["rows"]
        for country, count in sorted(s["country_distribution"].items(), key=lambda x: -x[1]):
            pct = round(100.0 * count / total, 2) if total > 0 else 0
            lines.append(f"| {country} | {count:,} | {pct}% |")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Name Length Stats ----
    lines.append("## 6. Business Name Length Statistics")
    lines.append("")
    lines.append("| Source | Min | Max | Mean | Median | P95 | Empty |")
    lines.append("|--------|-----|-----|------|--------|-----|-------|")
    for s in train_stats + test_stats:
        nl = s["name_length"]
        lines.append(f"| {s['label']} | {nl['min']} | {nl['max']} | {nl['mean']} | {nl['median']} | {nl['p95']} | {s['name_empty_count']} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Address Length Stats ----
    lines.append("## 7. Address Length Statistics")
    lines.append("")
    lines.append("| Source | Min | Max | Mean | Median | P95 | Empty |")
    lines.append("|--------|-----|-----|------|--------|-----|-------|")
    for s in train_stats + test_stats:
        al = s["address_length"]
        lines.append(f"| {s['label']} | {al['min']} | {al['max']} | {al['mean']} | {al['median']} | {al['p95']} | {s['address_empty_count']} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Sample Names ----
    lines.append("## 8. Sample Business Names")
    lines.append("")
    for s in train_stats:
        lines.append(f"### {s['label']}")
        lines.append("")
        lines.append("```")
        for name in s["sample_names"]:
            lines.append(name)
        lines.append("```")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Sample Addresses ----
    lines.append("## 9. Sample Addresses")
    lines.append("")
    for s in train_stats:
        lines.append(f"### {s['label']}")
        lines.append("")
        lines.append("```")
        for addr in s["sample_addresses"]:
            lines.append(addr)
        lines.append("```")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Ground Truth ----
    lines.append("## 10. Ground Truth Analysis")
    lines.append("")
    lines.append(f"- **Total Source 1 entities in GT:** {gt_stats['total_s1_entities']:,}")
    lines.append(f"- **Unique S1 IDs:** {gt_stats['unique_s1_ids']:,}")
    lines.append(f"- **Duplicate S1 IDs:** {gt_stats['duplicate_s1_ids']:,}")
    lines.append(f"- **Singleton (zero match) count:** {gt_stats['singleton_count']:,} ({gt_stats['singleton_pct']}%)")
    lines.append(f"- **One-to-many (>1 match) count:** {gt_stats['one_to_many_count']:,}")
    lines.append(f"- **S1→S2 total match pairs:** {gt_stats['s1_to_s2_match_count']:,}")
    lines.append(f"- **S1→S3 total match pairs:** {gt_stats['s1_to_s3_match_count']:,}")
    lines.append(f"- **Total match pairs:** {gt_stats['total_match_pairs']:,}")
    lines.append(f"- **Max matches for a single S1:** {gt_stats['max_matches_for_single_s1']} (entity: `{gt_stats['max_matches_s1_id']}`)")
    lines.append("")

    lines.append("### Match Count Distribution")
    lines.append("")
    lines.append(f"- Min: {gt_stats['match_count_distribution']['min']}")
    lines.append(f"- Max: {gt_stats['match_count_distribution']['max']}")
    lines.append(f"- Mean: {gt_stats['match_count_distribution']['mean']}")
    lines.append(f"- Median: {gt_stats['match_count_distribution']['median']}")
    lines.append(f"- P95: {gt_stats['match_count_distribution']['p95']}")
    lines.append("")

    lines.append("### Match Count Histogram")
    lines.append("")
    lines.append("| Matches | S1 Entity Count |")
    lines.append("|---------|-----------------|")
    for k, v in sorted(gt_stats["match_count_histogram"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {k} | {v:,} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Cross-Referencing ----
    lines.append("## 11. Ground Truth Cross-Reference Integrity")
    lines.append("")
    lines.append(f"- GT S1 IDs missing from train_source1: **{gt_stats['gt_s1_ids_missing_from_source1']}**")
    lines.append(f"- train_source1 IDs missing from GT: **{gt_stats['source1_ids_missing_from_gt']}**")
    lines.append(f"- Matched S2 IDs not in train_source2: **{gt_stats['matched_s2_not_in_source2']}**")
    lines.append(f"- Matched S3 IDs not in train_source3: **{gt_stats['matched_s3_not_in_source3']}**")
    lines.append("")

    if (gt_stats['gt_s1_ids_missing_from_source1'] == 0 and
        gt_stats['source1_ids_missing_from_gt'] == 0 and
        gt_stats['matched_s2_not_in_source2'] == 0 and
        gt_stats['matched_s3_not_in_source3'] == 0):
        lines.append("✅ All ground-truth references are valid.")
    else:
        lines.append("⚠️ Some ground-truth references are invalid — investigate before training.")
        if gt_stats.get("missing_from_s1_examples"):
            lines.append(f"  Missing S1 examples: {gt_stats['missing_from_s1_examples']}")
        if gt_stats.get("missing_s2_examples"):
            lines.append(f"  Missing S2 examples: {gt_stats['missing_s2_examples']}")
        if gt_stats.get("missing_s3_examples"):
            lines.append(f"  Missing S3 examples: {gt_stats['missing_s3_examples']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Test Set Overview ----
    lines.append("## 12. Test Set Summary")
    lines.append("")
    for s in test_stats:
        countries = ", ".join(f"{k}: {v:,}" for k, v in s["country_distribution"].items())
        lines.append(f"- **{s['label']}:** {s['rows']:,} records — Countries: {countries}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*End of data profile.*")

    return "\n".join(lines)


def main():
    p("=" * 60)
    p("  Business Entity Resolution — Data Audit")
    p("=" * 60)
    p("")

    # Check files exist
    all_files = [f for f, _ in TRAIN_FILES + TEST_FILES] + [TRAIN_GT]
    for f in all_files:
        if not f.exists():
            p(f"ERROR: Required file not found: {f}")
            sys.exit(1)
        p(f"  Found: {f.name} ({file_size_mb(f):.1f} MB)")
    p("")

    # Profile training sources ONE AT A TIME using chunks
    # We need to keep S1/S2/S3 ID sets for GT cross-referencing
    # But we process and free each source's DataFrame chunks as we go
    p("=" * 40)
    p("=== Phase 1: Training Sources ===")
    p("=" * 40)
    train_stats = []
    id_sets = {}

    for path, label in TRAIN_FILES:
        stats, ids = profile_source_chunked(path, label)
        train_stats.append(stats)
        id_sets[label] = ids
        p(f"    ID set size: {len(ids):,}")
        gc.collect()
        p("")

    # Profile ground truth (line-by-line, low memory)
    p("=" * 40)
    p("=== Phase 2: Ground Truth ===")
    p("=" * 40)
    gt_stats = profile_ground_truth(
        TRAIN_GT,
        id_sets["Train Source 1"],
        id_sets["Train Source 2"],
        id_sets["Train Source 3"],
    )
    p("")

    # Free training ID sets (can be very large for S2/S3)
    del id_sets
    gc.collect()

    # Profile test sources ONE AT A TIME
    p("=" * 40)
    p("=== Phase 3: Test Sources ===")
    p("=" * 40)
    test_stats = []
    for path, label in TEST_FILES:
        stats, ids = profile_source_chunked(path, label)
        test_stats.append(stats)
        del ids
        gc.collect()
        p("")

    # Generate report
    p("=" * 40)
    p("=== Phase 4: Generating Reports ===")
    p("=" * 40)
    md = generate_markdown(train_stats, test_stats, gt_stats)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / "data_profile.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    p(f"  Saved: {md_path}")

    # Save machine-readable stats
    all_stats = {
        "train_sources": train_stats,
        "test_sources": test_stats,
        "ground_truth": gt_stats,
    }
    json_path = REPORTS_DIR / "data_profile_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)
    p(f"  Saved: {json_path}")

    p("")
    p("=" * 60)
    p("  [OK] Data Audit Complete")
    p("=" * 60)


if __name__ == "__main__":
    main()
