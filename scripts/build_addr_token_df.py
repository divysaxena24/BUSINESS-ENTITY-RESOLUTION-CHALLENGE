#!/usr/bin/env python3
"""
Build Address Token Document Frequency (DF) map across S1, S2, S3.
DF is calculated over S1+S2+S3 (same population as name token DF).
Each token is counted at most once per record.
"""
import sys
import gc
import json
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))
from src import config
from src.normalization import normalize_address, address_tokens

def p(msg):
    print(msg, flush=True)

def main():
    p("="*60)
    p(" Building Address Token Document Frequency (DF)")
    p("="*60)

    df_counter = Counter()

    def process_file(path, label):
        p(f"Processing {label} ({path.name})...")
        total_rows = 0
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=config.CHUNK_SIZE):
            for addr in chunk["business_address"]:
                if not addr or not addr.strip():
                    continue
                toks = address_tokens(normalize_address(addr))
                for t in toks:
                    df_counter[t] += 1
            total_rows += len(chunk)
            if total_rows % 1_000_000 == 0:
                p(f"  ... {total_rows:,} rows")
        p(f"  Finished {label}: {total_rows:,} total rows.")
        return total_rows

    t1 = process_file(config.TRAIN_S1, "Source 1")
    t2 = process_file(config.TRAIN_S2, "Source 2")
    t3 = process_file(config.TRAIN_S3, "Source 3")

    total_docs = t1 + t2 + t3
    p(f"\nTotal documents processed: {total_docs:,}")
    p(f"Unique address tokens:     {len(df_counter):,}")

    # Save
    out_path = config.PROJECT_ROOT / "dataset" / "addr_token_df.json"
    p(f"Saving to {out_path} ...")
    with open(out_path, "w") as f:
        json.dump(dict(df_counter), f)

    # Stats
    most_common = df_counter.most_common(50)
    stats_path = config.PROJECT_ROOT / "dataset" / "addr_token_df_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total_documents": total_docs,
            "unique_tokens": len(df_counter),
            "top_50_tokens": most_common
        }, f, indent=2)

    p("\nTop 20 address tokens:")
    for tok, cnt in most_common[:20]:
        p(f"  {tok:30s} {cnt:>10,}")

    p("Done!")

if __name__ == "__main__":
    main()
