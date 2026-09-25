#!/usr/bin/env python3
"""
Build Document Frequency (DF) map for all normalized name tokens across S1, S2, S3.
"""
import sys
import gc
import json
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))
from src import config
from src.normalization import normalize_name, name_tokens

def p(msg):
    print(msg, flush=True)

def main():
    p("="*60)
    p(" Building Token Document Frequency (DF)")
    p("="*60)
    
    df_counter = Counter()
    
    def process_file(path, label):
        p(f"Processing {label} ({path.name})...")
        total_rows = 0
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=config.CHUNK_SIZE):
            for b_name in chunk["business_name"]:
                # Normalization and tokenization (returns a set of unique tokens per document)
                tokens = name_tokens(normalize_name(b_name))
                for t in tokens:
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
    p(f"Unique tokens found:       {len(df_counter):,}")
    
    # Save to JSON
    out_path = config.PROJECT_ROOT / "dataset" / "token_df.json"
    p(f"Saving to {out_path} ...")
    with open(out_path, "w") as f:
        # Save as a dict {token: count}
        json.dump(dict(df_counter), f)
        
    # Also save some stats for sanity check
    most_common = df_counter.most_common(50)
    stats_path = config.PROJECT_ROOT / "dataset" / "token_df_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total_documents": total_docs,
            "unique_tokens": len(df_counter),
            "top_50_tokens": most_common
        }, f, indent=2)
        
    p("Done!")

if __name__ == "__main__":
    main()
