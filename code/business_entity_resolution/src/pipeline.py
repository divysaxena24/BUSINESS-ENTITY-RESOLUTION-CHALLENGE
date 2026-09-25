"""
Main pipeline execution script. Handles train/val splitting, feature matrix creation,
model training, threshold tuning, and inference.
"""
import sys
import gc
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from . import config
from .blocking import load_s1_index, generate_candidates_chunked
from .features import compute_pair_features
from .evaluation import compute_detailed_metrics

def p(msg):
    print(msg, flush=True)

def load_ground_truth(gt_path, valid_s1_ids=None):
    """Load ground truth into {s1_id: set(sx_ids)}."""
    p(f"Loading Ground Truth from {gt_path.name}...")
    gt = defaultdict(set)
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f) # skip header
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            s1_id = parts[0]
            if valid_s1_ids is not None and s1_id not in valid_s1_ids:
                continue
            if len(parts) > 1 and parts[1]:
                matches = set(x.strip() for x in parts[1].split(",") if x.strip())
                gt[s1_id] = matches
            else:
                gt[s1_id] = set() # True singleton
    return gt

def build_feature_matrix(candidate_pairs, s1_data, s1_split_ids, ground_truth=None):
    """
    Given candidate pairs, generate features X and labels y.
    Only includes pairs where s1_id is in s1_split_ids.
    """
    X, y, pair_info = [], [], []
    
    for s1_id, sx_id, sx_norm_name, sx_norm_addr, sx_norm_country in candidate_pairs:
        if s1_id not in s1_split_ids:
            continue
            
        s1_norm_name, s1_norm_addr, s1_norm_country = s1_data[s1_id]
        is_s3 = sx_id.startswith("S3")
        
        feats = compute_pair_features(
            s1_norm_name, s1_norm_addr, s1_norm_country,
            sx_norm_name, sx_norm_addr, sx_norm_country,
            is_s3
        )
        
        X.append(feats)
        pair_info.append((s1_id, sx_id))
        
        if ground_truth is not None:
            label = 1 if sx_id in ground_truth.get(s1_id, set()) else 0
            y.append(label)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8) if y else None, pair_info

def run_pipeline(mode="dev"):
    """Run the training/validation baseline pipeline."""
    p("="*60)
    p(f" Starting Baseline Pipeline (Mode: {mode})")
    p("="*60)
    
    # 1. Dev Split
    dev_s1_ids = None
    if config.DEV_S1_COUNT:
        # Load exactly DEV_S1_COUNT IDs from GT to ensure we get a mix of singletons and matches
        all_gt = load_ground_truth(config.TRAIN_GT)
        all_s1 = list(all_gt.keys())
        random.seed(config.RANDOM_SEED)
        dev_s1_ids = set(random.sample(all_s1, min(config.DEV_S1_COUNT, len(all_s1))))
        p(f"DEV MODE: Subsampled {len(dev_s1_ids):,} S1 IDs.")
        del all_gt, all_s1
    
    # 2. Train/Val Split at S1 level
    s1_list = list(dev_s1_ids) if dev_s1_ids else list(load_ground_truth(config.TRAIN_GT).keys())
    random.seed(config.RANDOM_SEED)
    random.shuffle(s1_list)
    val_size = int(len(s1_list) * config.VAL_FRACTION)
    val_s1_ids = set(s1_list[:val_size])
    train_s1_ids = set(s1_list[val_size:])
    
    p(f"Split: {len(train_s1_ids):,} Train S1, {len(val_s1_ids):,} Val S1")
    
    # Load limited GT for val/train sets
    gt_train = load_ground_truth(config.TRAIN_GT, train_s1_ids)
    gt_val = load_ground_truth(config.TRAIN_GT, val_s1_ids)

    # 3. Blocking
    p("\n--- Phase 1: Blocking ---")
    s1_data, name_index, token_index = load_s1_index(config.TRAIN_S1, dev_s1_ids)
    
    candidate_pairs = []
    # Process S2
    s2_pairs, s1_cands_s2 = generate_candidates_chunked(
        config.TRAIN_S2, s1_data, name_index, token_index, "Train Source 2"
    )
    candidate_pairs.extend(s2_pairs)
    del s2_pairs, s1_cands_s2; gc.collect()
    
    # Process S3
    s3_pairs, s1_cands_s3 = generate_candidates_chunked(
        config.TRAIN_S3, s1_data, name_index, token_index, "Train Source 3"
    )
    candidate_pairs.extend(s3_pairs)
    del s3_pairs, s1_cands_s3; gc.collect()
    
    p(f"Total Candidate Pairs generated: {len(candidate_pairs):,}")
    
    # Measure blocking recall
    def measure_recall(gt_dict, cand_pairs, split_ids):
        # build cand dict: {s1: set(sx)}
        cands = defaultdict(set)
        for s1_id, sx_id, _, _, _ in cand_pairs:
            if s1_id in split_ids:
                cands[s1_id].add(sx_id)
        
        total_true = 0
        total_found = 0
        for s1_id, true_matches in gt_dict.items():
            total_true += len(true_matches)
            total_found += len(true_matches & cands[s1_id])
            
        recall = total_found / total_true if total_true else 0.0
        return recall, total_found, total_true, cands
        
    train_rec, f_tr, t_tr, train_cands = measure_recall(gt_train, candidate_pairs, train_s1_ids)
    val_rec, f_v, t_v, val_cands = measure_recall(gt_val, candidate_pairs, val_s1_ids)
    
    p(f"Train Blocking Recall: {train_rec:.4f} ({f_tr:,}/{t_tr:,})")
    p(f"Val Blocking Recall:   {val_rec:.4f} ({f_v:,}/{t_v:,})")
    
    # 4. Feature Gen
    p("\n--- Phase 2: Feature Engineering ---")
    p("Building Train Feature Matrix...")
    X_train, y_train, _ = build_feature_matrix(candidate_pairs, s1_data, train_s1_ids, gt_train)
    p(f"X_train shape: {X_train.shape}, y_train pos %: {(y_train.sum() / len(y_train)*100):.2f}%")
    
    p("Building Val Feature Matrix...")
    X_val, y_val, val_info = build_feature_matrix(candidate_pairs, s1_data, val_s1_ids, gt_val)
    p(f"X_val shape: {X_val.shape}")
    
    # Clean memory
    del candidate_pairs; gc.collect()
    
    # 5. Model Training
    p("\n--- Phase 3: Model Training (GBDT) ---")
    model = GradientBoostingClassifier(
        n_estimators=config.MODEL_N_ESTIMATORS,
        max_depth=config.MODEL_MAX_DEPTH,
        learning_rate=config.MODEL_LEARNING_RATE,
        min_samples_leaf=config.MODEL_MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_SEED
    )
    model.fit(X_train, y_train)
    
    # Feature importance
    p("\nFeature Importances:")
    importances = model.feature_importances_
    for name, imp in sorted(zip(config.FEATURE_NAMES, importances), key=lambda x: -x[1]):
        p(f"  {name:20s} {imp:.4f}")
        
    # 6. Threshold Sweep on Val
    p("\n--- Phase 4: Validation & Threshold Sweep ---")
    y_val_probs = model.predict_proba(X_val)[:, 1]
    
    best_f05 = -1
    best_thresh = 0.5
    best_metrics = {}
    
    for thresh in np.arange(config.THRESHOLD_MIN, config.THRESHOLD_MAX, config.THRESHOLD_STEP):
        preds = y_val_probs >= thresh
        # Build prediction dict
        pred_dict = defaultdict(set)
        for i, (s1_id, sx_id) in enumerate(val_info):
            if preds[i]:
                pred_dict[s1_id].add(sx_id)
                
        metrics = compute_detailed_metrics(gt_val, pred_dict)
        p(f"Thresh {thresh:.2f} -> F0.5: {metrics['macro_f05']:.4f} | Prec: {metrics['macro_precision']:.4f} | Rec: {metrics['macro_recall']:.4f}")
        
        if metrics['macro_f05'] > best_f05:
            best_f05 = metrics['macro_f05']
            best_thresh = thresh
            best_metrics = metrics
            
    p(f"\nBest Validation Threshold: {best_thresh:.2f}")
    p(f"Best Validation F0.5:      {best_metrics['macro_f05']:.4f}")
    
    # Save a report artifact
    report = {
        "dev_mode": mode == "dev",
        "train_s1_count": len(train_s1_ids),
        "val_s1_count": len(val_s1_ids),
        "blocking_recall_val": val_rec,
        "features": {name: float(imp) for name, imp in zip(config.FEATURE_NAMES, importances)},
        "best_threshold": float(best_thresh),
        "best_macro_f05": float(best_metrics['macro_f05']),
        "best_macro_precision": float(best_metrics['macro_precision']),
        "best_macro_recall": float(best_metrics['macro_recall']),
        "singleton_fp_rate": float(best_metrics['singleton_fp_rate'])
    }
    
    report_path = config.REPORTS_DIR / "pipeline_run.json"
    config.REPORTS_DIR.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    p(f"Done. Report saved to {report_path}")

if __name__ == "__main__":
    run_pipeline("dev")
