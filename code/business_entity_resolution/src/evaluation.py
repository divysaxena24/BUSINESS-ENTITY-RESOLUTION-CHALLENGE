"""
Evaluation module containing the official F0.5 calculation for the competition.
"""

def compute_f05_macro(ground_truth_dict, predictions_dict):
    """
    Compute macro-averaged F0.5 score over S1 entities.

    ground_truth_dict: { s1_id: set([matched_sx_ids]) }
    predictions_dict: { s1_id: set([predicted_sx_ids]) }
    """
    f05_scores = []
    
    # Evaluate over ALL ground truth S1 IDs (including singletons)
    for s1_id, true_matches in ground_truth_dict.items():
        pred_matches = predictions_dict.get(s1_id, set())
        
        tp = len(true_matches & pred_matches)
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        
        # Exact competition edge cases
        if tp == 0 and fp == 0 and fn == 0:
            # True negative (correctly predicted singleton)
            f05 = 1.0
        elif tp == 0:
            # False positive on singleton, or zero true positives
            f05 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            # F0.5 formula: (1 + beta^2) * P * R / (beta^2 * P + R) where beta = 0.5
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            
        f05_scores.append(f05)
        
    macro_f05 = sum(f05_scores) / len(f05_scores) if f05_scores else 0.0
    return macro_f05

def compute_detailed_metrics(ground_truth_dict, predictions_dict):
    """Compute F0.5, Precision, and Recall macro-averaged."""
    f05_scores = []
    precision_scores = []
    recall_scores = []
    
    singleton_fp = 0
    singleton_count = 0
    
    for s1_id, true_matches in ground_truth_dict.items():
        pred_matches = predictions_dict.get(s1_id, set())
        
        tp = len(true_matches & pred_matches)
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        
        is_singleton = (len(true_matches) == 0)
        if is_singleton:
            singleton_count += 1
            if fp > 0:
                singleton_fp += 1
        
        if tp == 0 and fp == 0 and fn == 0:
            f05, p, r = 1.0, 1.0, 1.0
        elif tp == 0:
            f05, p, r = 0.0, 0.0, 0.0
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f05 = (1.25 * p * r) / (0.25 * p + r)
            
        f05_scores.append(f05)
        precision_scores.append(p)
        recall_scores.append(r)
        
    n = len(f05_scores)
    
    return {
        "macro_f05": sum(f05_scores) / n if n else 0.0,
        "macro_precision": sum(precision_scores) / n if n else 0.0,
        "macro_recall": sum(recall_scores) / n if n else 0.0,
        "singleton_fp_rate": singleton_fp / singleton_count if singleton_count else 0.0
    }
