"""Investment Banking evaluation metrics: PR-AUC, ROC-AUC, Brier Score, Precision@K, and Lift@Top10%."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate_ma_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute M&A origination ranking and calibration metrics:
    - pr_auc (Average Precision)
    - roc_auc
    - brier_score
    - precision_at_K (e.g. K=10, 25, 50)
    - precision_top_10pct, recall_top_10pct, lift_top_10pct
    """
    if k_list is None:
        k_list = [10, 25, 50]

    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_prob, dtype=float)
    n = len(y_arr)
    if n == 0:
        raise ValueError("Cannot evaluate empty prediction arrays.")

    base_rate = float(np.mean(y_arr))
    total_positives = max(int(np.sum(y_arr)), 1)

    order = np.argsort(-p_arr)
    y_sorted = y_arr[order]

    top10_n = max(int(np.ceil(0.10 * n)), 1)
    top10_hits = int(np.sum(y_sorted[:top10_n]))
    precision_top_10pct = float(top10_hits / top10_n)
    recall_top_10pct = float(top10_hits / total_positives)
    lift_top_10pct = float(precision_top_10pct / base_rate) if base_rate > 0 else 0.0

    has_both_classes = len(np.unique(y_arr)) > 1
    roc_auc = float(roc_auc_score(y_arr, p_arr)) if has_both_classes else 0.5
    pr_auc = float(average_precision_score(y_arr, p_arr)) if has_both_classes else base_rate
    brier = float(brier_score_loss(y_arr, np.clip(p_arr, 0.0, 1.0)))

    metrics: Dict[str, float] = {
        "n_samples": float(n),
        "n_positives": float(np.sum(y_arr)),
        "base_rate": round(base_rate, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "precision_top_10pct": round(precision_top_10pct, 4),
        "recall_top_10pct": round(recall_top_10pct, 4),
        "lift_top_10pct": round(lift_top_10pct, 4),
    }

    for k in k_list:
        k_eff = min(max(int(k), 1), n)
        metrics[f"precision_at_{k}"] = round(float(np.sum(y_sorted[:k_eff]) / k_eff), 4)

    return metrics


def compute_decile_lift_table(y_true: np.ndarray, y_prob: np.ndarray, n_deciles: int = 10) -> pd.DataFrame:
    """Build a 10-decile gain & lift table sorted from highest predicted M&A probability (Decile 1) to lowest."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_prob, dtype=float)
    order = np.argsort(-p_arr)
    y_sorted = y_arr[order]
    p_sorted = p_arr[order]

    base_rate = max(float(np.mean(y_arr)), 1e-6)
    total_pos = max(int(np.sum(y_arr)), 1)

    indices = np.array_split(np.arange(len(y_arr)), n_deciles)
    cum_pos = 0
    rows = []
    for dec_idx, idx_group in enumerate(indices, start=1):
        if len(idx_group) == 0:
            continue
        dec_y = y_sorted[idx_group]
        dec_p = p_sorted[idx_group]
        hits = int(np.sum(dec_y))
        cum_pos += hits
        prec = float(hits / len(idx_group))
        lift = float(prec / base_rate)
        cum_recall = float(cum_pos / total_pos)

        rows.append(
            {
                "decile": dec_idx,
                "n_companies": int(len(idx_group)),
                "n_acquired": hits,
                "avg_predicted_prob": round(float(np.mean(dec_p)), 4),
                "actual_ma_rate": round(prec, 4),
                "lift": round(lift, 3),
                "cumulative_recall": round(cum_recall, 4),
            }
        )

    return pd.DataFrame(rows)
