"""Class imbalance strategies for rare-event M&A target prediction (~4%-6% positive rate)."""

from typing import Tuple, Union
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def prepare_imbalance_strategy(
    X_train: pd.DataFrame,
    y_train: Union[np.ndarray, pd.Series],
    strategy: str = "scale_pos_weight",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, float]:
    """
    Prepare training features/labels and positive-class weight according to the selected strategy:
    - 'scale_pos_weight' or 'class_weight': keeps training rows intact and returns pos_weight = N_neg / N_pos
    - 'smote': applies SMOTE oversampling strictly on the training fold (pos_weight = 1.0)
    - 'none': unweighted training (pos_weight = 1.0)
    """
    y_arr = np.asarray(y_train, dtype=int)
    n_pos = max(int((y_arr == 1).sum()), 1)
    n_neg = max(int((y_arr == 0).sum()), 1)
    ratio = float(n_neg) / float(n_pos)

    strat = strategy.lower().strip()
    if strat in ("scale_pos_weight", "class_weight", "balanced"):
        return X_train.copy(), y_arr, ratio

    if strat == "smote":
        k_neighbors = min(5, max(n_pos - 1, 1))
        if k_neighbors >= 1 and n_pos >= 3:
            smote = SMOTE(sampling_strategy=0.35, k_neighbors=k_neighbors, random_state=random_state)
            X_res, y_res = smote.fit_resample(X_train, y_arr)
            if not isinstance(X_res, pd.DataFrame):
                X_res = pd.DataFrame(X_res, columns=X_train.columns)
            return X_res, np.asarray(y_res, dtype=int), 1.0
        return X_train.copy(), y_arr, ratio

    if strat == "none":
        return X_train.copy(), y_arr, 1.0

    raise ValueError(f"Unknown imbalance strategy: {strategy}")
