"""Evaluate organism-specific XGBoost models with staged scaffold CV."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

sys.path.append(os.getcwd())

from script.utils.data_load import get_features_array


# Each phase retains the winning parameters from all preceding phases.
PHASE_GRIDS = (
    {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]},
    {"max_depth": [2, 3, 4, 5, 6, 7, 9], "min_child_weight": [1, 3, 5, 10]},
    {"subsample": [0.4, 0.6, 0.8, 1.0], "colsample_bytree": [0.4, 0.6, 0.8, 1.0]},
    {"gamma": [0.0, 0.1, 0.25, 1.0], "reg_lambda": [0.1, 0.25, 1.0, 10.0],
     "reg_alpha": [0.0, 0.1, 0.25, 1.0]},
)
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50


def r2_metric(predictions: np.ndarray, data: xgb.DMatrix) -> tuple[str, float]:
    """Native XGBoost metric; larger values are better."""
    return "r2", float(r2_score(data.get_label(), predictions))


@dataclass
class StagedSearchResult:
    best_estimator_: xgb.XGBRegressor
    best_params_: dict
    best_score_: float
    cv_results_: pd.DataFrame

    normalizer_: StandardScaler | None = None
    normalizer_start: int | None = None
    normalizer_end: int | None = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64).copy()
        if self.normalizer_ is not None:
            selected = slice(self.normalizer_start, self.normalizer_end)
            features[:, selected] = self.normalizer_.transform(features[:, selected])
        return self.best_estimator_.predict(features)


def fit_xgb_by_folds(
    training_data: pd.DataFrame,
    fold_column: str,
    output_column: str,
    features_start: int | str = 5,
    features_end: int | str | None = None,
    phase_grids: tuple[dict, ...] = PHASE_GRIDS,
    n_jobs: int = 1,
    random_state: int = 42,
    normalize_features: bool = False,
    normalizer_start: int | None = None,
    normalizer_end: int | None = None,
) -> StagedSearchResult:

    labels = training_data[fold_column]
    if labels.isna().any():
        raise ValueError("Fold labels must not contain missing values")
    codes, unique_labels = pd.factorize(labels, sort=False)
    folds = [
        (np.flatnonzero(codes != code), np.flatnonzero(codes == code))
        for code in range(len(unique_labels))
    ]
    features = np.asarray(
        get_features_array(training_data, features_start, features_end),
        dtype=np.float64,
    )
    target = training_data[output_column].to_numpy(dtype=np.float64)
    return _fit_xgb_with_folds(
        features, target, folds, phase_grids, n_jobs, random_state,
        normalize_features, normalizer_start, normalizer_end,
    )


def _fit_xgb_with_folds(
    features: np.ndarray,
    target: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    phase_grids: tuple[dict, ...],
    n_jobs: int,
    random_state: int,
    normalize_features: bool = False,
    normalizer_start: int | None = None,
    normalizer_end: int | None = None,
) -> StagedSearchResult:
    """Shared staged tuning and full-data refit for both fold strategies."""
    expected = [set(grid) for grid in PHASE_GRIDS]
    if len(phase_grids) != 4 or any(
        set(grid) != keys for grid, keys in zip(phase_grids, expected)
    ):
        raise ValueError("Provide exactly the four parameter groups in PHASE_GRIDS")
    for fold, (_, validation_indices) in enumerate(folds, start=1):
        validation_target = target[validation_indices]
        if len(validation_target) < 2 or np.ptp(validation_target) == 0:
            raise ValueError(
                f"Fold {fold}: R² requires at least two validation rows "
                "and a nonconstant target"
            )
    selected = slice(normalizer_start, normalizer_end)
    normalized_folds = []
    if normalize_features:
        for train_indices, validation_indices in folds:
            train_features = features[train_indices].copy()
            validation_features = features[validation_indices].copy()
            scaler = StandardScaler()
            train_features[:, selected] = scaler.fit_transform(train_features[:, selected])
            validation_features[:, selected] = scaler.transform(validation_features[:, selected])
            normalized_folds.append((
                xgb.DMatrix(train_features, label=target[train_indices], nthread=n_jobs),
                xgb.DMatrix(validation_features, label=target[validation_indices], nthread=n_jobs),
            ))

    dtrain = xgb.DMatrix(features, label=target, nthread=n_jobs)
    params = {
        "objective": "reg:squarederror", "tree_method": "hist",
        "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 1,
        "subsample": 1.0, "colsample_bytree": 1.0,
        "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0,
    }
    records = []
    for phase, grid in enumerate(phase_grids, start=1):
        print(f"Starting phase {phase}")
        best_r2 = -np.inf
        best_candidate = None
        for candidate in tqdm(ParameterGrid(grid)):
            candidate_params = {**params, **candidate}
            # xgb.cv invokes fpreproc once per fold, in the supplied order.
            prepared = iter(normalized_folds)

            def preprocess_fold(train, validation, parameters):
                train, validation = next(prepared)
                return train, validation, parameters

            history = xgb.cv(
                params={**candidate_params, "nthread": n_jobs,
                        "seed": random_state, "disable_default_eval_metric": 1},
                dtrain=dtrain, folds=folds,
                fpreproc=preprocess_fold if normalize_features else None,
                num_boost_round=N_ESTIMATORS,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                custom_metric=r2_metric, maximize=True,
                shuffle=False, seed=random_state, verbose_eval=False,
            )
            best_round = int(np.argmax(history["test-r2-mean"].to_numpy()))
            r2 = float(history["test-r2-mean"].iloc[best_round])
            records.append({
                "phase": phase, **candidate_params,
                "n_estimators": best_round + 1, "CV R2": r2,
            })
            if np.isfinite(r2) and r2 > best_r2:
                best_r2 = r2
                best_candidate = candidate_params
                best_n_estimators = best_round + 1
        if best_candidate is None:
            raise ValueError(f"Phase {phase} produced no finite CV R²")
        params = best_candidate
        print(f"Phase {phase}: R²={best_r2:.4f}, "
              f"n_estimators={best_n_estimators}, "
              f"parameters={ {key: params[key] for key in grid} }")

    best_params = {**params, "n_estimators": best_n_estimators}
    model = xgb.XGBRegressor(
        **best_params, n_jobs=n_jobs, random_state=random_state,
    )
    final_scaler = None
    final_features = features.copy()
    if normalize_features:
        final_scaler = StandardScaler()
        final_features[:, selected] = final_scaler.fit_transform(features[:, selected])
    model.fit(final_features, target)
    return StagedSearchResult(
        model, best_params, best_r2, pd.DataFrame(records),
        final_scaler, normalizer_start, normalizer_end,
    )