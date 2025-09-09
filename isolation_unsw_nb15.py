#!/usr/bin/env python3
# isolation_forest_nba.py

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest


# --------------------------- Logging -----------------------------------------

def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger("NBA-IForest")


# --------------------------- Config ------------------------------------------

@dataclass
class DatasetPreset:
    """Preset schema hints for common IDS datasets."""
    label_col: str
    normal_label_values: List[str]  # which values indicate normal/benign class (string or numeric cast to str)
    drop_cols: List[str]
    categorical_cols: Optional[List[str]] = None  # if None, auto-detect object/string cols


DATASET_PRESETS: Dict[str, DatasetPreset] = {
    # NSL-KDD (KDDTrain+.txt / KDDTest+.txt) – last two columns often: 'label', 'difficulty'
    "nsl_kdd": DatasetPreset(
        label_col="label",
        normal_label_values=["normal"],
        drop_cols=["difficulty"],
        categorical_cols=["protocol_type", "service", "flag"],
    ),
    # UNSW-NB15 CSV – has 'label' (0 normal, 1 attack) and 'attack_cat' (string)
    "unsw_nb15": DatasetPreset(
        label_col="label",
        normal_label_values=["0", "normal", "Normal"],
        drop_cols=["id"],
        categorical_cols=["proto", "service", "state"],
    ),
    # CICIDS2017 – 'Label' = BENIGN or attack names; many numeric cols
    "cicids2017": DatasetPreset(
        label_col="Label",
        normal_label_values=["BENIGN", "Benign", "0"],
        drop_cols=[],
        categorical_cols=None,  # auto-detect
    ),
}


@dataclass
class TrainConfig:
    """High-level training configuration."""
    dataset_name: Optional[str] = None           # one of DATASET_PRESETS keys or None for generic
    input_csv: str = ""
    sep: Optional[str] = None                    # None => auto-detect by pandas
    sample_rows: Optional[int] = None            # optional row cap for quick experiments
    artifacts_dir: str = "artifacts"
    random_state: int = 42
    n_estimators: int = 300
    max_samples: int | float = 256               # per-tree subsample (int or float in (0,1])
    contamination: Optional[float] = None        # if None and labels exist, inferred from label prevalence
    max_features: int | float = 1.0
    bootstrap: bool = False
    train_on_normals_only: bool = False          # semi-supervised: fit on normal traffic only
    class_balance_clip: Tuple[float, float] = (1e-5, 0.5)  # bounds for inferred contamination


# --------------------------- Data IO -----------------------------------------

class DataLoader:
    """Loads CSV and applies dataset-specific schema hints."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.preset = DATASET_PRESETS.get(cfg.dataset_name or "", None)

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading CSV: {self.cfg.input_csv}")
        df = pd.read_csv(self.cfg.input_csv, sep=self.cfg.sep, engine="python")
        if self.cfg.sample_rows:
            df = df.sample(self.cfg.sample_rows, random_state=self.cfg.random_state)
        logger.debug(f"Raw shape: {df.shape}")
        return df

    def split_features_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        # Apply drop columns if present
        if self.preset:
            for c in self.preset.drop_cols:
                if c in df.columns:
                    df = df.drop(columns=c)
        # Determine label column
        y: Optional[pd.Series] = None
        if self.preset and self.preset.label_col in df.columns:
            y = df[self.preset.label_col]
            X = df.drop(columns=[self.preset.label_col])
        else:
            # Generic/no labels available
            X = df
        logger.debug(f"Post-split shapes -> X: {X.shape}, y: {None if y is None else y.shape}")
        return X, y

    def detect_categoricals(self, X: pd.DataFrame) -> List[str]:
        if self.preset and self.preset.categorical_cols is not None:
            return [c for c in self.preset.categorical_cols if c in X.columns]
        # Auto-detect: object or string dtype
        cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
        logger.debug(f"Auto-detected categoricals: {cats}")
        return cats

    def make_binary_labels(self, y: pd.Series) -> pd.Series:
        """Map dataset labels to {0: normal, 1: attack}. Returns numeric series."""
        if y is None:
            raise ValueError("No label column found to create binary labels.")
        normal_set = set([str(v) for v in (self.preset.normal_label_values if self.preset else ["normal", "BENIGN", "0"])])
        y_bin = y.astype(str).apply(lambda v: 0 if v in normal_set else 1).astype(int)
        return y_bin


# --------------------------- Preprocessing -----------------------------------

class Preprocessor:
    """Builds a robust preprocessing pipeline."""

    def __init__(self, numeric_cols: List[str] | None = None, categorical_cols: List[str] | None = None):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.pipeline: Optional[ColumnTransformer] = None

    def build(self, X: pd.DataFrame) -> ColumnTransformer:
        # Auto-select if not provided
        if self.numeric_cols is None:
            self.numeric_cols = selector(dtype_include=np.number)(X)
        if self.categorical_cols is None:
            self.categorical_cols = selector(dtype_include=object)(X)

        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ]
        )

        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric_cols),
                ("cat", cat_pipe, self.categorical_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,  # allow sparse if many cats
        )
        return self.pipeline

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            self.build(X)
        return self.pipeline.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        return self.pipeline.transform(X)


# --------------------------- Model -------------------------------------------

class IsolationForestModel:
    """Wraps sklearn IsolationForest with NBA-friendly helpers."""
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        max_samples: int | float = 256,
        contamination: Optional[float] = None,
        max_features: int | float = 1.0,
        bootstrap: bool = False,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=-1,
        )
        self.model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        self.model = IsolationForest(**self.params)
        self.model.fit(X)
        return self

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Return 0 for inliers (normal), 1 for outliers (anomaly)."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        # sklearn: 1 = inlier, -1 = outlier
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Higher = more normal; lower (more negative) = more anomalous."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.decision_function(X)


# --------------------------- Evaluation --------------------------------------

class Evaluator:
    """Computes common IDS metrics if labels are available."""
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray] = None) -> Dict:
        # By convention here: anomaly = 1 (positive class)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return {
            "precision_anomaly": precision,
            "recall_anomaly": recall,
            "f1_anomaly": f1,
            "confusion_matrix": cm,
            "classification_report": report,
        }


# --------------------------- Orchestrator ------------------------------------

class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.loader = DataLoader(cfg)
        self.preproc = Preprocessor()
        self.model = IsolationForestModel(
            random_state=cfg.random_state,
            n_estimators=cfg.n_estimators,
            max_samples=cfg.max_samples,
            contamination=cfg.contamination,
            max_features=cfg.max_features,
            bootstrap=cfg.bootstrap,
        )

    def _ensure_artifacts_dir(self) -> None:
        os.makedirs(self.cfg.artifacts_dir, exist_ok=True)

    def _infer_contamination_from_labels(self, y_bin: pd.Series) -> float:
        anomaly_rate = float((y_bin == 1).mean())
        lo, hi = self.cfg.class_balance_clip
        inferred = float(np.clip(anomaly_rate, lo, hi))
        logger.info(f"Inferred contamination from labels: {inferred:.6f} (clip {lo}-{hi})")
        return inferred

    def run(self) -> Dict:
        # 1) Load
        df = self.loader.load()
        # X_df, y_series = self.loader.split_features_labels(df)
        X_df = df.drop(columns=["label"], axis=1, errors="ignore")
        y_series = df["label"]

        # 2) Optional semi-supervised training (fit on normals only)
        train_mask = slice(None)
        if self.cfg.train_on_normals_only and y_series is not None:
            y_bin = self.loader.make_binary_labels(y_series)
            train_mask = (y_bin == 0).values
            logger.info(f"Training on normals only: {train_mask.sum()} rows (of {len(X_df)})")

        # 3) Build & fit preprocessing on training subset, then transform all
        X_train_fit = X_df.iloc[train_mask] if isinstance(train_mask, np.ndarray) else X_df
        X_train = self.preproc.fit_transform(X_train_fit)
        X_all = self.preproc.transform(X_df)

        # 4) If labels present and contamination not set, infer it from prevalence (on all data)
        if y_series is not None and self.cfg.contamination is None:
            y_bin_all = self.loader.make_binary_labels(y_series)
            self.model.params["contamination"] = self._infer_contamination_from_labels(y_bin_all)

        # 5) Fit Isolation Forest
        self.model.fit(X_train)

        # 6) Predict & score
        y_pred_all = self.model.predict_labels(X_all)
        scores_all = self.model.anomaly_scores(X_all)

        # 7) Evaluate (if labels exist)
        metrics: Dict = {}
        import pdb
        pdb.set_trace()
        if y_series is not None:
            y_bin_eval = self.loader.make_binary_labels(y_series)
            metrics = Evaluator.evaluate(y_bin_eval.values, y_pred_all, scores_all)

        # 8) Persist artifacts
        self._ensure_artifacts_dir()
        joblib.dump(self.model, os.path.join(self.cfg.artifacts_dir, "model.joblib"))
        joblib.dump(self.preproc, os.path.join(self.cfg.artifacts_dir, "preprocessor.joblib"))

        if metrics:
            with open(os.path.join(self.cfg.artifacts_dir, "eval_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        summary = {
            "config": asdict(self.cfg),
            "n_samples": len(X_df),
            "n_features_in": X_df.shape[1],
            "has_labels": y_series is not None,
            "metrics": metrics,
        }
        logger.info("Training complete.")
        return summary


# --------------------------- CLI ---------------------------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Isolation Forest for NBA (Network Behavior Analysis)")
    p.add_argument("--dataset_name", type=str, default=None, choices=[None, *DATASET_PRESETS.keys()], help="Preset schema to apply")
    p.add_argument("--input_csv", type=str, required=True, help="Path to CSV (e.g., UNSW-NB15_1.csv, KDDTrain+.txt)")
    p.add_argument("--sep", type=str, default=None, help="CSV delimiter (None = auto-detect by pandas)")
    p.add_argument("--sample_rows", type=int, default=None, help="Optional row cap for quick runs")

    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    p.add_argument("--random_state", type=int, default=42)

    # IsolationForest params
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_samples", type=float, default=256)
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument("--max_features", type=float, default=1.0)
    p.add_argument("--bootstrap", action="store_true")

    p.add_argument("--train_on_normals_only", action="store_true", help="Fit model on normal traffic only (semi-supervised)")

    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv)")
    args = p.parse_args()

    setup_logging(args.verbose)

    return TrainConfig(
        dataset_name=args.dataset_name,
        input_csv=args.input_csv,
        sep=args.sep,
        sample_rows=args.sample_rows,
        artifacts_dir=args.artifacts_dir,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        contamination=args.contamination,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        train_on_normals_only=args.train_on_normals_only,
    )


def main() -> None:
    cfg = parse_args()
    trainer = Trainer(cfg)
    summary = trainer.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
