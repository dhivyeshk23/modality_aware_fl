"""Clinical/demographic modality data source: Dryad Genki CAD Evaluation dataset.

Only symptom/demographic/risk-factor fields are used as features (NOT radiologist
reads, CAD scores, or Xpert results - those are derived from the CXR/lab-test
modalities and would leak the label if used as "clinical-only" input).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "sex",
    "age_group",
    "shi",
    "c",
    "f",
    "wl",
    "ns",
    "c_f_wl_ns",
    "contact",
    "dm",
    "hiv",
    "history_tb",
]
LABEL_COLUMN = "tb_status"


@dataclass
class ClinicalSplit:
    X: np.ndarray
    y: np.ndarray


def load_dryad_table(xlsx_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df = df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN])
    return df


def make_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS].astype(np.float32).to_numpy()
    y = df[LABEL_COLUMN].astype(np.int64).to_numpy()
    return X, y


def load_clinical_clients(
    xlsx_path: str | Path, n_clients: int = 2, seed: int = 42
) -> list[ClinicalSplit]:
    """Split the Dryad cohort into `n_clients` simulated institutions (non-IID by
    random partition - each institution only ever sees its own patients)."""
    df = load_dryad_table(xlsx_path)
    X, y = make_feature_matrix(df)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    chunks = np.array_split(idx, n_clients)
    return [ClinicalSplit(X=X[c], y=y[c]) for c in chunks]


def train_val_test_split(
    split: ClinicalSplit, seed: int = 42
) -> dict[str, ClinicalSplit]:
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        split.X, split.y, test_size=0.3, random_state=seed, stratify=split.y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )
    return {
        "train": ClinicalSplit(X_train, y_train),
        "val": ClinicalSplit(X_val, y_val),
        "test": ClinicalSplit(X_test, y_test),
    }
