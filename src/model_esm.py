"""
model_esm.py — Drop-in replacement for model.py that uses ESM-2 embeddings.

Run embed.py first to generate models/esm_embeddings.npz, then run this.

Three feature modes are compared automatically:
  - esm_only         : ESM embeddings alone (640-dim)
  - physchem_only    : Your original 24 hand-crafted features
  - esm_plus_physchem: Both concatenated (664-dim) ← usually best

Usage:
    cd <project root>
    python src/model_esm.py
"""

import json
import os
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    BULK_FEATURE_COLS,
    CDR3_FEATURE_COLS,
    DB_PATH,
    DERIVED_FEATURE_COLS,
    RANDOM_SEED,
    SABDAB_SUMMARY_TSV,
    TEST_SIZE,
    clean_antigen_target,
)
from model import add_derived_features  # reuse your existing derived-feature logic

EMBED_PATH = Path("models") / "esm_embeddings.npz"
ALL_PHYSCHEM = BULK_FEATURE_COLS + CDR3_FEATURE_COLS + DERIVED_FEATURE_COLS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load pre-computed ESM-2 embeddings."""
    if not EMBED_PATH.exists():
        raise FileNotFoundError(
            f"{EMBED_PATH} not found. Run `python src/embed.py` first."
        )
    data = np.load(EMBED_PATH, allow_pickle=True)
    embeddings = data["embeddings"]           # (N, 640)
    pdb_ids    = data["pdb_ids"].tolist()     # list of N strings
    print(f"Loaded embeddings: {embeddings.shape} for {len(pdb_ids)} antibodies.")
    return embeddings, pdb_ids


def load_labels_and_physchem(pdb_ids: list[str]) -> pd.DataFrame:
    """
    Load physicochemical features + antigen-type labels for the given PDB IDs.
    Returns a DataFrame indexed by pdb_id, in the same order as pdb_ids.
    """
    conn = sqlite3.connect(DB_PATH)
    df_features = pd.read_sql_query("SELECT * FROM antibodies", conn)
    conn.close()

    df_labels = pd.read_csv(SABDAB_SUMMARY_TSV, sep="\t")
    df_labels = df_labels[["pdb", "antigen_type"]].dropna(subset=["antigen_type"])
    df_labels["target"] = df_labels["antigen_type"].apply(clean_antigen_target)
    df_labels = df_labels.dropna(subset=["target"])[["pdb", "target"]].drop_duplicates("pdb")

    df = pd.merge(df_features, df_labels, left_on="pdb_id", right_on="pdb", how="inner")
    df = add_derived_features(df)

    # Keep only rows whose PDB IDs appear in the embedding file, in the same order
    df = df.set_index("pdb_id").reindex(pdb_ids).dropna(subset=["target"])
    return df


# ---------------------------------------------------------------------------
# Feature matrix builders
# ---------------------------------------------------------------------------

def get_physchem_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[ALL_PHYSCHEM].fillna(0.0).values.astype(np.float32)


def build_feature_sets(
    embeddings: np.ndarray,
    df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Build the three feature matrices we'll compare.
    All rows are already aligned (same order, same index).
    """
    physchem = get_physchem_matrix(df)
    return {
        "esm_only"         : embeddings,
        "physchem_only"    : physchem,
        "esm_plus_physchem": np.concatenate([embeddings, physchem], axis=1),
    }


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

def build_candidates(random_seed: int) -> dict:
    """
    For ESM embeddings, LogReg + StandardScaler is often surprisingly competitive.
    RF handles high-dimensional input well without scaling.
    Adding a small MLP-style option via GradientBoosting is also worth trying.
    """
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",    # important for high-dim data
            random_state=random_seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "logistic_regression": make_pipeline(
            # PCA first: reduces 640-dim embeddings to 50 components before LR.
            # Captures ~90% of variance typically and prevents overfitting
            # on 461 samples.
            StandardScaler(),
            PCA(n_components=50, random_state=random_seed),
            LogisticRegression(
                max_iter=5000,
                C=1.0,
                random_state=random_seed,
                class_weight="balanced",
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def cv_score(model, X: np.ndarray, y, random_seed: int) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_seed)
    scores = cross_val_score(clone(model), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return {
        "cv_mean": float(scores.mean()),
        "cv_std" : float(scores.std()),
        "cv_min" : float(scores.min()),
        "cv_max" : float(scores.max()),
    }


def compare_feature_sets(
    feature_sets: dict[str, np.ndarray],
    y,
    random_seed: int,
) -> tuple[pd.DataFrame, str, np.ndarray]:
    """Score each feature set with RF and pick the best."""
    rf = RandomForestClassifier(
        n_estimators=200, random_state=random_seed, class_weight="balanced", n_jobs=-1
    )
    rows = []
    best_name, best_X, best_score = None, None, -1.0
    for name, X in feature_sets.items():
        stats = cv_score(rf, X, y, random_seed)
        rows.append({"feature_set": name, "n_features": X.shape[1], **stats})
        if stats["cv_mean"] > best_score:
            best_score, best_name, best_X = stats["cv_mean"], name, X
    result = pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)
    return result, best_name, best_X


def compare_models(
    X_train: np.ndarray,
    y_train,
    random_seed: int,
) -> tuple[pd.DataFrame, str, object]:
    candidates = build_candidates(random_seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    rows = []
    best_name, best_model, best_score = None, None, -1.0
    for name, model in candidates.items():
        scores = cross_val_score(clone(model), X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        stats = {
            "model"   : name,
            "cv_mean" : float(scores.mean()),
            "cv_std"  : float(scores.std()),
            "cv_min"  : float(scores.min()),
            "cv_max"  : float(scores.max()),
        }
        rows.append(stats)
        if stats["cv_mean"] > best_score:
            best_score, best_name, best_model = stats["cv_mean"], name, clone(model)
    result = pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)
    return result, best_name, best_model


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def train_esm_model() -> None:
    print("=== ESM-2 Antibody Classifier ===\n")

    # 1. Load data
    embeddings, pdb_ids = load_embeddings()
    df = load_labels_and_physchem(pdb_ids)

    # Align embeddings to the rows that survived the label merge
    embed_index = {pid: i for i, pid in enumerate(pdb_ids)}
    row_indices  = [embed_index[pid] for pid in df.index]
    embeddings   = embeddings[row_indices]

    y = df["target"]
    mask = y != "Hapten"
    embeddings, df, y = embeddings[mask], df[mask], y[mask]
    print(f"After dropping Hapten: {len(df)} samples")
    print(f"\nDataset: {len(df)} antibodies, {y.nunique()} classes")
    print(y.value_counts().to_string())
    print()

    # 2. Compare feature sets
    print("1. Comparing feature sets (RF, 5×3 CV)...")
    feature_sets = build_feature_sets(embeddings, df)
    feat_table, best_feat_name, X_best = compare_feature_sets(feature_sets, y, RANDOM_SEED)
    print(feat_table.assign(
        cv_mean=lambda f: (f["cv_mean"] * 100).round(2),
        cv_std =lambda f: (f["cv_std"]  * 100).round(2),
    )[["feature_set", "n_features", "cv_mean", "cv_std"]].to_string(index=False))
    print(f"\n  → Best feature set: {best_feat_name}\n")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_best, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # 4. Compare models
    print("2. Comparing models (5-fold CV on training split)...")
    model_table, best_model_name, best_model = compare_models(X_train, y_train, RANDOM_SEED)
    print(model_table.assign(
        cv_mean=lambda f: (f["cv_mean"] * 100).round(2),
        cv_std =lambda f: (f["cv_std"]  * 100).round(2),
    )[["model", "cv_mean", "cv_std"]].to_string(index=False))
    print(f"\n  → Best model: {best_model_name}\n")

    # 5. Final evaluation on held-out test set
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    holdout_acc = accuracy_score(y_test, predictions)

    print("3. Hold-out evaluation")
    print(f"   Accuracy: {holdout_acc * 100:.2f}%\n")
    print(classification_report(y_test, predictions, zero_division=0))

    # 6. Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/rf_esm_classifier.joblib")

    report = {
        "model"           : best_model_name,
        "feature_set"     : best_feat_name,
        "holdout_accuracy": float(holdout_acc),
        "feature_set_comparison": feat_table.to_dict(orient="records"),
        "model_comparison"      : model_table.to_dict(orient="records"),
        "classification_report" : classification_report(y_test, predictions, zero_division=0, output_dict=True),
    }
    with open("models/last_esm_experiment.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Model saved → models/rf_esm_classifier.joblib")
    print("Report saved → models/last_esm_experiment.json")


if __name__ == "__main__":
    train_esm_model()
