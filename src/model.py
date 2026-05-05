import os
import sqlite3
import pandas as pd
import joblib
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from config import (
    DB_PATH,
    SABDAB_SUMMARY_TSV,
    BULK_FEATURE_COLS,
    CDR_FEATURE_COLS,
    FEATURE_COLS,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    clean_antigen_target,
)


# ---------------------------------------------------------------------------
# CDR feature engineering
# ---------------------------------------------------------------------------

def _safe_cdr_features(seq) -> tuple[int, float, float]:
    """
    Compute length, GRAVY index, and net charge at pH 7 for a CDR sequence.
    Returns (0, 0.0, 0.0) for missing/invalid sequences so the row is kept.
    """
    if pd.isna(seq) or str(seq).strip() in ('', 'NA', 'nan'):
        return 0, 0.0, 0.0
    clean = str(seq).replace('X', '').upper()
    if not clean:
        return 0, 0.0, 0.0
    analysis = ProteinAnalysis(clean)
    return len(clean), analysis.gravy(), analysis.charge_at_pH(7.0)


def add_cdr_features(df: pd.DataFrame, tsv_path: str) -> pd.DataFrame:
    """
    Merge CDR3 sequences from the SAbDab TSV into df and compute per-CDR3
    biophysical features. Operates on a copy so the original is not mutated.

    SAbDab CDR3 columns used: 'cdrh3', 'cdrl3'
    These represent the antibody's antigen-binding loops and are far more
    informative for antigen-type classification than whole-chain bulk properties.
    """
    tsv = pd.read_csv(tsv_path, sep='\t')

    # SAbDab may have multiple rows per PDB (one per chain pair); keep one.
    cdr_cols = ['pdb']
    for col in ['cdrh3', 'cdrl3']:
        if col in tsv.columns:
            cdr_cols.append(col)
        else:
            print(f"  Warning: '{col}' column not found in TSV — filling with NaN.")
            tsv[col] = float('nan')
            cdr_cols.append(col)

    tsv_cdr = tsv[cdr_cols].drop_duplicates(subset=['pdb'])

    df = df.copy()
    df = df.merge(tsv_cdr, left_on='pdb_id', right_on='pdb', how='left')

    # Compute features for heavy CDR3
    h_feats = df['cdrh3'].apply(_safe_cdr_features)
    df['cdrh3_length'] = h_feats.apply(lambda x: x[0])
    df['cdrh3_gravy']  = h_feats.apply(lambda x: x[1])
    df['cdrh3_charge'] = h_feats.apply(lambda x: x[2])

    # Compute features for light CDR3
    l_feats = df['cdrl3'].apply(_safe_cdr_features)
    df['cdrl3_length'] = l_feats.apply(lambda x: x[0])
    df['cdrl3_gravy']  = l_feats.apply(lambda x: x[1])
    df['cdrl3_charge'] = l_feats.apply(lambda x: x[2])

    return df


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model():
    # ------------------------------------------------------------------
    # 1. Load & merge data
    # ------------------------------------------------------------------
    print("1. Loading data...")

    conn = sqlite3.connect(DB_PATH)
    df_features = pd.read_sql_query("SELECT * FROM antibodies", conn)
    conn.close()

    df_labels = pd.read_csv(SABDAB_SUMMARY_TSV, sep='\t')
    df_labels = df_labels[['pdb', 'antigen_type']].dropna(subset=['antigen_type']).copy()
    df_labels['target'] = df_labels['antigen_type'].apply(clean_antigen_target)

    # Drop multi-antigen entries (clean_antigen_target returns None for these)
    df_labels = df_labels.dropna(subset=['target'])
    df_labels = df_labels[['pdb', 'target']].drop_duplicates(subset=['pdb'])

    df = pd.merge(
        df_features,
        df_labels,
        left_on='pdb_id',
        right_on='pdb',
        how='inner',
        validate='one_to_one',
    )

    # ------------------------------------------------------------------
    # 2. Add CDR-derived features from TSV sequences
    # ------------------------------------------------------------------
    print("2. Computing CDR3 biophysical features...")
    df = add_cdr_features(df, SABDAB_SUMMARY_TSV)

    print(f"\n   Dataset ready: {len(df)} antibodies across {df['target'].nunique()} antigen classes.")
    print(f"   Classes: {df['target'].unique().tolist()}")
    print(f"   Features used: {FEATURE_COLS}\n")

    X = df[FEATURE_COLS]
    y = df['target']

    # ------------------------------------------------------------------
    # 3. Cross-validation
    #    With ~500 samples, a single 80/20 split has high variance.
    #    5-fold stratified CV gives a much more reliable accuracy estimate.
    # ------------------------------------------------------------------
    print("3. Cross-validating...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        class_weight='balanced',
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
    print(f"   Per-fold:    {[f'{s*100:.1f}%' for s in cv_scores]}\n")

    # ------------------------------------------------------------------
    # 4. Final train/test split for the held-out classification report
    # ------------------------------------------------------------------
    print("4. Training final model on 80% split for detailed evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    rf_model.fit(X_train, y_train)

    predictions = rf_model.predict(X_test)
    print(f"   Hold-out Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
    print(classification_report(y_test, predictions, zero_division=0))

    # ------------------------------------------------------------------
    # 5. Feature importance
    # ------------------------------------------------------------------
    print("\n--- Feature Importance ---")
    importances = sorted(
        zip(FEATURE_COLS, rf_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, importance in importances:
        bar = '█' * int(importance * 40)
        print(f"  {name:<20} {importance:.3f}  {bar}")

    # ------------------------------------------------------------------
    # 6. Save model
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf_model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()