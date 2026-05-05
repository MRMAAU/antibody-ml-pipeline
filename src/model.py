import json
import os
import sqlite3

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    BULK_FEATURE_COLS,
    CDR3_FEATURE_COLS,
    DB_PATH,
    DERIVED_FEATURE_COLS,
    ERROR_ANALYSIS_PATH,
    EXPERIMENT_REPORT_PATH,
    FEATURE_COMPARISON_PATH,
    FEATURE_COLS,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    SABDAB_SUMMARY_TSV,
    TEST_SIZE,
    clean_antigen_target,
)


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_ratio(numerator, denominator):
    numerator = _safe_float(numerator, 0.0)
    denominator = _safe_float(denominator, 0.0)
    return numerator / denominator if denominator else 0.0


def _sequence_fraction(sequence, letters):
    if pd.isna(sequence):
        return 0.0
    clean = str(sequence).replace('X', '').upper()
    if not clean:
        return 0.0
    return sum(clean.count(letter) for letter in letters) / len(clean)


def add_derived_features(df):
    df = df.copy()
    df['chain_length_ratio'] = df.apply(
        lambda row: _safe_ratio(row.get('heavy_chain_length'), row.get('light_chain_length')),
        axis=1,
    )
    df['chain_length_diff'] = df['heavy_chain_length'].fillna(0) - df['light_chain_length'].fillna(0)
    df['heavy_basic_frac'] = df['heavy_sequence'].apply(lambda seq: _sequence_fraction(seq, ('K', 'R', 'H')))
    df['light_basic_frac'] = df['light_sequence'].apply(lambda seq: _sequence_fraction(seq, ('K', 'R', 'H')))
    df['heavy_aromatic_frac'] = df['heavy_sequence'].apply(lambda seq: _sequence_fraction(seq, ('F', 'W', 'Y')))
    df['light_aromatic_frac'] = df['light_sequence'].apply(lambda seq: _sequence_fraction(seq, ('F', 'W', 'Y')))
    df['cdrh3_length'] = df['cdrh3_sequence'].fillna('').astype(str).str.replace('X', '', regex=False).str.len()
    df['cdrl3_length'] = df['cdrl3_sequence'].fillna('').astype(str).str.replace('X', '', regex=False).str.len()
    return df


def build_feature_matrix(df, feature_cols):
    return df[feature_cols].copy().fillna(0.0)


def load_training_frame():
    print('1. Loading data...')

    conn = sqlite3.connect(DB_PATH)
    df_features = pd.read_sql_query('SELECT * FROM antibodies', conn)
    conn.close()

    df_labels = pd.read_csv(SABDAB_SUMMARY_TSV, sep='\t')
    df_labels = df_labels[['pdb', 'antigen_type']].dropna(subset=['antigen_type']).copy()
    df_labels['target'] = df_labels['antigen_type'].apply(clean_antigen_target)
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
    df = add_derived_features(df)
    return df


def evaluate_random_forest_cv(X, y, random_seed):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_seed,
        class_weight='balanced',
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores


def compare_feature_sets(df, y, random_seed):
    feature_sets = {
        'bulk_only': BULK_FEATURE_COLS,
        'bulk_plus_cdr3': FEATURE_COLS,
        'bulk_plus_cdr3_plus_derived': FEATURE_COLS + DERIVED_FEATURE_COLS,
    }
    rows = []
    for name, features in feature_sets.items():
        scores = evaluate_random_forest_cv(build_feature_matrix(df, features), y, random_seed)
        rows.append({
            'feature_set': name,
            'n_features': len(features),
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max(),
        })
    result = pd.DataFrame(rows).sort_values('cv_mean', ascending=False).reset_index(drop=True)
    best_feature_set = result.iloc[0]['feature_set']
    feature_map = feature_sets[best_feature_set]
    return result, best_feature_set, feature_map


def build_candidate_models(random_seed):
    return {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            random_state=random_seed,
            class_weight='balanced',
        ),
        'logistic_regression': make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                random_state=random_seed,
                class_weight='balanced',
            ),
        ),
    }


def compare_models(X_train, y_train, random_seed):
    rows = []
    candidates = build_candidate_models(random_seed)
    best_name = None
    best_model = None
    best_score = float('-inf')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    for name, model in candidates.items():
        scores = cross_val_score(clone(model), X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        rows.append({
            'model': name,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max(),
        })
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name
            best_model = clone(model)

    return pd.DataFrame(rows).sort_values('cv_mean', ascending=False).reset_index(drop=True), best_name, best_model


def extract_model_feature_names(model, feature_cols):
    if hasattr(model, 'feature_importances_'):
        return feature_cols, model.feature_importances_
    if hasattr(model, 'named_steps') and 'logisticregression' in model.named_steps:
        return feature_cols, abs(model.named_steps['logisticregression'].coef_).mean(axis=0)
    return feature_cols, None


def top_confusions(y_true, y_pred, top_n=10):
    confusion = (
        pd.DataFrame({'true': y_true, 'pred': y_pred})
        .query('true != pred')
        .groupby(['true', 'pred'])
        .size()
        .reset_index(name='count')
        .sort_values(['count', 'true', 'pred'], ascending=[False, True, True])
        .head(top_n)
    )
    if not confusion.empty:
        confusion['count'] = confusion['count'].astype(int)
    return confusion


def train_model():
    df = load_training_frame()

    print(f"   Dataset ready: {len(df)} antibodies across {df['target'].nunique()} antigen classes.")
    print(f"   Classes: {df['target'].unique().tolist()}")
    print(f"   Bulk features: {BULK_FEATURE_COLS}")
    print(f"   CDR3 features: {CDR3_FEATURE_COLS}")
    print(f"   Derived features: {DERIVED_FEATURE_COLS}\n")

    y = df['target']

    print('2. Comparing feature sets...')
    feature_table, best_feature_set_name, best_feature_cols = compare_feature_sets(df, y, RANDOM_SEED)
    print(feature_table.assign(
        cv_mean=lambda frame: (frame['cv_mean'] * 100).round(2),
        cv_std=lambda frame: (frame['cv_std'] * 100).round(2),
        cv_min=lambda frame: (frame['cv_min'] * 100).round(2),
        cv_max=lambda frame: (frame['cv_max'] * 100).round(2),
    ).to_string(index=False))
    print(f"\n   Best feature set: {best_feature_set_name}\n")

    X = build_feature_matrix(df, best_feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print('3. Comparing candidate models...')
    model_table, best_model_name, best_model = compare_models(X_train, y_train, RANDOM_SEED)
    print(model_table.assign(
        cv_mean=lambda frame: (frame['cv_mean'] * 100).round(2),
        cv_std=lambda frame: (frame['cv_std'] * 100).round(2),
        cv_min=lambda frame: (frame['cv_min'] * 100).round(2),
        cv_max=lambda frame: (frame['cv_max'] * 100).round(2),
    ).to_string(index=False))
    print(f"\n   Best model: {best_model_name}\n")

    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    if hasattr(best_model, 'predict_proba'):
        prediction_confidence = best_model.predict_proba(X_test).max(axis=1)
    else:
        prediction_confidence = [None] * len(predictions)

    holdout_accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)

    print('4. Hold-out evaluation...')
    print(f"   Hold-out Accuracy: {holdout_accuracy * 100:.2f}%\n")
    print(classification_report(y_test, predictions, zero_division=0))

    holdout_results = df.loc[
        y_test.index,
        [
            'pdb_id',
            'heavy_chain_id',
            'light_chain_id',
            'heavy_sequence',
            'light_sequence',
            'cdrh3_sequence',
            'cdrl3_sequence',
            'target',
        ],
    ].copy()
    holdout_results = holdout_results.rename(columns={'target': 'true_label'})
    holdout_results['predicted_label'] = predictions
    holdout_results['prediction_confidence'] = prediction_confidence
    holdout_results['correct'] = holdout_results['true_label'] == holdout_results['predicted_label']

    errors = holdout_results.loc[~holdout_results['correct']].copy()
    errors = errors.sort_values(
        ['true_label', 'predicted_label', 'prediction_confidence'],
        ascending=[True, True, False],
    )
    confusion_summary = top_confusions(y_test, predictions)

    print('\n--- Top Confusions ---')
    if confusion_summary.empty:
        print('  None on this split.')
    else:
        for _, row in confusion_summary.iterrows():
            print(f"  {row['true']} -> {row['pred']}: {int(row['count'])}")

    print('\n--- Feature Importance ---')
    feature_names, importances = extract_model_feature_names(best_model, best_feature_cols)
    if importances is None:
        print('  Feature importances are not available for this model.')
    else:
        ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
        for name, importance in ranked:
            bar = '█' * int(float(importance) * 40)
            print(f"  {name:<24} {float(importance):.3f}  {bar}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    feature_table.to_csv(FEATURE_COMPARISON_PATH, index=False)
    model_table.to_csv(MODEL_COMPARISON_PATH, index=False)
    errors.to_csv(ERROR_ANALYSIS_PATH, index=False)

    experiment_report = {
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE,
        'n_rows': int(len(df)),
        'n_classes': int(df['target'].nunique()),
        'class_counts': {str(key): int(value) for key, value in df['target'].value_counts().to_dict().items()},
        'best_feature_set': best_feature_set_name,
        'best_model': best_model_name,
        'feature_columns': best_feature_cols,
        'feature_set_comparison': feature_table.to_dict(orient='records'),
        'model_comparison': model_table.to_dict(orient='records'),
        'holdout_accuracy': float(holdout_accuracy),
        'classification_report': report,
        'top_confusions': confusion_summary.to_dict(orient='records'),
    }
    with open(EXPERIMENT_REPORT_PATH, 'w', encoding='utf-8') as handle:
        json.dump(experiment_report, handle, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Feature comparison saved to {FEATURE_COMPARISON_PATH}")
    print(f"Model comparison saved to {MODEL_COMPARISON_PATH}")
    print(f"Holdout errors saved to {ERROR_ANALYSIS_PATH}")
    print(f"Experiment report saved to {EXPERIMENT_REPORT_PATH}")


if __name__ == '__main__':
    train_model()
