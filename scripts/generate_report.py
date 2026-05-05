#!/usr/bin/env python3
"""
Generate static and interactive reports from models/ artifacts.
Creates: reports/confusion_matrix.png, reports/feature_importances.png,
reports/confusion_matrix.html, reports/pr_curves.html, reports/cdrh3_logo.png, reports/report.html
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(r"c:\Users\MRM\OneDrive - Aalborg Universitet\Code stuff\antibody_ml")
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

# Load artifacts
LAST = MODELS / 'last_experiment.json'
HOLDOUT = MODELS / 'holdout_errors.csv'
MODEL_FILE = MODELS / 'rf_antigen_classifier.joblib'

if not HOLDOUT.exists():
    print('No holdout_errors.csv found in', MODELS)
    raise SystemExit(1)

errors = pd.read_csv(HOLDOUT)
model = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None

# Confusion matrix (static PNG)
labels = sorted(list(set(errors['true_label']) | set(errors['predicted_label'])))
cm = confusion_matrix(errors['true_label'], errors['predicted_label'], labels=labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Holdout Confusion Matrix')
plt.tight_layout()
conf_png = REPORTS / 'confusion_matrix.png'
plt.savefig(conf_png, dpi=150)
plt.close()
print('Wrote', conf_png)

# Feature importances (static PNG)
feat_png = REPORTS / 'feature_importances.png'
feature_names = None
if LAST.exists():
    with open(LAST, 'r', encoding='utf-8') as f:
        j = json.load(f)
        feature_names = j.get('feature_names')
if feature_names is None:
    # try columns from errors (exclude metadata)
    excluded = {'pdb_id','heavy_chain_id','light_chain_id','true_label','predicted_label','prediction_confidence','correct','heavy_sequence','light_sequence','cdrh3_sequence','cdrl3_sequence'}
    feature_names = [c for c in errors.columns if c not in excluded]

if model is None:
    print('No trained model to show feature importances for.')
else:
    if hasattr(model, 'feature_importances_'):
        imp = np.array(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        imp = np.abs(np.ravel(model.coef_))
    else:
        imp = np.zeros(len(feature_names))
    # Align feature names length with importance vector
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    if len(feature_names) != len(imp):
        # fallback: generate generic feature names matching importance length
        feature_names = [f'feat_{i}' for i in range(len(imp))]
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': imp})
    df_imp = df_imp.sort_values('importance', ascending=False).head(30)
    plt.figure(figsize=(8, max(4, 0.25*len(df_imp))))
    sns.barplot(data=df_imp, x='importance', y='feature', palette='viridis')
    plt.title('Top feature importances')
    plt.tight_layout()
    plt.savefig(feat_png, dpi=150)
    plt.close()
    print('Wrote', feat_png)

# Interactive confusion matrix (Plotly)
fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels, colorscale='Blues', text=cm, texttemplate='%{text}'))
fig.update_layout(title='Holdout Confusion Matrix (interactive)', xaxis_title='Predicted', yaxis_title='True')
conf_html = REPORTS / 'confusion_matrix.html'
fig.write_html(conf_html, include_plotlyjs='cdn')
print('Wrote', conf_html)

# PR curves per class (one-vs-rest using prediction_confidence if present)
pr_html = REPORTS / 'pr_curves.html'
if 'prediction_confidence' in errors.columns:
    # prediction_confidence is confidence for predicted class; to plot per-class we need model probabilities
    if model is not None and hasattr(model, 'predict_proba'):
        try:
            # rebuild X features used in model if present in errors; otherwise skip
            if hasattr(model, 'feature_names_in_'):
                Xcols = list(model.feature_names_in_)
            else:
                Xcols = [c for c in feature_names if c in errors.columns]
            if len(Xcols) == 0:
                print('No feature columns present in holdout_errors to compute PR curves; skipping')
            else:
                X = errors[Xcols].fillna(0).values
                probs = model.predict_proba(X)
                # map classes
                classes = model.classes_
                pr_fig = go.Figure()
                for i, cls in enumerate(classes):
                    y_true = (errors['true_label'] == cls).astype(int)
                    y_score = probs[:, i]
                    precision, recall, _ = precision_recall_curve(y_true, y_score)
                    pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=str(cls)))
                pr_fig.update_layout(title='Per-class Precision-Recall curves', xaxis_title='Recall', yaxis_title='Precision')
                pr_fig.write_html(pr_html, include_plotlyjs='cdn')
                print('Wrote', pr_html)
        except Exception as e:
            print('Failed to compute PR curves:', e)
    else:
        print('Model missing predict_proba; skipping PR curves')
else:
    print('No prediction_confidence column for PR curves; skipping')

# Sequence-logo for CDR-H3 (top freq)
cdrs = errors['cdrh3_sequence'].dropna().astype(str)
if len(cdrs)>=2:
    # take top N sequences or pad to same length by aligning left
    seqs = [s for s in cdrs if len(s)>0]
    maxlen = max(len(s) for s in seqs)
    AAs = list('ACDEFGHIKLMNPQRSTVWY')
    pwm = np.zeros((maxlen, len(AAs)), dtype=float)
    for s in seqs:
        for i, aa in enumerate(s):
            if aa in AAs:
                pwm[i, AAs.index(aa)] += 1
    pwm = pwm / pwm.sum(axis=1, keepdims=True)
    # plot logo-like bar stacks for first 20 positions
    logo_png = REPORTS / 'cdrh3_logo.png'
    fig, ax = plt.subplots(figsize=(min(12, 0.25*maxlen+2), 4))
    cumul = np.zeros(maxlen)
    for j, aa in enumerate(AAs):
        vals = pwm[:, j]
        ax.bar(range(maxlen), vals, bottom=cumul, label=aa)
        cumul += vals
    ax.set_xlabel('Position')
    ax.set_ylabel('Frequency')
    ax.set_title('CDR-H3 position frequencies (stacked)')
    ax.set_xlim(-0.5, maxlen-0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.tight_layout()
    fig.savefig(logo_png, dpi=150)
    plt.close()
    print('Wrote', logo_png)
else:
    print('Not enough CDR-H3 sequences to build logo')

# Build simple HTML report
report_file = REPORTS / 'report.html'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('<html><head><title>Antibody ML Report</title></head><body>')
    f.write('<h1>Antibody ML — Report</h1>')
    f.write('<h2>Confusion matrix (static)</h2>')
    f.write(f'<img src="{conf_png.name}" style="max-width:800px;">')
    f.write('<h2>Feature importances (static)</h2>')
    if feat_png.exists():
        f.write(f'<img src="{feat_png.name}" style="max-width:800px;">')
    f.write('<h2>Interactive plots</h2>')
    if conf_html.exists():
        f.write(f'<a href="{conf_html.name}">Interactive confusion matrix</a><br>')
    if pr_html.exists():
        f.write(f'<a href="{pr_html.name}">Per-class PR curves</a><br>')
    if 'logo_png' in locals() and logo_png.exists():
        f.write('<h2>CDR-H3 logo</h2>')
        f.write(f'<img src="{logo_png.name}" style="max-width:800px;">')
    f.write('<h2>Misclassifications (sample)</h2>')
    f.write(errors.head(20).to_html(index=False, escape=False))
    f.write('</body></html>')
print('Wrote', report_file)

print('\nAll done. Open reports/report.html to view the generated report (relative to project root).')
