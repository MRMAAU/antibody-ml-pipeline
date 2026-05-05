import os

# Shared paths
DATABASE_DIR = "database"
RAW_DATA_DIR = os.path.join("data", "raw")
DB_PATH = os.path.join(DATABASE_DIR, "sabdab_features.db")
SABDAB_SUMMARY_TSV = os.path.join(RAW_DATA_DIR, "sabdab_summary_sequences.tsv")
MODEL_PATH = os.path.join("models", "rf_antigen_classifier.joblib")
EXPERIMENT_REPORT_PATH = os.path.join("models", "last_experiment.json")
FEATURE_COMPARISON_PATH = os.path.join("models", "feature_set_comparison.csv")
MODEL_COMPARISON_PATH = os.path.join("models", "model_comparison.csv")
ERROR_ANALYSIS_PATH = os.path.join("models", "holdout_errors.csv")

# Bulk physicochemical features (from PDB parsing via parser.py)
BULK_FEATURE_COLS = [
    "heavy_mw",
    "heavy_pi",
    "heavy_gravy",
    "light_mw",
    "light_pi",
    "light_gravy",
]

# CDR3-derived features (extracted from sequences via parser.py using heuristic Cys/Trp detection)
# These capture antigen-binding loop properties which are more biologically relevant
# for antigen-type classification than whole-chain bulk properties.
CDR3_FEATURE_COLS = [
    "cdrh3_mw",
    "cdrh3_pi",
    "cdrh3_gravy",
    "cdrh3_charge",
    "cdrl3_mw",
    "cdrl3_pi",
    "cdrl3_gravy",
    "cdrl3_charge",
]

FEATURE_COLS = BULK_FEATURE_COLS + CDR3_FEATURE_COLS

# Simple sequence-derived features computed from stored heavy/light/CDR3 sequences.
DERIVED_FEATURE_COLS = [
    "heavy_chain_length",
    "light_chain_length",
    "chain_length_ratio",
    "chain_length_diff",
    "heavy_basic_frac",
    "light_basic_frac",
    "heavy_aromatic_frac",
    "light_aromatic_frac",
    "cdrh3_length",
    "cdrl3_length",
]

# Model configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Dataset build configuration
DEFAULT_SAMPLES_PER_CLASS = 250
DOWNLOAD_SLEEP_SECONDS = 0.2


def clean_antigen_target(antigen_type: str):
    """
    Map antigen labels to a single clean class string.

    Returns None for multi-antigen entries (e.g. 'protein | peptide') so they
    can be dropped cleanly with dropna(), rather than arbitrarily keeping the
    first label and training on mislabelled data.
    """
    val = str(antigen_type).strip()
    if "|" in val:
        return None
    return val