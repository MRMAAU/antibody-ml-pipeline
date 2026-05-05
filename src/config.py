import os

# Shared paths
DATABASE_DIR = "database"
RAW_DATA_DIR = os.path.join("data", "raw")
DB_PATH = os.path.join(DATABASE_DIR, "sabdab_features.db")
SABDAB_SUMMARY_TSV = os.path.join(RAW_DATA_DIR, "sabdab_summary_sequences.tsv")
MODEL_PATH = os.path.join("models", "rf_antigen_classifier.joblib")

# Bulk physicochemical features (from PDB parsing via parser.py)
BULK_FEATURE_COLS = [
    "heavy_mw",
    "heavy_pi",
    "heavy_gravy",
    "light_mw",
    "light_pi",
    "light_gravy",
]

# CDR-derived features (computed from SAbDab TSV sequences at training time)
# These capture antigen-binding loop properties rather than whole-chain bulk properties,
# which is far more biologically relevant for antigen-type classification.
CDR_FEATURE_COLS = [
    "cdrh3_length",
    "cdrh3_gravy",
    "cdrh3_charge",
    "cdrl3_length",
    "cdrl3_gravy",
    "cdrl3_charge",
]

FEATURE_COLS = BULK_FEATURE_COLS + CDR_FEATURE_COLS

# Model configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Dataset build configuration
DEFAULT_SAMPLES_PER_CLASS = 100
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