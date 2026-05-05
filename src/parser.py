import sqlite3
import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings
from Bio import BiopythonWarning
from config import DB_PATH, SABDAB_SUMMARY_TSV, RAW_DATA_DIR

warnings.simplefilter('ignore', BiopythonWarning)

# IMGT CDR3 definitions (approximate residue ranges in IMGT numbering)
# CDR3 spans positions 105-117 in IMGT for both heavy and light chains
# We'll use a heuristic: look for Cys at ~position 104 and Trp/Phe at ~position 117
# For simplicity, we extract the region between conserved Cys and Trp/Phe in the full sequence
CDR3_CYS_PATTERN = 'C'  # CDR3 starts after conserved Cys
CDR3_END_PATTERN = ['W', 'F']  # CDR3 typically ends with Trp or Phe


def get_chain_sequence(chain):
    sequence = ""
    for residue in chain.get_residues():
        if is_aa(residue, standard=True):
            sequence += seq1(residue.resname)
    return sequence


def calculate_biochemical_features(sequence):
    if not sequence:
        return 0.0, 0.0, 0.0
    clean_seq = sequence.replace('X', '')
    if not clean_seq:
        return 0.0, 0.0, 0.0
    analysis = ProteinAnalysis(clean_seq)
    return analysis.molecular_weight(), analysis.isoelectric_point(), analysis.gravy()


def calculate_charge_at_pH7(sequence):
    """Compute net charge at pH 7.0 for a sequence."""
    if not sequence:
        return 0.0
    clean_seq = sequence.replace('X', '')
    if not clean_seq:
        return 0.0
    try:
        analysis = ProteinAnalysis(clean_seq)
        return analysis.charge_at_pH(7.0)
    except:
        return 0.0


def extract_cdr3_heuristic(sequence):
    """
    Extract CDR3 from antibody sequence using a heuristic approach.
    CDR3 typically starts after a conserved Cys (position ~104 in IMGT)
    and ends with Trp/Phe (position ~117 in IMGT).
    
    This is a rough approximation; anarci with HMMER would be more accurate.
    """
    if not sequence or len(sequence) < 120:
        return ""
    
    # Find all Cys positions after position 90 (rough CDR2/CDR3 boundary)
    cys_positions = [i for i, aa in enumerate(sequence[90:], start=90) if aa == 'C']
    if not cys_positions:
        return ""
    
    # Use the last Cys (closest to CDR3 start)
    cdr3_start = cys_positions[-1] + 1
    if cdr3_start >= len(sequence):
        return ""
    
    # Find Trp or Phe after CDR3 start (should be within ~20 residues)
    cdr3_end = len(sequence)
    for i in range(cdr3_start, min(cdr3_start + 30, len(sequence))):
        if sequence[i] in ['W', 'F']:
            cdr3_end = i + 1
            break
    
    cdr3 = sequence[cdr3_start:cdr3_end]
    return cdr3 if cdr3 else ""


def extract_and_save_features(pdb_filepath, pdb_id, heavy_chain_id, light_chain_id):
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_id, pdb_filepath)
        model = structure[0]

        heavy_chain = model[heavy_chain_id]
        light_chain = model[light_chain_id]

        heavy_seq = get_chain_sequence(heavy_chain)
        light_seq = get_chain_sequence(light_chain)

        h_mw, h_pi, h_gravy = calculate_biochemical_features(heavy_seq)
        l_mw, l_pi, l_gravy = calculate_biochemical_features(light_seq)
        
        # Extract CDR3 sequences
        cdrh3_seq = extract_cdr3_heuristic(heavy_seq)
        cdrl3_seq = extract_cdr3_heuristic(light_seq)
        
        # Compute CDR3 features
        cdrh3_mw, cdrh3_pi, cdrh3_gravy = calculate_biochemical_features(cdrh3_seq)
        cdrh3_charge = calculate_charge_at_pH7(cdrh3_seq)
        
        cdrl3_mw, cdrl3_pi, cdrl3_gravy = calculate_biochemical_features(cdrl3_seq)
        cdrl3_charge = calculate_charge_at_pH7(cdrl3_seq)

    except KeyError as e:
        print(f"[{pdb_id}] Skipped: Chain {e} not found in structure.")
        return False
    except Exception as e:
        print(f"[{pdb_id}] Failed to parse: {e}")
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO antibodies
            (pdb_id, heavy_chain_id, light_chain_id, heavy_chain_length, light_chain_length,
             heavy_sequence, light_sequence, heavy_mw, heavy_pi, heavy_gravy, light_mw, light_pi, light_gravy,
             cdrh3_sequence, cdrl3_sequence, cdrh3_mw, cdrh3_pi, cdrh3_gravy, cdrh3_charge,
             cdrl3_mw, cdrl3_pi, cdrl3_gravy, cdrl3_charge)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pdb_id, heavy_chain_id, light_chain_id, len(heavy_seq), len(light_seq),
              heavy_seq, light_seq, h_mw, h_pi, h_gravy, l_mw, l_pi, l_gravy,
              cdrh3_seq, cdrl3_seq, cdrh3_mw, cdrh3_pi, cdrh3_gravy, cdrh3_charge,
              cdrl3_mw, cdrl3_pi, cdrl3_gravy, cdrl3_charge))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"[{pdb_id}] Database error: {e}")
        return False


def process_batch(tsv_path, raw_dir):
    print("Loading SAbDab TSV to map chain IDs...")
    df = pd.read_csv(tsv_path, sep='\t')

    # Filter for standard antibodies and drop duplicates
    df_filtered = df.dropna(subset=['Hchain', 'Lchain']).drop_duplicates(subset=['pdb'])

    # Pre-filter to only the PDB files we actually downloaded.
    # This avoids iterating thousands of TSV rows to process a few hundred files.
    pdb_files_on_disk = {
        f.replace('.pdb', '')
        for f in os.listdir(raw_dir)
        if f.endswith('.pdb')
    }
    df_filtered = df_filtered[df_filtered['pdb'].isin(pdb_files_on_disk)].copy()

    total = len(df_filtered)
    print(f"Found {total} downloaded PDB files to process.\nStarting extraction...\n")

    success_count = 0
    for i, (_, row) in enumerate(df_filtered.iterrows(), start=1):
        pdb_id = row['pdb']
        h_chain = row['Hchain']
        l_chain = row['Lchain']
        filepath = os.path.join(raw_dir, f"{pdb_id}.pdb")

        if extract_and_save_features(filepath, pdb_id, h_chain, l_chain):
            print(f"[{i}/{total}] [{pdb_id}] Successfully extracted features (H: {h_chain}, L: {l_chain})")
            success_count += 1
        else:
            print(f"[{i}/{total}] [{pdb_id}] Skipped.")

    print(f"\nBatch complete! Extracted biochemistry for {success_count}/{total} antibodies.")


if __name__ == "__main__":
    tsv_file = SABDAB_SUMMARY_TSV
    raw_directory = RAW_DATA_DIR

    if os.path.exists(tsv_file):
        process_batch(tsv_file, raw_directory)
    else:
        print("Error: Could not find sabdab_summary.tsv")