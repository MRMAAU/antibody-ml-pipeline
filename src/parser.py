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
             heavy_sequence, light_sequence, heavy_mw, heavy_pi, heavy_gravy, light_mw, light_pi, light_gravy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pdb_id, heavy_chain_id, light_chain_id, len(heavy_seq), len(light_seq),
              heavy_seq, light_seq, h_mw, h_pi, h_gravy, l_mw, l_pi, l_gravy))
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