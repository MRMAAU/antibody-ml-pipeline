import sqlite3
import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings
from Bio import BiopythonWarning

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
    db_path = 'database/sabdab_features.db'
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
        conn = sqlite3.connect(db_path)
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
    
    success_count = 0
    print("Starting extraction...\n")
    
    for _, row in df_filtered.iterrows():
        pdb_id = row['pdb']
        h_chain = row['Hchain']
        l_chain = row['Lchain']
        
        filepath = os.path.join(raw_dir, f"{pdb_id}.pdb")
        
        # Only process if we actually downloaded the file
        if os.path.exists(filepath):
            if extract_and_save_features(filepath, pdb_id, h_chain, l_chain):
                print(f"[{pdb_id}] Successfully extracted features (H: {h_chain}, L: {l_chain})")
                success_count += 1
                
    print(f"\nBatch complete! Extracted biochemistry for {success_count} antibodies.")

if __name__ == "__main__":
    tsv_file = "data/raw/sabdab_summary.tsv"
    raw_directory = "data/raw/"
    
    if os.path.exists(tsv_file):
        process_batch(tsv_file, raw_directory)
    else:
        print("Error: Could not find sabdab_summary.tsv")