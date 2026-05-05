import pandas as pd
import urllib.request
import os
import time

def download_sabdab_batch(tsv_path, num_samples=50):
    print("Loading SAbDab master index...")
    
    try:
        # Load the TSV file
        df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        print(f"Failed to read TSV. Make sure it downloaded correctly: {e}")
        return

    # 1. Filter the data
    # We only want standard antibodies (must have both Heavy and Light chains)
    # SAbDab uses 'nan' when a chain is missing. Pandas drops these with dropna()
    df_filtered = df.dropna(subset=['Hchain', 'Lchain'])
    
    # 2. Get unique PDBs (sometimes the same PDB has multiple entries)
    unique_pdbs = df_filtered.drop_duplicates(subset=['pdb']).head(num_samples)
    
    print(f"Found {len(unique_pdbs)} valid antibodies. Starting batch download...\n")
    
    # 3. Download the structures
    success_count = 0
    for index, row in unique_pdbs.iterrows():
        pdb_id = row['pdb']
        filepath = f"data/raw/{pdb_id}.pdb"
        
        # Skip if we already downloaded it
        if os.path.exists(filepath):
            print(f"[{pdb_id}] Already exists in data/raw/")
            success_count += 1
            continue
            
        # Fetch from the global Protein Data Bank
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            print(f"[{pdb_id}] Downloading...")
            urllib.request.urlretrieve(url, filepath)
            success_count += 1
            time.sleep(0.5)  # Pause for half a second to not overwhelm the RCSB server
        except Exception as e:
            print(f"[{pdb_id}] Failed to download: {e}")

    print(f"\nFinished! Successfully secured {success_count}/{num_samples} PDB files.")

if __name__ == "__main__":
    tsv_file = "data/raw/sabdab_summary.tsv"
    if os.path.exists(tsv_file):
        download_sabdab_batch(tsv_file, num_samples=50)
    else:
        print("Error: Could not find sabdab_summary.tsv in data/raw/")