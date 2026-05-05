import pandas as pd
import urllib.request
import os
import time

def download_stratified_batch(tsv_path, samples_per_class=100):
    print("Loading SAbDab master index...")
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        print(f"Failed to read TSV: {e}")
        return

    # 1. Filter for valid antibodies with known antigens
    df_filtered = df.dropna(subset=['Hchain', 'Lchain', 'antigen_type']).drop_duplicates(subset=['pdb'])
    
    # 2. Clean the target variable (just like we did in the model script)
    df_filtered['target'] = df_filtered['antigen_type'].astype(str).apply(lambda x: x.split(' |')[0].strip())
    
    # 3. Find categories that actually have enough data
    counts = df_filtered['target'].value_counts()
    valid_categories = counts[counts >= samples_per_class].index.tolist()
    
    print(f"Found {len(valid_categories)} categories with at least {samples_per_class} samples:")
    print(counts[valid_categories])
    
    # 4. Perform Stratified Sampling (grab equal amounts from each valid category)
    stratified_df = df_filtered[df_filtered['target'].isin(valid_categories)].groupby('target').head(samples_per_class)
    
    total_to_download = len(stratified_df)
    print(f"\nStarting batch download of {total_to_download} valid antibodies...\n")
    
    # 5. Download the structures
    success_count = 0
    for index, row in stratified_df.iterrows():
        pdb_id = row['pdb']
        target = row['target']
        filepath = f"data/raw/{pdb_id}.pdb"
        
        if os.path.exists(filepath):
            success_count += 1
            continue
            
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            print(f"[{success_count+1}/{total_to_download}] Downloading {pdb_id} ({target})...")
            urllib.request.urlretrieve(url, filepath)
            success_count += 1
            time.sleep(0.2)  # Pause to respect server limits
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")

    print(f"\nFinished! Successfully secured {success_count}/{total_to_download} PDB files.")

if __name__ == "__main__":
    tsv_file = "data/raw/sabdab_summary.tsv"
    if os.path.exists(tsv_file):
        # We are asking for 100 of EACH valid category
        download_stratified_batch(tsv_file, samples_per_class=100)
    else:
        print("Error: Could not find sabdab_summary.tsv in data/raw/")