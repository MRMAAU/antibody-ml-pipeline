import pandas as pd
import urllib.request
import os
import time
from config import (
    SABDAB_SUMMARY_TSV,
    RAW_DATA_DIR,
    DEFAULT_SAMPLES_PER_CLASS,
    DOWNLOAD_SLEEP_SECONDS,
    RANDOM_SEED,
    clean_antigen_target,
)

def download_stratified_batch(tsv_path, samples_per_class=100):
    print("Loading SAbDab master index...")
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        print(f"Failed to read TSV: {e}")
        return

    # 1. Filter for valid antibodies with known antigens
    df_filtered = df.dropna(subset=['Hchain', 'Lchain', 'antigen_type']).drop_duplicates(subset=['pdb'])

    # 2. Clean the target label.
    #    clean_antigen_target() returns None for multi-antigen entries (e.g. 'protein | peptide'),
    #    which we then drop — keeping an arbitrary first label would introduce noise.
    df_filtered = df_filtered.copy()
    df_filtered['target'] = df_filtered['antigen_type'].apply(clean_antigen_target)
    df_filtered = df_filtered.dropna(subset=['target'])

    # 3. Find categories that actually have enough data
    counts = df_filtered['target'].value_counts()
    valid_categories = counts[counts >= samples_per_class].index.tolist()

    print(f"Found {len(valid_categories)} categories with at least {samples_per_class} samples:")
    print(counts[valid_categories])

    # 4. Perform unbiased stratified sampling (equal random sample from each valid category)
    eligible_df = df_filtered[df_filtered['target'].isin(valid_categories)]
    sampled_groups = []
    for _, group in eligible_df.groupby('target'):
        sampled_groups.append(group.sample(n=samples_per_class, random_state=RANDOM_SEED))
    stratified_df = pd.concat(sampled_groups).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    total_to_download = len(stratified_df)
    print(f"\nStarting batch download of {total_to_download} valid antibodies...\n")

    # 5. Download the structures.
    #    Use enumerate so the counter is always accurate regardless of cache hits.
    success_count = 0
    for i, (_, row) in enumerate(stratified_df.iterrows(), start=1):
        pdb_id = row['pdb']
        target = row['target']
        filepath = os.path.join(RAW_DATA_DIR, f"{pdb_id}.pdb")

        if os.path.exists(filepath):
            success_count += 1
            continue

        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            print(f"[{i}/{total_to_download}] Downloading {pdb_id} ({target})...")
            urllib.request.urlretrieve(url, filepath)
            success_count += 1
            time.sleep(DOWNLOAD_SLEEP_SECONDS)  # Pause to respect server limits
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")

    print(f"\nFinished! Successfully secured {success_count}/{total_to_download} PDB files.")

if __name__ == "__main__":
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    tsv_file = SABDAB_SUMMARY_TSV
    if os.path.exists(tsv_file):
        download_stratified_batch(tsv_file, samples_per_class=DEFAULT_SAMPLES_PER_CLASS)
    else:
        print("Error: Could not find sabdab_summary.tsv in data/raw/")