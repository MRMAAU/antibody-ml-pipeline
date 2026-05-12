"""
embed.py — ESM-2 sequence embeddings for antibody CDR3 sequences.

Loads CDRH3 and CDRL3 sequences from the SQLite DB, embeds them using the
ESM-2 protein language model (smallest variant, CPU-friendly), and saves the
resulting feature matrix as a .npz file alongside the PDB IDs so it can be
joined back to labels in model.py.

Install once:
    pip install transformers torch

Usage:
    cd <project root>
    python src/embed.py              # embeds CDRH3 + CDRL3 (default)
    python src/embed.py --chains full  # embeds full heavy + light chains (slower)
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from config import DB_PATH

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Smallest ESM-2: 6 layers, 8M params, 320-dim hidden, fast on CPU
# Upgrade to facebook/esm2_t12_35M_UR50D (480-dim) if you want a boost
# and have ~15 min to wait.
ESM_MODEL_NAME = "facebook/esm2_t6_8M_650K"
EMBED_DIM = 320          # matches the model above; update if you swap models
BATCH_SIZE = 32          # safe for 8GB RAM on CDR3-length sequences
OUTPUT_PATH = Path("models") / "esm_embeddings.npz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sequences(chains: str = "cdr3") -> tuple[list[str], list[str], list[str]]:
    """
    Pull sequences from the DB.

    chains = "cdr3"  → embed CDRH3 and CDRL3 only  (fast, biologically focused)
    chains = "full"  → embed full heavy and light chains (slower, more context)

    Returns (pdb_ids, seq_A_list, seq_B_list).
    """
    conn = sqlite3.connect(DB_PATH)
    if chains == "cdr3":
        query = """
            SELECT pdb_id, cdrh3_sequence, cdrl3_sequence
            FROM antibodies
            WHERE cdrh3_sequence IS NOT NULL
              AND cdrl3_sequence IS NOT NULL
              AND cdrh3_sequence != ''
              AND cdrl3_sequence != ''
        """
    else:
        query = """
            SELECT pdb_id, heavy_sequence, light_sequence
            FROM antibodies
            WHERE heavy_sequence IS NOT NULL
              AND light_sequence IS NOT NULL
        """
    rows = conn.execute(query).fetchall()
    conn.close()

    pdb_ids = [r[0] for r in rows]
    seqs_a  = [r[1] for r in rows]
    seqs_b  = [r[2] for r in rows]
    print(f"Loaded {len(pdb_ids)} antibodies from DB.")
    return pdb_ids, seqs_a, seqs_b


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """
    Average the per-residue token embeddings, ignoring padding and special tokens.

    hidden_states : (batch, seq_len, hidden_dim)
    attention_mask: (batch, seq_len)   — 1 for real tokens, 0 for padding
    """
    # ESM uses [CLS] at position 0 and [EOS] at the last real position.
    # Mask those out so we only pool over actual amino-acid positions.
    mask = attention_mask.clone().float()
    mask[:, 0] = 0          # zero out [CLS]
    # zero out [EOS] — it sits at the first 0 in each row after the sequence
    for i in range(mask.shape[0]):
        seq_len = attention_mask[i].sum().item()  # includes [CLS] + AA + [EOS]
        if seq_len > 1:
            mask[i, int(seq_len) - 1] = 0

    mask_expanded = mask.unsqueeze(-1)                         # (B, L, 1)
    sum_embeddings = (hidden_states * mask_expanded).sum(1)   # (B, hidden)
    denom = mask_expanded.sum(1).clamp(min=1e-9)              # (B, 1)
    return (sum_embeddings / denom).cpu().numpy()             # (B, hidden)


def embed_sequences(
    model: EsmModel,
    tokenizer: AutoTokenizer,
    sequences: list[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Embed a list of sequences in batches.
    Returns (N, EMBED_DIM) float32 array.
    """
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(pooled)
            n_done = min(start + batch_size, len(sequences))
            print(f"  Embedded {n_done}/{len(sequences)}", end="\r")

    print()
    return np.vstack(all_embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_embeddings(chains: str = "cdr3") -> None:
    print(f"=== ESM-2 Embedding Pipeline ===")
    print(f"Model : {ESM_MODEL_NAME}")
    print(f"Chains: {chains}\n")

    # 1. Load sequences
    pdb_ids, seqs_a, seqs_b = load_sequences(chains)

    # 2. Load model (downloads ~31MB once, then cached in ~/.cache/huggingface)
    print("Loading ESM-2 model (first run downloads ~31MB)...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model     = EsmModel.from_pretrained(ESM_MODEL_NAME)

    label_a = "cdrh3" if chains == "cdr3" else "heavy"
    label_b = "cdrl3" if chains == "cdr3" else "light"

    # 3. Embed both chains
    print(f"\nEmbedding {label_a} sequences...")
    emb_a = embed_sequences(model, tokenizer, seqs_a)

    print(f"Embedding {label_b} sequences...")
    emb_b = embed_sequences(model, tokenizer, seqs_b)

    # 4. Concatenate: each antibody → (EMBED_DIM * 2,) vector
    combined = np.concatenate([emb_a, emb_b], axis=1)
    print(f"\nFinal embedding shape: {combined.shape}")  # (N, 640) for cdr3 mode

    # 5. Save
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        pdb_ids=np.array(pdb_ids),
        embeddings=combined,
        chains=np.array(chains),
        model_name=np.array(ESM_MODEL_NAME),
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chains",
        choices=["cdr3", "full"],
        default="cdr3",
        help="Which sequences to embed. 'cdr3' is fast and biologically focused.",
    )
    args = parser.parse_args()
    build_embeddings(chains=args.chains)
