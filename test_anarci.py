from anarci import number
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
import os

# Get a test file
files = [f for f in os.listdir('data/raw') if f.endswith('.pdb')][:1]
f = files[0]
path = os.path.join('data/raw', f)

# Extract heavy chain sequence
parser = PDBParser(QUIET=True)
struct = parser.get_structure('test', path)
chain = struct[0]['A']  # Usually 'A' is heavy
seq = ""
for residue in chain.get_residues():
    if is_aa(residue, standard=True):
        seq += seq1(residue.resname)

print(f"Heavy chain sequence length: {len(seq)}")
print(f"First 50 chars: {seq[:50]}")

# Try anarci numbering
result = number(seq)
print(f"Numbering result: {result}")
if result:
    print(f"Result[0] type: {type(result[0])}")
    print(f"Result[0][:10]: {result[0][:10] if result[0] else 'None'}")
