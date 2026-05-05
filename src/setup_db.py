import sqlite3
import os

def initialize_db():
    db_path = 'database/sabdab_features.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Nuke the old table so we can rebuild it fresh
    cursor.execute('DROP TABLE IF EXISTS antibodies')
    
    # 2. Build the new table with ALL columns
    cursor.execute('''
        CREATE TABLE antibodies (
            pdb_id TEXT PRIMARY KEY,
            heavy_chain_id TEXT,
            light_chain_id TEXT,
            heavy_chain_length INTEGER,
            light_chain_length INTEGER,
            heavy_sequence TEXT,
            light_sequence TEXT,
            heavy_mw REAL,
            heavy_pi REAL,
            heavy_gravy REAL,
            light_mw REAL,
            light_pi REAL,
            light_gravy REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Success! Database wiped and rebuilt with new columns at {db_path}")

if __name__ == "__main__":
    initialize_db()