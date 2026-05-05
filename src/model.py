import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings

# Suppress some scikit-learn warnings for cleaner output
warnings.filterwarnings('ignore')

def train_model():
    print("1. Loading Data...")
    # Load features from SQLite
    conn = sqlite3.connect('database/sabdab_features.db')
    df_features = pd.read_sql_query("SELECT * FROM antibodies", conn)
    conn.close()

    # Load labels from the master TSV
    df_labels = pd.read_csv('data/raw/sabdab_summary.tsv', sep='\t')
    
    # Merge them together based on the PDB ID
    df = pd.merge(df_features, df_labels[['pdb', 'antigen_type']], left_on='pdb_id', right_on='pdb')
    
    # Clean the target variable (drop any antibodies that don't have a known antigen type)
    df = df.dropna(subset=['antigen_type'])
    
    # Simplify the target: Many types are complex (e.g., "protein | protein"). 
    # We will just grab the first word to group them cleanly into broad categories.
    df['target'] = df['antigen_type'].astype(str).apply(lambda x: x.split(' |')[0].strip())
    
    print(f"Dataset ready! Training on {len(df)} antibodies.")
    print(f"Categories found: {df['target'].unique().tolist()}\n")

    # 2. Define Features (X) and Target (y)
    feature_cols = ['heavy_mw', 'heavy_pi', 'heavy_gravy', 'light_mw', 'light_pi', 'light_gravy']
    X = df[feature_cols]
    y = df['target']

    # 3. Split the data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("2. Training Random Forest...")
    # Initialize a forest with 100 decision trees
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    print("3. Evaluating Model...\n")
    predictions = rf_model.predict(X_test)
    
    # Print the accuracy and classification report
    print(f"Overall Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
    print(classification_report(y_test, predictions, zero_division=0))

    # 4. Feature Importance (The Biology Check)
    print("\n--- Feature Importance ---")
    importances = rf_model.feature_importances_
    for name, importance in zip(feature_cols, importances):
        print(f"{name}: {importance:.3f}")

if __name__ == "__main__":
    train_model()