import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model():
    # 1. Connect to the database and load data into Pandas
    db_path = 'database/sabdab_features.db'
    try:
        conn = sqlite3.connect(db_path)
        # We query just the numerical features we want the model to learn from
        df = pd.read_sql_query('''
            SELECT 
                pdb_id, 
                heavy_mw, heavy_pi, heavy_gravy, 
                light_mw, light_pi, light_gravy 
            FROM antibodies
        ''', conn)
        conn.close()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print(f"Dataset loaded! Total antibodies: {len(df)}")
    
    # --- THE DATA BOTTLENECK ---
    if len(df) < 10:
        print("\n[WARNING] You only have 1 row of data!")
        print("A Random Forest needs a dataset (usually 100+ rows) to train.")
        print("Here is what the Pandas DataFrame looks like right now:")
        print(df.head())
        return

    # 2. Define Features (X) and Target (y)
    # (Assuming we added a 'target_antigen' column to predict)
    X = df[['heavy_mw', 'heavy_pi', 'heavy_gravy', 'light_mw', 'light_pi', 'light_gravy']]
    y = df['target_antigen'] # We don't have this column yet!

    # 3. Split the data (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize and train the Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 5. Evaluate the model
    predictions = rf_model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    train_model()