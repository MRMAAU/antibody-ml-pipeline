import subprocess
import os

def run_step(command, description):
    print(f"\n>>> {description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error during {description}. Exiting.")
        exit(1)

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("database", exist_ok=True)

    # Sequence of execution
    run_step("python src/build_dataset.py", "Downloading PDB Files")
    run_step("python src/setup_db.py", "Resetting SQLite Database")
    run_step("python src/parser.py", "Extracting Biochemistry from PDBs")
    run_step("python src/model.py", "Training and Evaluating Model")

    print("\nPipeline execution finished successfully!")