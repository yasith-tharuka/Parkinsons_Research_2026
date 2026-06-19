import pandas as pd
from pathlib import Path

def preprocess_data(raw_data_path: Path, output_dir: Path):
    """
    Cleans the raw Parkinson's dataset by separating features, status labels, and patient IDs.
    Extracts group identifiers from patient name strings to ensure correct cross-validation.
    """
    print(f"Loading raw data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Extract Patient ID (e.g. 'phon_R01_S01_1' -> 'S01')
    df['patient_id'] = df['name'].apply(lambda x: x.split('_')[2])
    
    # Separate features, status, and patient groups
    features = df.drop(columns=["name", "status", "patient_id"])
    status = df["status"]
    groups = df["patient_id"]
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the output files
    features.to_csv(output_dir / "Parkinsons_cleaned.csv", index=False)
    status.to_csv(output_dir / "Parkinsons_status.csv", index=False)
    groups.to_csv(output_dir / "Parkinsons_groups.csv", index=False)
    
    print(f"Preprocessing complete. Processed files saved to: {output_dir}")
    print(f"Features: {features.shape[0]} samples, {features.shape[1]} features.")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    raw_path = PROJECT_ROOT / "data" / "raw" / "parkinsons.data"
    output_path = PROJECT_ROOT / "data" / "processed"
    preprocess_data(raw_path, output_path)
