import pandas as pd
from pathlib import Path
import urllib.request
import io
import arff

def preprocess_v2():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Parkinson's Disease Classification Dataset from OpenML...")
    
    # Dataset 42176 on OpenML is available as ARFF
    url = "https://openml.org/data/v1/download/21756046/parkinson-speech-uci.arff"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            dataset = arff.load(io.StringIO(response.read().decode('utf-8')))
            
            # create dataframe
            columns = [attr[0] for attr in dataset['attributes']]
            df = pd.DataFrame(dataset['data'], columns=columns)
            
        print(f"Dataset downloaded: {df.shape[0]} instances, {df.shape[1]} features.")
        
        # The dataset has an 'id' column for patients
        # The target column is 'class'
        if 'id' in df.columns:
            groups = df['id']
            df = df.drop(columns=['id'])
        else:
            groups = pd.Series(range(df.shape[0])) # Fallback
            
        if 'class' in df.columns:
            status = df['class']
            features = df.drop(columns=['class'])
        else:
            status = df.iloc[:, -1]
            features = df.iloc[:, :-1]
            
        # Convert target to numeric 0 and 1
        status = pd.to_numeric(status, errors='coerce')
        
        features.to_csv(output_dir / "Parkinsons_v2_features.csv", index=False)
        status.to_csv(output_dir / "Parkinsons_v2_status.csv", index=False)
        groups.to_csv(output_dir / "Parkinsons_v2_groups.csv", index=False)
        
        print("Preprocessed files saved to data/processed/")
    except Exception as e:
        print(f"Failed to download dataset: {e}")

if __name__ == "__main__":
    preprocess_v2()
