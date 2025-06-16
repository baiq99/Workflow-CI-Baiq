# automate_Baiq-Ega-Aulia.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import os

def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Preprocess dataset and save to output_path.
    """
    # Load dataset
    data = pd.read_csv(input_path)
    print(f"[INFO] Loaded data with shape: {data.shape}")

    # 1. Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data_imputed = data_imputed.convert_dtypes()
    print("[INFO] Missing values handled.")

    # 2. Remove duplicates
    before = data_imputed.shape[0]
    data_imputed.drop_duplicates(inplace=True)
    print(f"[INFO] Removed {before - data_imputed.shape[0]} duplicates.")

    # 3. Standardize numerical features
    num_cols = data_imputed.select_dtypes(include=['number']).columns.tolist()
    for col in num_cols:
        if not pd.api.types.is_numeric_dtype(data_imputed[col]):
            data_imputed[col] = pd.to_numeric(data_imputed[col], errors='coerce')
    data_imputed.dropna(subset=num_cols, inplace=True)

    scaler = StandardScaler()
    data_imputed[num_cols] = scaler.fit_transform(data_imputed[num_cols])
    print("[INFO] Standardized numerical features.")

    # 4. Remove outliers
    z_scores = np.abs(stats.zscore(data_imputed[num_cols]))
    if z_scores.shape[1] == len(num_cols):
        data_imputed = data_imputed[(z_scores < 3).all(axis=1)]
        print("[INFO] Outliers removed using z-score.")
    else:
        print("[WARN] Skipping outlier removal due to mismatch.")

    # 5. Encode categorical features
    cat_cols = data_imputed.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        data_imputed[col] = le.fit_transform(data_imputed[col].astype(str))
    print("[INFO] Encoded categorical features.")

    # 6. Optional binning
    if 'Administrative_Duration' in data_imputed.columns:
        try:
            data_imputed['Administrative_Duration_Bin'] = pd.qcut(
                data_imputed['Administrative_Duration'], q=4, labels=False, duplicates='drop'
            )
            print("[INFO] Binned 'Administrative_Duration'.")
        except Exception as e:
            print(f"[WARN] Binning failed: {e}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save preprocessed dataset
    data_imputed.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessed data saved to {output_path}")

    return data_imputed

if __name__ == "__main__":
    # Dynamic path for GitHub Actions and local use
    input_csv = os.path.join("Preprocessing", "Dataset", "online_shoppers_intention.csv")
    output_csv = os.path.join("Preprocessing", "Dataset", "online_shoppers_intention_preprocessed.csv")

    df = preprocess_data(input_csv, output_csv)

    print("\n✅ Preprocessing complete.")
    print(df.head())
