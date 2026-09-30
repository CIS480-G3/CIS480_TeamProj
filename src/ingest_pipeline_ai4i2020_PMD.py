# imports
import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo # import uc irvine library to fetch and format dataset

# AI4I 2020 PMD dataset: 601
DATASET_ID = 601

# establish save directory
# python file MUST remain in /src/
REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True) # create folder if needed

PROC_DATA_DIR = REPO_ROOT / "data" / "processed"
PROC_DATA_DIR.mkdir(parents=True, exist_ok=True) # create folder if needed


# fetch dataset
ai4i_2020_pmd = fetch_ucirepo(id=DATASET_ID)
  
# data (as pandas dataframes)
data = pd.DataFrame(ai4i_2020_pmd.data.original)

# dropping UID, Product ID and target columns (twf, hdf, pwf, osf, and rnf)
model_data = data.drop(columns=["UID","Product ID","TWF","HDF","PWF","OSF","RNF"])

# one-hot encoding type to (H)igh, (M)edium, and (L)ow 
model_data = pd.get_dummies(model_data, columns=["Type"], drop_first=False)

# format column names to snake_case
model_data.columns = model_data.columns.str.lower().str.replace(" ", "_")
model_data.head()

# SAVE ALL DATA
data.to_csv(RAW_DATA_DIR / "ai4i2020.csv")
print(f"Original dataset saved to: {Path(RAW_DATA_DIR / "ai4i2020.csv")}")

data.to_parquet(RAW_DATA_DIR / "ai4i2020.parquet")
print(f"Original dataset saved to: {Path(RAW_DATA_DIR / "ai4i2020.parquet")}")

model_data.to_csv(PROC_DATA_DIR / "ai4i2020_processed.csv", index=False)
print(f"Processed dataset saved to: {Path(PROC_DATA_DIR / "ai4i2020_processed.csv")}")
