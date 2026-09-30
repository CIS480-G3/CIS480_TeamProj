# imports
import pandas as pd
import pyarrow
from pathlib import Path
from ucimlrepo import fetch_ucirepo # import uc irvine library to fetch and format dataset~

# AI4I 2020 PMD dataset: 601
DATASET_ID = 601

# establish save directory
# ! Be sure this python file is not moved from /src/ !
REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True) # create folder if needed

PROC_DATA_DIR = REPO_ROOT / "data" / "processed"
PROC_DATA_DIR.mkdir(parents=True, exist_ok=True) # create folder if needed


# fetch dataset
ai4i_2020_pmd = fetch_ucirepo(id=DATASET_ID)
  
# data (as pandas dataframes)
data = pd.DataFrame(ai4i_2020_pmd.data.original)

# OPTIONAL: the dataset includes separated feature and target sets
# X = pd.DataFrame(ai4i_2020_pmd.data.features)
# y = pd.DataFrame(ai4i_2020_pmd.data.targets) 

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

data.to_csv(PROC_DATA_DIR / "ai4i2020_processed.csv")
print(f"Processed dataset saved to: {Path(PROC_DATA_DIR / "ai4i2020_processed.csv")}")

# OPTIONAL: save the separated sets as well
# X.to_csv(RAW_DATA_DIR / "ai4i2020_features.csv")
# X.to_parquet(RAW_DATA_DIR / "ai4i2020_features.parquet")
# print(f"Feature subset saved to: {RAW_DATA_DIR}")

# y.to_csv(RAW_DATA_DIR / "ai4i2020_targets.csv")
# y.to_parquet(RAW_DATA_DIR / "ai4i2020_targets.parquet")
# print(f"Target subset saved to: {RAW_DATA_DIR}")
