""" Functions for fetching and loading data into current script.

    **fetch_dataset_to_csv** -> Connects to internet and downloads dataset. Saves raw and processed versions to data folder as .csv files.

    **load_data** -> Returns a DataFrame from local csv file in data/... directory. Specify which csv to load ( 'processed' | 'raw' | 'sample' )
"""
import pandas as pd
from pathlib import Path
from typing import Literal

# ================== CONSTANTS =====================
# load_data source options
_SOURCES = Literal['processed','raw', 'sample']

# AI4I 2020 PMD dataset: 601
DATASET_ID = 601

# Model PREDICTORS and TARGET vars
PREDICTORS = [
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear'
]
TARGET = 'Machine failure'

# Directory reference constants (ONLY WORKS IF __file__ IS IN src/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'

OUTPUTS_DATA_DIR = REPO_ROOT / 'data' / 'outputs'

PROC_DATA_DIR = REPO_ROOT / 'data' / 'processed'
PROCESSED_DATA = PROC_DATA_DIR / 'ai4i2020_processed.csv'

RAW_DATA_DIR = REPO_ROOT / 'data' / 'raw'
RAW_DATA = RAW_DATA_DIR / 'ai4i2020.csv'

SAMPLE_DATA_DIR = REPO_ROOT / 'data' / 'sample'
SAMPLE_DATA = SAMPLE_DATA_DIR / 'sample_input.csv'


# ================== METHODS =====================
def load_data(source: _SOURCES = 'processed') -> pd.DataFrame:
    """Returns a DataFrame from local csv file in data/... directory. | source = 'processed', 'raw', or 'sample'.
    
    Raises FileNotFoundError if file doesn't exist at data/...
    """
    # select source path
    match source:
        case 'processed':
            data = PROCESSED_DATA
        case 'raw':
            data = RAW_DATA
        case 'sample':
            data = SAMPLE_DATA
        case _:
            raise(TypeError)

    # check if file exists
    if not data.is_file():
        raise FileNotFoundError(f'CSV file not found at: {data.resolve()}')

    return pd.read_csv(data)


def fetch_dataset_to_csv() -> str:
    """Fetches data from source, and creates processed version. Saves both to csv at data/...
    Will OVERWRITE previous raw and processed csv files!    
    Raises FileNotFoundError if file doesn't exist at location after calling save
    """
    from ucimlrepo import fetch_ucirepo # import uc irvine library to fetch and format dataset

    # establish save directory
    PROC_DATA_DIR.mkdir(parents=True, exist_ok=True) # create folder if needed
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True) # create folder if needed

    # fetch dataset
    ai4i_2020_pmd = fetch_ucirepo(id=DATASET_ID)
    
    # data (as pandas dataframes)
    assert ai4i_2020_pmd.data is not None
    data = pd.DataFrame(ai4i_2020_pmd.data.original)

    # dropping UID, Product ID and target columns (twf, hdf, pwf, osf, and rnf)
    model_data = data.drop(columns=['UID','Product ID','TWF','HDF','PWF','OSF','RNF'])

    # one-hot encoding type to (H)igh, (M)edium, and (L)ow 
    model_data = pd.get_dummies(model_data, columns=['Type'], drop_first=False)

    # format column names to snake_case
    # model_data.columns = model_data.columns.str.lower().str.replace(' ', '_')

    # SAVE ALL DATA
    data.to_csv(RAW_DATA, index=False)
    if not RAW_DATA.is_file():
            raise FileNotFoundError(f'CSV file not found at: {RAW_DATA.resolve()}')

    model_data.to_csv(PROCESSED_DATA, index=False)
    if not PROCESSED_DATA.is_file():
            raise FileNotFoundError(f'CSV file not found at: {PROCESSED_DATA.resolve()}')

    return f'All files saved successfully. ({RAW_DATA},{PROCESSED_DATA})'


# if run independently, save files to data folder
if __name__ == '__main__':
    print(fetch_dataset_to_csv())