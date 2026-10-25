

# Predictive Maintenance: Comparing a Trained Classifier Against a Baseline for Machine Failure Detection
This project explores general industrial predictive maintenance using a public, synthetic dataset. The hypothetical stakeholder is a maintenance manager looking to systematically identify and flag likely failures preemptively.

## Team Info
* **Authors:** Anthony Fuentes, Alex Johnson, Joey Brekan, Abraham Perez

## Business Problem & Objectives
* **Problem:** Unexpected machine failure leading to downtime.
* **Objective:** A trained classifier compared against a simple baseline. The MVP is designed to test operating measurements for a signal reliable enough to predict failures within test data.


## Data Source
Source DOI: 10.24432/C5HS5C   -   https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

A synthetic predictive maintenance dataset designed to allow for real-world machine failure prediction and analysis based on operating measures. Data includes 10,000 rows, 14, columns, and 0 NaN values.

|    | name                | role    | type        |
|----|---------------------|---------|-------------|
| 0  | UID                 | ID      | Integer     |
| 1  | Product ID          | ID      | Categorical |
| 2  | Type                | Feature | Categorical |
| 3  | Air temperature     | Feature | Continuous  |
| 4  | Process temperature | Feature | Continuous  |
| 5  | Rotational Speed    | Feature | Integer     |
| 6  | Torque              | Feature | Continuous  |
| 7  | Tool Wear           | Feature | Integer     |
| 8  | Machine Failure     | Target  | Integer     |
| 9  | TWF                 | Target  | Integer     |
| 10 | HDF                 | Target  | Integer     |
| 11 | PWF                 | Target  | Integer     |
| 12 | OSF                 | Target  | Integer     |
| 13 | RNF                 | Target  | Integer     |  

## Tech Stack & Tools
* **Language:** Python 3.14
* **Libraries:** Pandas, NumPy, Scikit-learn, [see reqirements.txt for more information]
* **Programs**: Python compatible IDE (VS Code), Jupyter Notebook, git

## Insights
*As of 10/15/2026*
Choosing a cutoff that results in a desired false positive value of 10% or less, the following was discovered:
* Selected Cutoff Value: **0.66**
* Test Recall: **72.06%**
* False Alarm Rate: **10.14%**  

A model with this cutoff value will catch 49 of 68 failures, compared to a baseline model which identified 0 of 68 failures. However, the current model incorrectly issues false alarms at a rate of 10.14%, slightly above our stated limit, and far above the baseline. Further model tuning is recommended.

## How to Run This Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/CIS480-G3/CIS480_TeamProj.git
   ```
   Once cloned, be sure to run all remaining commands from inside the CIS480_TeamProj directory.
   
3. **Install dependencies :**  
   *Optional: Create a python venv if desired:*
   ```bash
   python -m venv .venv

   source .venv/bin/activate
   ```
   **Install dependencies with:**
   ```bash
   pip install -r requirements.txt
   ``` 

4. **Recommended: Run the pipeline script**  
Open VS Code or similar and navigate to `src/pipeline.py` and **run script to ingest data to local device**.

Note: `sample_input.csv` is already included in github data, and can be accessed immediately without running pipeline.py. However, all scripts that reference the full raw (`ai4i21020.csv`) or prepared (`ai4i21020_processed.csv`) datasets require -**either**- the pipeline to run before execution -**or**- modification of the script to use the sample dataset.

## pipeline.py usage:
Functions for fetching and loading data into current script.
**fetch_dataset_to_csv** -> Connects to internet and downloads dataset. Saves raw and processed versions to data folder as `.csv` files. 
Usage: 
`fetch_dataset_to_csv()` 
Runs automatically when pipeline.py is run independently.

**load_data** -> Returns a DataFrame from local csv file in data/... directory. Specify which `.csv` to load ( 'processed' | 'raw' | 'sample' )
Usage:
`df = load_data()` returns data frame of `data/processed/ai4i2020_processed.csv`
`df_raw = load_data('raw')` returns data frame of `data/raw/ai4i2020.csv`
`df_sample = load_data('sample')` returns data frame of `data/sample/sample_input.csv`


**Constants:**
| var_name         | purpose		|
|------------------|------------|
| DATASET_ID       | ID of Dataset in ucimlrepo|
| PREDICTORS       | List of dataset predictors stored as str: 'Air temperature','Process temperature', 'Rotational speed','Torque', and 'Tool wear'|
| TARGET           | str: 'Machine failure'|
| REPO_ROOT        | OS agnostic Path object to root (`/`) directory path |
| DATA_DIR         | OS agnostic Path object to `data/` directory path |
| OUTPUTS_DATA_DIR | OS agnostic Path object to `data/outputs/` directory path |
| PROC_DATA_DIR    | OS agnostic Path object to `data/processed/` directory path |
| PROCESSED_DATA   | OS agnostic Path object to `data/processed/ai4i2020_processed.csv` file path |
| RAW_DATA_DIR     | OS agnostic Path object to `data/raw/` directory path |
| RAW_DATA         | OS agnostic Path object to `data/raw/ai4i2020.csv` file path |
| SAMPLE_DATA_DIR  | OS agnostic Path object to `data/sample/` directory path |
| SAMPLE_DATA      | OS agnostic Path object to `data/sample/sample_input.csv` file path |
