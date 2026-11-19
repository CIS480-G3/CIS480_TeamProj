

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
* **Libraries:** Pandas, NumPy, Scikit-learn, pytest, [see requirements.txt for more information]
* **Programs**: Python compatible IDE (VS Code), Jupyter Notebook, git

## Insights
*As of 11/14/2026*  
We sought to identify a cutoff which results in a desired false positive value of ~10% or less. With this in mind, our current model is described below:  
| Measure | Value |
|---------------|---------|
|Selected Cutoff Value| **0.66**|
|Test Recall| **72.06%**|
|False Alarm Rate| **10.14%**|
|Precision| **20%**|
|F1| **0.3131**|

A model with this cutoff value will catch 49 of 68 failures, compared to a baseline model which identified 0 of 68 failures. However, the current model incorrectly issues false alarms at a rate of 10.14%, slightly above our stated limit, and far above the baseline. Further model tuning is recommended.

------------------------

Based on all data points within the dataset, the most frequent operating conditions present are as follows:  
|            Feature|  Mutual Information Score|
|-------------------|--------------------|
|             Torque|            0.047131|
|   Rotational speed|            0.035788|
|          Tool wear|            0.009603|
|    Air temperature|            0.007452|
|Process temperature|            0.004626|

**Torque** is correlated most closely with machine failure, followed by **Rotational Speed** and **Tool Wear** as a distant third. It should be noted that this examination does not take into account the influence of combined features, such as temperature deltas, and the low MI scores across all individual features reflects this. Of these top three features, the difference between the mean normal operating condition and the reported condition at time of failure varies from -2% to +34%.

|                 | Normal Operation|  Failure Event|  Difference|  Difference (%)|
|-----------------|-----------------|---------------|------------|----------------|
|Torque           |        39.629655|      50.168142|   10.538486|       26.592425|
|Rotational speed |      1540.260014|    1496.486726|   43.773289|       -2.841942|
|Tool wear        |       106.693717|     143.781711|   37.087994|       34.761179|

## Limitations
- **Synthetic dataset:** The model was evaluated using the UCI AI4I 2020 synthetic predictive maintenance dataset. Its results may not represent actual industrial equipment, operating conditions, or failure patterns. Performance on real machine data has not been established.
- **Recall:** At the selected 0.66 cutoff, the model detected 49 of 68 failures (72.06% recall) on the test split. This result is based on one split with only 68 failures. Earlier five-seed comparisons showed recall ranging from 58.82% to 80.88%, indicating that performance can vary depending on the data split.
- **False alarms:** The model produced 196 false alarms among 1,932 non-failing machines, resulting in a 10.14% false alarm rate. This is slightly above the team's 10% evaluation guideline. The guideline is a team-selected target, not a requirement established by maintenance professionals.
- **Precision and F1:** Model precision was 20.0%, with an F1 score of 0.3131. This means most alerts in the test set did not correspond to an actual failure. These metrics should be considered alongside recall rather than presenting failure detection alone.
- **Cutoff selection:** The 0.66 cutoff was selected using a validation split rather than the test set. Although this avoids selecting the cutoff directly from test performance, the chosen threshold may not be optimal for different machines, datasets, or operational requirements.
- **Failure-rule behavior:** Overall classification metrics do not establish whether the model responds appropriately to individual failure conditions or machines near specific failure thresholds. The planned failure-rule analysis will examine this behavior, but its findings will be limited to the test rows and rules evaluated.
- **Operational use:** The model is an educational prototype, not a validated maintenance or safety system. It missed 19 of 68 failures at the selected cutoff. A negative prediction must not be interpreted as proof that equipment is safe, and model predictions must not replace inspections or qualified maintenance judgment.
- **Behavior near known failure boundaries:** The near-rules analysis found that the classifier did not consistently distinguish machines just above and below the dataset's OSF, PWF, and HDF failure boundaries. In the synthetic 1% boundary adjustments, the model alerted on both sides of the OSF, PWF low, and PWF high boundaries, while distinguishing the two HDF examples. These comparisons used only one selected test observation per boundary and are not sufficient to establish general behavior near the failure thresholds. The analysis suggests that additional feature engineering, including interaction features and machine Type, may be worth investigating.

## How to Run This Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/CIS480-G3/CIS480_TeamProj.git
   ```
   Once cloned, be sure to run all remaining commands from inside the CIS480_TeamProj directory.
   
2. **Install dependencies :**
   Requires Python 3.14 or newer. Check with `python --version`.

   *Optional: Create a python venv if desired:*
   ```bash
   python -m venv .venv

   source .venv/bin/activate
   ```
   **Install dependencies with:**
   ```bash
   pip install -r requirements.txt
   ``` 

3. **Recommended: Run the pipeline script in VS Code or similar.** Open `src/pipeline.py` and run it to ingest the data to your local device. This needs an internet connection.

   *Alternative: run it from the terminal, from the repo root:*
   ```bash
   python src/pipeline.py
   ```
   When it finishes, it prints `All files saved successfully.` and creates `data/raw/ai4i2020.csv` and `data/processed/ai4i2020_processed.csv`. If it fails, see "Data download and troubleshooting" below.

Note: `sample_input.csv` is already included in github data, and can be accessed immediately without running pipeline.py. However, all scripts that reference the full raw (`ai4i2020.csv`) or prepared (`ai4i2020_processed.csv`) datasets require -**either**- the pipeline to run before execution -**or**- modification of the script to use the sample dataset.

4. **Check the model results:**
   ```bash
   python notebooks/output2_matrices.py
   ```
   Compare the output with "Expected outputs" below.

5. **Run the tests:**
   ```bash
   pytest
   ```
   You should see `all tests pass`.

## Data download and troubleshooting
The full dataset downloads from the UCI repository through the ucimlrepo package, so running `src/pipeline.py` needs an internet connection.

If the download fails:
- No internet: connect and run `src/pipeline.py` again. The full run cannot work offline.
- Internet works but the UCI site is down: try again later.
- If it still fails, report it in a GitHub issue with the error message and what you tried.

`data/sample/sample_input.csv` is included in the repo, but it will not reproduce the expected outputs below.

## Expected outputs
Run the steps above from the repo root, in order.

1. `python src/pipeline.py` prints `All files saved successfully.` and creates `data/raw/ai4i2020.csv` and `data/processed/ai4i2020_processed.csv`. It does not print model results.
2. `python notebooks/output2_matrices.py` prints the results on the test split (2,000 machines, 20% of the 10,000 rows). You should see:

| Case | TN | FP | FN | TP | Recall % | False alarm % | Precision % | F1 |
|---|---|---|---|---|---|---|---|---|
| Baseline | 1932 | 0 | 68 | 0 | 0.00 | 0.00 | n/a | 0.0000 |
| Model at 0.66 | 1736 | 196 | 19 | 49 | 72.06 | 10.14 | 20.0 | 0.3131 |

3. `pytest` should end with `all tests pass`.

## pipeline.py usage:
Functions for fetching and loading data into current script.  

**fetch_dataset_to_csv** -> Connects to internet and downloads dataset. Saves raw and processed versions to data folder as `.csv` files.  
Usage:  
`fetch_dataset_to_csv()` saves processed and raw data to `data/processed/ai4i2020_processed.csv` and `data/raw/ai4i2020.csv` respectively.
Runs automatically when pipeline.py is run independently.

**load_data** -> Returns a DataFrame from local csv file in data/... directory. Specify which `.csv` to load ( 'processed' | 'raw' | 'sample' )  
Usage:  
`df = load_data()` returns data frame of `data/processed/ai4i2020_processed.csv`  
`df_raw = load_data('raw')` returns data frame of `data/raw/ai4i2020.csv`  
`df_sample = load_data('sample')` returns data frame of `data/sample/sample_input.csv`

**create_model** -> Returns a fitted sklearn LogisticRegression model based on the model expressed in confusion_matrix_numpy.py. Must be updated to reflect improvements or changes in features (predictors).
At present, the model does not incorporate scaling or other pipeline improvements. Until a more complex model is established, this method has very little utility.  
Usage:  
`create_model(X_train, y_train)` returns LogisticRegression model object trained on input data.

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

## Project Documentation

- [Project Handoff Guide](https://github.com/CIS480-G3/CIS480_TeamProj/tree/main/docs/HANDOFF.md)
- [Model Governance](https://github.com/CIS480-G3/CIS480_TeamProj/tree/main/docs/GOVERNANCE.md)
- [Release Checklist](https://github.com/CIS480-G3/CIS480_TeamProj/tree/main/docs/RELEASE_CHECKLIST.md)
