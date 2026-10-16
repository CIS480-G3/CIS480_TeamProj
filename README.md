# Predictive Maintenance: Comparing a Trained Classifier Against a Baseline for Machine Failure Detection

## Overview
This project explores general industrial predictive maintenance, using a public, synthetic dataset. The hypothetical stakeholder is a
maintenance manager looking to systematically identify and flag likely failures preemptively.

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
2. **Install dependencies :**  
   *Optional: Create a python venv if desired:*
   ```bash
   python -m venv .venv

   source .venv/bin/activate
   ```
   **Install dependencies with:**
   ```bash
   pip install -r requirements.txt
   ``` 

3. **Recommended: Run the pipeline script**  
Open VS Code or similar and navigate to `src/pipeline.py` and run.  
** Alternative: Run the notebook file**  
Open Jupyter Lab or VS Code and navigate to `notebooks/1.0_ingest_pipeline.ipynb` to ingest data to local device.

## Team Info
* **Authors:** Anthony Fuentes, Alex Johnson, Joey Brekan, Abraham Perez
