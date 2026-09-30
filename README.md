# CIS480_TeamProj
Repo for the group CIS480 data analytics project

## Overview
Predictive Maintenance: Comparing a Trained Classifier Against a Baseline for Machine Failure Detection

## Business Problem & Objectives
* **Problem:** TBD
* **Objective:** TBD

## Data Source
Source DOI: 10.24432/C5HS5C
A synthetic predictive maintenance dataset designed to allow for real-world machine failure prediction and analysis based on operating measures. Data includes 10,000 rows, 14, columns, and 0 NaN values.

                   name     role         type
0                   UID       ID      Integer   
1            Product ID       ID  Categorical   
2                  Type  Feature  Categorical   
3       Air temperature  Feature   Continuous   
4   Process temperature  Feature   Continuous   
5      Rotational speed  Feature      Integer   
6                Torque  Feature   Continuous   
7             Tool wear  Feature      Integer   
8       Machine failure   Target      Integer   
9                   TWF   Target      Integer   
10                  HDF   Target      Integer   
11                  PWF   Target      Integer   
12                  OSF   Target      Integer   
13                  RNF   Target      Integer   

## Tech Stack & Tools (Tentative)
* **Language:** Python 3.14 / R
* **Libraries:** Pandas, NumPy, Seaborn, Scikit-learn
* **BI / Visualization:** Tableau / Power BI

## Key Insights
TBD

## How to Run This Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/CIS480-G3/CIS480_TeamProj.git
   ```
2. **Install dependencies :**
   Optional: Create a python venv if desired:
   ```bash
   python -m venv .venv

   source .venv/bin/activate
   ```
   Install dependencies with:
   ```bash
   pip install -r requirements.txt
   ``` 

3. **Run the notebooks:** Open Jupyter Lab or VS Code and navigate to `notebooks/1.0_ingest_pipeline.ipynb` to begin.
   **- OR - Run the python script:** Open vs code or similar and navigate to /src/`ingest_pipeline_ai4i2020_PMD.py` and run

## Team Info
* **Authors:** Anthony Fuentes, Alex Johnson, Joey Brekan, Abraham Perez
