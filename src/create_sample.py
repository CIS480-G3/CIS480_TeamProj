"""Creates a stratified split of n (25) rows and saves it to a sample_input.csv file in data/samples.
Does not need to be ran locally. Sample file is included in git repo.
"""
import pandas as pd
import pipeline
from sklearn.model_selection import train_test_split

df = pipeline.load_data()

# create split for sample_input.csv
n_rows = 25

sample_split, _ = train_test_split(
    df,
    train_size=n_rows,
    stratify=df['Machine failure'],
    random_state=42
)

sample_split.to_csv(pipeline.SAMPLE_DATA, index=False)
