import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo


DATASET_ID = 601


def test_dataset_can_be_fetched():
    """UCI dataset 601 can be retrieved successfully."""
    dataset = fetch_ucirepo(id=DATASET_ID)

    assert dataset.data is not None
    assert dataset.data.original is not None


def test_dataset_shape():
    """The AI4I 2020 dataset contains the expected 10,000 rows."""
    dataset = fetch_ucirepo(id=DATASET_ID)
    data = pd.DataFrame(dataset.data.original)

    assert data.shape[0] == 10_000


def test_dataset_has_expected_columns():
    """The original dataset contains the expected columns."""
    dataset = fetch_ucirepo(id=DATASET_ID)
    data = pd.DataFrame(dataset.data.original)

    expected_columns = [
        "UID",
        "Product ID",
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]   

    assert list(data.columns) == expected_columns


def test_dataset_has_no_missing_values():
    """The original dataset contains no missing values."""
    dataset = fetch_ucirepo(id=DATASET_ID)
    data = pd.DataFrame(dataset.data.original)

    assert data.isnull().sum().sum() == 0


def test_product_ids_are_unique():
    """Every Product ID identifies a unique row."""
    dataset = fetch_ucirepo(id=DATASET_ID)
    data = pd.DataFrame(dataset.data.original)

    assert data["Product ID"].nunique() == len(data)


def test_processed_data_excludes_unneeded_columns():
    """Identifiers and failure-mode columns are excluded from model data."""
    dataset = fetch_ucirepo(id=DATASET_ID)
    data = pd.DataFrame(dataset.data.original)

    model_data = data.drop(
        columns=["UID", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    )

    excluded_columns = [
        "UID",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    for column in excluded_columns:
        assert column not in model_data.columns


def test_processed_data_preserves_row_count():
    """Processing does not add or remove observations."""
    dataset = fetch_ucirepo(id=DATASET_ID)
    data = pd.DataFrame(dataset.data.original)

    model_data = data.drop(
        columns=["UID", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    )

    model_data = pd.get_dummies(
        model_data,
        columns=["Type"],
        drop_first=False
    )

    assert len(model_data) == len(data)