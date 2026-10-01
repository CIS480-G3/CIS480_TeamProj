import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


DATA_PATH = "data/raw/ai4i2020.csv"

PREDICTORS = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

FORBIDDEN_COLUMNS = [
    "UID",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Machine failure",
]


def load_data():
    """Load the raw AI4I dataset."""
    return pd.read_csv(DATA_PATH)


def create_split():
    """Create the same reproducible train/test split used by the model."""
    df = load_data()

    X = df[PREDICTORS]
    y = df["Machine failure"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def test_expected_predictors_exist():
    """All five model predictors exist in the dataset."""
    df = load_data()

    for column in PREDICTORS:
        assert column in df.columns


def test_no_leakage_columns_are_predictors():
    """Identifiers, failure flags, and the target are not used as predictors."""
    df = load_data()
    X = df[PREDICTORS]

    for column in FORBIDDEN_COLUMNS:
        assert column not in X.columns


def test_train_test_split_size():
    """The dataset is split into 80% training and 20% testing data."""
    X_train, X_test, y_train, y_test = create_split()

    assert len(X_train) == 8000
    assert len(X_test) == 2000
    assert len(y_train) == 8000
    assert len(y_test) == 2000


def test_failure_exists_in_train_and_test():
    """Both datasets contain machine failures."""
    X_train, X_test, y_train, y_test = create_split()

    assert y_train.sum() > 0
    assert y_test.sum() > 0


def test_baseline_always_predicts_no_failure():
    """The baseline predicts zero for every test observation."""
    X_train, X_test, y_train, y_test = create_split()

    baseline = DummyClassifier(
        strategy="constant",
        constant=0
    )

    baseline.fit(X_train, y_train)
    predictions = baseline.predict(X_test)

    assert len(predictions) == len(y_test)
    assert (predictions == 0).all()


def test_logistic_regression_runs():
    """Logistic regression can train and generate predictions."""
    X_train, X_test, y_train, y_test = create_split()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_model_predictions_are_binary():
    """Model predictions are valid binary classification outputs."""
    X_train, X_test, y_train, y_test = create_split()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert set(predictions).issubset({0, 1})


def test_model_probabilities_are_valid():
    """Predicted failure probabilities fall between 0 and 1."""
    X_train, X_test, y_train, y_test = create_split()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    assert len(probabilities) == len(y_test)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()