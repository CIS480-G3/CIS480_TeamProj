# ============================================================
# Validation split cutoff selection (Week 8)
# ============================================================
# Pick the cutoff on a validation split from the training data.
# Then score that one cutoff on the test set, once.
# Same A04 split and model as threshold_sweep.py.
# ============================================================
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# --- Paths ---------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# optional first argument overrides the csv path, used for testing
CSV_PATH = (Path(sys.argv[1]) if len(sys.argv) > 1
            else SCRIPT_DIR.parent / 'data' / 'raw' / 'ai4i2020.csv')
OUTPUT_DIR = SCRIPT_DIR.parent / 'data' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FALSE_ALARM_LIMIT = 0.10

# --- Load data -----------------------------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

predictors = [
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear'
]
X = df[predictors]
y = df['Machine failure']

# ------------------------------------------------------------
# 1. Same A04 split, 80/20 stratified, random_state=42
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 2. Split the training data again into fit and validation
# ------------------------------------------------------------
# The test set is not touched until step 5
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print("Fit rows:", len(X_fit), "| failures:", int(y_fit.sum()))
print("Validation rows:", len(X_val), "| failures:", int(y_val.sum()))
print("Test rows:", len(X_test), "| failures:", int(y_test.sum()))
print()


def rates(y_true, scores, threshold):
    """Return confusion counts and rates at one threshold."""
    # turn probabilities into 0/1 predictions
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'Threshold': round(float(threshold), 2),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'False Alarm Rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }


# ------------------------------------------------------------
# 3. Sweep thresholds on the validation split only
# ------------------------------------------------------------
fit_model = LogisticRegression(max_iter=1000, class_weight='balanced')
fit_model.fit(X_fit, y_fit)
val_scores = fit_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.951, 0.01)
val_results = pd.DataFrame([rates(y_val, val_scores, t) for t in thresholds])
val_results['False Alarm %'] = (val_results['False Alarm Rate'] * 100).round(2)
val_results['Recall %'] = (val_results['Recall'] * 100).round(2)
val_results.to_csv(OUTPUT_DIR / 'validation_threshold_sweep.csv', index=False)

# ------------------------------------------------------------
# 4. Pick the cutoff with the same team rule
# ------------------------------------------------------------
# Highest recall with false alarm rate at or under 10%
eligible = val_results[val_results['False Alarm Rate'] <= FALSE_ALARM_LIMIT]
if eligible.empty:
    raise ValueError("No validation threshold meets the false alarm limit.")

chosen = eligible.sort_values(
    ['Recall', 'Threshold'], ascending=[False, True]
).iloc[0]
cutoff = float(chosen['Threshold'])

# ------------------------------------------------------------
# 5. Score that one cutoff on the test set, once
# ------------------------------------------------------------
# Refit on all training rows, same model as Output 2
final_model = LogisticRegression(max_iter=1000, class_weight='balanced')
final_model.fit(X_train, y_train)
test_scores = final_model.predict_proba(X_test)[:, 1]
test = rates(y_test, test_scores, cutoff)

# ------------------------------------------------------------
# 6. Print results and save a small evidence file
# ------------------------------------------------------------
print("=" * 60)
print("CUTOFF CHOSEN ON VALIDATION SPLIT")
print("=" * 60)
print(f"Cutoff: {cutoff:.2f}")
print(f"Validation recall: {chosen['Recall']:.2%} "
      f"({int(chosen['TP'])} of {int(y_val.sum())})")
print(f"Validation false alarm rate: {chosen['False Alarm Rate']:.2%} "
      f"({int(chosen['FP'])} of {int((y_val == 0).sum())})")
print()
print("=" * 60)
print("SCORED ONCE ON THE TEST SET")
print("=" * 60)
print(f"TN = {test['TN']}, FP = {test['FP']}, FN = {test['FN']}, TP = {test['TP']}")
print(f"Test recall: {test['Recall']:.2%} "
      f"({test['TP']} of {int(y_test.sum())})")
print(f"Test false alarm rate: {test['False Alarm Rate']:.2%} "
      f"({test['FP']} of {int((y_test == 0).sum())})")
print()
print("Earlier cutoff picked on the test set: 0.67")

summary = pd.DataFrame([{
    'Cutoff': cutoff,
    'Val Recall %': round(chosen['Recall'] * 100, 2),
    'Val False Alarm %': round(chosen['False Alarm Rate'] * 100, 2),
    'Test TN': test['TN'], 'Test FP': test['FP'],
    'Test FN': test['FN'], 'Test TP': test['TP'],
    'Test Recall %': round(test['Recall'] * 100, 2),
    'Test False Alarm %': round(test['False Alarm Rate'] * 100, 2),
}])
summary.to_csv(OUTPUT_DIR / 'validation_cutoff_summary.csv', index=False)
print(f"Saved results to {OUTPUT_DIR}")