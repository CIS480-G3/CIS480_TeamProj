# ============================================================
# T07 Threshold Sweep
# ============================================================
# Requirements:
#   - Same split as A04: 80/20 stratified, random_state=42
#   - Balanced logistic regression (class_weight='balanced')
#   - Test set: 2,000 machines (68 failures, 1,932 healthy)
#   - Threshold sweep 0.05 → 0.95
#   - Select highest recall where false alarm rate ≤ 10%
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# 1. Load data and recreate the A04 model
# ------------------------------------------------------------

# Adjust the path if your data lives elsewhere
df = pd.read_csv('../data/raw/ai4i2020.csv')

predictors = [
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear'
]

X = df[predictors]
y = df['Machine failure']

print("Full dataset shape:", df.shape)
print("Total failures:", y.sum())
print("Overall failure rate:", f"{y.mean():.2%}")
print()

# Same split as A04: 80/20 stratified, random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Balanced logistic regression (same as A04)
weighted_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

weighted_model.fit(X_train, y_train)

# ------------------------------------------------------------
# 2. Get the model's predicted probabilities on the test set
# ------------------------------------------------------------

# Probability of failure for each test machine
y_prob = weighted_model.predict_proba(X_test)[:, 1]

# Sanity checks — these must match T07's stated denominators
n_test = len(y_test)
n_failures = int(y_test.sum())
n_healthy = int((y_test == 0).sum())

print("Test set size:", n_test)
print("Real failures in test set:", n_failures)
print("Healthy machines in test set:", n_healthy)
print("First 10 predicted probabilities:", np.round(y_prob[:10], 3))
print()

# ------------------------------------------------------------
# 3. Threshold sweep 0.05 → 0.95
# ------------------------------------------------------------

thresholds = np.arange(0.05, 0.951, 0.01)

results = []

for threshold in thresholds:
    # Convert probabilities into 0/1 predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix (labels=[0,1] guarantees ordering)
    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    # Rates
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    results.append({
        'Threshold': round(float(threshold), 2),
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'Healthy Flagged': fp,
        'False Alarm Rate': false_alarm_rate,
        'Failures Caught': tp,
        'Recall': recall
    })

threshold_results = pd.DataFrame(results)

# Add readable percentage columns
threshold_results['False Alarm %'] = (
    threshold_results['False Alarm Rate'] * 100
).round(2)
threshold_results['Recall %'] = (
    threshold_results['Recall'] * 100
).round(2)

# ------------------------------------------------------------
# 4. Select the recommended cutoff
# ------------------------------------------------------------
# Team rule: highest recall among thresholds with FAR ≤ 10%

eligible = threshold_results[
    threshold_results['False Alarm Rate'] <= 0.10
]

if eligible.empty:
    raise ValueError(
        "No threshold satisfies the 10% false alarm constraint. "
        "Check the model or the split."
    )

recommended = eligible.sort_values(
    ['Recall', 'Threshold'],
    ascending=[False, True]
).iloc[0]

# ------------------------------------------------------------
# 5. Print the threshold table (Output 1)
# ------------------------------------------------------------

print("=" * 70)
print("THRESHOLD TABLE (T07 Output 1)")
print("=" * 70)

table_view = threshold_results[[
    'Threshold',
    'Healthy Flagged',
    'False Alarm %',
    'Failures Caught',
    'Recall %'
]].copy()

# Show a readable slice of the table around the decision region.
# (Full table is available as `threshold_results`.)
print(table_view.to_string(index=False))
print()

# ------------------------------------------------------------
# 6. Print the recommended cutoff and model confusion counts
# ------------------------------------------------------------

print("=" * 70)
print("RECOMMENDED CUTOFF")
print("=" * 70)
print(f"Threshold:            {recommended['Threshold']:.2f}")
print(f"Healthy flagged (FP): {int(recommended['FP'])} / {n_healthy}"
      f"  ({recommended['False Alarm Rate']:.2%})")
print(f"Failures caught (TP): {int(recommended['TP'])} / {n_failures}"
      f"  ({recommended['Recall']:.2%})")
print()
print("Model confusion matrix at chosen cutoff:")
print(f"  TN = {int(recommended['TN'])} / {n_healthy} healthy")
print(f"  FP = {int(recommended['FP'])} / {n_healthy} healthy")
print(f"  FN = {int(recommended['FN'])} / {n_failures} failures")
print(f"  TP = {int(recommended['TP'])} / {n_failures} failures")
print()
print("Baseline (always predict no failure):")
print(f"  TN = {n_healthy} / {n_healthy} healthy")
print(f"  FP = 0 / {n_healthy} healthy")
print(f"  FN = {n_failures} / {n_failures} failures")
print(f"  TP = 0 / {n_failures} failures")
print()

# ------------------------------------------------------------
# 7. Figure 1 — Recall and False Alarm Rate vs. Threshold
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_results['Threshold'],
    threshold_results['Recall %'],
    marker='o',
    markersize=3,
    label='Recall'
)

plt.plot(
    threshold_results['Threshold'],
    threshold_results['False Alarm %'],
    marker='o',
    markersize=3,
    label='False Alarm Rate'
)

plt.axhline(
    10,
    linestyle='--',
    color='red',
    label='10% False Alarm Limit'
)

plt.axvline(
    recommended['Threshold'],
    linestyle='--',
    color='green',
    label=f"Recommended Cutoff ({recommended['Threshold']:.2f})"
)

plt.xlabel('Classification Threshold')
plt.ylabel('Rate (%)')
plt.title('Figure 1. Recall and False Alarm Rate Across Classification Thresholds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. Ready-to-paste report sentence for Section 5.2
# ------------------------------------------------------------

print("=" * 70)
print("SECTION 5.2 REPLACEMENT TEXT")
print("=" * 70)
print(
    f"Among the tested thresholds from 0.05 through 0.95, the "
    f"recommended cutoff was {recommended['Threshold']:.2f}. "
    f"At this cutoff, the model flagged {int(recommended['FP'])} of "
    f"{n_healthy} healthy machines "
    f"({recommended['False Alarm Rate']:.2%} false alarm rate) and "
    f"caught {int(recommended['TP'])} of {n_failures} real failures "
    f"({recommended['Recall']:.2%} recall). This cutoff was selected "
    f"because it produced the highest recall while keeping the false "
    f"alarm rate at or below the team's 10% limit."
)