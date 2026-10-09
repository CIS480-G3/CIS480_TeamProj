import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

# Load the pipeline CSV
df = pd.read_csv('../data/raw/ai4i2020.csv')
predictors = ['Air temperature', 'Process temperature',
              'Rotational speed', 'Torque', 'Tool wear']
X = df[predictors]
y = df['Machine failure']

# Same split as the sweep
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Baseline always predicts no failure
base = DummyClassifier(strategy='constant', constant=0)
base.fit(X_train, y_train)
base_pred = base.predict(X_test)

# Model scores at the chosen cutoff
CUTOFF = 0.67
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, 1]
model_pred = (scores >= CUTOFF).astype(int)

# Build one table with both rows
rows = []
for name, pred in [('Baseline', base_pred), ('Model at 0.67', model_pred)]:
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    rows.append({'Case': name, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
                 'Recall %': round(100 * tp / (tp + fn), 2),
                 'False alarm %': round(100 * fp / (fp + tn), 2)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))

# Save as evidence file
out.to_csv('../results/output2_matrices.csv', index=False)