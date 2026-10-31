# Model Governance and Responsible Use

**Project:** Predictive Maintenance Classification  
**Team:** CIS480 Group 3  
**Owner:** Joey Brekan  
**Last Updated:** October 28, 2026

## 1. Purpose and Scope

This project uses the UCI AI4I 2020 Predictive Maintenance dataset to demonstrate how machine learning can help identify potential equipment failures. The model is intended for educational purposes only and is not approved for real-world maintenance, production, or safety decisions.

The selected classification cutoff is **0.66**, chosen using a validation split rather than the test set. On the test set, the model detected 49 of 68 failures, achieving 72.06% recall. It missed 19 failures and produced a 10.14% false alarm rate. Precision was 20.0%, and F1 was 0.3131.

The team accepted a 10% false alarm rate as its evaluation guideline. The observed result exceeds that guideline, so the model does not meet the target. Reporting precision and F1 alongside recall provides a more complete view of model performance, but it does not change the model's predictions or resolve its limitations.

## 2. Model Governance Checklist

| Governance question | Status | Evidence |
|---|---|---|
| Was the cutoff selected without using the test set? | Yes | Validation-split cutoff selection procedure |
| Were model results compared with a baseline? | Yes | Output 2: 49 of 68 failures detected; baseline detected 0 |
| Were false alarms and missed failures documented? | Yes | Output 2: 19 missed failures and 10.14% false alarm rate |
| Were limitations and prohibited uses documented? | Yes — team review pending | Sections 3 and 4 of this document |
| Was the 10% false alarm guideline established? | Yes | Team decision |
| Did the model meet the false alarm guideline? | No | Observed rate was 10.14% |
| Was the final `demo-w10` tag created and tested? | Yes | 21 tests passed in a fresh clone |
| Were final model results verified on `demo-w10`? | Yes | Output2 reproduced the expected confusion matrix and metrics |
| Did the team review the governance requirements? | Pending | Team review and confirmation |

## 3. Intended and Prohibited Uses

**Intended uses**
- Demonstrate a machine-failure classification workflow for educational purposes.
- Compare predictions with known outcomes and evaluate recall, precision, F1, and false alarm rate.
- Demonstrate how model outputs can inform further investigation in a simulated maintenance workflow.

**Prohibited uses**
- Do not use the model to make actual maintenance, production, repair, replacement, or safety decisions.
- Do not use predictions to replace inspections, established safety procedures, or qualified maintenance personnel.
- Do not assume a negative prediction means a machine is safe or will not fail.
- Do not claim the model is validated for real industrial equipment.
- Do not report recall alone as sufficient evidence of model quality.
- Do not describe the 10% false alarm guideline as met when the observed rate is 10.14%.

The model may be demonstrated in class as an educational decision-support tool. Human judgment, inspection, and established safety procedures must remain in place.

## 4. Responsibilities and Release Conditions

| Responsibility | Owner |
|---|---|
| Governance document and test verification | Joey Brekan |
| Final repository changes, README, and release tag | Alex Johnson |
| Risk register, limitations, and Section 7 recommendation | Abraham Perez |
| Final report, evidence links, and submission | Anthony Fuentes |
| Review and approval of governance requirements | All team members |

**Real-world use:** Not approved. Representative real-world data validation, review of the false alarm trade-off and missed-failure consequences, and approval from appropriate maintenance and safety stakeholders would be required before deployment could be considered.

## 5. Testing and Evidence

The demo-m08 version previously reported 16 passing tests. The subsequent metric-reporting change added five classification-metric tests. On October 28, 2026, the test suite on main reported 21 passing tests.

The updated Output 2 reports precision and F1 alongside recall. The recorded run reported precision of 20.0% and F1 of 0.3131 for the model at the 0.66 cutoff. For the baseline, precision was reported as not applicable and F1 was 0.0.

These results verify the recorded calculations and reporting behavior. The final demo-w10 release must still be verified using a fresh clone of that tag.

Model results from `demo-m08`:

| Measure | Result |
|---|---:|
| True negatives | 1,736 |
| False positives | 196 |
| False negatives | 19 |
| True positives | 49 |
| Recall | 72.06% |
| False alarm rate | 10.14% |

Model results from `demo-w09`:

| Measure | Result |
|---|---:|
| True negatives | 1,736 |
| False positives | 196 |
| False negatives | 19 |
| True positives | 49 |
| Recall | 72.06% |
| False alarm rate | 10.14% |

Evidence:
- [Classification metric tests](https://github.com/CIS480-G3/CIS480_TeamProj/blob/ec3fa84/tests/test_classification_metrics.py)
- [Output 2 at ec3fa84](https://github.com/CIS480-G3/CIS480_TeamProj/blob/ec3fa84/data/outputs/output2_matrices.csv)
- [Pull request #3](https://github.com/CIS480-G3/CIS480_TeamProj/pull/3)

## 6. Governance Decision

**Decision: Hold for real-world use.**

The model is an educational prototype with documented limitations. The team accepted the 10% false alarm guideline, but the observed rate exceeds it. Although the updated reporting makes model quality easier to evaluate, the model still missed 19 of 68 failures, and its real-world performance is unknown.
