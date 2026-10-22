# Model Governance and Responsible Use

**Project:** Predictive Maintenance Classification  
**Team:** CIS480 Group 3  
**Owner:** Joey Brekan  
**Last Updated:** October 21, 2026

## 1. Purpose and Scope

This project uses the synthetic UCI AI4I 2020 Predictive Maintenance dataset to demonstrate how machine learning can help identify potential equipment failures. The model is intended for educational purposes only and is not approved for real-world maintenance or safety decisions.

The selected classification cutoff is **0.66**, chosen using a validation split rather than the test set. On the existing test set, the model detected 49 of 68 failures, achieving 72.06% recall. It missed 19 failures and produced a 10.14% false alarm rate.

The team accepted a 10% false alarm rate as its evaluation guideline. The current result exceeds that guideline, so the model does not meet the target.

## 2. Model Governance Checklist

| Governance question | Status | Evidence |
|---|---|---|
| Was the cutoff selected without using the test set? | Yes | Validation-split cutoff selection procedure |
| Were model results compared with a baseline? | Yes | Output 2: 49 of 68 failures detected; baseline detected 0 |
| Were false alarms and missed failures documented? | Yes | Output 2: 19 missed failures and 10.14% false alarm rate |
| Were limitations and prohibited uses documented? | Yes — team review pending | Sections 3 and 4 of this document |
| Was the 10% false alarm guideline established? | Yes | Team decision |
| Did the model meet the false alarm guideline? | No | Observed rate was 10.14% |
| Was the final `demo-w09` tag created and tested? | Pending | Final tag and test output not yet available |
| Were final model results verified on `demo-w09`? | Pending | Regenerate Output 2 and compare results |
| Did the team review the governance requirements? | Pending | Team review and confirmation |

## 3. Prohibited Uses

- Do not use the model to make actual maintenance, production, or safety decisions.
- Do not use predictions to replace inspections or qualified maintenance personnel.
- Do not assume a negative prediction means a machine is safe.
- Do not claim the model is validated for real industrial equipment.
- Do not describe the 10% false alarm guideline as met when the observed rate is 10.14%.

The model may be demonstrated in class as an educational decision-support tool. Human inspection and established safety procedures must remain in place.

## 4. Responsibilities and Release Conditions

| Responsibility | Owner |
|---|---|
| Governance document and test verification | Joey Brekan |
| Final repository changes, README, and tag | Alex Johnson |
| Risk register, limitations, and Section 7 recommendation | Abraham Perez |
| Final report, evidence links, and submission | Anthony Fuentes |
| Review and approval of governance requirements | All team members |

**Classroom demonstration:** Before the final package is considered verified, the team must create `demo-w09`, run the required tests, reproduce the model results, and review the documentation.

**Real-world use:** Not approved. Additional testing on real machine data, review of the false alarm trade-off, and approval from appropriate maintenance and safety stakeholders would be required before deployment could be considered.

## 5. Testing and Evidence

The existing `demo-m08` record reports 16 passing tests. The final `demo-w09` tag has not yet been created, so its test results remain pending.

Expected model results from `demo-m08`:

| Measure | Result |
|---|---:|
| True negatives | 1,736 |
| False positives | 196 |
| False negatives | 19 |
| True positives | 49 |
| Recall | 72.06% |
| False alarm rate | 10.14% |

Before submission:

1. Create `demo-w09` and record its commit ID.
2. Run `pytest tests/ -v` on both tags and retain the results.
3. Regenerate Output 2 on `demo-w09` and compare the confusion matrix.
4. Add before-and-after test evidence to the T09 report.
5. Obtain team review and link this document from T09.

Results from `demo-m08` must not be treated as verification of `demo-w09`.

## 6. Governance Decision

**Decision: Hold for real-world use.**

The model is an educational prototype with documented limitations. The team has accepted the 10% false alarm guideline, but the current rate exceeds it. Final-tag testing, model-result verification, and team governance review remain pending.
