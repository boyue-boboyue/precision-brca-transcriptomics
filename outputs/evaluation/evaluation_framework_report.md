# Locked evaluation split report

Generated: 2026-09-09T16:11:27Z  
Random seed: `20260909`  
Analysis unit: patient (`case_barcode`)

## Locked partitions

- Primary four-class cohort: 945 patients; 756 development and 189 locked test.
- Five-class sensitivity cohort: 981 patients; 785 development and 196 locked test.
- The four-class assignments are unchanged inside the five-class sensitivity file; Normal-like patients were added with the same 80/20 target.
- No duplicate patient groups are present in the current case-level matrix. The generator nevertheless assigns folds at patient-group level and rejects conflicting labels within a patient.

## Nested cross-validation

- Outer loop: 5-fold stratified patient-level CV on the development set.
- Inner loop: 5-fold stratified patient-level CV separately precomputed inside every outer-training partition.
- `outer_fold` identifies the outer validation fold. `inner_fold_outer_K` is populated only for patients eligible for training in outer fold K.
- Locked-test rows have no outer or inner assignment and are rejected by the development-only runner.

## Leakage boundary

Every candidate estimator is a single scikit-learn Pipeline in this order: low-expression filter, median imputation, top-variance feature selection, standard scaling, classifier. Each transformer is fitted by GridSearchCV only on the relevant inner-training fold.

No model was fitted and the locked test set was not evaluated while creating this report.
