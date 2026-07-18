# Patient-wise EZ classification

Six standalone GUI scripts classify EZ versus Non-EZ using four mean features:

- `Hq Value`
- `evec avg`
- `deltaH`
- `frac_Avg`

Models are Logistic Regression, Random Forest, and XGBoost for adult and pediatric datasets. Each script opens a CSV file-selection window. Results are saved beside the selected CSV in a model-specific folder.

## Validation

- 10 patient-wise outer folds
- Adult scripts use the predefined adult patient folds
- Pediatric scripts use reproducible `StratifiedGroupKFold`
- Median imputation is fitted on each training fold only
- Logistic Regression scaling is fitted on each training fold only
- SMOTE is applied only to each training fold through an `imblearn` pipeline
- Test patients remain untouched
- Figures are saved as 600-DPI PNG files

## Required CSV columns

- `Subject_ID`
- `is_soz` (`0` = Non-EZ, `1` = EZ)
- `Hq Value`
- `evec avg`
- `deltaH`
- `frac_Avg`
- `Channel` is optional

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python 01_adult_logistic_regression_smote_gui.py
```

Select the appropriate CSV when the file dialog opens.
