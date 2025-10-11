# Loan Application Prediction
This repository contains a Jupyter Notebook (`LoanApplication.ipynb`) for analyzing loan application data to predict loan success or risk (high-risk loans labeled as "bad"). The project involves data preprocessing, feature engineering, and exploratory analysis to support credit risk modeling, potentially using logistic regression or scorecard techniques.
## Overview
The notebook processes historical loan application data and credit features to:

- Merge datasets and define a binary target variable (`TARGET`: 1 for high-risk loans where `Success`=0, 0 otherwise).
- Perform exploratory data analysis, including summary statistics and bad rate visualization.
- Set up for machine learning tasks like regression or classification (not fully implemented in the provided code).

The goal is to build a predictive model for credit risk, commonly used in financial institutions for loan approval decisions.
## Dataset
The notebook uses three CSV files downloaded from Google Drive:

- `loan_application.csv`: Contains loan application details (8847 entries, 7 columns):
- `UID`: Unique identifier
- `ApplicationDate`: Date of application
- `Amount`: Requested loan amount (500 to 20,000)
- `Term`: Repayment period (12 to 60 months)
- `EmploymentType`: Stated employment
- `LoanPurpose`: Purpose of the loan
- `Success`: Loan approval status (0 or 1)


`credit_features_subset.csv`: Credit bureau features (8847 entries, 14 columns):
Metrics like `ALL_AgeOfOldestAccount`, `ALL_CountActive`, `ALL_SumCurrentOutstandingBal`, etc.
Includes account counts, ages, balances, and default history.


`loan_data_dictionary.csv`: Data dictionary with column descriptions (20 entries, 2 columns: Name, Description).

Note: Some features contain negative values (-1), likely indicating missing data or special cases, which should be handled during preprocessing.
## Dependencies
The notebook requires the following Python packages:

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib and seaborn`: Visualization
- `scikit-learn`: Machine learning utilities
- `statsmodels`: Statistical modeling
- `gdown`: Download files from Google Drive
- `scorecardpy`: Credit scorecard development (WoE binning, Information Value)

**Install them using:**
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels gdown scorecardpy

## Notebook Structure

### 1. Setup:

Installs required libraries.
Imports libraries for data processing, visualization, and modeling.


### 2. Data Loading:

Downloads and loads the three CSV files.
Displays dataset info (e.g., column types, non-null counts).


### 3. Data Preprocessing:

Merges `loan_application.csv` and `credit_features_subset.csv` on `UID`.
Defines `TARGET` as `1 - Success` (1 for failed/high-risk loans).
Drops `UID`, `Success`, and `ApplicationDate`.
Prints the schema from `loan_data_dictionary.csv`.


### 4. Exploratory Analysis:

Generates summary statistics for numerical features (e.g., mean loan amount ~7560, term ~42 months).
Bins a `Score` column (assumed from a model) into 10 quantiles.
Calculates bad rate (proportion of `TARGET=1`) per bin.
Plots bad rate vs. score bins to check monotonicity (higher scores should have lower bad rates).



## Key Outputs

Summary Statistics: Descriptive stats for numerical columns, showing ranges and potential data issues (e.g., negative values).
Bad Rate Plot: Visualizes bad rates across score bins to validate model behavior (monotonic decrease is desirable).
Schema: Displays column descriptions for interpretability.



## Notes

Truncation: The provided notebook is truncated (e.g., plot output is incomplete). The full version likely includes model training (e.g., logistic regression or scorecard).
Data Quality: Negative values (-1) in credit features may require preprocessing (e.g., imputation or special handling).
Next Steps: Extend the notebook to include:
Feature binning using scorecardpy.woebin.
Logistic regression or scorecard model training.
Model evaluation (e.g., ROC AUC, confusion matrix).

