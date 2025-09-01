import pandas as pd
import statsmodels.formula.api as smf

try:
    df = pd.read_csv('LungCapData.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
except FileNotFoundError:
    print("Error: 'LungCapData.csv' not found. Please ensure the file is in the correct folder.")
    exit()

model_formula = 'LungCap ~ Age + Height + C(Smoke) + C(Gender) + C(Caesarean)'

try:
    # Model for the 10th percentile (the "lower end")
    model_q10 = smf.quantreg(model_formula, df).fit(q=0.10)

    # Model for the 50th percentile (the "median")
    model_q50 = smf.quantreg(model_formula, df).fit(q=0.50)

    # Model for the 90th percentile (the "higher end")
    model_q90 = smf.quantreg(model_formula, df).fit(q=0.90)

    print("Successfully fitted all three Quantile Regression models.")
except Exception as e:
    print(f"An error occurred during model fitting: {e}")
    exit()

coeffs_to_compare = ['Height', 'C(Smoke)[T.yes]']

comparison_df = pd.DataFrame({
    'Quantile_10': model_q10.params.loc[coeffs_to_compare],
    'Quantile_50_Median': model_q50.params.loc[coeffs_to_compare],
    'Quantile_90': model_q90.params.loc[coeffs_to_compare]
})

print("\n--- Comparison of Coefficients across Quantiles ---")
print(comparison_df.round(3))