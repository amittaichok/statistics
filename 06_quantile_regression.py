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
    model_q10 = smf.quantreg(model_formula, df).fit(q=0.10)
    model_q50 = smf.quantreg(model_formula, df).fit(q=0.50)
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

ci_q10 = model_q10.conf_int().loc[coeffs_to_compare].rename(columns={0: 'CI_10_low', 1: 'CI_10_high'})
ci_q50 = model_q50.conf_int().loc[coeffs_to_compare].rename(columns={0: 'CI_50_low', 1: 'CI_50_high'})
ci_q90 = model_q90.conf_int().loc[coeffs_to_compare].rename(columns={0: 'CI_90_low', 1: 'CI_90_high'})

comparison_with_ci = pd.concat([comparison_df, ci_q10, ci_q50, ci_q90], axis=1)

print("\n--- Comparison of Coefficients with 95% Confidence Intervals ---")
pd.set_option("display.max_columns", None)   # show all columns
pd.set_option("display.width", None)         # don't wrap lines
pd.set_option("display.colheader_justify", "center")  # cleaner header alignment
print(comparison_with_ci.round(3))
