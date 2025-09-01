import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols   # <-- fixed

try:
    df = pd.read_csv('LungCapData.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
except FileNotFoundError:
    print("Error: 'LungCapData.csv' not found. Please ensure the file is in the correct folder.")
    exit()

print("--- Results for Polynomial Model (Check p-value for I(Age**2)) ---")

polynomial_model_formula = 'LungCap ~ Age + I(Age**2) + Height + C(Smoke) + C(Gender) + C(Caesarean)'
polynomial_model = ols(polynomial_model_formula, data=df).fit()
print(polynomial_model.summary())

print("\n\n--- Results for Likelihood-Ratio Test ---")

full_model_formula = 'LungCap ~ Age + Height + C(Smoke) + C(Gender) + C(Caesarean)'
full_model = ols(full_model_formula, data=df).fit()

reduced_model_formula = 'LungCap ~ Age + Height'
reduced_model = ols(reduced_model_formula, data=df).fit()

lr_test_results = sm.stats.anova_lm(reduced_model, full_model)
print(lr_test_results)
