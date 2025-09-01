import pandas as pd
from scipy.stats import ttest_ind, levene, mannwhitneyu, chi2_contingency

try:
    df = pd.read_csv('LungCapData.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
except FileNotFoundError:
    print("Error: 'LungCapData.csv' not found. Please ensure the file is in the correct folder.")
    exit()

# Analysis for Gender vs. Lung Capacity
print("--- Analysis: Gender vs. Lung Capacity ---")
male_lungcap = df[df['Gender'] == 'male']['LungCap']
female_lungcap = df[df['Gender'] == 'female']['LungCap']

# T-test
t_stat_gender, p_val_gender = ttest_ind(male_lungcap, female_lungcap)
print(f"T-test result: t-statistic = {t_stat_gender:.3f}, p-value = {p_val_gender:.3e}")

# Assumption Check
levene_stat_gender, p_val_levene_gender = levene(male_lungcap, female_lungcap)
print(f"Levene's test for equal variances: p-value = {p_val_levene_gender:.3f}")

# Mann-Whitney U test
mwu_stat_gender, p_val_mwu_gender = mannwhitneyu(male_lungcap, female_lungcap, alternative='two-sided')
print(f"Mann-Whitney U test: p-value = {p_val_mwu_gender:.3e}")


# Analysis for Smoke vs. Lung Capacity
print("\n--- Analysis: Smoke vs. Lung Capacity ---")
smoker_lungcap = df[df['Smoke'] == 'yes']['LungCap']
nonsmoker_lungcap = df[df['Smoke'] == 'no']['LungCap']

# T-test
t_stat_smoke, p_val_smoke = ttest_ind(smoker_lungcap, nonsmoker_lungcap)
print(f"T-test result: t-statistic = {t_stat_smoke:.3f}, p-value = {p_val_smoke:.3f}")

# Assumption Check
levene_stat_smoke, p_val_levene_smoke = levene(smoker_lungcap, nonsmoker_lungcap)
print(f"Levene's test for equal variances: p-value = {p_val_levene_smoke:.3f}")

# Mann-Whitney U test
mwu_stat_smoke, p_val_mwu_smoke = mannwhitneyu(smoker_lungcap, nonsmoker_lungcap, alternative='two-sided')
print(f"Mann-Whitney U test: p-value = {p_val_mwu_smoke:.3f}")


# Analysis for Gender vs. Smoke
print("\n--- Analysis: Gender vs. Smoke ---")
contingency_table = pd.crosstab(df['Gender'], df['Smoke'])

# Chi-squared test of independence
chi2, p_val_chi2, dof, expected_freq = chi2_contingency(contingency_table)
print(f"Chi-squared test result: chi2-statistic = {chi2:.3f}, p-value = {p_val_chi2:.3f}")

# Assumption check
min_expected_freq = expected_freq.min()
print(f"Minimum expected frequency for Chi-squared test: {min_expected_freq:.2f}")

