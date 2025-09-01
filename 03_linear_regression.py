import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

CSV = "LungCapData.csv"

sns.set_theme(context="paper", style="whitegrid")
sns.set_context("paper", font_scale=1.6)  # bump all fonts at once

plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})

FIGSIZE = (8.5, 6.0)
DPI = 300

def savefig(fname):
    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(f"{fname}.svg", bbox_inches="tight")
    plt.show()

sns.set_palette("colorblind")
CB = sns.color_palette("colorblind")
c_scatter = CB[0]   # points
c_line    = CB[3]   # smooth/line
c_zero    = "0.2"   # neutral gray for zero-line

try:
    df = pd.read_csv(CSV)
except FileNotFoundError:
    print(f"Error: '{CSV}' not found. Put it next to this script.")
    sys.exit(1)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

for col in ["LungCap", "Age", "Height"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def yes_no(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower()
              .replace({"y":"yes","n":"no","1":"yes","0":"no","true":"yes","false":"no"})
              .replace({"yes":"Yes","no":"No"}))

def male_female(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower().replace({"m":"male","f":"female"})
    return s.replace({"male":"Male","female":"Female"})

if "Smoke" in df.columns:
    df["Smoke"] = yes_no(df["Smoke"]).astype("category")
if "Caesarean" in df.columns:
    df["Caesarean"] = yes_no(df["Caesarean"]).astype("category")
if "Gender" in df.columns:
    df["Gender"] = male_female(df["Gender"]).astype("category")

needed = ["LungCap", "Age", "Height", "Smoke", "Gender", "Caesarean"]
missing = [c for c in needed if c not in df.columns]
if missing:
    print(f"Error: Missing required columns: {missing}")
    sys.exit(1)

df = df.dropna(subset=needed).copy()

# Fit model
formula = "LungCap ~ Age + Height + C(Smoke) + C(Gender) + C(Caesarean)"
model = smf.ols(formula=formula, data=df).fit()

robust = model.get_robustcov_results(cov_type="HC3")

terms = list(model.params.index)
est = robust.params
se = robust.bse
ci = robust.conf_int()
if hasattr(ci, "iloc"):
    ci_low = ci.iloc[:, 0].to_numpy()
    ci_high = ci.iloc[:, 1].to_numpy()
else:
    ci_low = ci[:, 0]
    ci_high = ci[:, 1]
pvals = robust.pvalues

out = pd.DataFrame({
    "Term": terms,
    "Estimate": est,
    "Std. Error (HC3)": se,
    "95% CI Low": ci_low,
    "95% CI High": ci_high,
    "p-value": pvals
})

out["Term"] = (out["Term"]
    .str.replace(r"C\(Smoke\)\[T\.Yes\]", "Smoke: Yes vs No", regex=True)
    .str.replace(r"C\(Gender\)\[T\.Male\]", "Gender: Male vs Female", regex=True)
    .str.replace(r"C\(Caesarean\)\[T\.Yes\]", "Caesarean: Yes vs No", regex=True)
    .str.replace(r"^Intercept$", "Intercept", regex=True)
)

out = out.round({"Estimate":3, "Std. Error (HC3)":3, "95% CI Low":3, "95% CI High":3, "p-value":3})

print("\n=== Multiple Linear Regression: LungCap ~ Age + Height + Smoke + Gender + Caesarean ===")
print(f"n = {len(df)}, R^2 = {model.rsquared:.3f}, Adj. R^2 = {model.rsquared_adj:.3f}")
print(out.to_string(index=False))

print("\n--- Full OLS summary (classical SEs) ---\n")
print(model.summary())

# Diagnostics
resid  = model.resid
fitted = model.fittedvalues

fig, ax = plt.subplots(figsize=FIGSIZE)
sm.qqplot(resid, line="45", ax=ax, color=c_scatter, alpha=0.8)
if len(ax.lines) > 1:
    ax.lines[1].set_color(c_line)
    ax.lines[1].set_alpha(0.9)
    ax.lines[1].set_linewidth(2)

ax.set_title("Q–Q Plot of Residuals")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.tick_params(axis="both", labelsize=plt.rcParams["xtick.labelsize"])
plt.tight_layout()
plt.show()

plt.figure(figsize=FIGSIZE)
sns.residplot(
    x=fitted, y=resid, lowess=True,
    scatter_kws={"alpha": 0.7, "s": 50, "color": c_scatter,
                 "edgecolor": "white", "linewidths": 0.4},  # <-- fixed here
    line_kws={"color": c_line, "lw": 2, "alpha": 0.9}
)
plt.axhline(0, color=c_zero, lw=1.2, alpha=0.8)
plt.title("Residuals vs Fitted")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.tick_params(axis="both", labelsize=plt.rcParams["xtick.labelsize"])
plt.tight_layout()
plt.show()