import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import dedent

sns.set_theme(context="paper", style="whitegrid")
sns.set_context("paper", font_scale=1.6)

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
    """Save figures in both PNG (high-DPI) and SVG (vector) for publications."""
    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(f"{fname}.svg", bbox_inches="tight")
    plt.show()

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

try:
    df = pd.read_csv("LungCapData.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
except FileNotFoundError:
    raise SystemExit("Error: 'LungCapData.csv' not found. Place the file next to this script and rerun.")

gender_levels = df["Gender"].dropna().unique().tolist() if "Gender" in df else []
smoke_levels  = df["Smoke"].dropna().unique().tolist()  if "Smoke" in df else []

gender_palette = sns.color_palette("colorblind", n_colors=max(2, len(gender_levels) or 2))
smoke_palette  = sns.color_palette("colorblind", n_colors=max(2, len(smoke_levels) or 2))

sns.set_palette("colorblind")

NEUTRAL_DOT = "0.25"

# Statistics
print("=== Sample Size ===")
print(f"N = {len(df):,}")

cont_cols = ["LungCap", "Age", "Height"]
desc = df[cont_cols].describe().T.rename(columns={
    "count": "N", "mean": "Mean", "std": "SD",
    "min": "Min", "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Max"
})
print("\n=== Continuous Variables ===")
print(desc.round(2).to_string())

def cat_table(series, title):
    vc = series.value_counts(dropna=False)
    pct = (vc / vc.sum() * 100).round(1).astype(str) + "%"
    out = pd.DataFrame({"Count": vc, "Percent": pct})
    print(f"\n=== {title} ===")
    print(out.to_string())

cat_table(df["Gender"],    "Gender")
cat_table(df["Smoke"],     "Smoking Status")
cat_table(df["Caesarean"], "Caesarean Birth")

median_age = 13 if 13 in df["Age"].unique() else df["Age"].median()
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[-float("inf"), 13, float("inf")],
    labels=["Younger (≤13)", "Older (>13)"]
)

print("\n=== Cross-tab: Smoking by Age Group ===")
ct = pd.crosstab(df["Age_Group"], df["Smoke"], margins=True)
print(ct.to_string())

row_pct = pd.crosstab(df["Age_Group"], df["Smoke"], normalize="index") * 100
print("\nRow % (each age group sums to 100%):")
print(row_pct.round(1).astype(str) + "%")

# Visualizations
# Common plot kwargs
scatter_kws = dict(s=50, edgecolor="white", linewidth=0.5, alpha=0.9)

plt.figure(figsize=FIGSIZE)
ax = sns.scatterplot(
    data=df, x="Height", y="LungCap",
    hue="Gender", style="Gender",
    hue_order=gender_levels if gender_levels else None,
    style_order=gender_levels if gender_levels else None,
    palette=gender_palette,
    **scatter_kws
)
ax.set_title("Lung Capacity vs. Height")
ax.set_xlabel("Height (inches)")
ax.set_ylabel("Lung Capacity")
ax.legend(title="Gender", frameon=True)
savefig("fig_lungcap_vs_height")

plt.figure(figsize=FIGSIZE)
ax = sns.scatterplot(
    data=df, x="Age", y="LungCap",
    hue="Gender", style="Gender",
    hue_order=gender_levels if gender_levels else None,
    style_order=gender_levels if gender_levels else None,
    palette=gender_palette,
    **scatter_kws
)
ax.set_title("Lung Capacity vs. Age")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Lung Capacity")
ax.legend(title="Gender", frameon=True)
savefig("fig_lungcap_vs_age")

plt.figure(figsize=FIGSIZE)
ax = sns.boxplot(
    data=df, x="Gender", y="LungCap",
    order=gender_levels if gender_levels else None,
    palette=gender_palette, width=0.6
)
sns.stripplot(
    data=df, x="Gender", y="LungCap",
    order=gender_levels if gender_levels else None,
    size=4, color=NEUTRAL_DOT, alpha=0.35, jitter=0.18
)
ax.set_title("Lung Capacity by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Lung Capacity")
savefig("fig_lungcap_by_gender")

plt.figure(figsize=FIGSIZE)
ax = sns.boxplot(
    data=df, x="Smoke", y="LungCap",
    order=smoke_levels if smoke_levels else None,
    palette=smoke_palette, width=0.6
)
sns.stripplot(
    data=df, x="Smoke", y="LungCap",
    order=smoke_levels if smoke_levels else None,
    size=4, color=NEUTRAL_DOT, alpha=0.35, jitter=0.18
)
ax.set_title("Lung Capacity by Smoking Status")
ax.set_xlabel("Smoker")
ax.set_ylabel("Lung Capacity")
savefig("fig_lungcap_by_smoke")