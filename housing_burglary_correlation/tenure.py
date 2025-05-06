import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from burglary_data_2021 import process_burglary_data

# Load and process data
tenure_path = "../data/housing/tenure-households.xlsx"
tenure_df = pd.read_excel(tenure_path, sheet_name="2021")
tenure_df = tenure_df[tenure_df["local authority code"] != "E09000001"]

# Calculate percentages
total_col = "All Households"
tenure_cols = [
    "Owned outright",
    "Owned with a mortgage or loan",
    "Shared ownership ",
    "Rented from Local Authority",
    "Other social rented",
    "Private landlord or letting agency",
    "Other private rented",
    "Rent free",
]

for col in tenure_cols:
    tenure_df[f"pct_{col}"] = tenure_df[col] / tenure_df[total_col]

# Merge with burglary data
merged = tenure_df.merge(
    process_burglary_data("../data/crimes_metropolitan_2021"),
    left_on="LSOA code",
    right_on="LSOA",
    how="inner",
)

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
tenure_pct_cols = [f"pct_{col}" for col in tenure_cols]

for idx, col in enumerate(tenure_pct_cols):
    ax = axes.flatten()[idx]
    corr, p_val = pearsonr(merged[col], merged["burglary_count"])

    sns.regplot(
        data=merged,
        x=col,
        y="burglary_count",
        ax=ax,
        scatter_kws={"alpha": 0.3},
        line_kws={"color": "red"},
    )
    ax.set_title(f"{col.replace('pct_', '').title()}\n(r={corr:.2f}, p={p_val:.2e})")
    ax.set_xlabel("Percentage of Households")
    ax.set_ylabel("Burglary Count")

# Remove empty subplots
for idx in range(len(tenure_pct_cols), 9):
    axes.flatten()[idx].axis("off")

plt.tight_layout()
plt.show()
