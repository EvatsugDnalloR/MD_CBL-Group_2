import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from burglary_data_2021 import process_burglary_data


# Load and process data
occ_path = "../data/housing/occupancy_rating-bedrooms.xlsx"
occ_df = pd.read_excel(occ_path, sheet_name="2021")
occ_df = occ_df[occ_df["local authority code"] != "E09000001"]

# Calculate percentages
total_col = "All Households"
rating_cols = ["Occupancy rating: +2 or more", "+1", "0", "-1", "-2 or less"]

for col in rating_cols:
    occ_df[f"pct_{col}"] = occ_df[col] / occ_df[total_col]

# Create combined overcrowding metric
occ_df["pct_overcrowded"] = occ_df["pct_-1"] + occ_df["pct_-2 or less"]

# Merge with burglary data
merged = occ_df.merge(
    process_burglary_data("../data/crimes_metropolitan_2021"),
    left_on="LSOA code",
    right_on="LSOA",
    how="inner",
)

# Visualization
plot_cols = [
    "pct_Occupancy rating: +2 or more",
    "pct_+1",
    "pct_0",
    "pct_-1",
    "pct_-2 or less",
    "pct_overcrowded",
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, col in enumerate(plot_cols):
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
    ax.set_title(
        f"{col.replace('pct_', '').title()}\n(r={corr:.2f}, p={p_val:.4f})"
    )
    ax.set_xlabel("Percentage of Households")
    ax.set_ylabel("Burglary Count")

plt.tight_layout()
plt.show()
