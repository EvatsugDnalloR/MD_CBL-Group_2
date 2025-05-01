import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from burglary_data_2021 import process_burglary_data


# Load and process data
rooms_path = "../data/housing/number_of_rooms.xlsx"
rooms_df = pd.read_excel(rooms_path, sheet_name="2021")
rooms_df = rooms_df[rooms_df["local authority code"] != "E09000001"]

# Calculate percentages
total_col = "All households "
room_cols = [
    "1 room",
    "2 rooms",
    "3 rooms",
    "4 rooms",
    "5 rooms",
    "6 rooms",
    "7 rooms",
    "8 rooms",
    "9 or more rooms",
]

for col in room_cols:
    rooms_df[f"pct {col}"] = rooms_df[col] / rooms_df[total_col]

# Merge with burglary data
merged = rooms_df.merge(
    process_burglary_data("../data/crimes_metropolitan_2021"),
    left_on="LSOA code",
    right_on="LSOA",
    how="inner",
)

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
room_pct_cols = [f"pct {col}" for col in room_cols]

for idx, col in enumerate(room_pct_cols):
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
    ax.set_title(f"{col.title()}\n(r={corr:.2f}, p={p_val:.4f})")
    ax.set_xlabel("Percentage of Households")
    ax.set_ylabel("Burglary Count")

plt.tight_layout()
plt.show()
