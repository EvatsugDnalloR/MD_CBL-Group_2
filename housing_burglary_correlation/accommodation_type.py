import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from burglary_data_2021 import process_burglary_data

# Load accommodation data
acc_path = "../data/housing/accommodation_type.xlsx"
acc_df = pd.read_excel(acc_path, sheet_name="2021")

# Filter out City of London (E09000001)
acc_df = acc_df[acc_df['local authority code'] != 'E09000001']

# Calculate total households per LSOA
acc_df['total_households'] = acc_df['All households ']

# Select relevant columns
acc_cols = [
    'LSOA code',
    'Detached',
    'Semi-detached',
    'Terraced',
    'Purpose built flat',
    'Flat in a converted/ shared house (includes all households in shared dwellings)',
    'Flat in a commercial building',
    'total_households'
]
acc_df = acc_df[acc_cols]

# Rename columns for simplicity
acc_df.columns = [
    'LSOA',
    'detached',
    'semi_detached',
    'terraced',
    'purpose_built_flats',
    'shared_flats',
    'commercial_flats',
    'total_households'
]

# Merge accommodation and burglary data
merged_df = acc_df.merge(process_burglary_data("../data/crimes_metropolitan_2021"), on='LSOA', how='inner')


# Calculate correlations for each housing type
housing_types = [
    "detached",
    "semi_detached",
    "terraced",
    "purpose_built_flats",
    "shared_flats",
    "commercial_flats",
]

# Create figure grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()

# Plot each housing type
for idx, housing_type in enumerate(housing_types):
    ax = axes[idx]

    # Calculate correlation
    corr, p_value = pearsonr(merged_df[housing_type], merged_df["burglary_count"])

    # Create plot
    sns.regplot(
        data=merged_df,
        x=housing_type,
        y="burglary_count",
        ax=ax,
        scatter_kws={"alpha": 0.4, "color": "steelblue"},
        line_kws={"color": "darkred"},
    )

    # Formatting
    title = f"{housing_type.replace('_', ' ').title()}\n(r={corr:.2f}, p={p_value:.4f})"
    ax.set_title(title)
    ax.set_xlabel("Number of Houses")
    ax.set_ylabel('Burglary Count')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
