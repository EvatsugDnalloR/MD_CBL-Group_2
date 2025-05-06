import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from burglary_data_2021 import process_burglary_data

# Load car data
cars_path = "../data/housing/num_cars_or_vans.xlsx"
cars_df = pd.read_excel(cars_path, sheet_name="2021")

# Filter out City of London
cars_df = cars_df[cars_df['local authority code'] != 'E09000001']

# Create meaningful percentages
cars_df['pct_no_cars'] = cars_df['none'] / cars_df['All households ']
cars_df['pct_1_car'] = cars_df['one'] / cars_df['All households ']
cars_df['pct_2_cars'] = cars_df['two'] / cars_df['All households ']
cars_df['pct_3plus_cars'] = cars_df['three or more'] / cars_df['All households ']

# Select relevant columns
cars_df = cars_df[['LSOA code', 'pct_no_cars', 'pct_1_car', 'pct_2_cars', 'pct_3plus_cars']]

cars_df.columns = ['LSOA', 'pct_no_cars', 'pct_1_car', 'pct_2_cars', 'pct_3plus_cars']

merged_cars = cars_df.merge(process_burglary_data("../data/crimes_metropolitan_2021"), on='LSOA', how='inner')

car_van_pers = ['pct_no_cars', 'pct_1_car', 'pct_2_cars', 'pct_3plus_cars']

# Create figure grid
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
axes = axes.flatten()

# Plot each percentage type
for idx, percentage_type in enumerate(car_van_pers):
    ax = axes[idx]

    # Calculate correlation
    corr, p_value = pearsonr(merged_cars[percentage_type], merged_cars["burglary_count"])

    # Create plot
    sns.regplot(
        data=merged_cars,
        x=percentage_type,
        y="burglary_count",
        ax=ax,
        scatter_kws={"alpha": 0.4, "color": "steelblue"},
        line_kws={"color": "darkred"},
    )

    # Formatting
    title = f"{percentage_type.replace('_', ' ').title()}\n(r={corr:.2f}, p={p_value:.4f})"
    ax.set_title(title)
    ax.set_xlabel("Number of Houses")
    ax.set_ylabel('Burglary Count')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Weighted car index: 0*(no cars) + 1*(1 car) + 2*(2 cars) + 3*(3+ cars)
merged_cars['car_index'] = (
    0 * merged_cars['pct_no_cars'] +
    1 * merged_cars['pct_1_car'] +
    2 * merged_cars['pct_2_cars'] +
    3 * merged_cars['pct_3plus_cars']
)

# Check correlation with burglary
corr, p_value = pearsonr(merged_cars['car_index'], merged_cars['burglary_count'])
print(f"Car Index Correlation: r={corr:.2f}, p={p_value:.4f}")


# Create car ownership tiers
merged_cars['car_tier'] = pd.qcut(
    merged_cars['car_index'],
    q=5,
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)

# Plot burglary distribution across tiers
plt.figure(figsize=(10,6))
sns.boxplot(data=merged_cars, x='car_tier', y='burglary_count',
            order=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            palette='Blues')
plt.title("Burglary Counts by Car Ownership Tier")
plt.xlabel("Car Ownership Index Tier")
plt.ylabel("Burglary Count")
plt.show()
