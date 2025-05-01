import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Load and clean house price data
hp_path = "../data/housing/land-registry-house-prices-LSOA.xls"
hp_df = pd.read_excel(hp_path, sheet_name="Median")

# Filter out City of London and handle missing values
hp_df = hp_df[~hp_df["Area"].str.contains("City of London", na=False)]
hp_df = hp_df.replace(":", pd.NA).dropna(subset=["Code"])

# Select December columns for 2011-2017
dec_columns = [f"Year ending Dec {year}" for year in range(2011, 2018)]
hp_df = hp_df[["Code", "Area"] + dec_columns]

# Melt to long format and clean
hp_df = hp_df.melt(id_vars=["Code", "Area"], var_name="Year", value_name="Median_Price")
hp_df["Year"] = hp_df["Year"].str.extract("(\d{4})").astype(int)
hp_df["Median_Price"] = pd.to_numeric(hp_df["Median_Price"], errors="coerce")
hp_df = hp_df.dropna()

# Load burglary data for 2011-2017
burglary_counts = []
base_path = "../data/crimes_metropolitan_2011-2017/"

for year in range(2011, 2018):
    year_path = os.path.join(base_path, str(year), "*.csv")
    files = glob.glob(year_path)

    yearly_burglaries = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Filter and count burglaries
    yearly_counts = (
        yearly_burglaries[yearly_burglaries["Crime type"] == "Burglary"]["LSOA code"]
        .value_counts()
        .reset_index()
    )
    yearly_counts.columns = ["LSOA", f"Burglary_{year}"]
    yearly_counts["Year"] = year

    burglary_counts.append(yearly_counts)

burglary_df = pd.concat(burglary_counts)

# Merge datasets
merged = pd.merge(hp_df, burglary_df, left_on=["Code", "Year"], right_on=["LSOA", "Year"], how="inner")

# Analysis and visualisation
plt.figure(figsize=(14, 8))

# Plot individual year relationships
for idx, year in enumerate(range(2011, 2018)):
    ax = plt.subplot(2, 4, idx + 1)
    year_data = merged[merged["Year"] == year]

    sns.regplot(
        data=year_data,
        x="Median_Price",
        y=f"Burglary_{year}",
        scatter_kws={"alpha": 0.4, "color": "steelblue"},
        line_kws={"color": "darkred"},
    )

    corr, p_val = pearsonr(year_data["Median_Price"], year_data[f"Burglary_{year}"])
    ax.set_title(f"{year}\n(r={corr:.2f}, p={p_val:.4f})")
    ax.set_xlabel("Median Price (£)")
    ax.set_ylabel("Burglary Count")

plt.tight_layout()

# Plot temporal trend of correlation
correlations = []
for year in range(2011, 2018):
    year_data = merged[merged["Year"] == year]
    corr, _ = pearsonr(year_data["Median_Price"], year_data[f"Burglary_{year}"])
    correlations.append(corr)

plt.figure(figsize=(10, 6))
sns.lineplot(x=range(2011, 2018), y=correlations, marker="o")
plt.title("Correlation Trend: House Prices vs Burglaries (2011-2017)")
plt.xlabel("Year")
plt.ylabel("Pearson Correlation Coefficient")
plt.ylim(-1, 1)
plt.grid(True, alpha=0.3)
plt.show()
