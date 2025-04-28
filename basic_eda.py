import glob
import os

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Load all crime data
crime_files = glob.glob("data/crimes/**/*-street.csv", recursive=True)
df_list = []

for file in crime_files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

raw_df = pd.concat(df_list, ignore_index=True)

# Filter for Burglary crimes
burglary_df = raw_df[raw_df["Crime type"] == "Burglary"].copy()

# Handle missing locations
print(f"Initial burglary records: {len(burglary_df)}")
print(f"Records without location: {burglary_df['Location'].isnull().sum()}")

# Remove entries with no location data
burglary_clean = burglary_df.dropna(subset=["Longitude", "Latitude"])
burglary_clean = burglary_clean[burglary_clean["Location"] != "No Location"]

print(f"Cleaned burglary records: {len(burglary_clean)}")

# Convert Month to datetime
burglary_clean["Month"] = pd.to_datetime(burglary_clean["Month"], format="%Y-%m")

# Monthly trend analysis
monthly_trend = burglary_clean.resample("ME", on="Month").size()

plt.figure(figsize=(12, 6))
monthly_trend.plot(title="Monthly Burglary Trends")
plt.ylabel("Number of Burglaries")
plt.xlabel("Date")
plt.show()

# Outcome analysis
outcome_dist = burglary_clean["Last outcome category"].value_counts(normalize=True)

plt.figure(figsize=(15, 6))
sns.barplot(x=outcome_dist.values, y=outcome_dist.index)
plt.title("Distribution of Investigation Outcomes")
plt.xlabel("Percentage of Cases")
plt.show()

# Generate basic statistics
report = {
    "time_period": f"{burglary_clean['Month'].min().date()} to {burglary_clean['Month'].max().date()}",
    "total_burglaries": len(burglary_clean),
    "monthly_average": round(len(burglary_clean)/burglary_clean["Month"].nunique(), 1),
    "most_common_lsoa": burglary_clean["LSOA name"].mode()[0],
    "missing_data_percentage": f"{round((len(raw_df)-len(burglary_clean))/len(raw_df)*100, 1)}%",
    "clearance_rate": f"{round(outcome_dist.get('Investigation complete; no suspect identified', 0)*100, 1)}%"
}

print("\nBasic Statistics Report:")
for k, v in report.items():
    print(f"{k.replace('_', ' ').title()}: {v}")


# <--------------------------------------------------------------------------------------------------->


# Set GDAL path (modify to your Anaconda path)
os.environ["GDAL_DATA"] = r"D:\anaconda3\Library\share\gdal"

# Load all LSOA shapefiles
lsoa_path = "./data/lsoa/"
lsoa_files = [f for f in os.listdir(lsoa_path) if f.endswith(".shp")]

lsoa_gdf = gpd.GeoDataFrame()
for file in lsoa_files:
    temp_gdf = gpd.read_file(os.path.join(lsoa_path, file))
    lsoa_gdf = pd.concat([lsoa_gdf, temp_gdf], ignore_index=True)

# Check coordinate reference system (CRS)
print(f"\n{lsoa_gdf.crs}")  # Should be EPSG:27700 (UK Ordnance Survey) or similar

print("Columns in LSOA shapefiles:", lsoa_gdf.columns.tolist())

# Aggregate burglaries per LSOA
lsoa_counts = burglary_clean["LSOA code"].value_counts().reset_index()
lsoa_counts.columns = ["lsoa21cd", "Burglary Count"]

# Merge with LSOA geometries
lsoa_merged = lsoa_gdf.merge(lsoa_counts, on="lsoa21cd", how="left")
lsoa_merged["Burglary Count"] = lsoa_merged["Burglary Count"].fillna(0)
print(lsoa_merged["Burglary Count"].describe())  # Check value distribution

# Add actual burglary locations WITH CRS DEFINITION
gdf = gpd.GeoDataFrame(
    burglary_clean,
    geometry=gpd.points_from_xy(
        burglary_clean.Longitude,
        burglary_clean.Latitude
    ),
    crs="EPSG:4326"
)

# Convert to target CRS FIRST
gdf = gdf.to_crs(lsoa_merged.crs)  # EPSG:27700

# Filter crime points using London bounds
london_bbox = lsoa_merged.total_bounds
gdf = gdf.cx[london_bbox[0]:london_bbox[2], london_bbox[1]:london_bbox[3]]

# Verify conversion results
print("Crime points CRS after conversion:", gdf.crs)
print("Sample converted coordinates:\n", gdf.geometry.head(3))
print("Converted bounds:", gdf.total_bounds)

fig, ax = plt.subplots(figsize=(20, 15))  # Larger figure size

# Plot LSOA choropleth first
lsoa_merged.plot(
    column="Burglary Count",
    cmap="OrRd",  # More distinct color scheme
    linewidth=0.3,
    edgecolor="white",
    legend=True,
    scheme="quantiles",  # Better for skewed distributions
    legend_kwds={'loc': 'lower right'},
    alpha=0.8,  # Add transparency
    ax=ax
)

# # Then plot crime locations
# gdf.plot(
#     ax=ax,
#     markersize=3,
#     color="darkblue",
#     alpha=0.4,  # Reduced alpha for better choropleth visibility
#     marker="o"  # Smaller marker
# )

# Add context
ctx.add_basemap(ax, crs=lsoa_merged.crs, source=ctx.providers.CartoDB.Positron)

# Final touches
plt.title("Burglary Hotspots in Greater London (2022-2025)", fontsize=16)
plt.axis("off")

# Add scale bar
ax.add_artist(AnchoredSizeBar(ax.transData,
                              5000,  # 5km in EPSG:27700 units
                              "5 km",
                              loc="lower left",
                              pad=0.5,
                              color="black",
                              frameon=False,
                              size_vertical=100))

plt.show()
