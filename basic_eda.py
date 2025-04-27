import pandas as pd
import os
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Load all crime data
crime_files = glob.glob('data/crimes/**/*-street.csv', recursive=True)
df_list = []

for file in crime_files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

raw_df = pd.concat(df_list, ignore_index=True)

# Filter for Burglary crimes
burglary_df = raw_df[raw_df['Crime type'] == 'Burglary'].copy()

# Handle missing locations
print(f"Initial burglary records: {len(burglary_df)}")
print(f"Records without location: {burglary_df['Location'].isnull().sum()}")

# Remove entries with no location data
burglary_clean = burglary_df.dropna(subset=['Longitude', 'Latitude'])
burglary_clean = burglary_clean[burglary_clean['Location'] != 'No Location']

print(f"Cleaned burglary records: {len(burglary_clean)}")

# Convert Month to datetime
burglary_clean['Month'] = pd.to_datetime(burglary_clean['Month'], format='%Y-%m')

# Monthly trend analysis
monthly_trend = burglary_clean.resample('ME', on='Month').size()

plt.figure(figsize=(12, 6))
monthly_trend.plot(title='Monthly Burglary Trends')
plt.ylabel('Number of Burglaries')
plt.xlabel('Date')
plt.show()

# Outcome analysis
outcome_dist = burglary_clean['Last outcome category'].value_counts(normalize=True)

plt.figure(figsize=(15, 6))
sns.barplot(x=outcome_dist.values, y=outcome_dist.index)
plt.title('Distribution of Investigation Outcomes')
plt.xlabel('Percentage of Cases')
plt.show()

# Generate basic statistics
report = {
    "time_period": f"{burglary_clean['Month'].min().date()} to {burglary_clean['Month'].max().date()}",
    "total_burglaries": len(burglary_clean),
    "monthly_average": round(len(burglary_clean)/burglary_clean['Month'].nunique(), 1),
    "most_common_lsoa": burglary_clean['LSOA name'].mode()[0],
    "missing_data_percentage": f"{round((len(raw_df)-len(burglary_clean))/len(raw_df)*100, 1)}%",
    "clearance_rate": f"{round(outcome_dist.get('Investigation complete; no suspect identified', 0)*100, 1)}%"
}

print("\nBasic Statistics Report:")
for k, v in report.items():
    print(f"{k.replace('_', ' ').title()}: {v}")


# Set GDAL path (modify to your Anaconda path)
os.environ['GDAL_DATA'] = r'D:\anaconda3\Library\share\gdal'

# Load all LSOA shapefiles
lsoa_path = "./data/lsoa/"
lsoa_files = [f for f in os.listdir(lsoa_path) if f.endswith('.shp')]

lsoa_gdf = gpd.GeoDataFrame()
for file in lsoa_files:
    temp_gdf = gpd.read_file(os.path.join(lsoa_path, file))
    lsoa_gdf = pd.concat([lsoa_gdf, temp_gdf], ignore_index=True)

# Check coordinate reference system (CRS)
print(f"\n{lsoa_gdf.crs}")  # Should be EPSG:27700 (UK Ordnance Survey) or similar

print("Columns in LSOA shapefiles:", lsoa_gdf.columns.tolist())

# Aggregate burglaries per LSOA
lsoa_counts = burglary_clean['LSOA code'].value_counts().reset_index()
lsoa_counts.columns = ['lsoa21cd', 'Burglary Count']

# Merge with LSOA geometries
lsoa_merged = lsoa_gdf.merge(lsoa_counts, on="lsoa21cd", how="left")
lsoa_merged["Burglary Count"] = lsoa_merged["Burglary Count"].fillna(0)

fig, ax = plt.subplots(figsize=(12, 7))

# Plot LSOA areas colored by burglary count
lsoa_merged.plot(
    column="Burglary Count",
    cmap="Reds",
    linewidth=0.2,
    edgecolor="black",
    legend=True,
    scheme="natural_breaks",
    ax=ax,
)

# Add actual burglary locations WITH CRS DEFINITION
gdf = gpd.GeoDataFrame(
    burglary_clean,
    geometry=gpd.points_from_xy(
        burglary_clean.Longitude,
        burglary_clean.Latitude
    ),
    crs="EPSG:4326"  # Explicitly set WGS84 CRS
)

# Reproject to match LSOA CRS (EPSG:27700)
gdf = gdf.to_crs(lsoa_merged.crs)

# Now plot
gdf.plot(ax=ax, markersize=2, color="blue", alpha=0.3)
plt.title('Burglary Distribution Across London LSOAs')
plt.axis('off')
plt.show()
