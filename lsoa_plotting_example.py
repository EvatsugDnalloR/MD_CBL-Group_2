import os

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Load all burglary crime data
burglary_df = pd.read_csv("data/burglary.csv")

# Handle missing locations
print(f"Initial burglary records: {len(burglary_df)}")
print(f"Records without location: {burglary_df['Location'].isnull().sum()}")

# Remove entries with no location data
burglary_clean = burglary_df.dropna(subset=["Longitude", "Latitude"])
burglary_clean = burglary_clean[burglary_clean["Location"] != "No Location"]

print(f"Cleaned burglary records: {len(burglary_clean)}")


# Set GDAL path (modify to your Anaconda path)
os.environ["GDAL_DATA"] = r"D:\anaconda3\Library\share\gdal"

# Load all LSOA shapefiles
lsoa_path = "./data/lsoa_2021/"
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
