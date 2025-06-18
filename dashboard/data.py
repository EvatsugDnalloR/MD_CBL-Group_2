import os
import geopandas as gpd
import pandas as pd
from dashboard.allocation import allocate_police

current_directory = os.path.dirname(os.path.abspath(__file__))
static_directory = os.path.join(current_directory, "static")

# Load the necessary data files
london_boundaries = gpd.read_file(os.path.join(static_directory, "london_ward_boundaries.geojson")) # Load London ward boundaries for the map
predictions = allocate_police(pd.read_csv(os.path.join(static_directory, "prediction.csv")))
wards = pd.read_csv(os.path.join(static_directory, "list_london_wards.csv"))
burglary = pd.read_csv(os.path.join(static_directory, "structured_residential_burglary.csv"))

burglary["Date"] = pd.to_datetime({
    "year": burglary["Year"],
    "month": burglary["Month"],
    "day": 1
})  # Add date column to the data frames for easier processing in time series graph

predictions["Date"] = pd.to_datetime({
    "year": predictions["Year"],
    "month": predictions["Month"],
    "day": 1
})

# Load socioeconomic factors data
occupancy = pd.read_csv(os.path.join(static_directory, "occupancy_rating_bedrooms_wards.csv"))
cars_vans = pd.read_csv(os.path.join(static_directory, "cars_or_vans_wards.csv"))