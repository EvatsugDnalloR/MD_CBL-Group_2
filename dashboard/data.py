import geopandas as gpd
import pandas as pd
from allocation import allocate_police

# Load the necessary data files
london_boundaries = gpd.read_file("static/london_ward_boundaries.geojson") # Load London ward boundaries for the map
predictions = allocate_police(pd.read_csv("static/prediction.csv"))
wards= pd.read_csv("static/list_london_wards.csv")
burglary= pd.read_csv("static/structured_residential_burglary.csv")

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
occupancy = pd.read_csv("static/occupancy_rating_bedrooms_wards.csv")
cars_vans = pd.read_csv("static/cars_or_vans_wards.csv")
