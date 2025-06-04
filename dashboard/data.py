import pandas as pd
import geopandas as gpd
from allocation import allocate_police

london_boundaries = gpd.read_file("static/london_ward_boundaries.geojson")
predictions = allocate_police(pd.read_csv("static/prediction.csv"))
wards= pd.read_csv("static/list_london_wards.csv")
burglary= pd.read_csv("static/structured_residential_burglary.csv")

