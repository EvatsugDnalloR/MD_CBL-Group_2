import pandas as pd
from concat_eda_all_data import Dataset
from config import path

df = Dataset(path).eda()  # get cleaned dataset and perform EDA
