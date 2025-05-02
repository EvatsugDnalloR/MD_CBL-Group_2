import pandas as pd
from concat_eda_all_data import Dataset
from config import path, eda

df = Dataset(path, eda).perform_eda()  # get cleaned dataset and perform EDA
print(df)
