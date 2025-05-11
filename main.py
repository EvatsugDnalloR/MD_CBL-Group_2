from clean_data import Dataset
from config import path, eda

dataset = Dataset(path, eda)
df = dataset.perform_eda()  # get cleaned dataset and perform EDA if eda=True
df_residential = dataset.get_residential_burglaries(df)
