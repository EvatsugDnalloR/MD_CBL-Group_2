import pandas as pd
import random
from clean_data import Dataset
from config import path, eda

dataset = Dataset(path, eda)
df = dataset.clean_dataset()  # get cleaned dataset and perform EDA if eda=True
df_residential = dataset.get_residential_burglaries(df)

wards = list(df_residential['Ward name'].unique())
burg_pred = dict()

for w in wards:
    r = random.randint(3207, 7233)
    burg_pred[w] = r

df_allocation = pd.DataFrame(burg_pred.items(), columns=['Ward name', 'Prediction'])
max_pred = max(df_allocation['Prediction'])
df_allocation['Risk factor'] = round((df_allocation['Prediction'] / max_pred), 3)
df_allocation['Predicted officers'] = df_allocation['Risk factor'] * 100
print(df_allocation['Predicted officers'].describe())
