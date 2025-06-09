import pandas as pd
from clean_data import Dataset
from config import path, eda, finetune
from finetune_ward_models import FineTune


dataset = Dataset(path, eda)
df_all_burglary = dataset.clean_dataset()  # get cleaned dataset and perform EDA if eda=True
df = dataset.get_residential_burglaries(df_all_burglary)  # final dataset

# get fine-tuned orders for each ward model
df_orders = FineTune(df).fine_tune_all_metrics() if finetune else pd.read_csv(f"{path}/model_orders.csv")
