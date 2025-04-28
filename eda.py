"""
EDA on Feb. 2025 data.
"""
import pandas as pd

df_street = pd.read_csv("data/2025-02/2025-02-metropolitan-street.csv")
df_burglary = df_street[df_street['Crime type'] == 'Burglary']
print(df_burglary)
