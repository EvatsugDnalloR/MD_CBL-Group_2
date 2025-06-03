import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima.arima import auto_arima, ADFTest
import joblib  # to save models
import statistics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from clean_data import Dataset, Population
from config import path, eda

dataset = Dataset(path, eda)
df_all_burglary = dataset.clean_dataset()  # get cleaned dataset and perform EDA if eda=True
df = dataset.get_residential_burglaries(df_all_burglary)  # final dataset
df_pop = Population('lsoa_ward_data/land-area-population-density-london.xlsx').clean_population_df()

df_ward = pd.read_csv('lsoa_ward_data/LSOA_2021_to_Electoral_Ward_2024.csv')
lad_codes = dict(zip(df_ward['WD24CD'], df_ward['LAD24CD']))
df['LAD code'] = df['Ward code'].map(lad_codes)
df['key'] = df['Ward code'].astype(str) + '-' + df['Year'].astype(str)
df_pop['key'] = df_pop['New ward code'].astype(str) + '-' + df_pop['Year'].astype(str)
population_values = df_pop.set_index('key')['Population'].to_dict()
df['Population'] = df['key'].map(population_values)
df = df.drop(columns='key')


class FineTune:
    """
    Fine-tune SARIMAX model for each ward using population as an exogenous variable.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FineTune object.
        :param df: burglary dataframe with population data
        """
        self.df = df
        self.ward_codes = list(self.df['Ward code'].unique())

    def fine_tune_ward(self, w: str) -> tuple[float, float]:
        """
        Fine-tune, fit and save SARIMAX model for a given ward.
        :param w: ward code
        :return: RMSE and MAE of model's performance.
        """
        df_w = self.df[self.df['Ward code'] == w].copy()
        df_w['Month'] = pd.to_datetime(df_w['Year'].astype(str) + '-' + df_w['Month'].astype(str), format='%Y-%m')
        df_monthly = df_w.resample('M', on='Month').agg({'Month': 'count', 'Population': 'mean'})
        df_monthly.rename(columns={'Month': 'burglary_count', 'Population': 'avg_population'}, inplace=True)
        df_monthly = df_monthly.reindex(pd.date_range(start=df_monthly.index.min(), end='2025-12-31', freq='ME')).sort_index()
        df_monthly['avg_population'] = df_monthly['avg_population'].ffill().bfill()  # fill in missing values
        df_monthly['burglary_count'] = df_monthly['burglary_count'].fillna(0)  # missing burglaries amounts to 0 burglaries

        train = df_monthly[:'2022-12-31'].fillna(0)
        test = df_monthly['2023-01-31':'2025-02-28'].fillna(0)

        arima_model = auto_arima(train['burglary_count'], exog=train[['avg_population']], start_p=0, d=0, start_q=0,
                                 max_p=3, max_d=3, max_q=3, start_P=0, D=1, start_Q=0, max_P=3, max_Q=3, m=12,
                                 seasonal=True, error_action='ignore', suppress_warnings=True, stepwise=True,
                                 random_state=2025, n_fits=10,
                                 stationary=ADFTest(alpha=0.05).should_diff(train['burglary_count'])[1])

        model = SARIMAX(train['burglary_count'], exog=train[['avg_population']], order=arima_model.order,
                        seasonal_order=arima_model.seasonal_order).fit()
        forecast_test = model.get_forecast(steps=len(test), exog=test[['avg_population']]).predicted_mean

        # joblib.dump(model, f"models/{w}_model.pkl")

        rmse = np.sqrt(mean_squared_error(test['burglary_count'], forecast_test))
        mae = mean_absolute_error(test['burglary_count'], forecast_test)

        return rmse, mae

    def fine_tune_all_metrics(self) -> tuple[list, list]:
        """
        Fine-tune SARIMAX for all wards and compute metrics.
        :return: metrics from all wards.
        """
        all_rmse, all_mae = [], []

        for c in self.ward_codes:
            rmse, mae = self.fine_tune_ward(c)
            all_rmse.append(rmse)
            all_mae.append(mae)

        print(f"Average RMSE: {statistics.mean(all_rmse)}\nAverage MAE: {statistics.mean(all_mae)}")
        print(f"Median RMSE: {statistics.median(all_rmse)}\nMedian MAE: {statistics.median(all_mae)}")
        print(f"Min. RMSE: {min(all_rmse)}\nMin. MAE: {min(all_mae)}")
        print(f"Max. RMSE: {max(all_rmse)}\nMax. MAE: {max(all_mae)}")

        return all_rmse, all_mae


rmse, mae = FineTune(df).fine_tune_ward('E05014068')
print(rmse, mae)
