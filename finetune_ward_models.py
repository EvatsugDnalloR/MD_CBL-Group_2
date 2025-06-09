import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima.arima import auto_arima, ADFTest
import statistics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from clean_data import Population


class FineTune:
    """
    Fine-tune SARIMAX model for each ward using population as an exogenous variable.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FineTune object.
        :param df: burglary dataframe with population data
        """
        df_pop = Population('lsoa_ward_data/land-area-population-density-london.xlsx').clean_population_df()
        df_ward = pd.read_csv('lsoa_ward_data/LSOA_2021_to_Electoral_Ward_2024.csv')
        lad_codes = dict(zip(df_ward['WD24CD'], df_ward['LAD24CD']))
        df['LAD code'] = df['Ward code'].map(lad_codes)
        df['key'] = df['Ward code'].astype(str) + '-' + df['Year'].astype(str)
        df_pop['key'] = df_pop['New ward code'].astype(str) + '-' + df_pop['Year'].astype(str)
        population_values = df_pop.set_index('key')['Population'].to_dict()
        population_sqk = df_pop.set_index('key')['Population per square kilometre'].to_dict()
        df['Population'] = df['key'].map(population_values)
        df['Density'] = df['key'].map(population_sqk)
        self.df = df.drop(columns='key')
        self.ward_codes = list(self.df['Ward code'].unique())
        self.exog_list = ['Population', 'Density']

    def fine_tune_ward(self, w: str) -> tuple[float, float, tuple, tuple]:
        """
        Fine-tune, fit and save SARIMAX model for a given ward with no exogenous variable.
        :param w: ward code
        :return: metrics and orders of a model.
        """
        df_w = self.df[self.df['Ward code'] == w].copy()
        df_w['Month'] = pd.to_datetime(df_w['Year'].astype(str) + '-' + df_w['Month'].astype(str), format='%Y-%m')
        df_monthly = df_w.resample('M', on='Month').agg({'Month': 'count'})
        df_monthly.rename(columns={'Month': 'burglary_count'}, inplace=True)
        df_monthly = df_monthly.reindex(pd.date_range(start=df_monthly.index.min(), end='2025-12-31', freq='ME')).sort_index()
        df_monthly['burglary_count'] = df_monthly['burglary_count'].fillna(0)  # missing burglaries amounts to 0 burglaries

        train = df_monthly[:'2022-12-31'].fillna(0)
        test = df_monthly['2023-01-31':'2025-02-28'].fillna(0)

        fine_tuned_model = auto_arima(train['burglary_count'], start_p=0, d=0, start_q=0,
                                      max_p=3, max_d=3, max_q=3, start_P=0, D=1, start_Q=0, max_P=3, max_Q=3, m=12,
                                      seasonal=True, error_action='ignore', suppress_warnings=True, stepwise=True,
                                      random_state=2025, n_fits=10, trace=False,
                                      stationary=ADFTest(alpha=0.05).should_diff(train['burglary_count'])[1])

        model = SARIMAX(train['burglary_count'], order=fine_tuned_model.order,
                        seasonal_order=fine_tuned_model.seasonal_order).fit()
        forecast_test = model.get_forecast(steps=len(test)).predicted_mean

        rmse = np.sqrt(mean_squared_error(test['burglary_count'], forecast_test))
        mae = mean_absolute_error(test['burglary_count'], forecast_test)

        return rmse, mae, fine_tuned_model.order, fine_tuned_model.seasonal_order

    def fine_tune_ward_exog(self, w: str, exog_col: str) -> tuple[float, float, any, any]:
        """
        Fine-tune, fit and save SARIMAX model for a given ward with a given exogenous variable.
        :param w: ward code
        :param exog_col: exogenous variable col name
        :return: metrics and orders of a model.
        """
        df_w = self.df[self.df['Ward code'] == w].copy()
        df_w['Month'] = pd.to_datetime(df_w['Year'].astype(str) + '-' + df_w['Month'].astype(str), format='%Y-%m')
        df_monthly = df_w.resample('M', on='Month').agg({'Month': 'count', exog_col: 'mean'})
        df_monthly.rename(columns={'Month': 'burglary_count'}, inplace=True)
        df_monthly = df_monthly.reindex(pd.date_range(start=df_monthly.index.min(), end='2025-12-31', freq='ME')).sort_index()
        df_monthly[exog_col] = df_monthly[exog_col].ffill().bfill()  # fill in missing values
        df_monthly['burglary_count'] = df_monthly['burglary_count'].fillna(0)  # missing burglaries amounts to 0 burglaries

        train = df_monthly[:'2022-12-31'].fillna(0)
        test = df_monthly['2023-01-31':'2025-02-28'].fillna(0)

        fine_tuned_model = auto_arima(train['burglary_count'], exog=train[[exog_col]], start_p=0, d=0, start_q=0,
                                      max_p=3, max_d=3, max_q=3, start_P=0, D=1, start_Q=0, max_P=3, max_Q=3, m=12,
                                      seasonal=True, error_action='ignore', suppress_warnings=True, stepwise=True,
                                      random_state=2025, n_fits=10, trace=False,
                                      stationary=ADFTest(alpha=0.05).should_diff(train['burglary_count'])[1])

        model = SARIMAX(train['burglary_count'], exog=train[[exog_col]], order=fine_tuned_model.order,
                        seasonal_order=fine_tuned_model.seasonal_order).fit()
        forecast_test = model.get_forecast(steps=len(test), exog=test[[exog_col]]).predicted_mean

        rmse = np.sqrt(mean_squared_error(test['burglary_count'], forecast_test))
        mae = mean_absolute_error(test['burglary_count'], forecast_test)

        return rmse, mae, fine_tuned_model.order, fine_tuned_model.seasonal_order

    @staticmethod
    def print_metrics(all_rmse: list[float], all_mae: list[float], e: str):
        """
        Print metrics of the distribution of RMSE and MAE values.
        :param all_rmse: list of RMSE values.
        :param all_mae: list of MAE values.
        :param e: exogenous variable used.
        """
        print(f"Metrics with {e}:\n"
              f"Average RMSE: {np.mean(all_rmse):.3f}\nAverage MAE: {np.mean(all_mae):.3f}\n"
              f"Median RMSE: {statistics.median(all_rmse):.3f}\nMedian MAE: {statistics.median(all_mae):.3f}\n"
              f"Min. RMSE: {min(all_rmse):.3f}\nMin. MAE: {min(all_mae):.3f}\n"
              f"Max. RMSE: {max(all_rmse):.3f}\nMax. MAE: {max(all_mae):.3f}\n")

    def fine_tune_all_metrics(self) -> pd.DataFrame:
        """
        Fine-tune SARIMAX for all wards with all exogenous variables and compute metrics.
        :return: RMSE and MAE.
        """
        all_rmse, all_mae = [], []

        for c in self.ward_codes[:-1]:
            rmse, mae, order, seasonal_order = self.fine_tune_ward(c)  # get metrics without exog. var
            all_rmse.append(rmse)
            all_mae.append(mae)

        self.print_metrics(all_rmse, all_mae, 'no exogenous variables')

        for e in self.exog_list:
            all_rmse, all_mae, orders = [], [], []

            for c in self.ward_codes[:-1]:
                rmse, mae, order, seasonal_order = self.fine_tune_ward_exog(c, e)  # get metrics using each exog. var
                all_rmse.append(rmse)
                all_mae.append(mae)

                if e =='Density':
                    orders.append({'Ward code': c, 'Order': order, 'Seasonal order': seasonal_order})

            if e =='Density':
                df_orders = pd.DataFrame(orders)
                df_orders.to_csv('datasets/model_orders.csv', index=False)

            self.print_metrics(all_rmse, all_mae, e)

        return df_orders
