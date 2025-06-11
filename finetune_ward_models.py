import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima.arima import auto_arima, ADFTest
import statistics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class FineTune:
    """
    Fine-tune the SARIMA model for each ward.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FineTune object.
        :param df: burglary dataframe with population data
        """
        self.df = df
        self.ward_codes = list(self.df['Ward code'].unique())

    def fine_tune_ward(self, w: str) -> tuple[float, float, tuple, tuple]:
        """
        Fine-tune, fit and save the SARIMA model for a given ward.
        :param w: ward code
        :return: metrics and orders of a model.
        """
        df_w = self.df[self.df['Ward code'] == w].copy()
        df_w['Month'] = pd.to_datetime(df_w['Year'].astype(str) + '-' + df_w['Month'].astype(str), format='%Y-%m')
        df_monthly = df_w.resample('ME', on='Month').agg({'Month': 'count'})
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
        y_pred = model.get_forecast(steps=len(test)).predicted_mean

        rmse = np.sqrt(mean_squared_error(test['burglary_count'], y_pred))
        mae = mean_absolute_error(test['burglary_count'], y_pred)

        return rmse, mae, fine_tuned_model.order, fine_tuned_model.seasonal_order

    def fine_tune_all_metrics(self) -> pd.DataFrame:
        """
        Fine-tune SARIMA for all wards and compute metrics.
        :return: dataframe with fine-tuned SARIMA parameters for all wards.
        """
        all_rmse, all_mae, orders = [], [], []

        for c in self.ward_codes[:-1]:
            rmse, mae, order, seasonal_order = self.fine_tune_ward(c)
            all_rmse.append(rmse)
            all_mae.append(mae)
            orders.append({'Ward code': c, 'Order': order, 'Seasonal order': seasonal_order})

        df_orders = pd.DataFrame(orders)
        df_orders.to_csv('datasets/model_orders.csv', index=False)

        print(f"Average RMSE: {np.mean(all_rmse):.3f}\nAverage MAE: {np.mean(all_mae):.3f}\n"
              f"Median RMSE: {statistics.median(all_rmse):.3f}\nMedian MAE: {statistics.median(all_mae):.3f}\n"
              f"Min. RMSE: {min(all_rmse):.3f}\nMin. MAE: {min(all_mae):.3f}\n"
              f"Max. RMSE: {max(all_rmse):.3f}\nMax. MAE: {max(all_mae):.3f}\n")

        return df_orders
