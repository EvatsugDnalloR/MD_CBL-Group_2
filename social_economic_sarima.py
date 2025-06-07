import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
import os


class SocioEconomicSARIMA:
    def __init__(self, burglary_path: str, cars_path: str, occupancy_path: str):
        """
        Hybrid SARIMA forecasting model with socio-economic optimization

        Parameters:
        burglary_path (str): Path to residential burglary CSV
        cars_path (str): Path to car ownership Excel file
        occupancy_path (str): Path to bedroom occupancy Excel file
        """
        self.burglary_path = burglary_path
        self.cars_path = cars_path
        self.occupancy_path = occupancy_path
        self.cluster_orders = {
            0: {'normal': (1, 1, 3), 'seasonal': (0, 1, 3, 12)},
            1: {'normal': (2, 1, 2), 'seasonal': (0, 1, 2, 12)},
            2: {'normal': (3, 1, 1), 'seasonal': (0, 1, 1, 12)}
        }
        self.models = {}
        self.burglary_monthly = None
        self.optimization_params = None
        self.mae_model = None
        self.cluster_df = None
        self.forecast_results = None

    def load_and_preprocess(self):
        """Load and preprocess all required datasets"""
        # Load and process burglary data
        burglary = pd.read_csv(self.burglary_path)
        burglary["Month"] = pd.to_datetime(
            burglary["Year"].astype(str) + "-" + burglary["Month"].astype(str),
            format="%Y-%m"
        )
        self.burglary_monthly = burglary.groupby(
            ["Ward Code", "Month"]
        ).size().reset_index(name="Count")

        # Load and process socio-economic data
        cars_df = pd.read_excel(self.cars_path, sheet_name='2021')
        cars_df['NoCarPct'] = cars_df['none'] / cars_df['All households']

        occupancy_df = pd.read_excel(self.occupancy_path, sheet_name='2021')
        occupancy_df["ZeroRoomPct"] = occupancy_df["0"] / occupancy_df["All Households"]

        # Merge socio-economic data
        self.cluster_df = pd.merge(
            cars_df[['ward code', 'NoCarPct']],
            occupancy_df[['ward code', 'ZeroRoomPct']],
            on='ward code',
            how='inner'
        ).rename(columns={'ward code': 'ward_code'})

        print(f"Loaded {len(self.burglary_monthly)} burglary records")
        print(f"Loaded socio-economic data for {len(self.cluster_df)} wards")

    def perform_clustering(self, n_clusters=3):
        """Cluster wards based on socio-economic factors"""
        X = self.cluster_df[['NoCarPct', 'ZeroRoomPct']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_df['cluster'] = kmeans.fit_predict(X_scaled)

        # Print cluster distribution
        cluster_counts = self.cluster_df['cluster'].value_counts().sort_index()
        print("\nCluster Distribution:")
        for cluster_id, count in cluster_counts.items():
            print(f"Cluster {cluster_id}: {count} wards")

    def fit_models(self):
        """Train SARIMA models for all wards with cluster-specific parameters"""
        self.models = {}
        self.training_results = []

        for ward_code, group in self.burglary_monthly.groupby('Ward Code'):
            # Skip wards without socio-economic data
            if ward_code not in self.cluster_df['ward_code'].values:
                continue

            # Get cluster-specific orders
            cluster_id = self.cluster_df.loc[
                self.cluster_df['ward_code'] == ward_code, 'cluster'
            ].values[0]
            orders = self.cluster_orders[cluster_id]

            # Prepare time series
            series = group[['Month', 'Count']].set_index('Month').asfreq('MS')['Count']

            try:
                # Fit model
                model = SARIMAX(
                    series,
                    order=orders['normal'],
                    seasonal_order=orders['seasonal']
                )
                results = model.fit(disp=False)

                # Store model
                self.models[ward_code] = {
                    'model': results,
                    'orders': orders,
                    'cluster': cluster_id
                }

                # Generate in-sample predictions for MAE calculation
                preds = results.get_prediction().predicted_mean
                residuals = series - preds

                self.training_results.append({
                    'ward_code': ward_code,
                    'cluster': cluster_id,
                    'residuals': residuals.values,
                    'predicted': preds.values,
                    'actual': series.values
                })

            except Exception as e:
                print(f"Error fitting {ward_code}: {str(e)}")

        print(f"\nSuccessfully trained models for {len(self.models)} wards")

    def optimize_with_socio_economic(self):
        """Train MAE adjustment model using socio-economic factors"""
        # Prepare training data
        results_df = pd.DataFrame(self.training_results)
        results_df = pd.merge(
            results_df,
            self.cluster_df,
            on='ward_code',
            how='left'
        )

        # Compute MAE for each ward
        results_df['mae'] = results_df.apply(
            lambda x: np.mean(np.abs(x['residuals'])), axis=1
        )

        # Train linear model
        X = results_df[['NoCarPct', 'ZeroRoomPct']]
        y = results_df['mae']
        self.mae_model = LinearRegression()
        self.mae_model.fit(X, y)

        # Store optimization parameters
        self.optimization_params = {}
        for _, row in results_df.iterrows():
            # Predict MAE adjustment
            X_ward = [[row['NoCarPct'], row['ZeroRoomPct']]]
            mae_adj = self.mae_model.predict(X_ward)[0]

            # Determine direction
            mean_residual = np.mean(row['residuals'])
            sign = 1 if mean_residual >= 0 else -1

            self.optimization_params[row['ward_code']] = {
                'mae_adj': mae_adj,
                'sign': sign
            }

    def forecast_ward(self, ward_code, forecast_start, forecast_end):
        """
        Forecast burglary counts for a ward

        Parameters:
        ward_code (str): Ward identifier
        forecast_start (str): Start date in 'YYYY-MM' format
        forecast_end (str): End date in 'YYYY-MM' format

        Returns:
        pd.Series: Forecasted values
        """
        if ward_code not in self.models:
            raise ValueError(f"No model found for ward {ward_code}")

        # Generate base forecast
        model_info = self.models[ward_code]
        forecast_dates = pd.date_range(
            start=forecast_start,
            end=forecast_end,
            freq='MS'
        )
        steps = len(forecast_dates)
        forecast = model_info['model'].get_forecast(steps=steps)
        base_pred = forecast.predicted_mean

        # Apply socio-economic adjustment
        if ward_code in self.optimization_params:
            params = self.optimization_params[ward_code]
            adjusted_pred = base_pred + params['sign'] * params['mae_adj']
            return adjusted_pred
        return base_pred

    def forecast_all_wards(self, forecast_start, forecast_end):
        """Forecast burglary counts for all wards"""
        results = []
        forecast_dates = pd.date_range(
            start=forecast_start,
            end=forecast_end,
            freq='MS'
        )

        for ward_code in self.models.keys():
            try:
                preds = self.forecast_ward(ward_code, forecast_start, forecast_end)
                for date, pred in zip(forecast_dates, preds):
                    results.append({
                        'Ward Code': ward_code,
                        'Month': date.strftime('%Y-%m'),
                        'Forecast': max(0, pred)  # Ensure non-negative
                    })
            except Exception as e:
                print(f"Error forecasting {ward_code}: {str(e)}")

        self.forecast_results = pd.DataFrame(results)
        return self.forecast_results

    def save_forecasts(self, output_path):
        """Save forecast results to CSV"""
        if self.forecast_results is None:
            raise ValueError("No forecasts available. Run forecast_all_wards first.")

        self.forecast_results.to_csv(output_path, index=False)
        print(f"Saved forecasts to {output_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Configuration
    BURGLARY_PATH = 'data/residential_burglary.csv'
    CARS_PATH = 'data/housing/cars_or_vans_wards.xlsx'
    OCCUPANCY_PATH = 'data/housing/occupancy_rating_bedrooms_wards.xlsx'
    OUTPUT_PATH = 'forecasts_output/2025_burglary_forecasts.csv'

    # Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Initialize and run model
    print("=== Initializing SocioEconomicSARIMA Model ===")
    model = SocioEconomicSARIMA(
        burglary_path=BURGLARY_PATH,
        cars_path=CARS_PATH,
        occupancy_path=OCCUPANCY_PATH
    )

    print("\n=== Loading and Preprocessing Data ===")
    model.load_and_preprocess()

    print("\n=== Performing Ward Clustering ===")
    model.perform_clustering(n_clusters=3)

    print("\n=== Training SARIMA Models ===")
    model.fit_models()

    print("\n=== Optimizing with Socio-Economic Factors ===")
    model.optimize_with_socio_economic()

    print("\n=== Generating 2025 Forecasts ===")
    forecasts = model.forecast_all_wards(
        forecast_start='2025-03-01',
        forecast_end='2025-12-01'
    )

    print("\n=== Saving Results ===")
    model.save_forecasts(OUTPUT_PATH)

    print("\n=== Forecast Summary ===")
    print(f"Generated {len(forecasts)} forecasts")
    print(f"Time period: March 2025 - December 2025")
    print(f"Wards forecasted: {forecasts['Ward Code'].nunique()}")
