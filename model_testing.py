import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class CrimeForecastingModel:
    """
    Enhanced crime forecasting model integrating socioeconomic factors
    through SARIMAX framework
    """

    def __init__(self, sarima_order=(3, 1, 1), seasonal_order=(3, 1, 1, 12)):
        self.baseline_model = None
        self.sarima_order = sarima_order
        self.seasonal_order = seasonal_order
        self.models = {}
        self.scaler = StandardScaler()
        self.ward_baseline_effects = {}

    def prepare_socioeconomic_features(self, crime_data, car_data, occupancy_data):
        """
        Prepare socioeconomic features for integration
        """
        # Calculate percentages for key socioeconomic indicators
        car_features = car_data.copy()
        car_features['pct_no_car'] = car_features['none'] / car_features['All households']
        car_features['pct_one_car'] = car_features['one'] / car_features['All households']
        car_features['pct_multiple_cars'] = (car_features['two'] + car_features['three or more']) / car_features['All households']

        occupancy_features = occupancy_data.copy()
        occupancy_features['pct_optimal_occupancy'] = occupancy_features['0'] / occupancy_features['All Households']
        occupancy_features['pct_overcrowded'] = (occupancy_features['-1'] + occupancy_features['-2 or less']) / occupancy_features['All Households']
        occupancy_features['pct_underutilized'] = (occupancy_features['+1'] + occupancy_features['+2 or more']) / occupancy_features['All Households']

        # Merge features by ward code
        socio_features = car_features[['ward code', 'pct_no_car', 'pct_one_car', 'pct_multiple_cars']].merge(
            occupancy_features[['ward code', 'pct_optimal_occupancy', 'pct_overcrowded', 'pct_underutilized']],
            on='ward code'
        )

        return socio_features

    def create_ward_panels(self, crime_data, socio_features):
        """
        Create panel dataset combining crime time series with socioeconomic features
        """
        panels = {}

        # Group crime data by ward
        for ward_code in crime_data['ward_code'].unique():
            ward_crimes = crime_data[crime_data['ward_code'] == ward_code].copy()

            # Sort by date and create time series
            ward_crimes = ward_crimes.sort_values('date')
            ward_crimes.set_index('date', inplace=True)

            # Add socioeconomic features (constant for each ward)
            ward_socio = socio_features[socio_features['ward code'] == ward_code]
            if not ward_socio.empty:
                for col in ['pct_no_car', 'pct_optimal_occupancy', 'pct_overcrowded']:
                    ward_crimes[col] = ward_socio[col].iloc[0]

                panels[ward_code] = ward_crimes

        return panels

    def estimate_baseline_effects(self, panels):
        """
        Estimate ward-specific baseline effects using socioeconomic factors
        """
        ward_means = []
        socio_vars = []

        for ward_code, data in panels.items():
            if len(data) > 24:  # Require at least 2 years of data
                ward_means.append(data['crime_count'].mean())
                socio_vars.append([
                    data['pct_no_car'].iloc[0],
                    data['pct_optimal_occupancy'].iloc[0],
                    data['pct_overcrowded'].iloc[0]
                ])

        # Fit baseline effect model
        X = np.array(socio_vars)
        y = np.array(ward_means)

        # Simple linear regression for baseline effects
        from sklearn.linear_model import LinearRegression
        baseline_model = LinearRegression()
        baseline_model.fit(X, y)

        self.baseline_model = baseline_model
        return baseline_model

    def fit_ward_models(self, panels, use_external_regressors=True):
        """
        Fit SARIMAX models for each ward
        """
        successful_fits = 0

        for ward_code, data in panels.items():
            try:
                if len(data) < 36:  # Need sufficient data
                    continue

                # Prepare exogenous variables
                if use_external_regressors:
                    exog = data[['pct_no_car', 'pct_optimal_occupancy']].values
                else:
                    exog = None

                # Fit SARIMAX model
                model = SARIMAX(
                    data['crime_count'],
                    exog=exog,
                    order=self.sarima_order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                fitted_model = model.fit(disp=False, maxiter=100)
                self.models[ward_code] = fitted_model
                successful_fits += 1

            except Exception as e:
                print(f"Failed to fit model for ward {ward_code}: {str(e)}")
                continue

        print(f"Successfully fitted models for {successful_fits} wards")
        return successful_fits

    def forecast_ward(self, ward_code, steps=12, socio_features=None):
        """
        Generate forecasts for a specific ward
        """
        if ward_code not in self.models:
            return None

        model = self.models[ward_code]

        # Prepare exogenous variables for forecast period
        if socio_features is not None and ward_code in socio_features['ward code'].values:
            ward_socio = socio_features[socio_features['ward code'] == ward_code]
            exog_forecast = np.tile([
                ward_socio['pct_no_car'].iloc[0],
                ward_socio['pct_optimal_occupancy'].iloc[0]
            ], (steps, 1))
        else:
            exog_forecast = None

        # Generate forecast
        forecast = model.forecast(steps=steps, exog=exog_forecast)
        forecast_ci = model.get_forecast(steps=steps, exog=exog_forecast).conf_int()

        return {
            'forecast': forecast,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1]
        }

    def evaluate_model_performance(self, panels, test_periods=6):
        """
        Evaluate model performance using walk-forward validation
        """
        ward_performances = {}

        for ward_code, data in panels.items():
            if ward_code not in self.models or len(data) < 48:
                continue

            # Split data for validation
            train_data = data.iloc[:-test_periods]
            test_data = data.iloc[-test_periods:]

            try:
                # Refit model on training data
                exog_train = train_data[['pct_no_car', 'pct_optimal_occupancy']].values
                exog_test = test_data[['pct_no_car', 'pct_optimal_occupancy']].values

                model = SARIMAX(
                    train_data['crime_count'],
                    exog=exog_train,
                    order=self.sarima_order,
                    seasonal_order=self.seasonal_order
                )
                fitted_model = model.fit(disp=False)

                # Generate forecasts
                forecasts = fitted_model.forecast(steps=test_periods, exog=exog_test)

                # Calculate performance metrics
                mae = mean_absolute_error(test_data['crime_count'], forecasts)
                rmse = np.sqrt(mean_squared_error(test_data['crime_count'], forecasts))
                mape = np.mean(np.abs((test_data['crime_count'] - forecasts) / test_data['crime_count'])) * 100

                ward_performances[ward_code] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape
                }

            except Exception as e:
                continue

        return ward_performances

    def analyze_socioeconomic_impact(self, panels):
        """
        Analyze the impact of socioeconomic factors on predictions
        """
        impact_analysis = {}

        for ward_code, model in self.models.items():
            if hasattr(model, 'params') and len(model.params) > len(self.sarima_order):
                # Extract coefficients for socioeconomic variables
                # Assuming first external regressor is pct_no_car, second is pct_optimal_occupancy
                try:
                    socio_coeffs = model.params[-2:]  # Last two parameters should be exog variables
                    impact_analysis[ward_code] = {
                        'no_car_coeff': socio_coeffs.iloc[0],
                        'optimal_occupancy_coeff': socio_coeffs.iloc[1],
                        'model_aic': model.aic,
                        'model_bic': model.bic
                    }
                except:
                    continue

        return impact_analysis

# Example usage framework
def run_enhanced_forecasting_pipeline():
    """
    Example of how to run the enhanced forecasting pipeline
    """
    # Initialize model
    crime_model = CrimeForecastingModel()

    # Load and prepare data (placeholder - replace with actual data loading)
    # crime_data = pd.read_csv('residential_burglary.csv')
    # car_data = pd.read_csv('car_access_data.csv')
    # occupancy_data = pd.read_csv('occupancy_data.csv')

    # Prepare socioeconomic features
    # socio_features = crime_model.prepare_socioeconomic_features(crime_data, car_data, occupancy_data)

    # Create ward panels
    # panels = crime_model.create_ward_panels(crime_data, socio_features)

    # Estimate baseline effects
    # baseline_model = crime_model.estimate_baseline_effects(panels)

    # Fit ward-specific models
    # crime_model.fit_ward_models(panels, use_external_regressors=True)

    # Evaluate performance
    # performance = crime_model.evaluate_model_performance(panels)

    # Analyze socioeconomic impact
    # impact_analysis = crime_model.analyze_socioeconomic_impact(panels)

    print("Enhanced forecasting pipeline framework ready for implementation")

if __name__ == "__main__":
    run_enhanced_forecasting_pipeline()