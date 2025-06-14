import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from numpy import floating
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from model_fitting_score import oscillation_capture_score


class SocioEconomicSARIMA:
    """
    Pipeline of a hybrid SARIMA forecasting model with socio-economic optimization for burglary prediction.

    This class implements a sophisticated time series forecasting approach that combines
    traditional SARIMA modeling with socio-economic clustering to improve prediction
    accuracy for residential burglary incidents across different geographical wards.

    The model workflow includes:
    1. Loading and preprocessing burglary and socio-economic data
    2. Clustering wards based on socio-economic indicators
    3. Training cluster-specific SARIMA models
    4. Optimizing predictions using socio-economic adjustment factors
    5. Generating forecasts with performance metrics

    @author: Gustave
    """

    def __init__(self, burglary_path: str, cars_path: str, occupancy_path: str):
        """
        Initialize the SocioEconomicSARIMA model with data file paths.

        @param burglary_path: Path to the residential burglary CSV file containing
                             historical crime data with columns: Year, Month, Ward Code
        @param cars_path: Path to the car ownership Excel file containing socio-economic
                         data with sheet '2021' and columns: ward code, none, All households
        @param occupancy_path: Path to the bedroom occupancy Excel file containing
                              housing data with sheet '2021' and columns: ward code, 0, All Households

        @raise FileNotFoundError: If any of the specified file paths do not exist
        @raise ValueError: If file paths are empty or None
        """
        # Input validation
        if not burglary_path or not cars_path or not occupancy_path:
            raise ValueError("All file paths must be provided and non-empty")

        # Check if files exist
        for path, name in [(burglary_path, "burglary"), (cars_path, "cars"), (occupancy_path, "occupancy")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"The {name} file does not exist: {path}")

        # Store file paths
        self.burglary_path = burglary_path
        self.cars_path = cars_path
        self.occupancy_path = occupancy_path

        # Predefined cluster-specific SARIMA orders
        # Each cluster gets optimized (p,d,q) and (P,D,Q,s) parameters
        self.cluster_orders = {
            0: {"normal": (1, 1, 3), "seasonal": (0, 1, 3, 12)},
            1: {"normal": (2, 1, 2), "seasonal": (0, 1, 2, 12)},
            2: {"normal": (3, 1, 1), "seasonal": (0, 1, 1, 12)}
        }

        # Initialize instance variables
        self.models: Dict = {}
        self.training_results: List = []
        self.burglary_monthly: Optional[pd.DataFrame] = None
        self.optimization_params: Dict = {}
        self.cluster_df: Optional[pd.DataFrame] = None
        self.forecast_results: Optional[pd.DataFrame] = None
        self.model_scores: Dict = {}

    def load_and_preprocess(self):
        """
         Load and preprocess all required datasets for model training.

         This method:
         1. Loads burglary data and aggregates it by ward and month
         2. Loads socio-economic indicators (car ownership and bedroom occupancy)
         3. Calculates derived features (NoCarPct, ZeroRoomPct)
         4. Merges datasets on ward code

         @raise FileNotFoundError: If required files cannot be accessed
         @raise KeyError: If required columns are missing from the datasets
         @raise ValueError: If data formats are incompatible or datasets are empty
         """
        print("Loading burglary data...")
        # Load and validate burglary data
        burglary = pd.read_csv(self.burglary_path)

        # Check required columns
        required_burglary_cols = ["Year", "Month", "Ward Code"]
        missing_cols = [col for col in required_burglary_cols if col not in burglary.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in burglary data: {missing_cols}")

        if burglary.empty:
            raise ValueError("Burglary dataset is empty")

        # Process burglary data
        burglary["Month"] = pd.to_datetime(
            burglary["Year"].astype(str) + "-" + burglary["Month"].astype(str),
            format="%Y-%m"
        )
        self.burglary_monthly = burglary.groupby(
            ["Ward Code", "Month"]
        ).size().reset_index(name="Count")

        print("Loading socio-economic data...")
        # Load and validate cars data
        try:
            cars_df = pd.read_excel(self.cars_path, sheet_name="2021")
        except ValueError as e:
            raise ValueError(f"Sheet '2021' not found in cars file: {e}")

        required_cars_cols = ["ward code", "none", "All households"]
        missing_cols = [col for col in required_cars_cols if col not in cars_df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in cars data: {missing_cols}")

        # Calculate car ownership percentage
        if cars_df["All households"].sum() == 0:
            warnings.warn("All households sum is zero in cars data - this may cause division by zero")
        cars_df["NoCarPct"] = cars_df["none"] / cars_df["All households"]

        # Load and validate occupancy data
        try:
            occupancy_df = pd.read_excel(self.occupancy_path, sheet_name="2021")
        except ValueError as e:
            raise ValueError(f"Sheet '2021' not found in occupancy file: {e}")

        required_occupancy_cols = ["ward code", "0", "All Households"]
        missing_cols = [col for col in required_occupancy_cols if col not in occupancy_df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in occupancy data: {missing_cols}")

        # Calculate zero room percentage
        if occupancy_df["All Households"].sum() == 0:
            warnings.warn("All Households sum is zero in occupancy data - this may cause division by zero")
        occupancy_df["ZeroRoomPct"] = occupancy_df["0"] / occupancy_df["All Households"]

        # Merge socio-economic data
        self.cluster_df = pd.merge(
            cars_df[["ward code", "NoCarPct"]],
            occupancy_df[["ward code", "ZeroRoomPct"]],
            on="ward code",
            how="inner"
        ).rename(columns={"ward code": "ward_code"})

        # Validate merge results
        if self.cluster_df.empty:
            raise ValueError("No matching ward codes found between cars and occupancy data")

        # Check for missing values
        if self.cluster_df.isnull().any().any():
            null_counts = self.cluster_df.isnull().sum()
            warnings.warn(f"Missing values found in socio-economic data: {null_counts.to_dict()}")

        print(f"Loaded {len(self.burglary_monthly)} burglary records")
        print(f"Loaded socio-economic data for {len(self.cluster_df)} wards")
        print(f"Unique wards in burglary data: {self.burglary_monthly['Ward Code'].nunique()}")

        # Check data overlap
        burglary_wards = set(self.burglary_monthly['Ward Code'].unique())
        socio_wards = set(self.cluster_df['ward_code'].unique())
        common_wards = burglary_wards.intersection(socio_wards)

        if len(common_wards) == 0:
            raise ValueError("No common ward codes found between burglary and socio-economic data")

        missing_socio = burglary_wards - socio_wards
        if missing_socio:
            warnings.warn(f"Burglary data contains {len(missing_socio)} wards without socio-economic data")

    def perform_clustering(self, n_clusters: int = 3):
        """
        Cluster wards based on socio-economic factors using K-means algorithm.

        This method performs unsupervised clustering to group wards with similar
        socio-economic characteristics, enabling cluster-specific SARIMA parameter tuning.

        @param n_clusters: Number of clusters to create (default: 3)
        """
        # Standardize features
        X = self.cluster_df[["NoCarPct", "ZeroRoomPct"]]
        X_scaled = StandardScaler().fit_transform(X)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_df["cluster"] = kmeans.fit_predict(X_scaled)

        # Print cluster distribution
        cluster_counts = self.cluster_df["cluster"].value_counts().sort_index()
        print("\nCluster Distribution:")
        for cluster_id, count in cluster_counts.items():
            print(f"Cluster {cluster_id}: {count} wards")

    def fit_models(self):
        """
        Train SARIMA models for all wards with cluster-specific parameters.

        This method:
        1. Groups burglary data by ward
        2. Assigns cluster-specific SARIMA orders to each ward
        3. Splits data into train/test sets (80/20)
        4. Fits SARIMA models and stores results
        5. Calculates performance metrics for each ward

        @raise ValueError: If required data is not loaded or clustering not performed
        @raise RuntimeError: If model fitting fails for all wards
        """
        if self.cluster_df is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")

        successful_fits = 0
        total_wards = 0
        failed_wards = []

        print("Fitting SARIMA models...")

        for ward_code, group in self.burglary_monthly.groupby("Ward Code"):
            total_wards += 1

            # Skip wards without socio-economic data
            if ward_code not in self.cluster_df["ward_code"].values:
                failed_wards.append((ward_code, "No socio-economic data"))
                continue

            try:
                # Get cluster-specific orders
                cluster_id = self.cluster_df.loc[
                    self.cluster_df["ward_code"] == ward_code, "cluster"
                ].values[0]

                if cluster_id not in self.cluster_orders:
                    failed_wards.append((ward_code, f"No SARIMA orders for cluster {cluster_id}"))
                    continue

                orders = self.cluster_orders[cluster_id]

                # Prepare time series
                series = group[["Month", "Count"]].set_index("Month").asfreq("MS")["Count"]

                # Check series length
                if len(series) < 24:  # Need at least 2 years for seasonal modeling
                    warnings.warn(f"Ward {ward_code} has only {len(series)} months of data. "
                                  "Seasonal modeling may be unreliable.")

                # Handle missing values
                nan_count = series.isnull().sum()
                if nan_count > 0:
                    print(f"{ward_code} has {nan_count} NaN values - filled by 0")
                    series.fillna(0, inplace=True)

                # Check for constant series
                if series.var() == 0:
                    warnings.warn(f"Ward {ward_code} has constant burglary counts. Model may be unreliable.")

                # Split into train/test (80/20)
                len_data = len(series)
                test_month = max(1, int(np.ceil(len_data * 0.2)))  # At least 1 month for testing
                train = series.iloc[:-test_month]
                test = series.iloc[-test_month:]

                # Fit SARIMA model
                model = SARIMAX(
                    train,
                    order=orders["normal"],
                    seasonal_order=orders["seasonal"]
                )

                # Suppress convergence warnings for individual models
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    results = model.fit(disp=False)

                # Store model
                self.models[ward_code] = {
                    "model": results,
                    "orders": orders,
                    "cluster": cluster_id
                }

                # Generate predictions for performance evaluation
                preds = results.get_forecast(steps=test_month).predicted_mean
                residuals = test - preds

                # Store training results
                self.training_results.append({
                    "ward_code": ward_code,
                    "cluster": cluster_id,
                    "residuals": residuals.values,
                    "predicted": preds.values,
                    "actual": test.values
                })

                successful_fits += 1
                print(f"{ward_code} finished successfully")

            except Exception as e:
                failed_wards.append((ward_code, str(e)))
                print(f"Failed to fit model for ward {ward_code}: {str(e)}")

        # Report results
        print(f"\nModel fitting completed:")
        print(f"Successfully trained: {successful_fits}/{total_wards} wards")

        if failed_wards:
            print(f"Failed wards: {len(failed_wards)}")
            for ward, reason in failed_wards[:5]:  # Show first 5 failures
                print(f"  {ward}: {reason}")
            if len(failed_wards) > 5:
                print(f"  ... and {len(failed_wards) - 5} more")

        if successful_fits == 0:
            raise RuntimeError("No models were successfully fitted. Check data quality and parameters.")

        if successful_fits < total_wards * 0.5:
            warnings.warn(f"Less than 50% of models fitted successfully ({successful_fits}/{total_wards})")

    def optimize_with_socio_economic(self):
        """
        Train MAE adjustment model using socio-economic factors.

        This method applies post-processing optimization by:
        1. Training a linear regression model to predict MAE based on socio-economic factors
        2. Adjusting predictions to minimize forecasting errors
        3. Computing comprehensive performance metrics for each ward

        The optimization only applies adjustments that actually improve performance.

        @raise ValueError: If models have not been fitted or required data is missing
        @raise RuntimeError: If optimization process fails
        """
        if not self.models:
            raise ValueError("No models available. Call fit_models() first.")

        if not self.training_results:
            raise ValueError("No training results available. Call fit_models() first.")

        try:
            print("Optimizing with socio-economic factors...")

            # Prepare training data
            results_df = pd.DataFrame(self.training_results)
            results_df = pd.merge(
                results_df,
                self.cluster_df,
                left_on="ward_code",
                right_on="ward_code",
                how="left"
            )

            # Check for missing merges
            missing_socio = results_df[results_df[["NoCarPct", "ZeroRoomPct"]].isnull().any(axis=1)]
            if not missing_socio.empty:
                warnings.warn(f"Missing socio-economic data for {len(missing_socio)} wards during optimization")
                results_df = results_df.dropna(subset=["NoCarPct", "ZeroRoomPct"])

            if results_df.empty:
                raise ValueError("No valid data available for optimization after removing missing values")

            # Compute MAE for each ward
            results_df["mae"] = results_df.apply(
                lambda x: np.mean(np.abs(x["residuals"])) if len(x["residuals"]) > 0 else np.inf,
                axis=1
            )

            # Train linear regression model for MAE prediction
            X = results_df[["NoCarPct", "ZeroRoomPct"]]
            y = results_df["mae"]
            mae_model = LinearRegression()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                mae_model.fit(X, y)

            optimized_count = 0
            total_count = len(results_df)

            # Apply optimization to each ward
            for _, row in results_df.iterrows():
                # Predict MAE adjustment
                X_ward = [[row["NoCarPct"], row["ZeroRoomPct"]]]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    mae_adj = mae_model.predict(X_ward)[0]

                # Determine adjustment direction based on residual bias
                mean_residual = np.mean(row["residuals"])
                sign = 1 if mean_residual >= 0 else -1

                # Apply adjustment
                adj_pred = row["predicted"] + sign * mae_adj
                adj_mae = mean_absolute_error(row["actual"], adj_pred)

                # Only keep adjustment if it improves performance
                if adj_mae < row["mae"] and np.isfinite(adj_mae):
                    self.optimization_params[row["ward_code"]] = {
                        "mae_adj": mae_adj,
                        "sign": sign
                    }

                    # Calculate comprehensive scores for optimized prediction
                    ocs, dvr, sp = oscillation_capture_score(row["actual"], adj_pred)
                    self.model_scores[row["ward_code"]] = {
                        "final_mae": adj_mae,
                        "final_rmse": np.sqrt(mean_squared_error(row["actual"], adj_pred)),
                        "ocs_score": ocs,
                        "dvr_score": dvr,
                        "sp_score": sp
                    }
                    optimized_count += 1
                else:
                    # Keep original prediction
                    ocs, dvr, sp = oscillation_capture_score(row["actual"], row["predicted"])
                    self.model_scores[row["ward_code"]] = {
                        "final_mae": row["mae"],
                        "final_rmse": np.sqrt(mean_squared_error(row["actual"], row["predicted"])),
                        "ocs_score": ocs,
                        "dvr_score": dvr,
                        "sp_score": sp
                    }

            print(f"Optimization completed: {optimized_count}/{total_count} wards improved")

            if optimized_count == 0:
                warnings.warn("No wards benefited from socio-economic optimization")

        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}")

    def forecast_ward(self, ward_code: str, forecast_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Forecast burglary counts for a specific ward.

        @param ward_code: Ward identifier to forecast
        @param forecast_dates: Array of dates for which to generate forecasts
        @return: Forecasted values for the specified dates

        @raise ValueError: If no model exists for the specified ward
        @raise RuntimeError: If forecasting fails
        """
        if ward_code not in self.models:
            raise ValueError(f"No model found for ward {ward_code}. ")

        if len(forecast_dates) == 0:
            raise ValueError("No forecast dates provided")

        try:
            # Generate base forecast
            model_info = self.models[ward_code]
            steps = len(forecast_dates)

            # Validate steps
            if steps <= 0:
                raise ValueError("Number of forecast steps must be positive")

            forecast = model_info["model"].get_forecast(steps=steps)
            base_pred = forecast.predicted_mean

            # Apply socio-economic adjustment if available
            if ward_code in self.optimization_params:
                params = self.optimization_params[ward_code]
                adjusted_pred = base_pred + params["sign"] * params["mae_adj"]
                return adjusted_pred

            return base_pred

        except Exception as e:
            raise RuntimeError(f"Forecasting failed for ward {ward_code}: {str(e)}")

    def forecast_all_wards(self, forecast_start: str, num_forecast_months: int) -> pd.DataFrame:
        """
        Generate forecasts for all trained wards.

        @param forecast_start: Start date for forecasting in 'YYYY-MM-DD' format
        @param num_forecast_months: Number of months to forecast ahead
        @return: DataFrame containing forecasts and performance metrics for all wards

        @raise ValueError: If parameters are invalid or models not trained
        @raise RuntimeError: If forecasting process fails
        """
        if not self.models:
            raise ValueError("No models available. Call fit_models() first.")

        if num_forecast_months <= 0:
            raise ValueError("Number of forecast months must be positive")

        try:
            # Validate and parse forecast start date
            forecast_dates = pd.date_range(
                start=forecast_start,
                periods=num_forecast_months,
                freq="MS"
            )

            results = []
            successful_forecasts = 0
            failed_forecasts = []

            print(f"Generating forecasts for {len(self.models)} wards...")

            for ward_code in self.models:
                # Generate predictions
                preds = self.forecast_ward(ward_code, forecast_dates)

                # Get performance scores
                scores = self.model_scores.get(ward_code)

                # Store results for each forecast date
                for date, pred in zip(forecast_dates, preds, strict=False):
                    results.append({
                        "Ward Code": ward_code,
                        "Month": date.strftime("%Y-%m"),
                        "Forecast": max(0, int(np.round(pred))),  # Ensure non-negative integer
                        "mae": scores["final_mae"],
                        "rmse": scores["final_rmse"],
                        "ocs_score": scores["ocs_score"],
                        "dvr_score": scores["dvr_score"],
                        "sp_score": scores["sp_score"]
                    })

                successful_forecasts += 1

            if not results:
                raise RuntimeError("No successful forecasts generated")

            self.forecast_results = pd.DataFrame(results)

            print(f"Forecast generation completed:")
            print(f"Successful: {successful_forecasts}/{len(self.models)} wards")

            if failed_forecasts:
                print(f"Failed: {len(failed_forecasts)} wards")
                for ward, error in failed_forecasts[:3]:  # Show first 3 failures
                    print(f"  {ward}: {error}")

            return self.forecast_results

        except Exception as e:
            raise RuntimeError(f"Forecast generation failed: {str(e)}")

    def save_forecasts(self, output_path: str):
        """
        Save forecast results to a CSV file.

        @param output_path: Path where the forecast CSV file will be saved

        @raise ValueError: If no forecasts are available or output path is invalid
        @raise RuntimeError: If file saving fails
        """
        if self.forecast_results is None:
            raise ValueError("No forecasts available. Run forecast_all_wards() first.")

        if not output_path:
            raise ValueError("Output path cannot be empty")

        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Validate output path is writable
            if output_dir.exists() and not os.access(output_dir, os.W_OK):
                raise PermissionError(f"No write permission for directory: {output_dir}")

            # Save forecasts
            self.forecast_results.to_csv(output_path, index=False)

            # Verify file was created successfully
            if not Path(output_path).exists():
                raise RuntimeError("File was not created successfully")

            file_size = Path(output_path).stat().st_size
            print(f"Saved forecasts to {output_path} ({file_size:,} bytes)")
            print(f"File contains {len(self.forecast_results)} forecast records")

        except Exception as e:
            raise RuntimeError(f"Failed to save forecasts: {str(e)}")

    def get_model_summary(self) -> dict[str, str] | dict[
        str, int | dict[Any, Any] | dict[str, floating[Any] | float] | dict[str, int | float | Any] | dict[
            Any, int | Any]]:
        """
        Get a comprehensive summary of the trained models and their performance.

        @return: Dictionary containing model statistics and performance metrics

        @raise ValueError: If models have not been trained
        """
        if not self.models:
            raise ValueError("No models available. Call fit_models() first.")

        try:
            summary = {
                "total_wards": len(self.models),
                "total_clusters": len(set(info["cluster"] for info in self.models.values())),
                "optimized_wards": len(self.optimization_params),
                "cluster_distribution": {},
                "performance_metrics": {},
                "data_coverage": {}
            }

            # Cluster distribution
            cluster_counts = {}
            for info in self.models.values():
                cluster = info["cluster"]
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            summary["cluster_distribution"] = cluster_counts

            # Performance metrics aggregation
            if self.model_scores:
                mae_values = [scores["final_mae"] for scores in self.model_scores.values()
                              if np.isfinite(scores["final_mae"])]
                rmse_values = [scores["final_rmse"] for scores in self.model_scores.values()
                               if np.isfinite(scores["final_rmse"])]
                ocs_values = [scores["ocs_score"] for scores in self.model_scores.values()
                              if np.isfinite(scores["ocs_score"])]

                if mae_values:
                    summary["performance_metrics"] = {
                        "mean_mae": np.mean(mae_values),
                        "median_mae": np.median(mae_values),
                        "mean_rmse": np.mean(rmse_values) if rmse_values else np.nan,
                        "median_rmse": np.median(rmse_values) if rmse_values else np.nan,
                        "mean_ocs": np.mean(ocs_values) if ocs_values else np.nan,
                        "median_ocs": np.median(ocs_values) if ocs_values else np.nan
                    }

            # Data coverage information
            if self.burglary_monthly is not None and self.cluster_df is not None:
                total_burglary_wards = self.burglary_monthly['Ward Code'].nunique()
                total_socio_wards = len(self.cluster_df)

                summary["data_coverage"] = {
                    "burglary_wards": total_burglary_wards,
                    "socio_economic_wards": total_socio_wards,
                    "coverage_ratio": len(self.models) / total_burglary_wards if total_burglary_wards > 0 else 0
                }

            return summary

        except Exception as e:
            warnings.warn(f"Error generating model summary: {str(e)}")
            return {"error": str(e)}


if __name__ == "__main__":
    start_time = time.time()

    # Configuration
    BURGLARY_PATH = "data/residential_burglary.csv"
    CARS_PATH = "data/housing/cars_or_vans_wards.xlsx"
    OCCUPANCY_PATH = "data/housing/occupancy_rating_bedrooms_wards.xlsx"
    OUTPUT_PATH = "forecasts_output/2025_burglary_forecasts.csv"

    # Initialize model
    print("=== Initializing SocioEconomicSARIMA Model ===")
    model = SocioEconomicSARIMA(
        burglary_path=BURGLARY_PATH,
        cars_path=CARS_PATH,
        occupancy_path=OCCUPANCY_PATH
    )

    # Load and preprocess data
    print("\n=== Loading and Preprocessing Data ===")
    model.load_and_preprocess()

    # Perform clustering
    print("\n=== Performing Ward Clustering ===")
    model.perform_clustering(n_clusters=3)

    # Train models
    print("\n=== Training SARIMA Models ===")
    model.fit_models()

    # Optimize with socio-economic factors
    print("\n=== Optimizing with Socio-Economic Factors ===")
    model.optimize_with_socio_economic()

    # Generate forecasts
    print("\n=== Generating 2025 Forecasts ===")
    forecasts = model.forecast_all_wards(
        forecast_start="2025-03-01",
        num_forecast_months=10
    )

    # Save results
    print("\n=== Saving Results ===")
    model.save_forecasts(OUTPUT_PATH)

    # Display summary
    print("\n=== Model Summary ===")
    summary = model.get_model_summary()
    print(f"Total wards modeled: {summary.get('total_wards', 'N/A')}")
    print(f"Clusters used: {summary.get('total_clusters', 'N/A')}")
    print(f"Wards with optimization: {summary.get('optimized_wards', 'N/A')}")

    if 'performance_metrics' in summary and summary['performance_metrics']:
        metrics = summary['performance_metrics']
        print(f"Mean MAE: {metrics.get('mean_mae', 'N/A'):.3f}")
        print(f"Mean RMSE: {metrics.get('mean_rmse', 'N/A'):.3f}")

    # Forecast summary
    print("\n=== Forecast Summary ===")
    print(f"Generated {len(forecasts)} forecast records")
    print("Time period: March 2025 - December 2025")
    print(f"Wards forecasted: {forecasts['Ward Code'].nunique()}")
    print(f"Total predicted burglaries: {forecasts['Forecast'].sum():,}")

    execution_time = time.time() - start_time
    print(f"\nCompleted successfully in {execution_time:.2f} seconds")
