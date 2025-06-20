# 4CBLW00-20 Group 2

## Project Overview

We develop an automated, data-driven police demand forecasting system to help reduce residential burglaries 
in London. 
By combining time-series forecasting with socio-economic factors, 
our solution enables police forces to forecast the number of future burglaries in each ward and efficiently allocate resources
with our police allocation formula based on the burglary numbers.

### Key Features:
- Hybrid SARIMA forecasting with socio-economic optimisation
- Ward-level predictions with one-month granularity
- Oscillation capture metrics for model evaluation
- Dashboard integration for visual resource planning

## Methodology

Our hybrid approach combines three key components:

1. **Time-Series Forecasting**: SARIMA models capture temporal patterns in burglary data
2. **Socio-Economic Integration**: 
   - Ward clustering based on social-economic factors
   - Cluster-specific model parametrisation
   - MAE optimisation for models with linear regression fitted with social-economic factors
3. **Model Evaluation Metrics**:
   - Average Mean Absolute Error (MAE)
   - Average Root Mean Squared Error (RMSE)
   - Oscillation Capture Score (OCS)
   - Variability Ratio (VR)
   - Smoothness Penalty (SP)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EvatsugDnalloR/MD_CBL-Group_2.git
```
$\hspace{0.8cm}$ **Note**: make sure that `gitlfs` is installed before cloning, if not you can install it by running:
```bash
git lfs install
```
2. Install dependencies listed in `pyproject.toml`. If you use `uv` as package manager, run: 
```bash
uv sync
```
$\hspace{0.8cm}$ If you only need to run the dashboard,
   the only dependencies that you have to install is `numpy`, `pandas`, `geopandas`,
`plotly`, and `dash`.

## Usage

### Data Cleaning
All burglary data from the period December 2013 until February 2025 can be found in the `data/all_burglaries` folder.
This data was downloaded from the [UK police website](https://data.police.uk/data/).
The cleaning object can be found in `clean_data.py`, which outputs the already existing CSV
file `residential_burglary.csv` that can be found in the `data` folder.

### Socio-Economic Factors Correlation to the Residential Burglary Counts
The analysis of correlation between socio-economic factors and burglary counts across wards
can be found in the `social_economic_burglary_correlation` module.
At the end, two housing-related factors have been chosen: **Percentage of households owning no car** 
and **Percentage of Households with Occupancy Rating of 0** (no extra bedroom), both on ward level.

### Performance Comparison between **Naive Forecasting Model** and **Hybrid Forecasting Model**
The Python scripts and Jupyter Notebooks for performance testing of the models can be found in the `model_performance_testing` module, 
and all the scripts and notebooks mentioned below are inside this module.

The SARIMA model powers the naive model with an auto-fitting function from the `pmdarima` package, 
which is a common work-flow for time-series forecasting projects using SARIMA. 
To see the standard metrics (MAE and RMSE) of the naive model and all the parameters (i.e. the orders and seasonal orders) of SARIMA
determined by `pmdarima`, run the file `finetune_ward_models.py`, and the orders of the model will be outputted as a CSV file.
To see more further testing with self-defined fitting scores mentioned above (OCS, VR and SP), run the notebook
`autoarima_fitting.ipynb`, which also provides further visualisation of the models for each ward.

The Hybrid Forecasting model is described above, and to see a complete step-by-step workflow from loading data to model
performance after MAE optimisation, run the notebook `clustering_sarima_opt.ipynb`, which also provides
optional visualisation.

### Generate Forecast
Run the main forecasting pipeline `social_economic_sarima.py`, which will:
1. Load and preprocess the data 
2. Cluster wards by socio-economic factors 
3. Train optimised SARIMA models 
4. Generate forecasts from March 2025 to December 2025 
5. Save results to `forecasts_output/2025_burglary_forecasts.csv`

### Dashboard
All the dashboard-related files are in the `dashboard` module. 
To run the dashboard, run `python -m dashboard.app` in the root folder of the project. 
Note that the dashboard is running on the localhost http://127.0.0.1:8050/.

### Main Page Functionalities

*   **Data Filtering:** Select a specific year and month to display map with the filtered data.

*   **Map Visualisation:**
    *   Displays a map of London wards, colour-coded to indicate the predicted number of burglaries
    *   Users can select the map to show either burglary prediction or police officers allocated.
    *   Hovering over a ward reveals its code, name, burglary prediction and number of police officers allocated.

*   **Ward Navigation:**
    *   Search for a specific ward by name or code to navigate to its dedicated page.
    *   Clicking a ward on the map also navigates to its dedicated page.

*   **Socio-Economic Data:**
    *   A button allows users to display an additional map showing socio-economic factors alongside the main map.

### Ward Page Functionalities

*   **Ward Data:**
    *   Displays the name and code of the selected ward.
    *   Allows to filter data by year and month for this specific ward.
    *   Shows number of police officers allocated and burglary prediction or  historical burglary count for the selected year and month.

*   **Time-Series Graphs:**
    *   **Burglary Prediction Graph:** Shows historical and predicted burglary counts over time for the ward.
    *   **Police Allocation Graph:** Displays the number of police officers allocated to the ward from March 2025 to December 2025 as a bar chart.
    
## Dashboard Files 

*   `dashboard/app.py`: The main Dash application file that initialises the app and defines the overall layout.
*   `dashboard/pages/index.py`: Defines the layout and functionalities for the main dashboard page.
*   `dashboard/pages/ward.py`: Defines the layout and functionalities for the ward pages.
*   `dashboard/data.py`: Loads data used in the dashboard from the `dashboard/static` folder.
  
## Summary of Key Findings
1. Serious underfitting issue for the models fitted by `pmdarima`:
   - There exist wards with predictions without a seasonal order, causing the forecasting to be a straight line.
   - There exist wards with predictions of smooth exponential decreasing lines that are seriously underfitted.
2. Correlation between social-economic factors and burglary data properties (for data after 2020):
   - The Percentage of Households Owning No Car has a negative correlation with the Volatility of the burglaries of each 
ward with a relatively strong correlation coefficient of about -0.47
   - The Percentage of Households Owning No Car has a positive correlation with the Average Absolute Difference of the
burglaries of each ward with a relatively strong correlation coefficient of about 0.52
   - The Percentage of Households with Bedroom Occupancy Rating of 0 has a negative correlation with the Volatility 
of the burglaries of each ward with a relatively strong correlation coefficient of about -0.47
   - The Percentage of Households with Bedroom Occupancy Rating of 0 has a positive correlation with the
Average Absolute Difference of the burglaries of each ward with a relatively strong correlation coefficient of about 0.48
3. Improvement of the Hybrid Model:
   - With only cluster-specific SARIMA order optimisation, compared to the best outcome of auto-fitted models, 
there is on average a 5% improvement on both MAE and RMSE scores, while for the best performed cluster, 
there is a 15% improvement on those scores.
   - With further MAE optimisation, compared to the best outcome of auto-fitted models, 
there is on average 15% improvement on both the MAE and RMSE scores, while for the best performed cluster, 
there is a 25% improvement on those scores.
   - The underfitting issue has been resolved, which can be shown from both the improvement on the fitting scores and visualisations.

## Contributors:
- Data Cleaning and Fine-Tune Ward Models - **Juliette Hattingh-Haasbroek** (1779192)
- Hybrid Model Architecture, Model Performance Testing, Fitting Scores, and the Main Forecasting Pipeline - **Gustave Rolland** (1957635)
- Dashboard - **Roberto Tormo Navarro** (1936786)
