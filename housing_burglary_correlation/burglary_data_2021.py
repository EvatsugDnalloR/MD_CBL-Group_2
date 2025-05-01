import pandas as pd
import glob


def process_burglary_data(data_path):
    """
    Function to load and process burglary crime data from multiple CSV files.

    Args:
        data_path (str): Path to the folder containing crime data CSV files.

    Returns:
        pd.DataFrame: Dataframe containing burglary count aggregated by LSOA code.
    """
    crime_files = glob.glob(f"{data_path}/*-metropolitan-street.csv")
    crime_dfs = []

    # Load crime data from multiple files
    for file in crime_files:
        df = pd.read_csv(file)
        crime_dfs.append(df)

    # Concatenate and filter only burglary crimes_city_and_metropolitan
    burglary_df = pd.concat(crime_dfs, ignore_index=True)
    burglary_df = burglary_df[burglary_df["Crime type"] == "Burglary"]

    # Aggregate burglary count by LSOA
    lsoa_burglary = burglary_df["LSOA code"].value_counts().reset_index()
    lsoa_burglary.columns = ["LSOA", "burglary_count"]

    return lsoa_burglary
