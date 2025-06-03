import pandas as pd
pd.set_option('display.max_columns', None)  # show all columns
import xlwings as xw
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import requests
import json
import joblib
from config import use_api


def population_df(file_path: str) -> pd.DataFrame:
    """
    Create dataframe with population data from all years.
    :param file_path: path to Excel file
    :return: population dataframe.
    """
    years = ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
    wb = xw.Book(file_path)
    sheet = wb.sheets['Ward']
    all_data = []

    for year in years:
        sheet.range('E1').value = year
        wb.app.calculate()
        data = sheet.range('A3:K628').value

        df_year = pd.DataFrame(data, columns=['Ward code', 'Ward name', 'Borough', 'Population', 'Hectares',
                                              'Square Kilometres', 'Population per hectare',
                                              'Population per square kilometre', 'Unnamed: 8',
                                              'Census population (2011)', 'Population per hectare.1'])
        df_year['Year'] = int(year)
        all_data.append(df_year)

    df = pd.concat(all_data, ignore_index=True)
    wb.close()

    return df


def new_ward_codes_df(old_codes: list[str]) -> dict[str, list]:
    """
    Use a JSON API to convert the 2022 ward codes to the current 2024 ward codes.
    The API returns the same dictionary as in `population_new_ward_codes.pkl`.
    :param old_codes: list of old ward codes
    :return: dictionary with the old wards as keys and their new ward codes as values.
    """
    new_codes = {}

    for c in old_codes:
        r = requests.get(f"https://findthatpostcode.uk/areas/{c}")
        data = json.loads(r.text)
        new_codes[c] = data['data']['attributes']['successor']

    return new_codes


df_all_pop = population_df('lsoa_ward_data/land-area-population-density-london.xlsx')
old_ward_codes = list(df_all_pop['Ward code'].unique())
new_ward_codes = new_ward_codes_df(old_ward_codes) if use_api else joblib.load('lsoa_ward_data/population_new_ward_codes.pkl')
df_all_pop['New ward code'] = df_all_pop['Ward code'].map(lambda k: new_ward_codes[k] if new_ward_codes[k] else [k])
df_all_pop  = df_all_pop.explode('New ward code', ignore_index=True)
cols_to_average = ['Population', 'Square Kilometres', 'Population per square kilometre']
df_all_pop = df_all_pop.groupby(['New ward code', 'Year'])[cols_to_average].mean().reset_index()
print(df_all_pop)
