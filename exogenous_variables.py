import pandas as pd
pd.set_option('display.max_columns', None)  # show all columns
import xlwings as xw
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import requests
import json
import joblib
from config import use_api


class Population:
    """
    Create the population dataframe from all years for all wards.
    :attributes: file_path
    """
    def __init__(self, file_path: str):
        """
        Initialize the Population object.
        :param file_path: path to Excel file containing population data
        """
        self.file_path = file_path

    def population_df(self) -> pd.DataFrame:
        """
        Create dataframe with population data from all years.
        :return: population dataframe.
        """
        wb = xw.Book(self.file_path)
        sheet = wb.sheets['Ward']
        all_data = []

        for year in ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']:
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

    def new_ward_codes_df(self, old_codes: list[str]) -> dict[str, list]:
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

    def clean_population_df(self) -> pd.DataFrame:
        """
        Obtain population data from all years for all wards.
        Rename the old ward codes to the new codes.
        Aggregate the rows for each ward per year by averages.
        This method returns the same dataframe as in `population.csv`.
        :return: population dataframe with necessary columns.
        """
        df = self.population_df()
        old_ward_codes = list(df['Ward code'].unique())
        new_ward_codes = self.new_ward_codes_df(old_ward_codes) if use_api else joblib.load('lsoa_ward_data/population_new_ward_codes.pkl')
        df['New ward code'] = df['Ward code'].map(lambda k: new_ward_codes[k] if new_ward_codes[k] else [k])
        df  = df.explode('New ward code', ignore_index=True)
        cols_to_average = ['Population', 'Square Kilometres', 'Population per square kilometre']
        df = df.groupby(['New ward code', 'Year'])[cols_to_average].mean().reset_index()

        return df


df_pop = Population('lsoa_ward_data/land-area-population-density-london.xlsx').clean_population_df()
print(df_pop)
