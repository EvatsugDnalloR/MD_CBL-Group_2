import pandas as pd
pd.set_option('display.max_columns', None)  # show all columns
import xlwings as xw
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def add_population(file_path: str) -> pd.DataFrame:
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

df_all_pop = add_population('lsoa_ward_data/land-area-population-density-london.xlsx')
print(df_all_pop)
