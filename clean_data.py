import pandas as pd
import plotly.express as px
from config import path, eda
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Dataset:
    """
    Make one dataset containing burglary data from all dates.
    """
    def __init__(self, path: str = path, eda: bool = eda) -> None:
        """
        Initialize the Dataset object.
        :param path: path to directory to store dataframe, make sure it exists!
        :param eda: perform EDA on data
        """
        self.path = path
        self.eda = eda
        self.df_names = []

    @staticmethod
    def add_zero(n: int) -> str:
        """
        Add 0 in front of a single digit to correct the syntax for CSV files.
        :param n: month
        :return: corrected syntax for CSV files.
        """
        return f"0{n}" if len(str(n)) == 1 else str(n)

    def make_dataset(self) -> pd.DataFrame:
        """
        Add all dataframes together into a single dataframe.
        :return: burglary data for all dates.
        """
        for y in range(2013, 2026):
            for m in range(1, 13):
                m = self.add_zero(m)  # e.g. 5 changes to 05
                try:
                    df_y_m = pd.read_csv(f"data/all_burglaries/{str(y)}-{m}-metropolitan-street-burglary.csv")
                    self.df_names.append(df_y_m)
                except:  # csv with this year and month doesn't exist, move on toward next combo
                    continue

        df = pd.concat(self.df_names)  # add all dataframes together

        return df

    def clean_dataset(self) -> pd.DataFrame:
        """
        Perform EDA and clean the dataset.
        Ensure all LSOAs are renamed to their LSOA 2021 codes and names.
        Add ward code and name for each row based on LSOA.
        :return: cleaned dataset.
        """
        df = self.make_dataset()
        print(f"The dataset has {len(df)} rows.\n"
              f"There are {sum(df['Longitude'].isnull())} rows with no crime locations." if self.eda else "")

        df = df.dropna(subset=['Longitude', 'Latitude'])  # remove rows with empty locations
        df = df.drop(['Context', 'Crime type'], axis=1)  # remove empty and unnecessary columns
        df['Year'] = df['Month'].str.extract(r'(\d{4})-\d{2}').astype(int)  # separate month and year
        df['Month'] = df['Month'].str.extract(r'\d{4}-(\d{2})').astype(int)

        df_lsoa_2011_to_lsoa_2021 = pd.read_csv('lsoa_ward_data/LSOA_2011_to_LSOA_2021.csv')
        lsoa_codes = dict(zip(df_lsoa_2011_to_lsoa_2021['LSOA11CD'], df_lsoa_2011_to_lsoa_2021['LSOA21CD']))
        lsoa_names = dict(zip(df_lsoa_2011_to_lsoa_2021['LSOA11NM'], df_lsoa_2011_to_lsoa_2021['LSOA21NM']))

        df_ward = pd.read_csv('lsoa_ward_data/LSOA_2021_to_Electoral_Ward_2024.csv')
        ward_codes = dict(zip(df_ward['LSOA21CD'], df_ward['WD24CD']))
        ward_names = dict(zip(df_ward['LSOA21NM'], df_ward['WD24NM']))
        lad_codes = dict(zip(df_ward['WD24CD'], df_ward['LAD24CD']))

        df['LSOA code'] = df['LSOA code'].map(lsoa_codes).fillna(df['LSOA code'])
        df['LSOA name'] = df['LSOA name'].map(lsoa_names).fillna(df['LSOA name'])
        df['Ward code'] = df['LSOA code'].map(ward_codes)
        df['Ward name'] = df['LSOA name'].map(ward_names)
        df['LAD code'] = df['Ward code'].map(lad_codes)  # check if all burglaries took place in London

        in_london_mask = df['LAD code'].str.startswith('E09')  # all burglaries in London
        df_outside = df[~in_london_mask]
        print(f"There are {len(df_outside)} burglaries that took place outside of London." if self.eda else "")
        df = df[in_london_mask]

        city_of_london_mask = df['LSOA name'].str.contains('City of London')
        df_col = df[city_of_london_mask]
        print(f"There are {len(df_col)} burglaries that took place in the City of London." if self.eda else "")
        df = df[~city_of_london_mask]

        df = df.drop(['LAD code', 'Unnamed: 0'], axis=1)
        df.to_csv(f"{self.path}/burglary.csv", index=False)  # save cleaned dataset

        return df

    def get_residential_burglaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all non-residential burglaries.
        :param df: cleaned dataset with all burglaries.
        :return: dataset with non-residential burglaries.
        """
        not_residential_list = ["Supermarket", "Parking Area", "Sports/Recreation Area", "Shopping Area",
                                "Petrol Station", "Further/Higher Educational Building", "Nightclub",
                                "Pedestrian Subway", "Theatre/Concert Hall", "Hospital", "Conference/Exhibition Centre",
                                "Police Station", "Park/Open Space", "Bus/Coach Station"]  # non-residential locations
        not_residential_pattern = "|".join(not_residential_list)
        not_residential_mask = df['Location'].str.contains(not_residential_pattern)
        df_not = df[~not_residential_mask]  # remove all non-residential locations

        if self.eda:
            print(f"There are {len(df) - len(df_not)} non-residential burglaries.\n"
                  f"The cleaned dataset contains {len(df_not)} rows.\n" if self.eda else "")
            df_count = df.groupby(['Year', 'Month']).size().reset_index(name='count')  # get crime counts per month per year
            df_count = df_count.pivot(index='Month', columns='Year', values='count')
            df_count = df_count.sort_index()
            fig = px.line(df_count, markers=True, title="Number of Residential Burglaries per Month",
                          labels={"Month": "Month", "value": "Count"}, line_shape='linear')
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(1, 13))), xaxis_title="Month",
                              yaxis_title="Count", legend_title="Year")
            fig.show()

        df_not.to_csv(f"{self.path}/residential_burglary.csv", index=False)

        return df_not


dataset = Dataset(path, eda)
df_all_burglary = dataset.clean_dataset()  # get cleaned dataset and perform EDA if eda=True
df = dataset.get_residential_burglaries(df_all_burglary)  # final dataset
