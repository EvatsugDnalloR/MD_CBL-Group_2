import pandas as pd
import plotly.express as px


class Dataset():
    """
    Make one dataset containing burglary data from all dates.
    """
    def __init__(self, path: str) -> None:
        """
        Initialize the Dataset object.
        :param path: path to directory to store dataframe, make sure it exists!
        """
        self.path = path
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
                    df_y_m = pd.read_csv(f"all_data/{str(y)}-{m}-metropolitan-street-burglary.csv")
                    df_y_m = df_y_m[df_y_m['Crime type'] == 'Burglary']
                    self.df_names.append(df_y_m)
                except:  # csv with this year and month doesn't exist, move on toward next combo
                    continue

        df = pd.concat(self.df_names)  # add all dataframes together

        return df

    def clean_dataset(self) -> pd.DataFrame:
        """
        Perform EDA and clean the dataset.
        :return: cleaned dataset.
        """
        df = self.make_dataset()
        print(f"The dataset has {len(df)} rows and {len(df.columns)} columns.\n"
              f"There are {sum(df['Longitude'].isnull())} rows with no crime locations.\n")

        df = df.dropna(subset=['Longitude', 'Latitude'])  # remove rows with empty locations
        df = df.drop(['Context', 'Crime type', 'Unnamed: 0'], axis=1)  # remove empty and unnecessary columns
        df['Year'] = df['Month'].str.extract(r'(\d{4})-\d{2}').astype(int)  # separate month and year
        df['Month'] = df['Month'].str.extract(r'\d{4}-(\d{2})').astype(int)
        df.to_csv(f"{self.path}/all_dates_burglary.csv")  # save cleaned dataset

        return df

    def eda(self) -> pd.DataFrame:
        """
        Perform EDA on cleaned dataset.
        :return: cleaned dataset.
        """
        df = self.clean_dataset()
        print(f"Investigation outcomes:\n{df['Last outcome category'].value_counts()}\n\n"
              f"Burglaries per LSOA code:\n{df['LSOA name'].value_counts()}")

        df_count = df.groupby(['Year', 'Month']).size().reset_index(name='count')  # get crime counts per month per year
        df_count = df_count.pivot(index='Month', columns='Year', values='count')
        df_count = df_count.sort_index()
        fig = px.line(df_count, markers=True, title="Number of Crimes per Month",
                      labels={"Month": "Month", "value": "Count"}, line_shape='linear')
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(1, 13))), xaxis_title="Month",
                          yaxis_title="Count", legend_title="Year")
        fig.show()

        return df
