import json
from datetime import datetime

import geopandas as gpd
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

london_boundaries = gpd.read_file("static/london_ward_boundaries.geojson")
burglary = pd.read_csv("static/residential_burglary.csv")

app = Dash(__name__, title="London Residential Burglary & Police Allocation Dashboard")

app.layout = html.Div([
    html.H1("London Residential Burglary & Police Allocation Dashboard"),

    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': year}
                     for year in sorted(burglary['Year'].unique())],
            value=max(burglary['Year'].unique())  # Default to most recent year
        ),

        html.Label("Select Month:"),
        dcc.Dropdown(
            id='month-dropdown',
            options=[
                {'label': 'January', 'value': 1},
                {'label': 'February', 'value': 2},
                {'label': 'March', 'value': 3},
                {'label': 'April', 'value': 4},
                {'label': 'May', 'value': 5},
                {'label': 'June', 'value': 6},
                {'label': 'July', 'value': 7},
                {'label': 'August', 'value': 8},
                {'label': 'September', 'value': 9},
                {'label': 'October', 'value': 10},
                {'label': 'November', 'value': 11},
                {'label': 'December', 'value': 12}
            ],
            value=datetime.now().month  # Default to the current month

        ),
    ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

    html.Div([
        dcc.Graph(
            id='burglary-map',
            config={'modeBarButtonsToRemove': ['zoom', 'pan', 'select', 'zoomIn', 'zoomOut','autoScale', 'resetScale', 'lasso2d'],'displaylogo': False},
            style={'width': '100%', 'height': '700px'}
        ),

    ]),

])


@app.callback(Output('burglary-map', 'figure'), [Input('year-dropdown', 'value'), Input('month-dropdown', 'value')])
def update_dashboard(selected_year, selected_month):
    """
    Updates the burglary map figure based on the selected year and month.  Data is
    filtered to match the specified time frame.
    """
    filtered_burglary = burglary[(burglary['Year'] == selected_year) & (burglary['Month'] == selected_month)]

    if filtered_burglary.empty:
        fig = px.scatter_map(center={"lat": 51.5074, "lon": -0.1278}, zoom=9, map_style="carto-positron")
        fig.update_layout(annotations=[
            dict(text="Data not available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                 font=dict(size=20))], margin={"r": 0, "t": 0, "l": 0, "b": 0})

        return fig

    burglary_by_ward = filtered_burglary.groupby(['Ward code']).size().reset_index(name='burglary_count')

    filtered_boundaries = london_boundaries.merge(burglary_by_ward, left_on="WD24CD", right_on="Ward code", how="left")
    # Sets burglary count to 0 for wards with no occurrences in the burglary dataframe
    filtered_boundaries['burglary_count'] = filtered_boundaries['burglary_count'].fillna(0)

    boundaries_json = json.loads(filtered_boundaries.to_json())

    fig = px.choropleth_map(filtered_boundaries, geojson=boundaries_json, locations=filtered_boundaries.index,
                            color='burglary_count', hover_name="WD24NM",
                            hover_data=['WD24NM', 'WD24CD', 'burglary_count'],
                            labels={'burglary_count': 'Burglary Count', 'WD24NM': 'Ward name', 'WD24CD': 'Ward code'},
                            color_continuous_scale="Blues", map_style="carto-positron", zoom=9,
                            center={"lat": 51.5074, "lon": -0.1278}, opacity=0.7,

                            )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig


if __name__ == '__main__':
    app.run(debug=True)
