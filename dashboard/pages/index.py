
from datetime import datetime
import plotly.express as px
from dash import dcc, html, no_update, register_page, callback
from dash.dependencies import Input, Output,State
from dashboard.data import london_boundaries, predictions, wards
from numpy import linspace
years= sorted(predictions['Year'].unique())
current_year,current_month=datetime.now().year, datetime.now().month


register_page(__name__,path="/", title="London Residential Burglary & Police Allocation Dashboard")

layout=html.Div([

    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': year}
                     for year in years],
            value=current_year  # Default to most recent year
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
            value=current_month  # Default to the current month

        ),
    ], style={'width': '30%', 'margin': '0 auto', 'padding': '10px'}),

    html.Div(
        html.Button(
            "Load Data",
            id="load-data-btn",
            n_clicks=0,
            style={
                "padding": "8px 20px",
                "fontSize": "16px",
                "backgroundColor": "#007BFF",
                "color": "#FFFFFF",
                "border": "none",
                "borderRadius": "4px",
                "cursor": "pointer"
            }
        ),
        style={'textAlign': 'center', 'marginBottom': '15px'}
    ),


    html.Div(id='content'),

])



def load_data(year,month):
    filtered_predictions = predictions[(predictions['Year'] == year) & (predictions['Month'] == month)]

    if filtered_predictions.empty:
        return True, "No data available for the selected year and month."

    return False,london_boundaries.merge(filtered_predictions, left_on="WD24CD", right_on="Ward_Code", how="left")



@callback(Output('content', 'children'), [Input('load-data-btn', 'n_clicks')],
    [State('year-dropdown', 'value'), State('month-dropdown', 'value')],
    prevent_initial_call=True,running=[(Output("load-data-btn", "disabled"), True, False)])
def load_map(_,selected_year, selected_month):
    """
    Loads data based on the selected year and month and returns a choropleth map and search bar.
    """
    data_empty, data = load_data(selected_year, selected_month)
    if data_empty:
            return html.Div(f"No data available for {selected_month}/{selected_year}.",style={"textAlign": "center","padding": "60px","fontSize": "20px"})
    else:
        fig= create_map(data, london_boundaries)

        search_bar = html.Div([
            html.Div([
                html.Label("Search for a ward:"),
                dcc.Dropdown(
                    id='ward-search',
                    options=[{'label': f"{row['Ward name']} ({row['Ward code']})",
                              'value': row['Ward code']}
                             for _, row in wards.iterrows()],
                    placeholder="Type to search for a ward...",
                    style={"width": "100%"}
                ),
                html.Button(
                    "View Ward Data",
                    id="view-ward-btn",
                    style={
                        "marginTop": "10px",
                        "padding": "5px 15px",
                        "backgroundColor": "#28a745",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "4px",
                        "cursor": "pointer"
                    }
                )
            ], style={"width": "50%", "marginBottom": "20px", "padding": "10px",
          "border": "1px solid #ddd", "borderRadius": "5px", "margin": "12px auto"})
        ])

        feature_dropdown = html.Div([
            html.Label("Select feature for map:"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[

                    { 'label': 'Burglary prediction', 'value': 1},
                    {'label': 'Police officers allocated', 'value': 2},
                    {'label': 'Education', 'value': 3},
                    { 'label': 'Municipality', 'value': 4},


                ],
                value=1  # Default to the current month

            ),
        ], style={'width': '30%', 'margin': '0 auto', 'padding': '10px'})

        return html.Div([
            search_bar,
            feature_dropdown,
            dcc.Graph(
                id="map",
                figure=fig,
                style={"width": "100%", "height": "700px"},
                config={
                    "doubleClick": "reset+autosize",
                    "modeBarButtonsToRemove": [
                        "zoom", "pan", "select", #"zoomIn", "zoomOut",
                        "autoScale", "resetScale", "lasso2d", "toImage", "resetView",
                    ],
                    "displaylogo": False,
                },
            )])


def create_map(data,boundaries):

    """
    Creates a choropleth map of London wards with burglary predictions and police allocation.
    """
    center = {'lat': 51.5074, 'lon': -0.1278}  # London coordinates
    zoom = 9  # Default zoom level for the map

    p98 = data['prediction'].quantile(0.98) # 95th to eliminate outliers

    fig = px.choropleth_map(data, geojson=boundaries, locations="Ward_Code",
                                featureidkey="properties.WD24CD",
                                color='prediction', hover_name="WD24NM",
                                hover_data=['Ward_Code', 'prediction', 'officers', 'risk'],
                                labels={'Ward_Code': 'Ward code','officers': 'Police Officers allocated','risk': 'Risk Factor',"prediction": 'Burglary Prediction'},
                                color_continuous_scale="Blues",
                                map_style="carto-positron", zoom=zoom,range_color=(0, p98),
                                center=center, opacity=0.7)



    scale = linspace(0, p98, 5)
    scale_labels = [f"{int(v)}" for v in scale]
    scale_labels[-1] = f"{int(p98)}+"  # add the “+” to the last label
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Predicted burglary count",
            tickvals=scale,
            ticktext=scale_labels,
            ticks="outside"
        )
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig

@callback(
    Output('redirect', 'pathname'),
    Input('view-ward-btn', 'n_clicks'),
    State('ward-search', 'value'),
    prevent_initial_call=True,
    running=[(Output("view-ward-btn", "disabled"), True, False)]
)
def redirect_to_ward_page(_, ward_code):
    """
    Redirects to the ward page when the user clicks the View Ward Data button.
    """
    if ward_code:
        return f"/ward/{ward_code.lower()}"
    else:
        return no_update


@callback(
    Output('redirect', 'pathname',allow_duplicate=True),
    Input('map', 'clickData'),
    prevent_initial_call=True,
    running=[(Output("map", "disabled"), True, False)]
)
def redirect_to_ward_from_map(ward_selected):
    """
    Redirects to the ward page when the user clicks on a ward in the map.
    """
    if ward_selected==None:
        return no_update
    ward_code = ward_selected['points'][0]['customdata'][0]
    return f"/ward/{ward_code.lower()}"

