
from datetime import datetime
import plotly.express as px
from dash import dcc, html, no_update, register_page, callback
from dash.dependencies import Input, Output,State
from dashboard.data import london_boundaries, predictions, wards, population, cars_vans, occupancy
from numpy import linspace


years= sorted(predictions['Year'].unique())
current_year,current_month=datetime.now().year, datetime.now().month


register_page(__name__,path="/", title="London Residential Burglary & Police Allocation Dashboard")

layout=html.Div([

    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(
            id='year-dropdown-index',
            options=[{'label': str(year), 'value': year}
                     for year in years],
            value=current_year  # Default to most recent year
        ),

        html.Label("Select Month:"),
        dcc.Dropdown(
            id='month-dropdown-index',
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
    [State('year-dropdown-index', 'value'), State('month-dropdown-index', 'value')],
    prevent_initial_call=True,running=[(Output("load-data-btn", "disabled"), True, False)])
def load_map(_,selected_year, selected_month):
    """
    Loads data based on the selected year and month and returns a choropleth map and search bar.
    """
    data_empty, data = load_data(selected_year, selected_month)
    if data_empty:
            return html.Div(f"No data available for {selected_month}/{selected_year}.",style={"textAlign": "center","padding": "60px","fontSize": "20px"})
    else:
        fig= create_map(data, london_boundaries,"prediction")

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
                ],
                value=1

            ),
        ], style={'width': '30%', 'margin': '0 auto', 'padding': '10px'})

        socio_economic_factors = html.Div([
            html.Button(
                "Show Socio-Economic Factors",
                id="toggle-socio-btn",
                n_clicks=0,
                style={
                    "padding": "8px 20px",
                    "fontSize": "14px",
                    "backgroundColor": "#28a745",
                    "color": "#FFFFFF",
                    "border": "none",
                    "borderRadius": "4px",
                    "cursor": "pointer",
                    "marginBottom": "15px"
                }
            ),

            html.Div(id="socio-economic-section")
        ], style={'textAlign': 'center', 'marginTop': '20px'})
        maps_container = html.Div([
            dcc.Graph(
                id="map",
                figure=fig,
                style={"width": "100%", "height": "700px"},
                config={
                    "doubleClick": "reset+autosize",
                    "modeBarButtonsToRemove": [
                        "zoom", "pan", "select",  # "zoomIn", "zoomOut",
                        "autoScale", "resetScale", "lasso2d", "toImage", "resetView",
                    ],
                    "displaylogo": False,
                },
            )
        ], id="maps-container", style={"display": "flex", "flexDirection": "row", "gap": "10px"})

        return html.Div([
            search_bar,
            feature_dropdown,
            socio_economic_factors,
            maps_container,



        ])


def create_map(data,boundaries,factor,title="Predicted burglary count"):

    """
    Creates a choropleth map of London wards with burglary predictions and police allocation.


    """
    center = {'lat': 51.5074, 'lon': -0.1278}  # London coordinates
    zoom = 9 # Default zoom level for the map

    p98 = data[factor].quantile(0.98) # 95th to eliminate outliers

    fig = px.choropleth_map(data, geojson=boundaries, locations="Ward_Code",
                                featureidkey="properties.WD24CD",
                                color=factor, hover_name="WD24NM",
                                hover_data=['Ward_Code', 'prediction', 'officers'],
                                labels={'Ward_Code': 'Ward code','officers': 'Police Officers allocated',"prediction": 'Burglary Prediction'},
                                color_continuous_scale="Blues",
                                map_style="carto-positron", zoom=zoom,range_color=(0, p98),
                                center=center, opacity=0.7,title=title)



    scale = linspace(0, p98, 5)
    scale_labels = [f"{int(v)}" for v in scale]
    scale_labels[-1] = f"{int(p98)}+"  # add the “+” to the last label
    fig.update_layout(

        coloraxis_colorbar=dict(
            title=title,
            tickvals=scale,
            ticktext=scale_labels,
            ticks="outside",
            len=0.9

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
    if ward_selected is None:
        return no_update
    ward_code = ward_selected['points'][0]['customdata'][0]
    return f"/ward/{ward_code.lower()}"


@callback(Output('map', 'figure'),
          [Input('feature-dropdown', 'value')],
          [State('year-dropdown-index', 'value'),
           State('month-dropdown-index', 'value')])

def update_map(feature, selected_year, selected_month):
    """

    """
    data_empty, data = load_data(selected_year, selected_month)

    if data_empty:
        return no_update

    if feature == 1:  # Burglary prediction
        return create_map(data, london_boundaries,"prediction")

    elif feature == 2:  # Police officers allocated
        return create_map(data, london_boundaries, "officers",title="Police Officers Allocated")

    else:
        return no_update


@callback(
    [Output('maps-container', 'children'), Output('toggle-socio-btn', 'children'),Output('socio-economic-section', 'children')],
    [Input('toggle-socio-btn', 'n_clicks')],
    [State('year-dropdown-index', 'value'),
     State('month-dropdown-index', 'value')],
    prevent_initial_call=True,
    running=[(Output("toggle-socio-btn", "disabled"), True, False)]
)
def socio_economic_map(n_clicks, selected_year, selected_month):
    """
    Toggle the display of socio-economic factors map next to the main map.
    """
    data_empty, data = load_data(selected_year, selected_month)



    main_fig = create_map(data, london_boundaries, "prediction")


    # If button clicked odd number of times, show both maps
    if n_clicks and n_clicks % 2 == 1:
        # Create socio-economic data and map

        # Return both maps side by side
        maps_content = [
            dcc.Graph(
                id="map",
                figure=main_fig,
                style={"width": "50%", "height": "700px"},
                config={
                    "doubleClick": "reset+autosize",
                    "modeBarButtonsToRemove": [
                        "zoom", "pan", "select",
                        "autoScale", "resetScale", "lasso2d", "toImage", "resetView",
                    ],
                    "displaylogo": False,
                },
            ),
            dcc.Graph(
                id="socio-map",
                style={"width": "50%", "height": "700px"},
                config={
                    "doubleClick": "reset+autosize",
                    "modeBarButtonsToRemove": [
                        "zoom", "pan", "select",
                        "autoScale", "resetScale", "lasso2d", "toImage", "resetView",
                    ],
                    "displaylogo": False,
                },
            )
        ]
        button_text = "Hide Socio-Economic Factors"
        socio_economic_options = html.Div([
            html.Label("Select socio-economic factor:"),
            dcc.Dropdown(
                id='socio-economic-dropdown',
                options=[
                    {'label': "Population", 'value': 1},
                    {'label': "Cars or vans", 'value': 2},
                    {'label': "Occupancy", 'value': 3}
                ],
                value=1,
                style={'width': "200px",'margin': '0 auto', 'padding': '10px'}
            )
        ], style={"textAlign": "center", "marginBottom": "10px"})
    else:
        # Return only the main map
        maps_content = [
            dcc.Graph(
                id="map",
                figure=main_fig,
                style={"width": "100%", "height": "700px"},
                config={
                    "doubleClick": "reset+autosize",
                    "modeBarButtonsToRemove": [
                        "zoom", "pan", "select",
                        "autoScale", "resetScale", "lasso2d", "toImage", "resetView",
                    ],
                    "displaylogo": False,
                },
            )
        ]
        button_text = "Show Socio-Economic Factors"
        socio_economic_options=None

    return maps_content, button_text,socio_economic_options


@callback(
    Output('socio-map', 'figure'),
    Input('socio-economic-dropdown', 'value'),
    [State('year-dropdown-index', 'value'),
     State('month-dropdown-index', 'value')],
    prevent_initial_call=True
)
def update_socio_economic_map(socio_factor, selected_year, selected_month):
    """
    Updates the socio-economic map when a different factor is selected.
    """
    data_empty, data = load_data(selected_year, selected_month)

    if data_empty:
        return no_update

    if socio_factor == 1:
        filtered_population=population[population["Year"]==2025]
        socio_data = data.merge(filtered_population, left_on="Ward_Code", right_on="Ward code", how="left")
        return create_map(socio_data, london_boundaries, "population_density", title="Population")

    elif socio_factor == 2:
        socio_data = data.merge(cars_vans, left_on="Ward_Code", right_on="Ward code", how="left")
        return create_map(socio_data, london_boundaries, "none_pct", title="% households with no cars or vans")

    elif socio_factor == 3:
        socio_data = data.merge(occupancy, left_on="Ward_Code", right_on="Ward code", how="left")
        return create_map(socio_data, london_boundaries, "0_pct", title="%")

    else:
        return no_update
