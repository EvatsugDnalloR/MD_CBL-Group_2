
from datetime import datetime

import plotly.express as px
from dash import callback, dcc, html, no_update, register_page
from dash.dependencies import Input, Output, State
from numpy import linspace

from dashboard.data import (
    cars_vans,
    london_boundaries,
    occupancy,
    predictions,
    wards,
)

years= sorted(predictions["Year"].unique()) #Get years from the predictions data for the dropdown
current_year,current_month=datetime.now().year, datetime.now().month


register_page(__name__,path="/", title="London Residential Burglary & Police Allocation Dashboard")

#Main layout of the home page
layout=html.Div([

    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(
            id="year-dropdown-index",
            options=[{"label": str(year), "value": year}
                     for year in years],
            value=current_year  # Default to most recent year
        ),

        html.Label("Select Month:"),
        dcc.Dropdown(
            id="month-dropdown-index",
            options=[
                {"label": "January", "value": 1},
                {"label": "February", "value": 2},
                {"label": "March", "value": 3},
                {"label": "April", "value": 4},
                {"label": "May", "value": 5},
                {"label": "June", "value": 6},
                {"label": "July", "value": 7},
                {"label": "August", "value": 8},
                {"label": "September", "value": 9},
                {"label": "October", "value": 10},
                {"label": "November", "value": 11},
                {"label": "December", "value": 12}

            ],
            value=current_month  # Default to the current month

        ),
    ], style={"width": "30%", "margin": "0 auto", "padding": "10px"}),

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
        style={"textAlign": "center", "marginBottom": "15px"}
    ),


    html.Div(id="content")  # Container for storing the map and search bar when year and month are selected

])


def load_data(year,month):

    """
    Loads the prediction data for the selected year and month and merges it with the London boundaries based on the ward code.
    If no data is available for the selected year and month, it returns a message indicating that.
    """
    filtered_predictions = predictions[(predictions["Year"] == year) & (predictions["Month"] == month)]

    if filtered_predictions.empty:
        return True, "No data available for the selected year and month."

    return False,london_boundaries.merge(filtered_predictions, left_on="WD24CD", right_on="Ward_Code", how="left")


@callback(Output("content", "children"), [Input("load-data-btn", "n_clicks")],
    [State("year-dropdown-index", "value"), State("month-dropdown-index", "value")],
    prevent_initial_call=True,running=[(Output("load-data-btn", "disabled"), True, False)])
def load_content(_,selected_year, selected_month):
    """
    Loads a  burglary prediction map for the selected year and month,a search bar for wards, a dropdown to select the feature to display on the map and a button to show the socio-economic factors map.
    """
    data_empty, data = load_data(selected_year, selected_month)
    if data_empty:
            return html.Div(f"No data available for {selected_month}/{selected_year}.",style={"textAlign": "center","padding": "60px","fontSize": "20px"})
    else:
        fig= create_map(data, london_boundaries,"prediction") # Create the  map with burglary predictions

        # Search bar allows the user to search for a ward by name or code
        search_bar = html.Div([
            html.Div([
                html.Label("Search for a ward:"),
                dcc.Dropdown(
                    id="ward-search",
                    options=[{"label": f"{row['Ward name']} ({row['Ward code']})",
                              "value": row["Ward code"]}
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

        # Dropdown to select the feature to display on the map (burglary prediction or police officers allocated)
        feature_dropdown = html.Div([
            html.Label("Select feature for map:"),
            dcc.Dropdown(
                id="feature-dropdown",
                options=[

                    { "label": "Burglary prediction", "value": 1},
                    {"label": "Police officers allocated", "value": 2},
                ],
                value=1

            ),
        ], style={"width": "30%", "margin": "0 auto", "padding": "10px"})
        # Button to show or hide the socio-economic factors map
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
            html.Div(id="socio-economic-options") #Container for  dropdown with socio-economic factors
        ], style={"textAlign": "center", "marginTop": "20px"})

        # Container for the burglary prediction map and the socio-economic factors map
        maps_container = html.Div([
            html.Div([
            html.H3("Predicted Residential Burglary Count",id="burglary_map_title", style={"textAlign": "center"}),
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
            )], id="burglary-map-section", style={"width": "100%"}),

            html.Div([
                html.H3(id="socio_economic_title", style={"textAlign": "center"}),
                dcc.Graph(
                    id="socio-map",
                    style={"width": "100%", "height": "700px","display": "none"},
                    config={
                        "doubleClick": "reset+autosize",
                        "modeBarButtonsToRemove": [
                            "zoom", "pan", "select",
                            "autoScale", "resetScale", "lasso2d", "toImage", "resetView",
                        ],
                        "displaylogo": False,
                    },
                )],id="socio-economic-section", style={"display":"none"})

        ], id="maps-container", style={"display": "flex", "flexDirection": "row", "gap": "10px"})

        return html.Div([
            search_bar,
            feature_dropdown,
            socio_economic_factors,
            maps_container,
        ])


def create_map(data,boundaries,factor,title=None):

    """
    Creates a choropleth map of London wards with the specified factor for the given data.
    """
    center = {"lat": 51.5074, "lon": -0.1278}  # London coordinates
    zoom = 9 # Default zoom level for the map

    p98 = data[factor].quantile(0.98) # 98th percentile to eliminate outliers when scaling the color scale
    #
    fig = px.choropleth_map(data, geojson=boundaries, locations="Ward_Code",
                                featureidkey="properties.WD24CD",
                                color=factor, hover_name="WD24NM",
                                hover_data=["Ward_Code", "prediction", "officers"],
                                labels={"Ward_Code": "Ward code","officers": "Police Officers allocated","prediction": "Burglary Prediction"},
                                color_continuous_scale="Blues",
                                map_style="carto-positron", zoom=zoom,range_color=(0, p98),
                                center=center, opacity=0.7,)

    scale = linspace(0, p98, 5) # Create a scale for the color bar using the 98th percentile
    scale_labels = [f"{int(v)}" for v in scale]
    scale_labels[-1] = f"{int(p98)}+"  # add the “+” to the last label
    fig.update_layout(

        coloraxis_colorbar=dict(
            tickvals=scale,
            ticktext=scale_labels,
            ticks="outside",
            len=0.9,
            title=title

        )
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig

@callback(
    Output("redirect", "pathname"),
    Input("view-ward-btn", "n_clicks"),
    State("ward-search", "value"),
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
    Output("redirect", "pathname",allow_duplicate=True),
    Input("map", "clickData"),
    prevent_initial_call=True,
    running=[(Output("map", "disabled"), True, False)]
)
def redirect_to_ward_from_map(ward_selected):
    """
    Redirects to the ward page when the user clicks on a ward in the map.
    """
    if ward_selected is None:
        return no_update
    ward_code = ward_selected["points"][0]["customdata"][0]
    return f"/ward/{ward_code.lower()}"

@callback(
    Output("redirect", "pathname",allow_duplicate=True),
    Input("socio-map", "clickData"),
    prevent_initial_call=True,
    running=[(Output("socio-map", "disabled"), True, False)]
)
def redirect_to_ward_from_map(ward_selected):
    """
    Redirects to the ward page when the user clicks on a ward in the socio-economic factors map.
    """
    if ward_selected is None:
        return no_update
    ward_code = ward_selected["points"][0]["customdata"][0]
    return f"/ward/{ward_code.lower()}"


@callback(Output("map", "figure"),
          Output("burglary_map_title", "children"),
          [Input("feature-dropdown", "value")],
          [State("year-dropdown-index", "value"),
           State("month-dropdown-index", "value")]
          , prevent_initial_call=True)

def update_map(feature, selected_year, selected_month):
    """
    Updates the map based on the selected feature (burglary prediction or police officers allocated)
    """
    data_empty, data = load_data(selected_year, selected_month)

    if data_empty:
        return no_update

    if feature == 1:  # Burglary prediction
        return create_map(data, london_boundaries,"prediction"), "Predicted Residential Burglary Count"

    elif feature == 2:  # Police officers allocated
        return create_map(data, london_boundaries, "officers"), "Police Officers Allocated"

    else:
        return no_update,no_update

@callback(
    [Output("burglary-map-section", "style"), Output("toggle-socio-btn", "children"),
     Output("socio-economic-section", "style"), Output("socio-economic-options", "children")],
    [Input("toggle-socio-btn", "n_clicks")],
    prevent_initial_call=True,
    running=[(Output("toggle-socio-btn", "disabled"), True, False)]
)
def socio_economic_map(n_clicks):
    """
    Shows or hides the socio-economic factors map when the button is clicked.
    """
    # If button clicked an odd number of times, it shows both maps
    if n_clicks and n_clicks % 2 == 1:

        button_text = "Hide Socio-Economic Factors"
        socio_economic_options = html.Div([
            html.Label("Select socio-economic factor:"),
            dcc.Dropdown(
                id="socio-economic-dropdown",
                options=[
                    {"label": "Cars or vans", "value": 1},
                    {"label": "Occupancy", "value": 2}
                ],
                value=1,  # Default to Cars or vans
                style={"width": "40%","margin": "0 auto" }
            )
        ], style={"textAlign": "center", "marginBottom": "10px"})

        socio_economic_section_style = {"display": "block","width":"50%"} # Show the socio-economic section
        burglary_map_section_style = {"width": "50%"}
    else:
        button_text = "Show Socio-Economic Factors"
        socio_economic_section_style= {"display": "none"} # Hide the socio-economic section
        burglary_map_section_style = {"width": "100%"}
        socio_economic_options = None

    return burglary_map_section_style, button_text,socio_economic_section_style,socio_economic_options


@callback(
    Output("socio-map", "figure"),
    Output("socio_economic_title", "children"),
    Output("socio-map", "style"),
    Input("socio-economic-dropdown", "value"),
    State("year-dropdown-index", "value"),State("month-dropdown-index", "value"),
    prevent_initial_call=True
)
def update_socio_economic_map(socio_factor, selected_year, selected_month):
    """
    Merges the socio-economic data and returns a map based on the selected socio-economic factor.
    """
    data_empty, data = load_data(selected_year, selected_month) # Load burglary predictions and police officers allocated with london boundaries

    if data_empty:
        return no_update
    style_map={"width": "100%", "height": "700px"}

    if socio_factor == 1: # Number of Cars or Vans
        socio_data = data.merge(cars_vans, left_on="Ward_Code", right_on="Ward code", how="left")
        return create_map(socio_data, london_boundaries, "%None", title="%")," % Households with no cars or vans 2021",style_map

    elif socio_factor == 2: # Occupancy
        socio_data = data.merge(occupancy, left_on="Ward_Code", right_on="Ward code", how="left")
        return create_map(socio_data, london_boundaries, "0_pct", title="%")," % Households with exactly the required number of bedrooms 2021",style_map

    else:
        return no_update,no_update,no_update
