
import plotly.graph_objs as go
from dash import Input, Output, State, callback, dcc, html, register_page

from dashboard.data import burglary, predictions, wards

register_page(__name__, path_template="/ward/<ward_code>")


years = sorted(burglary["Year"].unique())

graph_config = {
    "doubleClick": "reset+autosize",
    "modeBarButtonsToRemove": [
        "zoom", "pan", "select",   "zoomIn", "zoomOut",
        "autoScale", "resetScale", "lasso2d", "toImage", "resetView",
    ],
    "displaylogo": False,
}


def layout(ward_code=None):

    """
    Creates layout for the ward page
    """

    if ward_code.upper() in wards["Ward code"].values:

        ward_code = ward_code.upper()

        ward_name= wards[wards["Ward code"] == ward_code]["Ward name"].values[0]
        return html.Div([
            dcc.Store("ward_code", data=ward_code),
            html.Div(
                dcc.Link(
                    "Go back to main page",
                    href="/",
                    style={"display": "inline-block","padding": "10px 20px","marginTop": "10px","marginBottom": "20px","backgroundColor": "#646665","color": "white","borderRadius": "5px","fontWeight": "bold","textDecoration": "none",}
                ),
                style={"textAlign": "center", "width": "100%"}
            ),
            html.H2(f"Ward {ward_name} ({ward_code}) Data", style={"textAlign": "center"}),

            html.Div([
                html.Label("Select Year:"),
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[{"label": str(year), "value": year}
                             for year in years],
                    value=years[-1]

                ),

                html.Label("Select Month:"),
                dcc.Dropdown(
                    id="month-dropdown",
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

                    ]

                ),
            ], style={"width": "30%", "margin": "0 auto", "padding": "10px"}),

            html.Div(
                html.Button(
                    "View Data",
                    id="view-data-btn",
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

            html.Div(id="ward-content"),

            html.Div([

                html.Div([
                    dcc.Graph(
                        id="burglary-prediction-graph",
                        config=graph_config,
                        figure=burglary_prediction_graph(ward_code)
                    )
                ], style={"flex": "1", "padding": "10px"}),

                html.Div([
                    dcc.Graph(
                        id="police-allocation-graph", figure=police_allocation_graph(ward_code),
                        config=graph_config
                    )

                ], style={"flex": "1", "padding": "10px"}),

            ], style={"display": "flex", "flexDirection": "row"})
        ])
    else:

        return html.Div(
            html.H2("Ward Not Found", style={"textAlign": "center"}))


@callback(Output("ward-content", "children"),Input("view-data-btn", "n_clicks"),
          [State("year-dropdown", "value"), State("month-dropdown", "value"), State("ward_code", "data")],
          prevent_initial_call=True, running=[(Output("view-data-btn", "disabled"), True, False)])
def load_ward_data(_,selected_year, selected_month, ward_code):
    predictions_filtered = predictions[(predictions["Ward_Code"] == ward_code) & (predictions["Year"] == selected_year)
                                       & (predictions["Month"] == selected_month)]
    filtered_burglary = burglary[(burglary["Ward code"] == ward_code) & (burglary["Year"] == selected_year)
                                 & (burglary["Month"] == selected_month)]

    data = []

    if not predictions_filtered.empty:
        pred_row=predictions_filtered.iloc[0]
        data.append(
            html.Div([
                html.P(f"Police Allocation: {pred_row['officers']}", style={"textAlign": "center"}),
                html.P(f"Prediction: {pred_row['prediction']}", style={"textAlign": "center"})
            ])
        )
    if not filtered_burglary.empty:
        data.append(
            html.Div([
                html.P(f"Burglary Count: {filtered_burglary.iloc[0]['burglary_count']}", style={"textAlign": "center"}),
            ])
        )

    return html.Div(data)


def burglary_prediction_graph(ward_code):

    filtered_predictions = predictions[
        (predictions["Ward_Code"] == ward_code)
    ].sort_values("Date")

    filtered_burglary= burglary[
        (burglary["Ward code"] == ward_code)
    ].sort_values("Date")

    fig = go.Figure()

    # Historical burglary
    fig.add_trace(go.Scatter(
        x=filtered_burglary["Date"],
        y=filtered_burglary["burglary_count"],
        mode="lines",
        name="Historical burglary",
        legendgroup="Historical",
        line=dict(color="purple")
    ))

    # Predicted burglary
    fig.add_trace(go.Scatter(
        x=filtered_predictions["Date"],
        y=filtered_predictions["prediction"],
        mode="lines",
        name="Predicted burglary",
        legendgroup="Predicted",
        line=dict(color="blue")
    ))

    fig.update_xaxes(
        type="date",
        tickformat="%Y-%m",
        rangeslider_visible=True,
        showgrid=False,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(count=4, label="4y", step="year", stepmode="backward"),
                dict(step="all")
            ]
        ),

    )

    fig.update_layout(
        xaxis_title="Month-Year",
        yaxis_title="Burglary Count",)
    return fig


def police_allocation_graph(ward_code):
    filtered_predictions = predictions[predictions["Ward_Code"] == ward_code]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=filtered_predictions["Date"],
            y=filtered_predictions["officers"],
            name="Police Allocation",
            marker_color="lightblue",
            text=filtered_predictions["officers"],
            textposition="auto"
        )

    )

    fig.update_layout(
        title={
            "text": "Police Allocation",
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis_title="Month-Year",
        yaxis_title="Number of Officers",
        barmode="group",

    )

    return fig