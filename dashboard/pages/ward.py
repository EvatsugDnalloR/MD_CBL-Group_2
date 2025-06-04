
from dash import  dcc, html,register_page,callback,Output,Input,State
from dashboard.data import burglary, predictions, wards
import plotly.graph_objs as go

register_page(__name__, path_template="/ward/<ward_code>")


years= sorted(burglary['Year'].unique())




def layout(ward_code=None, **kwargs):

    if ward_code.upper() in wards["Ward code"].values:

        ward_code = ward_code.upper()

        ward_name= wards[wards["Ward code"] == ward_code]["Ward name"].values[0]
        return html.Div([
            dcc.Store('ward_code', data=ward_code),
            html.H2(f"Ward {ward_name} ({ward_code}) Data", style={"textAlign": "center"}),

            html.Div([
                html.Label("Select Year:"),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': str(year), 'value': year}
                             for year in years],
                    value=years[-1]

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
                        {'label': 'December', 'value': 13}

                    ]


                ),
            ], style={'width': '30%', 'margin': '0 auto', 'padding': '10px'}),

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
                style={'textAlign': 'center', 'marginBottom': '15px'}
            ),

            html.Div(id='ward-content'),



            html.Div([

                html.Div([
                    dcc.Graph(
                        id="burglary-prediction-graph"
                    ),
                html.Div([

                    html.Div(
                        "Select Year Range",
                        style={"textAlign": "center", "marginBottom": "10px"}
                    ),
                    dcc.RangeSlider(
                        id="year-range-slider",
                        min=2013,
                        max=2025,
                        step=1,
                        marks={year: str(year) for year in range(2013, 2026)},
                        value=[2013, 2025]
                    )], style={"margin": "20px 0"}),

                    html.Div(
                        "Select Month Range",
                        style={"textAlign": "center", "marginBottom": "10px"}
                    ),

                    dcc.RangeSlider(
                        id="month-range-slider",
                        min=1,
                        max=12,
                        step=1,
                        marks={month: str(month) for month in range(1, 13)},
                        value=[1, 12]
                    ),


                ], style={"flex": "1", "padding": "10px"}),


                html.Div([
                    dcc.Graph(
                        id="police-allocation-graph",figure=police_allocation_graph(ward_code)
                    )


                ], style={"flex": "1", "padding": "10px"}),


            ], style={"display": "flex", "flexDirection": "row"})
        ])
    else:

        return html.Div(
            html.H2("Ward Not Found", style={"textAlign": "center"}))


@callback(Output('ward-content', 'children'),Input('view-data-btn', 'n_clicks')
          , [State('year-dropdown', 'value'), State('month-dropdown', 'value'), State('ward_code', 'data')],
    prevent_initial_call=True, running=[(Output("view-data-btn", "disabled"), True, False)])

def load_ward_data(_,selected_year, selected_month, ward_code):
    predictions_filtered = predictions[(predictions["Ward_Code"] == ward_code) &(predictions["Year"] == selected_year) & (predictions["Month"] == selected_month)]
    filtered_burglary = burglary[(burglary["Ward code"] == ward_code) & (burglary["Year"] == selected_year) & (burglary["Month"] == selected_month)]


    data=[]

    if predictions_filtered.empty:
        data.append(html.H3("No predictions available for the selected year and month.", style={"textAlign": "center"}))
    else:

        pred_row=predictions_filtered.iloc[0]
        data.append(
            html.Div([
                html.P(f"Police Allocation: {pred_row['officers']}", style={"textAlign": "center"}),
                html.P(f"Risk Factor: {pred_row['risk']}", style={"textAlign": "center"}),
                html.P(f"Prediction: {pred_row['prediction']}", style={"textAlign": "center"}),
            ])
        )
    if filtered_burglary.empty:
        data.append(html.H3("No burglary data available for the selected year and month.", style={"textAlign": "center"}))
    else:
        data.append(
            html.Div([
                html.P(f"Burglary Count: {filtered_burglary.iloc[0]['burglary_count']}", style={"textAlign": "center"}),
            ])
        )

    return html.Div(data)

@callback( Output("burglary-prediction-graph", "figure"),[Input('year-range-slider', 'value'), Input('month-range-slider', 'value')]
           , State('ward_code', 'data'))
def burglary_prediction_graph( year_range, month_range, ward_code):

    filtered_predictions = predictions[
        (predictions["Ward_Code"] == ward_code) &
        (predictions["Year"].between(year_range[0], year_range[1])) &
        (predictions["Month"].between(month_range[0], month_range[1]))
    ]

    filtered_predictions.loc[:, "Year_Month"] = (
        filtered_predictions["Year"].astype(str) + "-" +
        filtered_predictions["Month"].astype(str).str.zfill(2)
    )

    filtered_predictions = filtered_predictions.sort_values("Year_Month")

    filtered_burglary= burglary[
        (burglary["Ward code"] == ward_code) &
        (burglary["Year"].between(year_range[0], year_range[1])) &
        (burglary["Month"].between(month_range[0], month_range[1]))
    ]

    filtered_burglary.loc[:, "Year_Month"] = (
        filtered_burglary["Year"].astype(str) + "-" +
        filtered_burglary["Month"].astype(str).str.zfill(2)
    )

    filtered_burglary = filtered_burglary.sort_values("Year_Month")

    fig=go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_burglary["Year_Month"],
            y=filtered_burglary["burglary_count"],
            mode='lines+markers',
            name='Burglary Count',
            line=dict(color='purple',shape='spline')
        )

    )

    fig.add_trace(
        go.Scatter(
            x=filtered_predictions["Year_Month"],
            y=filtered_predictions["prediction"],
            mode='lines+markers',
            name='Burglary Prediction',
            line=dict(color='blue',shape='spline')
        )
    )



    return fig

def police_allocation_graph(ward_code):


    filtered_predictions = predictions[predictions["Ward_Code"] == ward_code]
    filtered_predictions["Year_Month"] = (
            predictions["Year"].astype(str) + "-" +
            predictions["Month"].astype(str).str.zfill(2)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=filtered_predictions["Year_Month"],
            y=filtered_predictions["officers"],
            name='Police Allocation',
            marker_color='blue',
            text=filtered_predictions["officers"],
            textposition='auto'
        )
    )

    fig.update_layout(
        title="Police Allocation",
        xaxis_title="Year-Month",
        yaxis_title="Number of Officers",
        barmode='group'
    )

    return fig
