from dash import Dash, dcc, html, page_container


app = Dash(__name__, title="London Residential Burglary & Police Allocation Dashboard",suppress_callback_exceptions=True, use_pages=True )




app.layout = html.Div([
    dcc.Location(id='redirect',refresh="callback-nav"),
    html.H1(
        "London Residential Burglary & Police Allocation Dashboard",
        style={"textAlign": "center"}
    ),page_container

])



if __name__ == '__main__':
    app.run(debug=True)
