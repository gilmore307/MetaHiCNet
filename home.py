import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Part 1: Define the layout of the homepage
app.layout = dbc.Container([
    # Header Section
    html.Div([
        html.H1("MetaHiCNet", className="display-3 text-center"),
        html.P("A platform for normalizing and visualizing microbial Hi-C interaction networks.", className="lead text-center text-muted")
    ], style={'marginTop': '20px', 'padding': '20px'}),

    # Features Section
    html.Div([
        html.H3("Key Features", className="text-center my-4"),
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Upload Data", className="text-center"),
                html.P("Easily upload raw Hi-C data for analysis.", className="text-center")
            ]), width=4),
            dbc.Col(html.Div([
                html.H5("Normalize Data", className="text-center"),
                html.P("Apply state-of-the-art normalization methods.", className="text-center")
            ]), width=4),
            dbc.Col(html.Div([
                html.H5("Visualize Results", className="text-center"),
                html.P("View interactive visualizations of your data.", className="text-center")
            ]), width=4)
        ])
    ], className="my-5"),

    # Call to Action
    html.Div([
        html.H4("Get Started Now", className="text-center my-4"),
        dbc.Button("Proceed to App", href="/app", color="success", size="lg", className="d-block mx-auto")
    ], className="my-5"),

    # Footer Section
    html.Div([
        html.P("All resources and tools on this website are freely accessible to the public. For questions, please contact us at yuxuan.du@utsa.edu.", 
               style={'textAlign': 'center','padding': '20px','backgroundColor': 'lightgray', 'marginTop': '20px'})
    ])

], fluid=True)

# Part 2: Run the app
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
