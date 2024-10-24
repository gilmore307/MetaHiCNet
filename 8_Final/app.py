import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from stages.a_file_upload import create_upload_component, register_callbacks

# Define stages mapping for each method
stages_mapping = {
    'method1': ['File Uploading', 'Data Processing', 'Normalization', 'Spurious Contact Removal', 'Visualization'],
    'method2': ['File Uploading', 'Normalization', 'Spurious Contact Removal', 'Visualization'],
    'method3': ['File Uploading', 'Visualization']
}

# Initialize the Dash app with Bootstrap theme
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)


# Function to generate the flowchart using buttons for stages_mapping
def create_flowchart(stage_highlight, method='method1'):
    method_stages = stages_mapping.get(method, stages_mapping['method1'])
    buttons = []
    for idx, stage in enumerate(method_stages):
        color = 'primary' if stage == stage_highlight else 'light'
        buttons.append(
            dbc.Button(stage, color=color, disabled=True, className="mx-2 my-2")
        )
        if idx < len(method_stages) - 1:
            # Add an arrow between stages
            buttons.append(html.Span("→", style={'font-size': '24px', 'margin': '0 10px'}))
    return html.Div(buttons, style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})

# Define the layout of the app
app.layout = dbc.Container([
    html.H1("Meta Hi-C Visualization", className="my-4 text-center"),

    # Store to hold the current stage for each method
    dcc.Store(id='current-stage-method1', data='File Uploading'),
    dcc.Store(id='current-stage-method2', data='File Uploading'),
    dcc.Store(id='current-stage-method3', data='File Uploading'),

    dcc.Tabs(id='tabs-method', value='method1', children=[
        # Method 1: Raw Data Uploads
        dcc.Tab(label='First-time users: Upload raw data', value='method1', children=[
            html.Div(id='flowchart-container-method1'),
            html.Div(id='upload-component-container-method1'),
            dbc.Button("Validate All Files", id="validate-button", color="success", className="mt-3"),
            html.Div(id="validation-output", style={'padding': '0px', 'color': 'green'}),
        ]),

        # Method 2: Unnormalized Data Uploads
        dcc.Tab(label='Change normalization method: Upload unnormalized data', value='method2', children=[
            html.Div(id='flowchart-container-method2'),
            html.Div(id='upload-component-container-method2'),
            dbc.Button("Validate All Files", id="validate-button-unnormalized", color="success", className="mt-3"),
            html.Div(id="validation-output-unnormalized", style={'padding': '0px', 'color': 'green'})
        ]),

        # Method 3: Normalized Data Uploads
        dcc.Tab(label='Continue previous visualization: Upload normalized data', value='method3', children=[
            html.Div(id='flowchart-container-method3'),
            html.Div(id='upload-component-container-method3'),
            dbc.Button("Validate All Files", id="validate-button-normalized", color="success", className="mt-3"),
            html.Div(id="validation-output-normalized", style={'padding': '0px', 'color': 'green'})
        ]),
    ]),
], fluid=True)

# Callback to update the flowchart and conditionally show/hide the upload components based on the stage
@app.callback(
    [Output('flowchart-container-method1', 'children'),
     Output('upload-component-container-method1', 'children'),
     Output('flowchart-container-method2', 'children'),
     Output('upload-component-container-method2', 'children'),
     Output('flowchart-container-method3', 'children'),
     Output('upload-component-container-method3', 'children')],
    [Input('tabs-method', 'value'),
     Input('current-stage-method1', 'data'),
     Input('current-stage-method2', 'data'),
     Input('current-stage-method3', 'data')]
)
def update_layout(selected_method, stage_method1, stage_method2, stage_method3):
    upload_component1 = None
    upload_component2 = None
    upload_component3 = None
    
    # Handle Method 1
    if selected_method == 'method1':
        flowchart1 = create_flowchart(stage_method1, method='method1')
        if stage_method1 == 'File Uploading':
            upload_component1 = html.Div([
                dbc.Row([
                    dbc.Col(create_upload_component(
                        'raw-contig-info',
                        'Upload Contig Information File (.csv)', 
                        '/examples/contig_information.csv',
                        "This file must include the following columns: 'Contig', 'Restriction sites', 'Length', 'Coverage', and 'Self-contact'."
                    )),
                    dbc.Col(create_upload_component(
                        'raw-contig-matrix',
                        'Upload Raw Contact Matrix File (.npz)', 
                        '/examples/raw_contact_matrix.npz',
                        "The Unnormalized Contact Matrix must include the following keys: 'indices', 'indptr', 'format', 'shape', 'data'."
                    )),
                ]),
                dbc.Row([
                    dbc.Col(create_upload_component(
                        'raw-binning-info',
                        'Upload Binning Information File (.csv)', 
                        '/examples/binning_information.csv',
                        "This file must include the following columns: 'Contig', 'Bin', and 'Type'."
                    )),
                    dbc.Col(create_upload_component(
                        'raw-bin-taxonomy',
                        'Upload Bin Taxonomy File (.csv)', 
                        '/examples/taxonomy.csv',
                        "This file must include the following columns: 'Bin', 'Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Plasmid ID'."
                    )),
                ]),
            ])
        return flowchart1, upload_component1, None, None, None, None

    # Handle Method 2
    if selected_method == 'method2':
        flowchart2 = create_flowchart(stage_method2, method='method2')
        if stage_method2 == 'File Uploading':
            upload_component2 = html.Div([
                dbc.Row([
                    dbc.Col(create_upload_component(
                        'unnormalized-data-folder',
                        'Upload Unnormalized Data Folder (.7z)', 
                        '/examples/unnormalized_information.7z',
                        "Please upload the 'unnormalized_information' folder generated from your previous visualization.  \n"
                        "It must include the following files: 'bin_info_final.csv', 'contig_info_final.csv', 'raw_contact_matrix.npz'."
                    )),
                ]),
            ])
        return None, None, flowchart2, upload_component2, None, None

    # Handle Method 3
    if selected_method == 'method3':
        flowchart3 = create_flowchart(stage_method3, method='method3')
        if stage_method3 == 'File Uploading':
            upload_component3 = html.Div([
                dbc.Row([
                    dbc.Col(create_upload_component(
                        'normalized-data-folder',
                        'Upload Visualization Data Folder (.7z)', 
                        '/examples/normalized_information.7z',
                        "Please upload the 'normalized_information' folder generated from your previous visualization.  \n"
                        "It must include the following files: 'bin_info_final.csv', 'contig_info_final.csv', 'contig_contact_matrix.npz', 'bin_contact_matrix.npz'."
                    )),
                ]),
            ])
        return None, None, None, None, flowchart3, upload_component3

# Register all the callbacks from file_upload.py
register_callbacks(app)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)