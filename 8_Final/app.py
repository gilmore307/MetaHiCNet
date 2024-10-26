import os
import uuid
import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from stages.a_uploading import (create_upload_layout_method1,create_upload_layout_method2,create_upload_layout_method3,register_upload_callbacks)
from stages.b_processing import process_data, create_processed_data_preview, register_processing_callbacks
import logging
import io

# Initialize logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
logger.addHandler(stream_handler)

# Define stages mapping for each method
stages_mapping = {
    'method1': ['File Uploading', 'Data Processing', 'Normalization', 'Spurious Contact Removal', 'Visualization'],
    'method2': ['File Uploading', 'Normalization', 'Spurious Contact Removal', 'Visualization'],
    'method3': ['File Uploading', 'Visualization']
}

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

# Function to generate the flowchart using buttons for stages_mapping
def create_flowchart(current_stage, method='method1'):
    method_stages = stages_mapping.get(method, stages_mapping['method1'])
    buttons = []
    for idx, stage in enumerate(method_stages):
        color = 'primary' if stage == current_stage else 'success' if method_stages.index(current_stage) > idx else 'light'
        buttons.append(dbc.Button(stage, color=color, disabled=True, className="mx-2 my-2"))
        if idx < len(method_stages) - 1:
            buttons.append(html.Span("→", style={'font-size': '24px', 'margin': '0 10px'}))
    return html.Div(buttons, style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})

# Textarea to display logger messages
log_text = dcc.Textarea(
    id="log-box",
    value="Logger Initialized...\n",  # Initial log message
    style={
        'width': '100%',
        'height': '200px',
        'resize': 'none',
        'border': '1px solid #ccc',
        'padding': '10px',
        'overflow': 'auto',
        'backgroundColor': '#f9f9f9',
        'color': '#333',
        'fontFamily': 'monospace'
    },
    readOnly=True
)

# Define the layout of the app
app.layout = dbc.Container([
    html.H1("Meta Hi-C Visualization", className="my-4 text-center"),
    dcc.Store(id='current-stage-method1', data='File Uploading'),
    dcc.Store(id='current-stage-method2', data='File Uploading'),
    dcc.Store(id='current-stage-method3', data='File Uploading'),
    dcc.Store(id='user-folder', data=str(uuid.uuid4())),

    dcc.Tabs(id='tabs-method', value='method1', children=[
        dcc.Tab(label='First-time users: Upload raw data', value='method1', children=[
            html.Div(id='flowchart-container-method1'),
            html.Div(id='dynamic-content-method1'),
        ]),
        dcc.Tab(label='Change normalization method: Upload unnormalized data', value='method2', children=[
            html.Div(id='flowchart-container-method2'),
            html.Div(id='dynamic-content-method2'),
        ]),
        dcc.Tab(label='Continue previous visualization: Upload normalized data', value='method3', children=[
            html.Div(id='flowchart-container-method3'),
            html.Div(id='dynamic-content-method3'),
        ]),
    ]),
    dcc.Interval(id="log-interval", interval=2000, n_intervals=0),  # Update every 2 seconds
    log_text
], fluid=True)

# Callback to update the flowchart and display content for Method 1
@app.callback(
    [Output('flowchart-container-method1', 'children'),
     Output('dynamic-content-method1', 'children')],
    [Input('tabs-method', 'value'),
     Input('current-stage-method1', 'data')],
    [State('user-folder', 'data')]
)
def update_layout_method1(selected_method, stage_method1, user_folder):
    if selected_method != 'method1':
        return no_update, no_update

    flowchart1 = create_flowchart(stage_method1, method='method1')

    if stage_method1 == 'File Uploading':
        upload_component1 = create_upload_layout_method1()
        return flowchart1, upload_component1

    elif stage_method1 == 'Data Processing':
        contig_info_path = os.path.join('assets/output', user_folder, 'contig_information.csv')
        binning_info_path = os.path.join('assets/output', user_folder, 'binning_information.csv')
        taxonomy_path = os.path.join('assets/output', user_folder, 'bin_taxonomy.csv')

        combined_data = process_data(contig_info_path, binning_info_path, taxonomy_path, user_folder)
        preview_component = create_processed_data_preview(combined_data)
        return flowchart1, preview_component

    return flowchart1, None

# Callback to update the flowchart and display content for Method 2
@app.callback(
    [Output('flowchart-container-method2', 'children'),
     Output('dynamic-content-method2', 'children')],
    [Input('tabs-method', 'value'),
     Input('current-stage-method2', 'data')],
    [State('user-folder', 'data')]
)
def update_layout_method2(selected_method, stage_method2, user_folder):
    if selected_method != 'method2':
        return no_update, no_update

    flowchart2 = create_flowchart(stage_method2, method='method2')

    if stage_method2 == 'File Uploading':
        upload_component2 = create_upload_layout_method2()
        return flowchart2, upload_component2

    return flowchart2, None

# Callback to update the flowchart and display content for Method 3
@app.callback(
    [Output('flowchart-container-method3', 'children'),
     Output('dynamic-content-method3', 'children')],
    [Input('tabs-method', 'value'),
     Input('current-stage-method3', 'data')],
    [State('user-folder', 'data')]
)
def update_layout_method3(selected_method, stage_method3, user_folder):
    if selected_method != 'method3':
        return no_update, no_update

    flowchart3 = create_flowchart(stage_method3, method='method3')

    if stage_method3 == 'File Uploading':
        upload_component3 = create_upload_layout_method3()
        return flowchart3, upload_component3

    return flowchart3, None

# Periodic callback to update the log box content
@app.callback(
    Output("log-box", "value"),
    [Input("log-interval", "n_intervals")]
)
def update_log_box(n):
    log_stream.seek(0)  # Go to the start of the log
    log_content = log_stream.read()  # Read the current log content
    return log_content  # Display the latest logs

# Register all the callbacks from a_uploading
register_upload_callbacks(app)
register_processing_callbacks(app)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)