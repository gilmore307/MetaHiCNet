import os
import uuid
import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from stages.a_uploading import create_upload_layout_method1, create_upload_layout_method2, create_upload_layout_method3, register_upload_callbacks
from stages.b_processing import process_data, create_processed_data_preview

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
        if stage == current_stage:
            color = 'primary'
        elif method_stages.index(current_stage) > idx:
            color = 'success'
        else:
            color = 'light'

        buttons.append(
            dbc.Button(stage, color=color, disabled=True, className="mx-2 my-2")
        )

        if idx < len(method_stages) - 1:
            buttons.append(html.Span("→", style={'font-size': '24px', 'margin': '0 10px'}))

    return html.Div(buttons, style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})

# Define the layout of the app
app.layout = dbc.Container([
    html.H1("Meta Hi-C Visualization", className="my-4 text-center"),

    # Store components to hold the current stage for each method
    dcc.Store(id='current-stage-method1', data='File Uploading'),
    dcc.Store(id='current-stage-method2', data='File Uploading'),
    dcc.Store(id='current-stage-method3', data='File Uploading'),
    dcc.Store(id='user-folder', data=str(uuid.uuid4())),

    dcc.Tabs(id='tabs-method', value='method1', children=[
        dcc.Tab(label='First-time users: Upload raw data', value='method1', children=[
            html.Div(id='flowchart-container-method1'),
            html.Div(id='dynamic-content-method1'),  # Placeholder div for Method 1 content
        ]),
        dcc.Tab(label='Change normalization method: Upload unnormalized data', value='method2', children=[
            html.Div(id='flowchart-container-method2'),
            html.Div(id='dynamic-content-method2'),  # Placeholder div for Method 2 content
        ]),
        dcc.Tab(label='Continue previous visualization: Upload normalized data', value='method3', children=[
            html.Div(id='flowchart-container-method3'),
            html.Div(id='dynamic-content-method3'),  # Placeholder div for Method 3 content
        ]),
    ]),
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

# Register all the callbacks from a_uploading
register_upload_callbacks(app)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
