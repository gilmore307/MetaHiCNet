import uuid
import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from stages.a_preparation import (
    create_upload_layout_method1, 
    create_upload_layout_method2, 
    create_upload_layout_method3,
    register_preparation_callbacks)
from stages.b_normalization import (
    create_normalization_layout, 
    register_normalization_callbacks)
import logging
import io

# Part 1: Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
logger.addHandler(stream_handler)

stages_mapping = {
    'method1': ['File Preparation', 'Normalization', 'Spurious Contact Removal', 'Visualization'],
    'method2': ['File Preparation', 'Normalization', 'Spurious Contact Removal', 'Visualization'],
    'method3': ['File Preparation', 'Visualization']
}

# Part 2: Function to generate the flowchart using buttons for stages_mapping
def create_flowchart(current_stage, method='method1'):
    method_stages = stages_mapping.get(method, stages_mapping['method1'])
    buttons = []
    for idx, stage in enumerate(method_stages):
        color = 'primary' if stage == current_stage else 'success' if method_stages.index(current_stage) > idx else 'light'
        buttons.append(dbc.Button(stage, color=color, disabled=True, className="mx-2 my-2"))
        if idx < len(method_stages) - 1:
            buttons.append(html.Span("→", style={'font-size': '24px', 'margin': '0 10px'}))
    return html.Div(buttons, style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})

# Part 3: Define the layout of the app
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

app.layout = dbc.Container([
    html.H1("Meta Hi-C Visualization", className="my-4 text-center"),
    dcc.Store(id='current-stage-method1', data='File Preparation'),
    dcc.Store(id='current-stage-method2', data='File Preparation'),
    dcc.Store(id='current-stage-method3', data='File Preparation'),
    dcc.Store(id='preparation-status-method1', data=False),
    dcc.Store(id='preparation-status-method2', data=False),
    dcc.Store(id='preparation-status-method3', data=False),
    dcc.Store(id='normalization-status-method-1', data=False),
    dcc.Store(id='normalization-status-method-2', data=False),
    dcc.Store(id='user-folder', data=str(uuid.uuid4())),

    dcc.Tabs(id='tabs-method', value='method1', children=[
        dcc.Tab(label='First-time users: Upload raw data', value='method1', children=[
            html.Div(id='flowchart-container-method1'),
            html.Div(id='dynamic-content-method1'),
            dbc.Row([
                dbc.Col(dbc.Button(id="execute-button-method1", children="Prepare Data", color="success", style={'width': '100%'}), width=2),
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dbc.Button("Previous", id="previous-button-method1", color="secondary", style={'width': '100%'}, disabled=True), width=2),
                        dbc.Col(dbc.Button("Next", id="next-button-method1", color="primary", style={'width': '100%'}, disabled=True), width=2),
                    ], justify="end", className="g-2")
                ], width=10)
            ], justify="between", align="center", className="mt-3")
        ]),
        dcc.Tab(label='Change normalization method: Upload unnormalized data', value='method2', children=[
            html.Div(id='flowchart-container-method2'),
            html.Div(id='dynamic-content-method2'),
            dbc.Row([
                dbc.Col(dbc.Button(id="execute-button-method2", children="Prepare Data", color="success", style={'width': '100%'}), width=2),
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dbc.Button("Previous", id="previous-button-method2", style={'width': '100%'}, color="secondary", disabled=True), width=2),
                        dbc.Col(dbc.Button("Next", id="next-button-method2", color="primary", style={'width': '100%'}, disabled=True), width=2),
                    ], justify="end", className="g-2")
                ], width=10)
            ], justify="between", align="center", className="mt-3")
        ]),
        dcc.Tab(label='Continue previous visualization: Upload normalized data', value='method3', children=[
            html.Div(id='flowchart-container-method3'),
            html.Div(id='dynamic-content-method3'),
            dbc.Row([
                dbc.Col(dbc.Button(id="execute-button-method3", children="Prepare Data", color="success", style={'width': '100%'}), width=2),
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dbc.Button("Previous", id="previous-button-method3", color="secondary", style={'width': '100%'}, disabled=True), width=2),
                        dbc.Col(dbc.Button("Next", id="next-button-method3", color="primary", style={'width': '100%'}, disabled=True), width=2),
                    ], justify="end", className="g-2")
                ], width=10)
            ], justify="between", align="center", className="mt-3")
        ]),
    ]),
    dcc.Interval(id="log-interval", interval=2000, n_intervals=0),  # Update every 2 seconds
    log_text
], fluid=True)

# Part 4: Callback to manage button states and text for method 1
@app.callback(
    [Output('execute-button-method1', 'children'),
     Output('next-button-method1', 'disabled'),
     Output('previous-button-method1', 'disabled')],
    [Input('current-stage-method1', 'data'),
     Input('preparation-status-method1', 'data')]
)
def manage_buttons_method1(current_stage, preparation_status):
    stages = stages_mapping['method1']
    current_index = stages.index(current_stage)

    button_text = "Prepare Data" if current_stage == 'File Preparation' else "Undetermined"
    previous_disabled = (current_index == 0)
    next_disabled = not preparation_status

    return button_text, next_disabled, previous_disabled

# Callback to manage button states and text for method 2
@app.callback(
    [Output('execute-button-method2', 'children'),
     Output('next-button-method2', 'disabled'),
     Output('previous-button-method2', 'disabled')],
    [Input('current-stage-method2', 'data'),
     Input('preparation-status-method2', 'data')]
)
def manage_buttons_method2(current_stage, preparation_status):
    stages = stages_mapping['method2']
    current_index = stages.index(current_stage)

    button_text = "Prepare Data" if current_stage == 'File Preparation' else "Undetermined"
    previous_disabled = (current_index == 0)
    next_disabled = not preparation_status

    return button_text, next_disabled, previous_disabled

# Callback to manage button states and text for method 3
@app.callback(
    [Output('execute-button-method3', 'children'),
     Output('next-button-method3', 'disabled'),
     Output('previous-button-method3', 'disabled')],
    [Input('current-stage-method3', 'data'),
     Input('preparation-status-method3', 'data')]
)
def manage_buttons_method3(current_stage, preparation_status):
    stages = stages_mapping['method3']
    current_index = stages.index(current_stage)

    button_text = "Prepare Data" if current_stage == 'File Preparation' else "Undetermined"
    previous_disabled = (current_index == 0)
    next_disabled = not preparation_status

    return button_text, next_disabled, previous_disabled

# Part 5: Callback to handle stage navigation
@app.callback(
    [Output(f'current-stage-{method}', 'data') for method in ['method1', 'method2', 'method3']],
    [Input(f'next-button-{method}', 'n_clicks') for method in ['method1', 'method2', 'method3']] +
    [Input(f'previous-button-{method}', 'n_clicks') for method in ['method1', 'method2', 'method3']],
    [State(f'current-stage-{method}', 'data') for method in ['method1', 'method2', 'method3']],
    prevent_initial_call=True
)
def navigate_stages(*args):
    next_clicks = args[:3]
    previous_clicks = args[3:6]
    current_stages = args[6:9]
    results = []

    for idx, method in enumerate(['method1', 'method2', 'method3']):
        stages = stages_mapping[method]
        current_stage = current_stages[idx]
        current_index = stages.index(current_stage)

        if next_clicks[idx]:
            if current_index < len(stages) - 1:
                results.append(stages[current_index + 1])
            else:
                results.append(current_stage)
        elif previous_clicks[idx]:
            if current_index > 0:
                results.append(stages[current_index - 1])
            else:
                results.append(current_stage)
        else:
            results.append(current_stage)

    return results

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

    if stage_method1 == 'File Preparation':
        upload_component1 = create_upload_layout_method1()
        return flowchart1, upload_component1

    elif stage_method1 == 'Normalization':
        # Placeholder for normalization component
        normalization_component1 = create_normalization_layout(method_id=1)
        return flowchart1, normalization_component1

    elif stage_method1 == 'Spurious Contact Removal':
        # Placeholder for spurious contact removal content
        spurious_component = html.Div("Spurious contact removal content goes here.")
        return flowchart1, spurious_component

    elif stage_method1 == 'Visualization':
        # Placeholder for visualization content
        visualization_component = html.Div("Visualization content goes here.")
        return flowchart1, visualization_component

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

    if stage_method2 == 'File Preparation':
        upload_component2 = create_upload_layout_method2()
        return flowchart2, upload_component2

    elif stage_method2 == 'Normalization':
        # Placeholder for normalization content
        normalization_component2 = create_normalization_layout(method_id=2)
        return flowchart2, normalization_component2

    elif stage_method2 == 'Spurious Contact Removal':
        spurious_component = html.Div("Spurious contact removal content goes here.")
        return flowchart2, spurious_component

    elif stage_method2 == 'Visualization':
        visualization_component = html.Div("Visualization content goes here.")
        return flowchart2, visualization_component

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

    if stage_method3 == 'File Preparation':
        upload_component3 = create_upload_layout_method3()
        return flowchart3, upload_component3

    elif stage_method3 == 'Visualization':
        visualization_component = html.Div("Visualization content goes here.")
        return flowchart3, visualization_component

    return flowchart3, None

# Part 6: Periodic callback to update the log box content
@app.callback(
    Output("log-box", "value"),
    [Input("log-interval", "n_intervals")]
)
def update_log_box(n):
    log_stream.seek(0)  # Go to the start of the log
    log_content = log_stream.read()  # Read the current log content
    return log_content  # Display the latest logs

# Part 7: Run the Dash app
register_preparation_callbacks(app)
register_normalization_callbacks(app, method_id=1)
register_normalization_callbacks(app, method_id=2)

if __name__ == '__main__':
    app.run_server(debug=True)