import uuid
import dash
from dash import dcc, html, no_update
from dash.exceptions import PreventUpdate
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
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks='initial_duplicate'  # Set globally
)
app.enable_dev_tools(debug=True)

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
logger.addHandler(stream_handler)

stages_mapping = {
    'method1': ['Preparation', 'Normalization', 'Spurious Contact Removal', 'Visualization'],
    'method2': ['Preparation', 'Normalization', 'Spurious Contact Removal', 'Visualization'],
    'method3': ['Preparation', 'Visualization']
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
    dcc.Store(id='current-stage-method1', data='Preparation'),
    dcc.Store(id='current-stage-method2', data='Preparation'),
    dcc.Store(id='current-stage-method3', data='Preparation'),
    dcc.Store(id='preparation-status-method1', data=False),
    dcc.Store(id='preparation-status-method2', data=False),
    dcc.Store(id='preparation-status-method3', data=False),
    dcc.Store(id='normalization-status-method-1', data=False),
    dcc.Store(id='normalization-status-method-2', data=False),
    dcc.Store(id='user-folder', data=str(uuid.uuid4())),

    dcc.Tabs(id='tabs-method', value='method1', children=[
        dcc.Tab(label="Upload and Prepare Raw Hi-C Data (First-Time Users)", value='method1', children=[
            html.Div(id='flowchart-container-method1'),
            html.Div(id='dynamic-content-method1')
        ]),
        dcc.Tab(label="Upload Unnormalized Data for New Normalization Method", value='method2', children=[
            html.Div(id='flowchart-container-method2'),
            html.Div(id='dynamic-content-method2')
        ]),
        dcc.Tab(label="Resume Visualization with Previously Normalized Data", value='method3', children=[
            html.Div(id='flowchart-container-method3'),
            html.Div(id='dynamic-content-method3')
        ]),
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Previous", id="previous-button", color="secondary", style={'width': '100%'}, disabled=True), width=2),
        dbc.Col(dbc.Button("Prepare Data", id="execute-button", color="success", style={'width': '100%'}), width=2),
        dbc.Col(dbc.Button("Next", id="next-button", color="secondary", style={'width': '100%'}, disabled=True), width=2),
    ], justify="between", align="center", className="mt-3"),
    dcc.Interval(id="log-interval", interval=2000, n_intervals=0),  # Update every 2 seconds
    log_text
], fluid=True)

@app.callback(
    [Output('current-stage-method1', 'data'),
     Output('current-stage-method2', 'data'),
     Output('current-stage-method3', 'data')],
    [Input('next-button', 'n_clicks'),
     Input('previous-button', 'n_clicks')],
    [State('tabs-method', 'value'),
     State('current-stage-method1', 'data'),
     State('current-stage-method2', 'data'),
     State('current-stage-method3', 'data')],
    prevent_initial_call=True
)
def update_current_stage(next_click, prev_click, selected_method, stage_method1, stage_method2, stage_method3):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Identify which button was clicked
    if ctx.triggered[0]['prop_id'] == 'next-button.n_clicks':
        action = 'next'
    elif ctx.triggered[0]['prop_id'] == 'previous-button.n_clicks':
        action = 'previous'
    else:
        raise PreventUpdate

    # Get the current stage and list of stages for the selected method
    stages = stages_mapping[selected_method]
    current_stage = stage_method1 if selected_method == 'method1' else stage_method2 if selected_method == 'method2' else stage_method3
    current_index = stages.index(current_stage)

    # Determine the new stage based on the action
    if action == 'next' and current_index < len(stages) - 1:
        new_stage = stages[current_index + 1]
    elif action == 'previous' and current_index > 0:
        new_stage = stages[current_index - 1]
    else:
        new_stage = current_stage  # No change if at start or end of stages

    # Return the updated stages based on the selected method
    if selected_method == 'method1':
        return new_stage, stage_method2, stage_method3
    elif selected_method == 'method2':
        return stage_method1, new_stage, stage_method3
    else:
        return stage_method1, stage_method2, new_stage

@app.callback(
    [Output('previous-button', 'disabled'),
     Output('next-button', 'disabled'),
     Output('previous-button', 'color'),
     Output('next-button', 'color')],
    [Input('current-stage-method1', 'data'),
     Input('current-stage-method2', 'data'),
     Input('current-stage-method3', 'data'),
     Input('preparation-status-method1', 'data'),
     Input('preparation-status-method2', 'data'),
     Input('preparation-status-method3', 'data'),
     Input('normalization-status-method-1', 'data'),
     Input('normalization-status-method-2', 'data')],
    [State('tabs-method', 'value')]
)
def update_navigate_button_states(stage_method1, stage_method2, stage_method3,
                         prep_status1, prep_status2, prep_status3,
                         norm_status1, norm_status2, selected_method):
    
    # Determine the current stage based on selected method
    current_stage = {
        'method1': stage_method1,
        'method2': stage_method2,
        'method3': stage_method3
    }[selected_method]

    # Check if the previous button should be disabled (if at the first stage)
    stages = stages_mapping[selected_method]
    prev_disabled = (stages.index(current_stage) == 0)
    
    # Set the next button as disabled by default
    next_disabled = True
    
    # Logic to enable the Next button based on the stage and validation status
    if current_stage == 'Preparation' and {
        'method1': prep_status1,
        'method2': prep_status2,
        'method3': prep_status3
    }[selected_method]:
        next_disabled = False
    elif current_stage == 'Normalization' and {
        'method1': norm_status1,
        'method2': norm_status2
    }.get(selected_method, False):  # Only methods 1 and 2 have a normalization stage
        next_disabled = False
    elif current_stage in ['Spurious Contact Removal', 'Visualization']:
        next_disabled = False

    # Set colors based on enabled/disabled state
    prev_color = 'secondary' if prev_disabled else 'primary'
    next_color = 'secondary' if next_disabled else 'primary'

    return prev_disabled, next_disabled, prev_color, next_color

@app.callback(
    [Output('execute-button', 'children'),  # Set button text
     Output('execute-button', 'disabled'),  # Set button enabled/disabled
     Output('execute-button', 'color')],    # Set button color
    [Input('current-stage-method1', 'data'),
     Input('current-stage-method2', 'data'),
     Input('current-stage-method3', 'data'),
     Input('preparation-status-method1', 'data'),
     Input('preparation-status-method2', 'data'),
     Input('preparation-status-method3', 'data'),
     Input('normalization-status-method-1', 'data'),
     Input('normalization-status-method-2', 'data')],
    [State('tabs-method', 'value')]
)
def update_execute_button(stage_method1, stage_method2, stage_method3,
                          prep_status1, prep_status2, prep_status3,
                          norm_status1, norm_status2, selected_method):

    # Determine the current stage and preparation/normalization status based on the selected method
    if selected_method == 'method1':
        current_stage = stage_method1
        prep_status = prep_status1
        norm_status = norm_status1
    elif selected_method == 'method2':
        current_stage = stage_method2
        prep_status = prep_status2
        norm_status = norm_status2
    else:
        current_stage = stage_method3
        prep_status = prep_status3
        norm_status = None  # Method 3 does not have normalization

    # Set button text based on the current stage
    if current_stage == 'Preparation':
        button_text = "Prepare Data"
        disable_button = prep_status  # Disable if preparation is complete for this method
    elif current_stage == 'Normalization':
        button_text = "Normalize Data"
        disable_button = norm_status  # Disable if normalization is complete for this method
    elif current_stage == 'Spurious Contact Removal':
        button_text = "Remove Spurious Contacts"
        disable_button = False  # Adjust condition if specific requirements are added
    elif current_stage == 'Visualization':
        button_text = "Generate Visualization"
        disable_button = False  # Adjust condition if specific requirements are added
    else:
        button_text = "Execute"  # Default text in case of any unexpected stage
        disable_button = True  # Disable by default for unexpected cases

    # Set button color based on its disabled state
    button_color = 'secondary' if disable_button else 'success'

    return button_text, disable_button, button_color

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

    if stage_method1 == 'Preparation':
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

    if stage_method2 == 'Preparation':
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

    if stage_method3 == 'Preparation':
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