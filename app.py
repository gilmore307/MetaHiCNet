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
from stages.c_visualization import (
    create_visualization_layout, 
    register_visualization_callbacks)
import logging
import redis
import os
import io
import datetime
import shutil

# Part 1: Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks='initial_duplicate'  # Set globally
)
app.enable_dev_tools(debug=os.getenv("DEBUG", "False") == "True", dev_tools_hot_reload=False)

# Initialize the global logger and attach the Dash logger handler
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
logger.addHandler(stream_handler)

# Connect to Redis using REDISCLOUD_URL from environment variables
redis_url = os.getenv("REDISCLOUD_URL")
r = redis.from_url(redis_url, decode_responses=True)
try:
    r.ping()
    logger.info("Connected to Redis!")
except redis.ConnectionError:
    logger.info("Failed to connect to Redis.")
    
stages_mapping = {
    'method1': ['Preparation', 'Normalization', 'Visualization'],
    'method2': ['Preparation', 'Normalization', 'Visualization'],
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
app.layout = dbc.Container([
    # Stores for app state management
    dcc.Store(id='current-method', data='method1'),
    dcc.Store(id='current-stage', data='Preparation'),
    dcc.Store(id='preparation-status-method1', data=False),
    dcc.Store(id='preparation-status-method2', data=False),
    dcc.Store(id='preparation-status-method3', data=False),
    dcc.Store(id='normalization-status', data=False),
    dcc.Store(id='user-folder', storage_type='session'), # Unique folder for each visit
    dcc.Location(id='url', refresh=False),  # Detects a new session/visit
    dcc.Interval(id="log-interval", interval=2000, n_intervals=0),  # Update every 2 seconds
    dcc.Interval(id='refresh-ttl', interval=5*60*1000),

    # Main content container, dynamically updated by callback
    html.Div(id="main-content", children=[
        # Header
        html.H1("Meta Hi-C Visualization", className="my-4 text-center"),
        
        # Tabs for different methods
        dcc.Tabs(id='tabs-method', value='method1', children=[
            dcc.Tab(label="Upload and Prepare Raw Hi-C Data (First-Time Users)", value='method1', id='tab-method1'),
            dcc.Tab(label="Upload Unnormalized Data for New Normalization Method", value='method2', id='tab-method2'),
            dcc.Tab(label="Resume Visualization with Previously Normalized Data", value='method3', id='tab-method3')
        ]),
        
        # Flowchart container
        html.Div(id='flowchart-container', className="my-4"),

        # Loading spinner and primary dynamic content area
        dcc.Loading(
            id="loading-spinner",
            type="default",
            delay_show=500,
            children=[
                html.Div(id='dynamic-content', className="my-4"),
                
                # Navigation buttons
                dbc.Row([
                    dbc.Col(dbc.Button("Previous", id="previous-button", color="secondary", style={'width': '100%'}, disabled=True), width=2),
                    dbc.Col(dbc.Button("Prepare Data", id="execute-button", color="success", style={'width': '100%'}), width=2),
                    dbc.Col(dbc.Button("Next", id="next-button", color="secondary", style={'width': '100%'}, disabled=True), width=2),
                ], justify="between", align="center", className="mt-3")
            ]
        ),
        
        # Log interval and log display
        dcc.Textarea(
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
    ])
], fluid=True)

@app.callback(
    Output('user-folder', 'data'),
    Input('url', 'pathname'),  # Trigger on page load
    prevent_initial_call=True  # Prevent this callback from running on app start
)
def initialize_user_folder(pathname):
    # Generate a unique folder ID for the session
    unique_folder = str(uuid.uuid4())
    logger.info(f"Session Code: {unique_folder}")
    return unique_folder

@app.callback(
    Input('refresh-ttl', 'n_intervals'),
    State('user-folder', 'data')
)
def refresh_session_ttl(n, user_folder):
    if user_folder:
        try:
            # Set the session TTL in seconds
            SESSION_TTL = 900  # 15 minutes in seconds
            
            # Refresh TTL for active session data in Redis
            r.expire(user_folder, SESSION_TTL)

            # Check the remaining TTL in seconds
            remaining_ttl = r.ttl(user_folder)
            if remaining_ttl == -2:
                # If the TTL is -2, the session has expired; proceed with cleanup
                session_folder_path = os.path.join("output", user_folder)
                if os.path.exists(session_folder_path):
                    shutil.rmtree(session_folder_path)
                    logger.info(f"{datetime.datetime.now()} - Session expired. Folder deleted for user: {user_folder}")
            else:
                # Calculate remaining time in minutes and seconds for logging
                remaining_minutes, remaining_seconds = divmod(remaining_ttl, 60)
                logger.info(f"{datetime.datetime.now()} - Session active. {remaining_minutes} minutes {remaining_seconds} seconds remaining.")
        
        except Exception as e:
            # Log any errors that occur during TTL refresh or cleanup
            logger.error(f"{datetime.datetime.now()} - Error refreshing TTL or deleting folder: {str(e)}")


@app.callback(
    [Output('tab-method1', 'disabled'),
     Output('tab-method2', 'disabled'),
     Output('tab-method3', 'disabled')],
    [Input('current-stage', 'data'), 
     Input('tabs-method', 'value')]
)
def disable_other_tabs(current_stage, selected_method):
    # Only allow changing tabs in the Preparation stage
    if current_stage != 'Preparation':
        return [True, True, True]
    return [False, False, False] 

@app.callback(
    [Output('current-method', 'data'),
     Output('current-stage', 'data')],
    [Input('tabs-method', 'value'),
     Input('next-button', 'n_clicks'),
     Input('previous-button', 'n_clicks')],
    [State('current-method', 'data'),
     State('current-stage', 'data')]
)
def update_current_method_and_stage(selected_method, next_click, prev_click, current_method, current_stage):
    # Determine the new stage based on which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        

    method = selected_method
    stages = stages_mapping[method]
    current_index = stages.index(current_stage)

    if ctx.triggered[0]['prop_id'] == 'next-button.n_clicks' and current_index < len(stages) - 1:
        stage = stages[current_index + 1]
    elif ctx.triggered[0]['prop_id'] == 'previous-button.n_clicks' and current_index > 0:
        stage = stages[current_index - 1]
    else:
        stage = current_stage

    return method, stage

@app.callback(
    [Output('previous-button', 'disabled'),
     Output('next-button', 'disabled'),
     Output('previous-button', 'color'),
     Output('next-button', 'color')],
    [Input('current-stage', 'data'),
     Input('preparation-status-method1', 'data'),
     Input('preparation-status-method2', 'data'),
     Input('preparation-status-method3', 'data'),
     Input('normalization-status', 'data')],  # Single normalization status for all methods
    [State('current-method', 'data')]
)
def update_navigate_button_states(current_stage, prep_status1, prep_status2, prep_status3,
                                  norm_status, selected_method):
    
    stages = stages_mapping[selected_method]
    prev_disabled = (stages.index(current_stage) == 0)
    next_disabled = True
    
    # Logic to enable the Next button based on the current stage and status
    if current_stage == 'Preparation' and {
        'method1': prep_status1,
        'method2': prep_status2,
        'method3': prep_status3
    }[selected_method]:
        next_disabled = False
    elif current_stage == 'Normalization' and norm_status:
        next_disabled = False

    # Set button colors based on their enabled/disabled state
    prev_color = 'secondary' if prev_disabled else 'primary'
    next_color = 'secondary' if next_disabled else 'primary'

    return prev_disabled, next_disabled, prev_color, next_color

@app.callback(
    [Output('execute-button', 'children'),  # Set button text
     Output('execute-button', 'disabled'),  # Set button enabled/disabled
     Output('execute-button', 'color')],    # Set button color
    [Input('current-stage', 'data'),
     Input('preparation-status-method1', 'data'),
     Input('preparation-status-method2', 'data'),
     Input('preparation-status-method3', 'data'),
     Input('normalization-status', 'data')],  # Single normalization status
    [State('current-method', 'data')]
)
def update_execute_button(current_stage, prep_status1, prep_status2, prep_status3,
                          norm_status, selected_method):

    # Determine preparation status based on the selected method
    prep_status = {
        'method1': prep_status1,
        'method2': prep_status2,
        'method3': prep_status3
    }[selected_method]

    # Set button text and disable conditions based on the current stage
    if current_stage == 'Preparation':
        button_text = "Prepare Data"
        disable_button = prep_status  # Disable if preparation is complete for this method
    elif current_stage == 'Normalization':
        button_text = "Normalize Data"
        disable_button = norm_status  # Disable if normalization is complete
    else:
        button_text = "Execute"  # Default text in case of any unexpected stage
        disable_button = True  # Disable by default for unexpected cases

    # Set button color based on its disabled state
    button_color = 'secondary' if disable_button else 'success'

    return button_text, disable_button, button_color

@app.callback(
    Output('main-content', 'children'),  # Replace the entire content inside main-content
    [Input('current-stage', 'data')],
    [State('current-method', 'data')]
)
def update_main_content(current_stage, selected_method):
    if current_stage == 'Visualization':
        # Replace main-content with visualization layout
        return create_visualization_layout()
    else:
        # Default content for other stages
        return [
            html.H1("Meta Hi-C Visualization", className="my-4 text-center"),
            
            dcc.Tabs(id='tabs-method', value=selected_method, children=[
                dcc.Tab(label="Upload and Prepare Raw Hi-C Data (First-Time Users)", value='method1', id='tab-method1'),
                dcc.Tab(label="Upload Unnormalized Data for New Normalization Method", value='method2', id='tab-method2'),
                dcc.Tab(label="Resume Visualization with Previously Normalized Data", value='method3', id='tab-method3')
            ]),
            
            html.Div(id='flowchart-container', className="my-4"),

            # Loading spinner and primary dynamic content area
            dcc.Loading(
                id="loading-spinner",
                type="default",
                delay_show=500,
                children=[
                    html.Div(id='dynamic-content', className="my-4"),
                    
                    # Navigation buttons
                    dbc.Row([
                        dbc.Col(dbc.Button("Previous", id="previous-button", color="secondary", style={'width': '100%'}, disabled=True), width=2),
                        dbc.Col(dbc.Button("Prepare Data", id="execute-button", color="success", style={'width': '100%'}), width=2),
                        dbc.Col(dbc.Button("Next", id="next-button", color="secondary", style={'width': '100%'}, disabled=True), width=2),
                    ], justify="between", align="center", className="mt-3")
                ]
            ),
            
            # Log interval and log box
            dcc.Interval(id="log-interval", interval=2000, n_intervals=0),  # Update every 2 seconds
            dcc.Textarea(
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
        ]

@app.callback(
    [Output('flowchart-container', 'children'),  # Single container for flowchart
     Output('dynamic-content', 'children')],  # Add output for URL redirection
    [Input('tabs-method', 'value'),  # Track selected method
     Input('current-stage', 'data')],  # Track current stage
    [State('user-folder', 'data')]
)
def update_layout(selected_method, current_stage, user_folder):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]
        logger.info(f"Triggered by element ID: {trigger['prop_id']} with value: {trigger['value']}")
    else:
        logger.info("Callback triggered without a specific input (e.g., initial load).")

    # Generate the flowchart for the current method and stage
    flowchart = create_flowchart(current_stage, method=selected_method)

    # Check the current method and stage to render the appropriate content
    if current_stage == 'Preparation':
        if selected_method == 'method1':
            content = create_upload_layout_method1()
        elif selected_method == 'method2':
            content = create_upload_layout_method2()
        elif selected_method == 'method3':
            content = create_upload_layout_method3()
        else:
            return no_update, no_update
    elif current_stage == 'Normalization':
        content = create_normalization_layout()
    else:
        return no_update, no_update

    return flowchart, content

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
register_normalization_callbacks(app)
register_visualization_callbacks(app)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))  # Default to 8050 if PORT is not set
    app.run_server(debug=os.getenv("DEBUG", "False") == "True", port=port, host="0.0.0.0")