import uuid
import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import logging
import redis
import os
import shutil
import json
from stages.a_preparation import (
    create_upload_layout_method1, 
    create_upload_layout_method2, 
    create_upload_layout_method3,
    register_preparation_callbacks)
from stages.b_normalization import (
    create_normalization_layout, 
    register_normalization_callbacks)
from stages.c_results import (
    results_layout, 
    register_results_callbacks)
from stages.d_visualization import (
    create_visualization_layout, 
    register_visualization_callbacks)

# Part 1: Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks='initial_duplicate'
)
app.enable_dev_tools(debug=False)
server = app.server

# Connect to Redis using REDISCLOUD_URL from environment variables
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.StrictRedis.from_url(redis_url, decode_responses=False)

# Initialize the logger
class SessionLogHandler(logging.Handler):
    def __init__(self, session_id, app):
        super().__init__()
        self.session_id = session_id
        self.app = app
    
    def emit(self, record):
        log_entry = self.format(record)
        redis_key = f"{self.session_id}:log"

        # Add symbols based on log level
        if record.levelname == "ERROR":
            log_entry = f"❌ {log_entry}"
        elif record.levelname == "INFO":
            log_entry = f"✅ {log_entry}"
        elif record.levelname == "WARNING":
            log_entry = f"⚠️ {log_entry}"
        
        current_logs = r.get(redis_key)
        if current_logs is None:
            current_logs = []
        else:
            current_logs = json.loads(current_logs)
        current_logs.append(log_entry)
        
        r.set(redis_key, json.dumps(current_logs), ex=SESSION_TTL)

logger = logging.getLogger('app_logger')
logger.setLevel(logging.INFO)
                   
SESSION_TTL = 300
    
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
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/home", id="home-link")),
            dbc.NavItem(dbc.NavLink("Help", id="open-help-modal", href="#")),
            dbc.NavItem(dbc.NavLink("GitHub", href="https://github.com/gilmore307/MetaHiCNet", target="_blank")),
            dbc.NavItem(dbc.NavLink("Contact", href="https://github.com/gilmore307/MetaHiCNet/issues", target="_blank")),
        ],
        brand="MetaHiCNet",
        brand_href="/"
    ),
    html.Div([
        html.H1("MetaHiCNet", className="display-3 text-center", style={'color': 'black'}),
        html.P("A platform for normalizing and visualizing microbial Hi-C interaction networks.", className="lead text-center", style={'color': 'black'})
    ], style={
        'marginTop': '20px',
        'marginBottom': '20px',
        'padding': '40px',
        'background': 'linear-gradient(to right, rgba(0, 0, 255, 0.4), rgba(0, 255, 0, 0.4), rgba(255, 0, 0, 0.4))',
        'color': 'black',
        'borderRadius': '10px',
    }),
    dcc.Store(id='current-method', data='method1', storage_type='session'),
    dcc.Store(id='current-stage', data='Preparation', storage_type='session'),
    dcc.Store(id='preparation-status-method1', data=False, storage_type='session'),
    dcc.Store(id='preparation-status-method2', data=False, storage_type='session'),
    dcc.Store(id='preparation-status-method3', data=False, storage_type='session'),
    dcc.Store(id='normalization-status', data=False, storage_type='session'),
    dcc.Store(id='visualization-status', data='results', storage_type='session'),
    dcc.Store(id='user-folder', storage_type='session'),
    dcc.Interval(id="ttl-interval", interval=SESSION_TTL*250),
    dcc.Location(id='url', refresh=True),
    html.Div(id="dummy-output", style={"display": "none"}),
    html.Div(id="main-content", children=[]),
    html.Div([
        html.Hr(style={'borderTop': '2px solid #ccc', 'marginBottom': '20px'}),  # Add a line above the footer
        html.P("MetaHiCNet is an open-source tool designed to assist in Hi-C data analysis. All resources and tools on this website are freely accessible to the public.", className="text-center", style={'lineHeight': '0.5'}),
        html.P("For questions, please contact us at yuxuan.du@utsa.edu.", className="text-center", style={'lineHeight': '0.5'}),
    ], style={'marginTop': '20px'}),
    dbc.Modal(
        [
            dbc.ModalHeader("Help"),
            dbc.ModalBody(),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-help-modal", className="ml-auto")
            ),
        ],
        id="help-modal",
        is_open=False,
        size="xl"
    ),
], fluid=True)
        
@app.callback(
    Output('url', 'href'),
    Input('home-link', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_home(n_clicks):
    if n_clicks:
        return "/home"

@app.callback(
    [Output('user-folder', 'data'),
     Output('current-method', 'data', allow_duplicate=True),
     Output('current-stage', 'data', allow_duplicate=True),
     Output('preparation-status-method1', 'data', allow_duplicate=True),
     Output('preparation-status-method2', 'data', allow_duplicate=True),
     Output('preparation-status-method3', 'data', allow_duplicate=True),
     Output('normalization-status', 'data', allow_duplicate=True)],
     Input('url', 'pathname'),
     prevent_initial_call=True
)
def setup_user_id(pathname):
    # Generate a unique session ID for the user
    unique_folder = str(uuid.uuid4())
    print(f"New user starts session: {unique_folder}")
    
    # Store the session marker key with TTL in Redis
    r.set(f"{unique_folder}", "", ex=SESSION_TTL)

    # Initialize the session log handler dynamically with the generated session ID
    session_log_handler = SessionLogHandler(session_id=unique_folder, app=app)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    session_log_handler.setFormatter(formatter)
    logger.addHandler(session_log_handler)
    
    logger.info("App initiated")
    logger.info(f"Session created with ID: {unique_folder}")

    # Return initial states
    return unique_folder, "method1", "Preparation", False, False, False, False

@app.callback(
    Input('ttl-interval', 'n_intervals'),
    State('user-folder', 'data')
)
def refresh_and_cleanup(n, user_folder):
    # 1. Refresh TTL for active session
    if user_folder:
        # Refresh TTL for all keys with the user_folder prefix
        keys_to_refresh = r.keys(f"{user_folder}*")
        for key in keys_to_refresh:
            r.expire(key, SESSION_TTL)

    # 2. Perform cleanup for expired session folders on disk
    output_path = "output"
    if os.path.exists(output_path):
        # List all session folders in the output directory
        for folder_name in os.listdir(output_path):
            session_folder_path = os.path.join(output_path, folder_name)
            
            # Check if the Redis session key for this folder still exists
            if not r.exists(f"{folder_name}"):  # Base key without suffix
                # If the base key doesn't exist in Redis, the session has expired
                if os.path.exists(session_folder_path):
                    shutil.rmtree(session_folder_path)
                    print(f"User terminated session: {user_folder}")

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
    Output('current-stage', 'data'),
    [Input('current-stage', 'data'),
     Input('preparation-status-method1', 'data'),
     Input('preparation-status-method2', 'data'),
     Input('preparation-status-method3', 'data'),
     Input('normalization-status', 'data')],
    [State('current-method', 'data')]
)
def advance_stage(current_stage, prep_status1, prep_status2, prep_status3, norm_status, selected_method):
    stages = stages_mapping[selected_method]
    current_index = stages.index(current_stage)

    # Determine preparation status based on the selected method
    prep_status = {
        'method1': prep_status1,
        'method2': prep_status2,
        'method3': prep_status3
    }[selected_method]

    # Logic to auto-advance to the next stage if conditions are met
    if current_stage == 'Preparation' and prep_status and current_index < len(stages) - 1:
        return stages[current_index + 1]
    elif current_stage == 'Normalization' and norm_status and current_index < len(stages) - 1:
        return stages[current_index + 1]

    return current_stage

@app.callback(
    Output('execute-button', 'children'),
    Input('current-stage', 'data')
)
def update_execute_button(current_stage):
    if current_stage == 'Preparation':
        button_text = "Prepare and Validate Data"
    elif current_stage == 'Normalization':
        button_text = "Normalize Data"
    else:
        button_text = "Execute"

    return button_text

@app.callback(
    Output('load-button', 'style'),
    [Input('current-method', 'data'),
     Input('current-stage', 'data')]
)
def show_quick_load_button(method, stage):
    if method == 'method1' and stage == 'Preparation':
        # Show the quick load button when in method1 preparation stage
        return {'display': 'inline-block', 'width': '300px', 'margin-left': '100px'}
    else:
        # Hide the button for other methods or stages
        return {'display': 'none'}

@app.callback(
    Output('main-content', 'children'),
    [Input('current-stage', 'data'),
     Input('visualization-status', 'data')],
    [State('user-folder', 'data'),
     State('current-method', 'data')]
)
def update_main_content(current_stage, visualization_status, user_folder, selected_method):
    if current_stage == 'Visualization':
        if visualization_status == 'results':
            logger.info("Prepareing normalization results...")
            
            return results_layout(user_folder)
        else:
            return create_visualization_layout()
    else:
        # Default content for other stages
        return [

            
            dcc.Tabs(id='tabs-method', value=selected_method, children=[
                dcc.Tab(label="Upload and prepare raw Hi-C interaction data (First-Time Users)", value='method1', id='tab-method1'),
                dcc.Tab(label="Upload Unnormalized Data to Apply New Normalization Method", value='method2', id='tab-method2'),
                dcc.Tab(label="Upload Normalized Data to Retrieve Visualization", value='method3', id='tab-method3')
            ]),
            
            html.Div(id='flowchart-container', className="my-4"),

            # Loading spinner and primary dynamic content area
            dcc.Loading(
                id="loading-spinner",
                type="default",
                children=[
                    html.Div(id='dynamic-content', className="my-4"),
                    html.Div(id='blank-element', style={'display': 'none'}),
                    dbc.Row([
                        dbc.Col(dbc.Button("Prepare Data", id="execute-button", color="success", style={'width': '300px'})),
                        dbc.Col(dbc.Button('Load Example Files', id='load-button', style={'width': '300px'}), className="d-flex justify-content-end")
                    ], className="d-flex justify-content-between")
                ]
            ),
            
            # Log interval and log box
            dcc.Interval(id="log-interval", interval=2000, n_intervals=0),  # Update every 2 seconds
            dcc.Textarea(
                id="log-box",
                style={
                    'width': '100%',
                    'height': '200px',
                    'resize': 'none',
                    'border': '1px solid #ccc',
                    'padding': '10px',
                    'overflowY': 'scroll', 
                    'backgroundColor': '#f9f9f9',
                    'color': '#333',
                    'fontFamily': 'Arial, sans-serif',
                },
                readOnly=True
            )
        ]

@app.callback(
    [Output('flowchart-container', 'children'),
     Output('dynamic-content', 'children'),
     Output('current-method', 'data')],  # Add Output for current-method
    [Input('tabs-method', 'value'),
     Input('current-stage', 'data')],
    [State('user-folder', 'data')]
)
def update_layout(selected_method, current_stage, user_folder):
    # Generate the flowchart for the current method and stage
    flowchart = create_flowchart(current_stage, method=selected_method)

    # Determine the appropriate content based on the current stage and method
    if current_stage == 'Preparation':
        if selected_method == 'method1':
            content = create_upload_layout_method1()
        elif selected_method == 'method2':
            content = create_upload_layout_method2()
        elif selected_method == 'method3':
            content = create_upload_layout_method3()
        else:
            return no_update, no_update, no_update
    elif current_stage == 'Normalization':
        content = create_normalization_layout()
    else:
        return no_update, no_update, no_update

    # Return flowchart, content, and updated current-method
    return flowchart, content, selected_method

@app.callback(
    Output("help-modal", "is_open"),
    [Input("open-help-modal", "n_clicks"), Input("close-help-modal", "n_clicks")],
    [dash.dependencies.State("help-modal", "is_open")]
)
def toggle_modal(open_clicks, close_clicks, is_open):
    # If the Help link is clicked, open the modal
    if open_clicks or close_clicks:
        return not is_open
    return is_open

@app.callback(
    Output('log-box', 'value'),
    Input('log-interval', 'n_intervals'),
    State('user-folder', 'data')
)
def update_log_box(n_intervals, session_id):
    redis_key = f"{session_id}:log"
    logs = r.get(redis_key)
    
    if logs:
        logs = json.loads(logs.decode())
        return "\n".join(logs)
    
    return "No logs yet."

app.clientside_callback(
    """
    function(n_intervals) {
        const logBox = document.getElementById('log-box');
        if (logBox) {
            logBox.scrollTop = logBox.scrollHeight;  // Scroll to the bottom
        }
        return null;  // No output needed
    }
    """,
    Output('log-box', 'value', allow_duplicate=True),  # Dummy output to trigger the callback
    Input('log-interval', 'n_intervals')
)

# Part 7: Run the Dash app
register_preparation_callbacks(app)
register_normalization_callbacks(app)
register_visualization_callbacks(app)
register_results_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)