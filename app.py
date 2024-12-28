import numpy as np
import uuid
import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
from dash import callback_context
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
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
from stages.description import modal_body

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
            dbc.NavItem(dbc.NavLink("Home", id="home", href="#")),
            dbc.NavItem(dbc.NavLink("Run", id="run", href="#")),
            dbc.NavItem(dbc.NavLink("Help", id="open-help-modal", href="#")),
            dbc.NavItem(dbc.NavLink("GitHub", href="https://github.com/gilmore307/MetaHiCNet", target="_blank")),
            dbc.NavItem(dbc.NavLink("Issues", href="https://github.com/gilmore307/MetaHiCNet/issues", target="_blank")),
        ],
        brand="MetaHiCNet",
        brand_href="/"
    ),
    html.Div([
        html.H1("MetaHiCNet", className="display-3 text-center", style={'color': 'black'}),
        html.P("A platform for normalizing and visualizing microbial Hi-C interaction networks.", className="lead text-center", style={'color': 'black'})
    ], style={
        'marginBottom': '10px',
        'padding': '30px',
        'background': 'linear-gradient(to right, rgba(0, 0, 255, 0.4), rgba(0, 255, 0, 0.4), rgba(255, 0, 0, 0.4))',
        'color': 'black',
        'borderRadius': '10px',
    }),
    dcc.Store(id='current-method', data='method1', storage_type='session'),
    dcc.Store(id='current-stage', data='Preparation', storage_type='session'),
    dcc.Store(id='home-status', data=True, storage_type='session'),
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
        html.Div([
            dcc.Markdown(
                """
                MetaHiCNet is an open-source tool designed to assist in Hi-C data analysis. All resources and tools on this website are freely accessible to the public.
    
                For questions, please contact us at [yuxuan.du@utsa.edu](mailto:yuxuan.du@utsa.edu).
                """,
                style={'textAlign': 'center', 'color': '#555', 'fontSize': '14px'}
            )
        ], style={'padding': '10px'})
    ], style={
        'marginTop': '20px',
        'padding': '20px',
        'backgroundColor': '#f9f9f9',
        'borderTop': '1px solid #ddd',
        'textAlign': 'center',
        'borderRadius': '5px'
    }),
        
    dbc.Modal(
        [
            dbc.ModalHeader("Help"),
            dbc.ModalBody(modal_body),
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
     Output('home-status', 'data'),
    [Input("home", "n_clicks"), 
     Input("run", "n_clicks")],
     prevent_initial_call=True
)
def home(home_clicks, run_clicks):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'home':
        return True
    
    if triggered_id == 'run':
        return False
    
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
    Output('current-stage', 'data', allow_duplicate=True),
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
     Input('visualization-status', 'data'),
     Input('home-status', 'data')],
    [State('user-folder', 'data'),
     State('current-method', 'data')]
)
def update_main_content(current_stage, visualization_status, home_status, user_folder, selected_method):
    if home_status:
        
        # Cytoscape elements
        cyto_elements = [{'data': {'id': 'f_CAG-74', 'label': 'f_CAG-74', 'label_size': 20, 'size': 1, 'color': '#FFFFFF', 'parent': None, 'visible': 'element'}, 'position': {'x': -40.89731077491495, 'y': -292.34870981472704}, 'style': {'text-margin-y': -5, 'font-style': 'italic'}}, 
                         {'data': {'id': 'f_Lachnospiraceae', 'label': 'f_Lachnospiraceae', 'label_size': 20, 'size': 1, 'color': '#FFFFFF', 'parent': None, 'visible': 'element'}, 'position': {'x': -132.17943380158638, 'y': 266.9486848790008}, 'style': {'text-margin-y': -5, 'font-style': 'italic'}}, 
                         {'data': {'id': 'f_Oscillospiraceae', 'label': 'f_Oscillospiraceae', 'label_size': 20, 'size': 1, 'color': '#FFFFFF', 'parent': None, 'visible': 'element'}, 'position': {'x': 148.3860186713183, 'y': 156.05953336132904}, 'style': {'text-margin-y': -5, 'font-style': 'italic'}}, 
                         {'data': {'id': 'f_UBA644', 'label': 'f_UBA644', 'label_size': 20, 'size': 1, 'color': '#FFFFFF', 'parent': None, 'visible': 'element'}, 'position': {'x': -287.901062481941, 'y': -47.39844478043631}, 'style': {'text-margin-y': -5, 'font-style': 'italic'}}, 
                         {'data': {'id': 'f_Campylobacteraceae', 'label': 'f_Campylobacteraceae', 'label_size': 20, 'size': 1, 'color': '#FFFFFF', 'parent': None, 'visible': 'element'}, 'position': {'x': 223.01336932438124, 'y': -110.05310110900723}, 'style': {'text-margin-y': -5, 'font-style': 'italic'}}, 
                         {'data': {'id': 'f_Siphoviridae_v', 'label': 'f_Siphoviridae_v', 'label_size': 20, 'size': 1, 'color': '#FFFFFF', 'parent': None, 'visible': 'element'}, 'position': {'x': 0.0, 'y': 0.0}, 'style': {'text-margin-y': -5, 'font-style': 'italic'}}, 
                         {'data': {'id': 'vMAG_12', 'label': 'vMAG_12', 'label_size': 15, 'size': 15, 'color': '#AE445A', 'parent': 'f_Siphoviridae_v', 'visible': 'element'}, 'position': {'x': 0, 'y': 0}, 'style': {'text-margin-y': -5, 'font-style': 'normal'}}, 
                         {'data': {'id': 'MAG_542', 'label': 'MAG_542', 'label_size': 15, 'size': 10, 'color': '#00FFFF', 'parent': 'f_CAG-74', 'visible': 'element'}, 'position': {'x': -70.39206589804775, 'y': -265.3290980442661}, 'style': {'text-margin-y': -5, 'font-style': 'normal'}}, 
                         {'data': {'id': 'MAG_524', 'label': 'MAG_524', 'label_size': 15, 'size': 10, 'color': '#89CFF0', 'parent': 'f_Lachnospiraceae', 'visible': 'element'}, 'position': {'x': -161.67418892471918, 'y': 293.96829664946176}, 'style': {'text-margin-y': -5, 'font-style': 'normal'}}, 
                         {'data': {'id': 'MAG_909', 'label': 'MAG_909', 'label_size': 15, 'size': 10, 'color': '#008B8B', 'parent': 'f_Oscillospiraceae', 'visible': 'element'}, 'position': {'x': 118.8912635481855, 'y': 183.07914513179}, 'style': {'text-margin-y': -5, 'font-style': 'normal'}}, 
                         {'data': {'id': 'MAG_1018', 'label': 'MAG_1018', 'label_size': 15, 'size': 10, 'color': '#008B8B', 'parent': 'f_Oscillospiraceae', 'visible': 'element'}, 'position': {'x': 153.33156449511915, 'y': 99.70758950395455}, 'style': {'text-margin-y': -5, 'font-style': 'normal'}}, 
                         {'data': {'id': 'MAG_919', 'label': 'MAG_919', 'label_size': 15, 'size': 10, 'color': '#6495ED', 'parent': 'f_UBA644', 'visible': 'element'}, 'position': {'x': -317.3958176050737, 'y': -20.37883300997535}, 'style': {'text-margin-y': -5, 'font-style': 'normal'}}, 
                         {'data': {'id': 'MAG_962', 'label': 'MAG_962', 'label_size': 15, 'size': 10, 'color': '#007FFF', 'parent': 'f_Campylobacteraceae', 'visible': 'element'}, 'position': {'x': 193.51861420124845, 'y': -83.03348933854629}, 'style': {'text-margin-y': -5, 'font-style': 'normal'}}, 
                         {'data': {'source': 'vMAG_12', 'target': 'MAG_542', 'width': 1, 'color': '#bbb', 'visible': 'element', 'selectable': False}}, {'data': {'source': 'vMAG_12', 'target': 'MAG_524', 'width': 1, 'color': '#bbb', 'visible': 'element', 'selectable': False}}, 
                         {'data': {'source': 'vMAG_12', 'target': 'MAG_909', 'width': 1, 'color': '#bbb', 'visible': 'element', 'selectable': False}}, {'data': {'source': 'vMAG_12', 'target': 'MAG_1018', 'width': 1, 'color': '#bbb', 'visible': 'element', 'selectable': False}}, 
                         {'data': {'source': 'vMAG_12', 'target': 'MAG_919', 'width': 1, 'color': '#bbb', 'visible': 'element', 'selectable': False}}, {'data': {'source': 'vMAG_12', 'target': 'MAG_962', 'width': 1, 'color': '#bbb', 'visible': 'element', 'selectable': False}}]

        # Cytoscape Stylesheet for nodes and edges
        cyto_stylesheet = [
            {
                'selector': 'node',
                'style': {
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'background-color': 'data(color)',
                    'label': 'data(label)',
                    'font-size': 'data(label_size)',
                    'border-color': 'data(border_color)',
                    'border-width': 'data(border_width)',
                    'parent': 'data(parent)',
                    'display': 'data(visible)'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'width': 'data(width)',
                    'line-color': 'data(color)',
                    'opacity': 0.6,
                    'display': 'data(visible)'
                }
            }
        ]
        
        return html.Div([
            html.Div([
                html.H4("Normalize Your Hi-C Interaction Data", className="text-center"),
                html.P(
                    "Explore up to 5 powerful normalization methods and advanced spurious contact removal techniques to refine your Hi-C contact matrices with our website.",
                    className="text-center",
                    style={'color': '#555'}
                ),
            ]),

            html.Div(
                html.Img(
                    src='/assets/home/normalization.png',
                    style={'width': '60%', 'display': 'block', 'margin': '0 auto'},
                ),
                className="text-center"
            ),
            
            html.Hr(),
            
            html.Div([
                html.H4("Play with Your Hi-C Interaction Network", className="text-center"),
                html.P(
                    "Flexibly explore and visualize your interaction networks using Cytoscape. Interact with nodes, adjust layouts, and experiment with your data in real time.",
                    className="text-center",
                    style={'color': '#555'}
                ),
            ], style={'marginTop': '30px'}),
            
            cyto.Cytoscape(
                id='cytoscape-playground',
                elements=cyto_elements,
                style={'width': '100%', 'height': '400px'},
                layout={'name': 'preset'},
                stylesheet=cyto_stylesheet
            ),
        ])
    
    else:
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
    [Input("open-help-modal", "n_clicks"), 
     Input("close-help-modal", "n_clicks")],
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