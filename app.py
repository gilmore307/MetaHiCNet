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
modal_body = dbc.ModalBody(
    html.Div(
        [
            html.H2("Section 1: Preparation", className="mb-3"),
            html.P("MetaHiCNet accommodates both new and returning users with tailored input requirements.", 
                   className="mb-4"),

            html.H4("For New Users:", className="mt-3"),
            html.Div([
                html.H5("1. Contig Information File", className="mt-3"),
                html.P([
                    "This file includes the following columns:",
                    html.Ul([
                        html.Li([
                            html.Strong("‘Contig index’"), ", ",
                            html.Strong("‘Number of restriction sites’"), ", and ",
                            html.Strong("‘Contig length’"), " (required)."
                        ]),
                        html.Li([
                            html.Strong("‘Contig coverage’"), " (optional): If not provided, it will be estimated by dividing the diagonal value in the raw Hi-C contact matrix by the ‘Contig length’."
                        ])
                    ]),
                    "This file can be directly generated from common Meta Hi-C analysis pipelines, such as ",
                    html.Strong("MetaCC"), " and ", html.Strong("HiCBin"), "."
                ], className="mb-3"),

                html.H5("2. Hi-C Contact Matrix", className="mt-3"),
                html.P([
                    "The file can be provided in one of the following formats:",
                    html.Ul([
                        html.Li([
                            html.Strong(".txt or .csv format"), ": Should contain the columns ",
                            html.Strong("‘Contig_name1’"), ", ",
                            html.Strong("‘Contig_name2’"), ", and ",
                            html.Strong("‘Contacts’"), "."
                        ]),
                        html.Li([
                            html.Strong(".npz format"), ": Should be either a ",
                            html.Strong("NumPy dense matrix"), " or a ",
                            html.Strong("SciPy sparse matrix"), "."
                        ])
                    ]),
                    "This file can be directly generated from common Meta Hi-C analysis pipelines, such as MetaCC and HiCBin.",
                    html.Br(),
                    html.Strong("Note:"), " The row and column indices of the Hi-C Contact Matrix must match the row indices of the Contig Information File. ",
                ], className="mb-3"),

                html.H5("3. Binning Information File (Optional)", className="mt-3"),
                html.P([
                    "Skip this step if your goal is solely to normalize the Hi-C contact matrix.",
                    html.Br(),
                    "File format: ",
                    html.Ul([
                        html.Li([
                            html.Strong("‘Contig index’"), " and ",
                            html.Strong("‘Bin index’"), " (specifying the bin to which each contig belongs)."
                        ])
                    ]),
                    "This file can be obtained from the binning results of ", html.Strong("Meta Hi-C analysis pipelines"), " or any other binners you select."
                ], className="mb-3"),

                html.H5("4. Taxonomy Information File (Optional)", className="mt-3"),
                html.P([
                    "Skip this step if your goal is solely to normalize the Hi-C contact matrix.",
                    html.Br(),
                    "File format:",
                    html.Ul([
                        html.Li([
                            html.Strong("‘Bin index’")
                        ]),
                        html.Li([
                            html.Strong("‘Category’"), ": The taxonomic category of each bin (",
                            html.Strong("‘chromosome’"), ", ",
                            html.Strong("‘virus’"), ", ",
                            html.Strong("‘plasmid’"), ", or ",
                            html.Strong("‘Unclassified’"), "). Unclassified bins can also be left blank."
                        ]),
                        html.Li([
                            "Additional ", html.Strong("Taxonomic Columns"), " for taxonomic information (e.g., family, genus, species)."
                        ])
                    ])
                ], className="mb-3"),
            ]),

            html.H4("For Returning Users:", className="mt-3"),
            html.P([
                "Returning users can upload compressed files generated in the ",
                html.Strong("Hi-C Contact Normalization Results"), " page to restore their progress.",
                html.Br(),
                html.Ul([
                    html.Li([
                        html.Strong("Unnormalized Data"), ": Upload the ",
                        html.Strong("unnormalized_information.7z"), " file to move directly to the normalization stage."
                    ]),
                    html.Li([
                        html.Strong("Normalized Data"), ": Upload the ",
                        html.Strong("normalized_information.7z"), " file to proceed directly to the visualization stage."
                    ])
                ])
            ], className="mb-0"),
            
            html.H4("For a Try-Out:", className="mt-3"),
            html.P([
                "Simply click the ", html.Strong("Load Example Files"), " button to automatically load pre-prepared files, allowing you to move directly to the next step without uploading any data."
            ], className="mb-3"),
            
            html.Hr(),
            
            # Section 2: Normalization
            html.H2("Section 2: Normalization", className="mt-5 mb-3"),
            html.P("MetaHiCNet offers several normalization methods for Hi-C contact matrices. Choose the method based on your dataset’s characteristics and analysis needs.", className="mb-4"),

            # Raw Method
            html.H4("1. Raw", className="mt-3"),
            html.P([
                html.Strong("What is this method?"), " The Raw method filters out low-contact values to reduce noise in the Hi-C contact matrix."
            ]),
            html.P([
                html.Strong("How does it work?"), " It removes values below a certain threshold percentile to denoise the data."
            ]),
            html.P([
                html.Strong("When should I use it?"), " Use Raw option when your data has already been normalized or you dont need to normalize your data."
            ]),
            html.P([
                html.Strong("Parameter Settings:"),
                html.Ul([
                    html.Li("Threshold: Set between 0-10 (default 5); higher values remove more noise.")
                ])
            ]),

            # normCC Method
            html.H4("2. normCC", className="mt-3"),
            html.P([
                html.Strong("What is this method?"), " The normCC method applies a negative binomial (NB) distribution model to adjust for biases due to contig coverage, length, and restriction sites."
            ]),
            html.P([
                html.Strong("How does it work?"), " It predicts expected Hi-C contacts and scales the raw value to eliminate biases."
            ]),
            html.P([
                html.Strong("When should I use it?"), " Use normCC when contigs have high overlap and contamination."
            ]),
            html.P([
                html.Strong("Parameter Settings:"),
                html.Ul([
                    html.Li("Threshold: Set between 0-10 (default 5); controls denoising.")
                ])
            ]),

            # HiCzin Method
            html.H4("3. HiCzin", className="mt-3"),
            html.P([
                html.Strong("What is this method?"), " The HiCzin method applies log transformation and Zero-Inflated Negative Binomial model to normalize Hi-C contact matrices with many zeros."
            ]),
            html.P([
                html.Strong("How does it work?"), " It models interactions using logarithmic scaling and adjusts for biases like coverage, length, and restriction sites."
            ]),
            html.P([
                html.Strong("When should I use it?"), " Use HiCzin when dealing with sparse matrices containing many zero or near-zero values."
            ]),
            html.P([
                html.Strong("Parameter Settings:"),
                html.Ul([
                    html.Li("Threshold: Set between 0-10 (default 5); removes low-contact values.")
                ])
            ]),

            # bin3C Method
            html.H4("4. bin3C", className="mt-3"),
            html.P([
                html.Strong("What is this method?"), " The bin3C method applies bistochastic normalization to balance the rows and columns of the Hi-C contact matrix."
            ]),
            html.P([
                html.Strong("How does it work?"), " It iteratively rescales the matrix using the Kantorovich-Rubinstein regularization algorithm until balance is achieved."
            ]),
            html.P([
                html.Strong("When should I use it?"), " Use bin3C when balancing the matrix is critical for downstream interpretation like clustering or binning."
            ]),
            html.P([
                html.Strong("Parameter Settings:"),
                html.Ul([
                    html.Li("Threshold: Set between 0-10 (default 5); removes low-contact values."),
                    html.Li("Max Iterations: Default 1000; increase for slow-converging datasets."),
                    html.Li("Tolerance: Default 1e-6; lower for higher precision.")
                ])
            ]),

            # MetaTOR Method
            html.H4("5. MetaTOR", className="mt-3"),
            html.P([
                html.Strong("What is this method?"), " The MetaTOR method applies square root normalization to stabilize variance of Hi-C contact matrix."
            ]),
            html.P([
                html.Strong("How does it work?"), " The method works by scaling the raw contact values based on the diagonal elements of the contact matrix, which represent self-interactions."
            ]),
            html.P([
                html.Strong("When should I use it?"), " Use MetaTOR when you need to stabilize variance across the Hi-C matrix for consistency in interaction contacts."
            ]),
            html.P([
                html.Strong("Parameter Settings:"),
                html.Ul([
                    html.Li("Threshold: Set between 0-10 (default 5); removes low-contact values.")
                ])
            ]),
            
            # Additional Options
            html.H4("6. Additional Options (Advanced Options to Process Data After Normalization)", className="mt-4"),
            html.P("The following options can be enabled to further process the data after normalization:", className="mb-3"),

            # Remove Unclassified Contigs
            html.H5("Remove Unclassified Contigs:", className="mt-3"),
            html.P([
                "Check this box to exclude contigs or bins that are not classified in any taxonomic levels. Enabling this option helps reduce the size of your dataset and speeds up processing."
            ], className="mb-3"),

            # Remove Host-Host Interactions
            html.H5("Remove Host-Host Interactions:", className="mt-3"),
            html.P([
                "Check this box to remove all interactions between contigs or bins labeled as chromosomes. Enabling this option can significantly reduce the amount of Hi-C contacts and accelerate processing."
            ], className="mb-3"),

            # Important Note
            html.P([
                html.Strong("Important Note:"), " Please do not enable the Remove Unclassified Contigs and Remove Host-Host Interactions options if you have not provided the Binning Information File and Taxonomy Information File during the data upload process. These files are required to process these options correctly."
            ], className="mt-3"),
            
            html.H4("7. Normalization Results", className="mt-4"),
            html.H5("Figure: Relationship between raw interaction counts and the product of the number of restriction sites, length, and coverage between contig pairs.", className="mt-3"),  # Sub-header for Figure
            html.P([
                "The figure illustrates how raw Hi-C interaction counts are influenced by three bias factors: the number of restriction sites, contig length, and coverage between pairs of contigs.",
            ], className="mb-3"),
            html.P([
                html.Strong("X-axis:"), " Represents the product of one of the bias factors (restriction sites, length, or coverage).",
                html.Br(),
                html.Strong("Y-axis:"), " Represents the raw interaction counts."
            ], className="mb-3"),

            html.H5("Table: Pearson Correlation Coefficients (Absolute Value) Between Normalized Hi-C Contacts and the Product of Each of the Three Factors of Explicit Biases", className="mt-3"),  # Sub-header for Table
            html.P([
                html.Strong("Smaller values"), " are better, indicating that the normalization method has effectively removed the bias.",
                html.Br(),
                html.Strong("Larger values"), " suggest that the normalization method has not fully corrected for the corresponding bias.",
                html.Br(),
                html.Strong("Ideal Outcome: "), "Correlations close to 0 indicate successful bias removal."
            ], className="mb-3"),
            
            html.Hr(),
            
            # Section 3: Visualization
            html.H2("Section 3: Visualization", className="mt-5 mb-3"),
            html.P("Use the following visualization options to explore and analyze Hi-C interaction data in different ways.", className="mb-4"),

            # Switch Visualization Network
            html.H4("Switch Visualization Network", className="mt-3"),
            html.P([
                "Click this button to switch from the current visualization to the normalization results view."
            ], className="mb-3"),

            # Reset Button
            html.H4("Reset Selection", className="mt-3"),
            html.P([
                "Click this button to clear all selections and reset the visualization to Cross-Taxa Hi-C Interaction."
            ], className="mb-3"),

            # Tooltip Toggle Container
            html.H4("Enable or Disable Tooltips", className="mt-3"),
            html.P([
                "Check this box to enable tooltips that provide contextual information about the components of this app."
            ], className="mb-3"),

            # Dropdowns
            html.H4("Visualization and Selection Dropdowns", className="mt-3"),
            html.P([
                "Use these dropdown menus to explore different visualization modes or select annotations and bins for detailed analysis.",
                html.Br(),
                "The options selected here control the content displayed in the visualizations and tables.",
                html.Br(),
                "The dropdown offers three visualization options: ",
                html.Ul([
                    html.Li("Taxonomic Framework: Displays a hierarchical treemap showing how annotations are grouped and scaled by a selected metric, such as coverage or classification."),
                    html.Li("Cross-Taxa Hi-C Interaction: Focuses on interactions between annotations at a taxonomic level, shown as a Cytoscape graph and bar charts summarizing interaction metrics."),
                    html.Li("Cross-Bin Hi-C Interactions: Explores relationships between individual bins and their connections within the dataset, emphasizing specific bins of interest.")
                ])
            ], className="mb-3"),

            # Legend Container
            html.H4("Color Legend and Taxonomy Level Selector", className="mt-3"),
            html.P([
                "1. ", html.Strong("Color Legend:"), " Colors are consistently applied across the Cytoscape graph, bar chart, and tables to represent categories or annotations at the selected taxonomy level. Use the legend to identify categories by their assigned colors.",
                html.Ul([
                    html.Li("Viruses are reddish colors."),
                    html.Li("Plasmids are greenish colors."),
                    html.Li("Chromosomes are bluish colors.")
                ]),
                "2. ", html.Strong("Taxonomy Selector:"), " The taxonomy selector affects: ",
                html.Ul([
                    html.Li("Taxonomy Visualization: Adjusts how nodes are annotated and grouped in the hierarchy, such as by phylum or genus."),
                    html.Li("Bin Visualization: Influences how nodes (bins) are distributed in the network, grouping them according to the selected taxonomy level."),
                    html.Li("Contact Table: Changes the level of aggregation of the contact table by defining rows and columns based on the selected taxonomy level.")
                ])
            ], className="mb-3"),

            # Bar Chart Container
            html.H4("Bar Chart", className="mt-3"),
            html.P([
                "1. ", html.Strong("Chart Types:"), " The bar chart can display the following types of charts: ",
                html.Ul([
                    html.Li("Fraction of Classified Bins by Taxonomic Ranks: Shows the percentage of bins classified at each taxonomic level (e.g., phylum, genus)."),
                    html.Li("Across Taxonomy Hi-C Contacts: Summarizes Hi-C contact strengths for each taxonomic annotation."),
                    html.Li("Hi-C Contacts with Selected Bin: Highlights the contact strengths between the selected bin and other bins.")
                ]),
                "2. ", html.Strong("Scroll Bar:"), " A horizontal scroll bar allows you to navigate through bars when there are too many to display at once."
            ], className="mb-3"),

            # Information Table Container
            html.H4("Information Table", className="mt-3"),
            html.P([
                "1. ", html.Strong("Filter, Sort, and Search:"), " Use column headers to sort rows or apply filters to narrow down results. You can also use the search box in the headers to find specific bins or annotations quickly.",
                html.Br(),
                "2. ", html.Strong("Bin Selection:"), " Click a row to select a bin, updating the Cytoscape graph, bar chart, and other visualizations.",
                html.Br(),
                "3. ", html.Strong("Automatic Filtering:"), " The table updates dynamically based on selections. ",
                html.Ul([
                    html.Li("Taxa Selected: Shows bins within the selected taxa."),
                    html.Li("Bin Selected: Shows bins interacting with the selected bin.")
                ]),
                "4. ", html.Strong("Filter Checkbox:"), " Enabling 'Only show elements present in the diagram' checkbox displays only bins or annotations visible in the Cytoscape graph.",
                html.Br(),
                "5. ", html.Strong("Color Coding:"), " ",
                html.Ul([
                    html.Li("Index Column: Matches the node colors in the Cytoscape graph."),
                    html.Li("Taxonomy Column: Color-coded by the selected taxonomy category."),
                    html.Li("Numeric Columns: Higher values are represented with deeper colors.")
                ])
            ], className="mb-3"),

            # Treemap Graph Container
            html.H4("Treemap Graph", className="mt-3"),
            html.P([
                "1. ", html.Strong("Hierarchy Representation:"), " Taxa of finer levels (e.g., species) are nested within rectangles of their broader levels (e.g., genus, domain).",
                html.Br(),
                "2. ", html.Strong("Color Coding:"), " Rectangles are color-coded by taxonomic level.",
                html.Ul([
                    html.Li("Darker Colors: Represent finer taxonomic levels, such as species or genus."),
                    html.Li("Lighter Colors: Represent broader taxonomic levels, such as domain or phylum.")
                ]),
                "3. ", html.Strong("Size Representation:"), " The size of each rectangle reflects the total coverage within that taxa.",
                html.Br(),
                "4. ", html.Strong("Click:"), " ",
                html.Ul([
                    html.Li("Click on a rectangle to explore it further in related visualizations."),
                    html.Li("Click on the header of the treemap to return to broader taxonomic levels.")
                ])
            ], className="mb-3"),

            # Cytoscape Graph Container
            html.H4("Cytoscape Graph", className="mt-3"),
            html.P([
                "This is a network graph visualizing relationships between annotations or bins based on Hi-C interactions.",
                html.Br(),
                "1. ", html.Strong("Node Distribution:"), " The graph dynamically adjusts positions to emphasize these relationships.",
                html.Ul([
                    html.Li("Nodes are distributed using a force-directed layout. Nodes closer to each other indicate stronger Hi-C interactions."),
                    html.Li("Selected nodes or bins are fixed at the center of the graph for focused analysis."),
                    html.Li("Nodes representing bins are distributed spatially within their annotation groups.")
                ]),
                "2. ", html.Strong("Interactive Node Selection:"), " ",
                html.Ul([
                    html.Li("Click on a node to select it. The selection updates related visualizations, such as the information table, bar chart, and contact table."),
                    html.Li("Selected nodes are visually highlighted with a border, and their connections are emphasized.")
                ])
            ], className="mb-3"),

            # Contact Table Container
            html.H4("Contact Table", className="mt-3"),
            html.P([
                "This table displays pairwise Hi-C contact values between taxa, providing a detailed view of their interactions.",
                html.Br(),
                "1. ", html.Strong("Hi-C Contact Values:"), " Each cell represents the interaction strength between the taxa in the corresponding row and column.",
                html.Br(),
                "2. ", html.Strong("Row Annotation Selection:"), " Click on a row to select its corresponding annotation or bin.",
                html.Br(),
                "3. ", html.Strong("Sorting:"), " ",
                html.Ul([
                    html.Li("Click the header of numeric columns to sort rows by the values in ascending or descending order. This helps identify bins or annotations with the strongest or weakest interactions."),
                    html.Li("Click the header of the 'Index' column to reset the sorting and return to the initial state.")
                ]),
                "4. ", html.Strong("Color Coding:"), " ",
                html.Ul([
                    html.Li("Higher contact values are highlighted with deeper colors, making it easy to identify strong interactions at a glance."),
                    html.Li("The row annotation is color-coded consistently with its type, matching the color scheme used in other visualizations.")
                ])
            ], className="mb-3"),
        ],
        style={"maxHeight": "70vh", "overflowY": "auto"}
    )
)
    
app.layout = dbc.Container([
    # Stores for app state management
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/home", id="home-link")),
            dbc.NavItem(dbc.NavLink("Help", id="open-help-modal", href="#")),
            dbc.NavItem(dbc.NavLink("GitHub", href="https://github.com/gilmore307/MetaHiCNet", target="_blank")),
            dbc.NavItem(dbc.NavLink("Comments", href="https://github.com/gilmore307/MetaHiCNet/issues", target="_blank")),
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