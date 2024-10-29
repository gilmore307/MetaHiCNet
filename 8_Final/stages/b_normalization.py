import dash
from dash import dcc, html, no_update
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.express as px
import logging
from helper import (
    run_normalization, 
    preprocess
)

# Set up logging
logger = logging.getLogger("app_logger")

def create_normalization_layout():
    normalization_methods = [
        {'label': 'Raw - Removing contacts below a certain threshold for noise reduction.', 'value': 'Raw'},
        {'label': 'normCC - GLM-based normalization to adjust for varying coverage and signal.', 'value': 'normCC'},
        {'label': 'HiCzin - Logarithmic scaling method for normalizing Hi-C contact frequencies.', 'value': 'HiCzin'},
        {'label': 'bin3C - Sinkhorn-Knopp algorithm for balancing matrix rows and columns.', 'value': 'bin3C'},
        {'label': 'MetaTOR - Square root normalization to stabilize variance.', 'value': 'MetaTOR'}
    ]
    
    layout = html.Div([
        html.Div([
            html.Label("Select Normalization Method:"),
            dcc.Dropdown(
                id='normalization-method',
                options=normalization_methods,
                value='Raw',  # Default value
                style={'width': '100%'}
            )
        ], className="my-3"),
        html.Div(id='normalization-parameters', className="my-3"), 
        # Store component for the normalized matrix
        dcc.Store(id='normalized-matrix-store', storage_type='memory')
    ])

    return layout

def register_normalization_callbacks(app):
    @app.callback(
        [Output('normalization-parameters', 'children')],
        [Input('normalization-method', 'value')]
    )
    def update_parameters(selected_method):
        if selected_method == 'Raw' or selected_method == 'normCC':
            return [html.Div([
                html.Label("Threshold Percentage for Denoising (default: 5): Contacts below this percentile will be removed to reduce noise."),
                dcc.Input(
                    id='thres-input',
                    type='number',
                    value=5,
                    placeholder="Threshold percentage (0-100)",
                    style={'width': '100%'}
                )
            ])]
        elif selected_method == 'HiCzin':
            return [html.Div([
                html.Label("Epsilon (default: 1): A small value added to avoid zero values in calculations."),
                dcc.Input(
                    id='epsilon-input',
                    type='number',
                    value=1,
                    placeholder="Epsilon value",
                    style={'width': '100%'}
                ),
                html.Label("Threshold Percentage for Denoising (default: 5): Contacts below this percentile will be removed to reduce noise."),
                dcc.Input(
                    id='thres-input',
                    type='number',
                    value=5,
                    placeholder="Threshold percentage (0-100)",
                    style={'width': '100%'}
                )
            ])]
        elif selected_method == 'bin3C':
            return [html.Div([
                html.Label("Epsilon (default: 1): A small value added to avoid zero values in calculations."),
                dcc.Input(
                    id='epsilon-input',
                    type='number',
                    value=1,
                    placeholder="Epsilon value",
                    style={'width': '100%'}
                ),
                html.Label("Maximum Iterations (default: 1000): Controls the number of iterations for the Sinkhorn-Knopp algorithm."),
                dcc.Input(
                    id='max-iter-input',
                    type='number',
                    value=1000,
                    placeholder="Maximum iterations for convergence",
                    style={'width': '100%'}
                ),
                html.Label("Tolerance for Convergence (default: 1e-6): Defines the precision for convergence. Lower values increase precision."),
                dcc.Input(
                    id='tol-input',
                    type='number',
                    value=1e-6,
                    placeholder="Tolerance for convergence",
                    style={'width': '100%'}
                ),
                html.Label("Threshold Percentage for Denoising (default: 5): Contacts below this percentile will be removed to reduce noise."),
                dcc.Input(
                    id='thres-input',
                    type='number',
                    value=5,
                    placeholder="Threshold percentage (0-100)",
                    style={'width': '100%'}
                )
            ])]
        elif selected_method == 'MetaTOR':
            return [html.Div([
                html.Label("Epsilon (default: 1): A small value added to avoid zero values in calculations."),
                dcc.Input(
                    id='epsilon-input',
                    type='number',
                    value=1,
                    placeholder="Epsilon value",
                    style={'width': '100%'}
                ),
                html.Label("Threshold Percentage for Denoising (default: 5): Contacts below this percentile will be removed to reduce noise."),
                dcc.Input(
                    id='thres-input',
                    type='number',
                    value=5,
                    placeholder="Threshold percentage (0-100)",
                    style={'width': '100%'}
                )
            ])]
        else:
            return no_update

    @app.callback(
        [Output('normalization-status', 'data'),
         Output('dynamic-heatmap', 'children'),
         Output('normalized-matrix-store', 'data')],
        [Input('execute-button', 'n_clicks')],
        [State('normalization-method', 'value'),
         State('epsilon-input', 'value'),
         State('thres-input', 'value'),
         State('max-iter-input', 'value'),
         State('user-folder', 'data'),
         State('current-method', 'data'),
         State('current-stage', 'data')],
        prevent_initial_call=True
    )
    def execute_normalization(n_clicks, method, epsilon, threshold, max_iter, user_folder, selected_method, current_stage):
        # Only trigger if in the 'Normalization' stage for the selected methods
        logger.info(f"Running normalization for {selected_method} using {method}...")
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'execute-button':
            raise PreventUpdate
        if selected_method not in ['method1', 'method2'] or current_stage != 'Normalization':
            raise PreventUpdate
        if n_clicks is None:
            raise PreventUpdate
    
        logger.info(f"Running normalization for {selected_method} using {method}...")
        
        # Preprocess the data needed for normalization
        contig_info, contact_matrix = preprocess(user_folder)
        
        if contig_info is None or contact_matrix is None:
            logger.error("Error reading files from folder. Please check the uploaded data.")
            return False, html.Div("Error reading files from folder. Please check the uploaded data."), no_update
        
        # Run the normalization
        normalized_matrix = run_normalization(method, contig_info, contact_matrix, epsilon, threshold, max_iter)
        
        if normalized_matrix is None or normalized_matrix.nnz == 0:
            logger.error("Normalization failed or produced an empty matrix.")
            return False, html.Div("Normalization failed or produced an empty matrix."), no_update
        
        dense_matrix = normalized_matrix.toarray()
        fig = px.imshow(dense_matrix, color_continuous_scale='Viridis')
        heatmap = dcc.Graph(figure=fig)
        
        # Convert the sparse matrix to a dictionary for storage in dcc.Store
        normalized_matrix_data = {
            'data': normalized_matrix.data.tolist(),
            'row': normalized_matrix.row.tolist(),
            'col': normalized_matrix.col.tolist(),
            'shape': normalized_matrix.shape
        }
        
        logger.info(f"Normalization successful for {selected_method}.")
        return True, heatmap, normalized_matrix_data