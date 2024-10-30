from dash import dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.express as px
import logging
import os
import py7zr
import numpy as np
from helper import (
    preprocess_normalization,
    run_normalization,
    generating_bin_information
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
        html.Div(id='heatmap-container', className="my-3"),
        dcc.Store(id='normalized-matrix-store', storage_type='memory')
    ])

    return layout

def register_normalization_callbacks(app):
    @app.callback(
        Output('normalization-parameters', 'children'),
        Input('normalization-method', 'value')
    )
    def update_parameters(normalization_method):
        # Define parameters for each method and set all input boxes initially disabled
        return html.Div([
            html.Div([
                dcc.Checklist(
                    id='remove-unmapped-contigs',
                    options=[{'label': 'Remove Unmapped Contigs', 'value': 'remove_unmapped'}],
                    value=[],  # Default is unchecked
                    style={'margin-right': '20px'}
                ),
                dcc.Checklist(
                    id='remove-host-host',
                    options=[{'label': 'Remove Host-Host Interactions', 'value': 'remove_host'}],
                    value=[],  # Default is unchecked
                )
            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),
            html.Label("Threshold Percentage for Denoising (default: 5): Contacts below this percentile will be removed to reduce noise."),
            dcc.Input(
                id='thres-input',
                type='number',
                value=5,
                placeholder="Threshold percentage (0-100)",
                style={'width': '100%'},
                disabled=normalization_method not in ['Raw', 'normCC', 'HiCzin', 'bin3C', 'MetaTOR']
            ),
            html.Label("Epsilon (default: 1): A small value added to avoid zero values in calculations."),
            dcc.Input(
                id='epsilon-input',
                type='number',
                value=1,
                placeholder="Epsilon value",
                style={'width': '100%'},
                disabled=normalization_method not in ['HiCzin', 'bin3C', 'MetaTOR']
            ),
            html.Label("Maximum Iterations (default: 1000): Controls the number of iterations for the Sinkhorn-Knopp algorithm."),
            dcc.Input(
                id='max-iter-input',
                type='number',
                value=1000,
                placeholder="Maximum iterations for convergence",
                style={'width': '100%'},
                disabled=normalization_method != 'bin3C'
            ),
            html.Label("Tolerance for Convergence (default: 1e-6): Defines the precision for convergence. Lower values increase precision."),
            dcc.Input(
                id='tol-input',
                type='number',
                value=1e-6,
                placeholder="Tolerance for convergence",
                style={'width': '100%'},
                disabled=normalization_method != 'bin3C'
            )
        ])

    @app.callback(
        [Output('normalization-status', 'data')],
        [Input('execute-button', 'n_clicks')],
        [State('normalization-method', 'value'),
         State('epsilon-input', 'value'),
         State('thres-input', 'value'),
         State('max-iter-input', 'value'),
         State('tol-input', 'value'),
         State('remove-unmapped-contigs', 'value'),
         State('remove-host-host', 'value'),
         State('user-folder', 'data'),
         State('current-method', 'data'),
         State('current-stage', 'data')],
        prevent_initial_call=True
    )
    def execute_normalization(n_clicks, normalization_method, epsilon, threshold, max_iter, tolerance,
                              remove_unmapped_contigs, remove_host_host, user_folder, selected_method, current_stage):
        # Only trigger if in the 'Normalization' stage for the selected methods
        if not n_clicks or selected_method not in ['method1', 'method2'] or current_stage != 'Normalization':
            raise PreventUpdate
        
        logger.info(f"Running normalization for {selected_method} using {normalization_method}...")
    
        # Set default values if states are missing
        epsilon = epsilon if epsilon is not None else 1
        threshold = threshold if threshold is not None else 5
        max_iter = max_iter if max_iter is not None else 1000
        tolerance = tolerance if tolerance is not None else 1e-6
    
        # Convert checkbox values to booleans
        remove_unmapped_contigs = 'remove_unmapped' in remove_unmapped_contigs
        remove_host_host = 'remove_host' in remove_host_host
    
        # Preprocess the data needed for normalization
        contig_info, contact_matrix = preprocess_normalization(user_folder)
        if contig_info is None or contact_matrix is None:
            logger.error("Error reading files from folder. Please check the uploaded data.")
            return [False]
    
        # Define normalization parameters based on the selected method
        normalization_params = {
            "method": normalization_method,
            "contig_df": contig_info,
            "contact_matrix": contact_matrix,
            "epsilon": epsilon,
            "threshold": threshold,
            "max_iter": max_iter,
            "tolerance": tolerance
        }
    
        # Run normalization
        normalized_matrix = run_normalization(**normalization_params)
        if normalized_matrix is None or normalized_matrix.nnz == 0:
            logger.error("Normalization failed or produced an empty matrix.")
            return [False]
    
        logger.info(f"Normalization for {normalization_method} completed successfully.")
    
        # Perform bin information generation after normalization
        logger.info("Generating bin level information table and contact matrix...")
        bin_data, contig_info, bin_contact_matrix, contig_contact_matrix = generating_bin_information(
            contig_info,
            normalized_matrix,
            remove_unmapped_contigs,
            remove_host_host
        )
    
        # Define the output path and save the files individually
        user_output_path = f'assets/output/{user_folder}'
        os.makedirs(user_output_path, exist_ok=True)
    
        bin_info_final_path = os.path.join(user_output_path, 'bin_info_final.csv')
        contig_info_final_path = os.path.join(user_output_path, 'contig_info_final.csv')
        bin_contact_matrix_path = os.path.join(user_output_path, 'bin_contact_matrix.npz')
        contig_contact_matrix_path = os.path.join(user_output_path, 'contig_contact_matrix.npz')
    
        # Save each file
        bin_data.to_csv(bin_info_final_path, index=False)
        contig_info.to_csv(contig_info_final_path, index=False)
        np.savez_compressed(
            bin_contact_matrix_path,
            data=bin_contact_matrix.data,
            row=bin_contact_matrix.row,
            col=bin_contact_matrix.col,
            shape=bin_contact_matrix.shape
        )
        np.savez_compressed(
            contig_contact_matrix_path,
            data=contig_contact_matrix.data,
            row=contig_contact_matrix.row,
            col=contig_contact_matrix.col,
            shape=contig_contact_matrix.shape
        )
    
        # Compress saved files into normalized_information.7z
        normalized_archive_path = os.path.join(user_output_path, 'normalized_information.7z')
        with py7zr.SevenZipFile(normalized_archive_path, 'w') as archive:
            archive.write(bin_info_final_path, 'bin_info_final.csv')
            archive.write(contig_info_final_path, 'contig_info_final.csv')
            archive.write(bin_contact_matrix_path, 'bin_contact_matrix.npz')
            archive.write(contig_contact_matrix_path, 'contig_contact_matrix.npz')
    
        logger.info("Bin information generation, file saving, and compression completed successfully.")
    
        return [True]
