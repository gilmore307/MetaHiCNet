from dash import dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.sparse import coo_matrix, spdiags, isspmatrix_csr
import statsmodels.api as sm
from itertools import combinations, product
from joblib import Parallel, delayed
import logging
import os
import pandas as pd
import py7zr
import numpy as np
from stages.helper import (
    save_to_redis,
    get_indexes,
    calculate_submatrix_sum
)

# Set up logging
logger = logging.getLogger("app_logger")

def preprocess_normalization(user_folder, assets_folder='output'):
    try:
        logger.info("Starting data preprocessing...")
        
        # Locate the folder path for the data preparation output
        folder_path = os.path.join(assets_folder, user_folder)
        logger.info(f"Folder path for data preparation output: {folder_path}")

        # Define paths for the files within the folder
        contig_info_path = os.path.join(folder_path, 'contig_info_final.csv')
        contact_matrix_path = os.path.join(folder_path, 'raw_contact_matrix.npz')

        # Read the contig information file as a pandas DataFrame
        logger.info(f"Reading contig information file from: {contig_info_path}")
        contig_info = pd.read_csv(contig_info_path)

        # Load the contact matrix from .npz and reconstruct it as a sparse COO matrix
        logger.info(f"Loading contact matrix from: {contact_matrix_path}")
        contact_matrix_data = np.load(contact_matrix_path)
        data = contact_matrix_data['data']
        row = contact_matrix_data['row']
        col = contact_matrix_data['col']
        shape = tuple(contact_matrix_data['shape'])
        contact_matrix = coo_matrix((data, (row, col)), shape=shape)

        logger.info("Data preprocessing completed successfully.")
        return contig_info, contact_matrix

    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        return None, None

def run_normalization(method, contig_df, contact_matrix, epsilon=1, threshold=5, max_iter=1000, tolerance=0.000001):
    # Ensure contact_matrix is in coo format for consistency across methods
    contact_matrix = contact_matrix.tocoo()
    
    def safe_square(array):
        array = np.clip(array, -1e10, 1e10)  # Clip large values for safety
        return array ** 2
    
    def safe_divide(array, divisor):
        divisor = np.where(divisor == 0, epsilon, divisor)  # Add epsilon to avoid zero division
        return array / divisor

    def standardize(array):
        std = np.std(array)
        return np.zeros_like(array) if std == 0 else (array - np.mean(array)) / std

    def denoise(matrix, threshold):
        matrix = matrix.tocoo()
        threshold_value = np.percentile(matrix.data, threshold)
        mask = matrix.data > threshold_value
        return coo_matrix((matrix.data[mask].astype(int), (matrix.row[mask], matrix.col[mask])), shape=matrix.shape)

    def _bisto_seq(m, max_iter, tol):
        # Make a copy of the original matrix 'm' for later use
        _orig = m.copy()
    
        # Replace zero diagonals with ones to prevent potential scale explosion
        m = m.tolil()  # Convert matrix to LIL format for efficient indexing
        is_zero_diag = m.diagonal() == 0  # Check if there are zeros on the diagonal
        if np.any(is_zero_diag):
            # Set the diagonal elements that are zero to one
            ix = np.where(is_zero_diag)
            m[ix, ix] = 1
    
        # Ensure the matrix is in CSR format for efficient matrix operations
        if not isspmatrix_csr(m):
            m = m.tocsr()
    
        # Initialize variables
        n = m.shape[0]  # Number of rows (and columns, assuming square matrix)
        e = np.ones(n)  # A vector of ones
        x = e.copy()    # Initialize x as a copy of e
        delta = 0.1     # Lower bound for y
        Delta = 3       # Upper bound for y
        g = 0.9         # Step size factor
        etamax = 0.1    # Maximum eta value
        eta = etamax    # Initial eta
        stop_tol = tol * 0.5  # Stopping tolerance
        rt = tol ** 2        # Residual tolerance (squared)
        v = x * m.dot(x)     # Initial value for v (Ax)
        rk = 1 - v           # Residual (1 - Ax)
        rho_km1 = rk.T.dot(rk)  # Initial rho (dot product of rk and rk)
        rho_km2 = rho_km1       # Initialize rho_km2 with the same value as rho_km1
        rout = rho_km1        # Set rout as the initial rho
        rold = rout           # Previous value of rout
        n_iter = 0            # Initialize iteration counter
        i = 0                 # Inner loop iteration counter
        y = np.empty_like(e)  # Vector y for updating
    
        # Main loop for the iterative process
        while rout > rt and n_iter < max_iter:
            i += 1
            k = 0  # Inner loop counter
            y[:] = e  # Reset y to vector of ones
            inner_tol = max(rout * eta ** 2, rt)  # Set tolerance for inner loop
            
            # Inner loop for balancing the matrix
            while rho_km1 > inner_tol:
                k += 1
                if k == 1:
                    Z = rk / v  # Precompute Z
                    p = Z  # Set p to Z in the first iteration
                    rho_km1 = rk.T.dot(Z)  # Update rho
                else:
                    # Compute beta and update p in subsequent iterations
                    beta = rho_km1 / rho_km2
                    p = Z + beta * p
                
                # Compute w and alpha for the line search
                w = x * m.dot(x * p) + v * p
                alpha = rho_km1 / p.T.dot(w)  # Compute step size alpha
                ap = alpha * p  # Compute the step size applied to p
                ynew = y + ap  # Update y with the new value
                
                # Check for bound violations (either below delta or above Delta)
                if np.amin(ynew) <= delta:
                    if delta == 0:
                        break
                    ind = np.where(ap < 0)[0]
                    gamma = np.amin((delta - y[ind]) / ap[ind])
                    y += gamma * ap  # Adjust y to stay within bounds
                    break
                if np.amax(ynew) >= Delta:
                    ind = np.where(ynew > Delta)[0]
                    gamma = np.amin((Delta - y[ind]) / ap[ind])
                    y += gamma * ap  # Adjust y to stay within bounds
                    break
    
                y = ynew  # Update y if no bounds are violated
                rk = rk - alpha * w  # Update the residual
                rho_km2 = rho_km1  # Store the old value of rho
                Z = rk / v  # Update Z
                rho_km1 = np.dot(rk.T, Z)  # Update rho
    
                # Check for NaN values in x
                if np.any(np.isnan(x)):
                    raise RuntimeError('Scale vector has developed invalid values (NaNs)!')
            
            # Update x with the new y values
            x *= y
            v = x * m.dot(x)  # Update v
            rk = 1 - v  # Recalculate the residual
            rho_km1 = np.dot(rk.T, rk)  # Update rho
            rout = rho_km1  # Update the residual norm
            n_iter += k + 1  # Update the iteration count
            rat = rout / rold  # Compute the relative change in residual
            rold = rout  # Update old residual value
            res_norm = np.sqrt(rout)  # Compute the residual norm
            eta = g * rat  # Update eta
            eta = max(min(eta, etamax), stop_tol / res_norm)  # Enforce eta bounds
    
        # Check if the algorithm converged
        if n_iter >= max_iter:
            logger.error(f'Maximum number of iterations ({max_iter}) reached without convergence')
        
        # Return the balanced matrix and the scale vector 'x'
        X = spdiags(x, 0, n, n, 'csr')  # Create a diagonal matrix with x
        balanced_matrix = X.T.dot(_orig.dot(X)) * 1000000
        return balanced_matrix 

    try:
        if method == 'Raw':
            logger.info("Running Raw normalization.")
            return denoise(contact_matrix, threshold)

        elif method == 'normCC':
            logger.info("Running normCC normalization.")
            signal = contact_matrix.max(axis=1).toarray().ravel()
            coverage = contig_df['Contig coverage'].values
            contact_matrix.setdiag(0)

            df = contig_df.copy()
            df['Contig coverage'] = coverage
            df['signal'] = signal

            logger.info("Performing log transformations for normCC.")
            df['log_site'] = np.log(df['The number of restriction sites'] + epsilon)
            df['log_len'] = np.log(df['Contig length'])
            df['log_coverage'] = np.log(df['Contig coverage'] + epsilon)

            exog = df[['log_site', 'log_len', 'log_coverage']]
            exog = sm.add_constant(exog)
            endog = df['signal']
            glm_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(alpha=1))
            res = glm_nb.fit()

            expected_signal = np.exp(np.dot(exog, res.params))
            scal = np.max(expected_signal)

            normalized_data = [scal * v / np.sqrt(expected_signal[i] * expected_signal[j])
                               for i, j, v in zip(contact_matrix.row, contact_matrix.col, contact_matrix.data)]

            normalized_matrix = coo_matrix((normalized_data, (contact_matrix.row, contact_matrix.col)),
                                           shape=contact_matrix.shape)
            return denoise(normalized_matrix, threshold)

        elif method == 'HiCzin':
            logger.info("Running HiCzin normalization.")
            contact_matrix.setdiag(0)
            coverage = contig_df['Contig coverage'].replace(0, epsilon).values

            map_x = contact_matrix.row
            map_y = contact_matrix.col
            map_data = contact_matrix.data
            index = map_x < map_y
            map_x, map_y, map_data = map_x[index], map_y[index], map_data[index]

            sample_site = standardize(np.log(contig_df['The number of restriction sites'].iloc[map_x].values * contig_df['The number of restriction sites'].iloc[map_y].values))
            sample_len = standardize(np.log(contig_df['Contig length'].iloc[map_x].values * contig_df['Contig length'].iloc[map_y].values))
            sample_cov = standardize(np.log(coverage[map_x] * coverage[map_y]))

            data_hiczin = pd.DataFrame({
                'sample_site': sample_site,
                'sample_len': sample_len,
                'sample_cov': sample_cov,
                'sampleCon': map_data
            })

            exog = data_hiczin[['sample_site', 'sample_len', 'sample_cov']]
            exog = sm.add_constant(exog)
            endog = data_hiczin['sampleCon']

            glm_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(alpha=1))
            res = glm_nb.fit()

            expected_signal = np.exp(np.dot(exog, res.params))
            normalized_data = map_data / expected_signal

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (map_x, map_y)), shape=contact_matrix.shape
            )
            normalized_contact_matrix += normalized_contact_matrix.transpose()

            return denoise(normalized_contact_matrix, threshold)

        elif method == 'bin3C':
            logger.info("Running bin3C normalization.")
            num_sites = contig_df['The number of restriction sites'].values + epsilon
            normalized_data = [v / (num_sites[i] * num_sites[j])
                               for i, j, v in zip(contact_matrix.row, contact_matrix.col, contact_matrix.data)]

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (contact_matrix.row, contact_matrix.col)), shape=contact_matrix.shape
            )

            bistochastic_matrix = _bisto_seq(normalized_contact_matrix, max_iter, tolerance)
            return denoise(bistochastic_matrix, threshold)

        elif method == 'MetaTOR':
            logger.info("Running MetaTOR normalization.")
            signal = contact_matrix.diagonal() + epsilon
            normalized_data = [v / np.sqrt(signal[i] * signal[j])
                               for i, j, v in zip(contact_matrix.row, contact_matrix.col, contact_matrix.data)]

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (contact_matrix.row, contact_matrix.col)), shape=contact_matrix.shape
            )

            return denoise(normalized_contact_matrix, threshold)

    except Exception as e:
        logger.error(f"Error during {method} normalization: {e}")
        return None

def generating_bin_information(contig_info, contact_matrix, remove_unmapped_contigs=False, remove_host_host=False):
    # Ensure dense matrix for processing
    dense_matrix = contact_matrix.toarray()

    # Handle unmapped contigs
    if remove_unmapped_contigs:
        unmapped_contigs = contig_info[contig_info['Category'] == "unmapped"].index.tolist()
        contig_info = contig_info.drop(unmapped_contigs).reset_index(drop=True)
        
        # Mask for rows/columns to keep
        keep_mask = np.ones(dense_matrix.shape[0], dtype=bool)
        keep_mask[unmapped_contigs] = False
        dense_matrix = dense_matrix[keep_mask, :][:, keep_mask]

    # Aggregate bin data
    bin_info = contig_info.groupby('Bin index').agg({
        'Contig index': lambda x: ', '.join(x),
        'The number of restriction sites': 'sum',
        'Contig length': 'sum',
        'Contig coverage': lambda x: (x * contig_info.loc[x.index, 'Contig length']).sum() / contig_info.loc[x.index, 'The number of restriction sites'].sum(),
        'Within-contig Hi-C contacts': 'sum',
        'Category': 'first',
        'Domain': 'first',
        'Kingdom': 'first',
        'Phylum': 'first',
        'Class': 'first',
        'Order': 'first',
        'Family': 'first',
        'Genus': 'first',
        'Species': 'first'
    }).reset_index()
    
    bin_info['Contig coverage'] = bin_info['Contig coverage'].astype(float).map("{:.2f}".format)

    chromosome_rows = contig_info[contig_info['Category'] == 'chromosome']
    non_chromosome_rows = contig_info[contig_info['Category'] != 'chromosome']
    
    grouped_chromosome_rows = chromosome_rows.groupby('Bin index').agg({
        'Contig index': lambda x: ', '.join(x),
        'The number of restriction sites': 'sum',
        'Contig length': 'sum',
        'Contig coverage': lambda x: (x * chromosome_rows.loc[x.index, 'Contig length']).sum() / chromosome_rows.loc[x.index, 'The number of restriction sites'].sum(),
        'Within-contig Hi-C contacts': 'sum',
        'Category': 'first',
        'Domain': 'first',
        'Kingdom': 'first',
        'Phylum': 'first',
        'Class': 'first',
        'Order': 'first',
        'Family': 'first',
        'Genus': 'first',
        'Species': 'first'
    }).reset_index()
    
    grouped_chromosome_rows['Contig coverage'] = grouped_chromosome_rows['Contig coverage'].astype(float).map("{:.2f}".format)
    
    # Combine grouped chromosome rows and non-chromosome rows
    contig_info = pd.concat([grouped_chromosome_rows, non_chromosome_rows], ignore_index=True)
    
    # Create a mapping for temporary renaming
    rename_map = {
        'virus': 'a_virus',
        'plasmid': 'b_plasmid',
        'chromosome': 'c_chromosome'
    }
    reverse_map = {v: k for k, v in rename_map.items()}
    
    bin_info['Category'] = bin_info['Category'].replace(rename_map)
    contig_info['Category'] = contig_info['Category'].replace(rename_map)

    bin_info = bin_info.sort_values(by='Category', ascending=True)
    contig_info = contig_info.sort_values(by='Category', ascending=True)

    bin_info['Category'] = bin_info['Category'].replace(reverse_map)
    contig_info['Category'] = contig_info['Category'].replace(reverse_map)

    unique_bins = bin_info['Bin index']
    bin_indexes_dict = get_indexes(unique_bins, contig_info, 'Bin index')
    host_bin = bin_info[bin_info['Category'] == 'chromosome']['Bin index'].tolist()
    non_host_bin = bin_info[~bin_info['Category'].isin(['chromosome'])]['Bin index'].tolist()
    bin_contact_matrix = pd.DataFrame(0.0, index=unique_bins, columns=unique_bins)
    
    # Step 1: Get unique contig indices
    unique_contigs = contig_info['Contig index'].unique()
    contig_indexes_dict = get_indexes(unique_contigs, contig_info, 'Contig index')
    host_contig = contig_info[contig_info['Category'] == 'chromosome']['Contig index'].tolist()
    non_host_contig = contig_info[~contig_info['Category'].isin(['chromosome'])]['Contig index'].tolist()
    contig_contact_matrix = pd.DataFrame(0.0, index=unique_contigs, columns=unique_contigs)


    # Combine pairs of all necessary interactions
    if remove_host_host:
        bin_non_host_pairs = list(combinations(non_host_bin, 2))
        bin_non_host_self_pairs = [(x, x) for x in non_host_bin]
        bin_host_non_host_pairs = list(product(host_bin, non_host_bin))
        bin_all_pairs = bin_non_host_pairs + bin_non_host_self_pairs + bin_host_non_host_pairs
        
        chromosome_indices = contig_info[contig_info['Category'] == 'chromosome'].index.tolist()
        for i in chromosome_indices:
            for j in chromosome_indices:
                dense_matrix[i, j] = 0
        
        contig_non_host_pairs = list(combinations(non_host_contig, 2))
        contig_non_host_self_pairs = [(x, x) for x in non_host_contig]
        contig_host_non_host_pairs = list(product(host_contig, non_host_contig))
        contig_all_pairs = contig_non_host_pairs + contig_non_host_self_pairs + contig_host_non_host_pairs
        
        # Mask out chromosome-chromosome contacts in the dense matrix (zero out these values)
        chromosome_indices = contig_info[contig_info['Category'] == 'chromosome'].index.tolist()
        for i in chromosome_indices:
            for j in chromosome_indices:
                dense_matrix[i, j] = 0
    else:
        bin_non_self_pairs = list(combinations(unique_bins, 2))
        bin_self_pairs = [(x, x) for x in unique_bins]
        bin_all_pairs = bin_non_self_pairs + bin_self_pairs
        
        contig_non_self_pairs = list(combinations(unique_contigs, 2))
        contig_self_pairs = [(x, x) for x in unique_contigs]
        contig_all_pairs = contig_non_self_pairs + contig_self_pairs
        
    results = Parallel(n_jobs=-1)(
        delayed(calculate_submatrix_sum)(pair, bin_indexes_dict, dense_matrix) for pair in bin_all_pairs
    )
    
    for annotation_i, annotation_j, value in results:
        bin_contact_matrix.at[annotation_i, annotation_j] = value
        bin_contact_matrix.at[annotation_j, annotation_i] = value
        
    results = Parallel(n_jobs=-1)(
        delayed(calculate_submatrix_sum)(pair, contig_indexes_dict, dense_matrix) for pair in contig_all_pairs
    )
    
    # Step 2: Fill the contig contact matrix with the calculated values
    for annotation_i, annotation_j, value in results:
        contig_contact_matrix.at[annotation_i, annotation_j] = value
        contig_contact_matrix.at[annotation_j, annotation_i] = value

    # Convert to COO sparse matrices for storage
    bin_contact_matrix = coo_matrix(bin_contact_matrix)
    contig_contact_matrix = coo_matrix(contig_contact_matrix)

    return bin_info, contig_info, bin_contact_matrix, contig_contact_matrix

def create_normalization_layout():
    normalization_methods = [
        {'label': 'Raw - Removing contacts below a certain threshold for noise reduction.', 'value': 'Raw'},
        {'label': 'normCC - GLM-based normalization to adjust for varying coverage and signal.', 'value': 'normCC'},
        {'label': 'HiCzin - Logarithmic scaling method for normalizing Hi-C contact frequencies.', 'value': 'HiCzin'},
        {'label': 'bin3C - Kantorovich-Rubinstein regularization for balancing matrix rows and columns.', 'value': 'bin3C'},
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
        
        html.Div([
            html.Div([
                dcc.Checklist(
                    id='remove-unmapped-contigs',
                    options=[{'label': 'Remove Unmapped Contigs', 'value': 'remove_unmapped'}],
                    value=['remove_unmapped'],
                    style={'margin-right': '20px'}
                ),
                dcc.Checklist(
                    id='remove-host-host',
                    options=[{'label': 'Remove Host-Host Interactions', 'value': 'remove_host'}],
                    value=['remove_host'],
                )
            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),

            # Threshold input
            html.Div([
                html.Label("Threshold Percentage for Denoising (default: 5): Contacts below this percentile will be removed to reduce noise."),
                dcc.Input(
                    id='thres-input',
                    type='number',
                    value=5,
                    placeholder="Threshold percentage (0-100)",
                    style={'width': '100%'}
                )
            ], id='thres-container', className="my-3"),

            # Epsilon input
            html.Div([
                html.Label("Epsilon (default: 1): A small value added to avoid zero values in calculations."),
                dcc.Input(
                    id='epsilon-input',
                    type='number',
                    value=1,
                    placeholder="Epsilon value",
                    style={'width': '100%'}
                )
            ], id='epsilon-container', className="my-3"),

            # Max iterations input
            html.Div([
                html.Label("Maximum Iterations (default: 1000): Controls the number of iterations for the Sinkhorn-Knopp algorithm."),
                dcc.Input(
                    id='max-iter-input',
                    type='number',
                    value=1000,
                    placeholder="Maximum iterations for convergence",
                    style={'width': '100%'}
                )
            ], id='max-iter-container', className="my-3"),

            # Tolerance input
            html.Div([
                html.Label("Tolerance for Convergence (default: 1e-6): Defines the precision for convergence. Lower values increase precision."),
                dcc.Input(
                    id='tol-input',
                    type='number',
                    value=1e-6,
                    placeholder="Tolerance for convergence",
                    style={'width': '100%'}
                )
            ], id='tol-container', className="my-3")

        ], id='normalization-parameters', className="my-3")
    ])

    return layout

def register_normalization_callbacks(app):
    @app.callback(
        [Output('thres-container', 'style'),
         Output('epsilon-container', 'style'),
         Output('max-iter-container', 'style'),
         Output('tol-container', 'style')],
        Input('normalization-method', 'value')
    )

    def update_parameters(normalization_method):
        # Determine styles based on the selected normalization method
        thres_style = {'display': 'block'} if normalization_method in ['Raw', 'normCC', 'HiCzin', 'bin3C', 'MetaTOR'] else {'display': 'none'}
        epsilon_style = {'display': 'block'} if normalization_method in ['HiCzin', 'bin3C', 'MetaTOR'] else {'display': 'none'}
        max_iter_style = {'display': 'block'} if normalization_method == 'bin3C' else {'display': 'none'}
        tol_style = {'display': 'block'} if normalization_method == 'bin3C' else {'display': 'none'}
        
        return thres_style, epsilon_style, max_iter_style, tol_style

    @app.callback(
        [Output('normalization-status', 'data'),
         Output('blank-element', 'children', allow_duplicate=True)],
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
        
        contig_info, contact_matrix = preprocess_normalization(user_folder)
        
        if contig_info is None or contact_matrix is None:
            logger.error("Error reading files from folder. Please check the uploaded data.")
            return False, ""
    
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
            return False, ""
    
        logger.info(f"Normalization for {normalization_method} completed successfully.")
    
        # Perform bin information generation after normalization
        logger.info("Generating bin level information table and contact matrix...")
        bin_info, contig_info, bin_contact_matrix, contig_contact_matrix = generating_bin_information(
            contig_info,
            normalized_matrix,
            remove_unmapped_contigs,
            remove_host_host
        )
        logger.info("Bin information generation completed successfully.")
        
        # Define the output path and save the files individually
        user_output_path = f'output/{user_folder}'
        os.makedirs(user_output_path, exist_ok=True)
    
        bin_info_final_path = os.path.join(user_output_path, 'bin_info_final.csv')
        contig_info_final_path = os.path.join(user_output_path, 'contig_info_final.csv')
        bin_contact_matrix_path = os.path.join(user_output_path, 'bin_contact_matrix.npz')
        contig_contact_matrix_path = os.path.join(user_output_path, 'contig_contact_matrix.npz')
    
        # Save each file
        bin_info.to_csv(bin_info_final_path, index=False)
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
    
        logger.info("File saving completed successfully.")
    
        # Redis keys specific to each user folder
        bin_info_key = f'{user_folder}:bin-information'
        bin_matrix_key = f'{user_folder}:bin-dense-matrix'
        contig_info_key = f'{user_folder}:contig-information'
        contig_matrix_key = f'{user_folder}:contig-dense-matrix'
        
        # Save the loaded data to Redis with keys specific to the user folder
        save_to_redis(bin_info_key, bin_info)       
        save_to_redis(bin_matrix_key, bin_contact_matrix)
        save_to_redis(contig_info_key, contig_info)
        save_to_redis(contig_matrix_key, contig_contact_matrix)

        logger.info("Data loaded and saved to Redis successfully.")
        
        return True, ""