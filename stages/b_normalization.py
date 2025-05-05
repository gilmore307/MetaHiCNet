from dash import dcc, html, dash_table
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
from scipy.sparse import save_npz, load_npz
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

        # Define paths for the files within the folder
        contig_info_path = os.path.join(folder_path, 'contig_info_final.csv')
        contig_info = pd.read_csv(contig_info_path)
        
        contact_matrix_path = os.path.join(folder_path, 'unnormalized_contig_matrix.npz')
        contact_matrix = load_npz(contact_matrix_path).tocoo()

        return contig_info, contact_matrix

    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        return None, None

def run_normalization(method, contig_df, contact_matrix, alpha = 1, epsilon=1, threshold=5, max_iter=1000, tolerance=0.000001):
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
        
        # Mask values based on the threshold
        threshold_value = np.percentile(matrix.data, threshold)
        mask = matrix.data > threshold_value
        
        # Apply denoise: keeping values above threshold
        filtered_data = matrix.data[mask]
        filtered_rows = matrix.row[mask]
        filtered_cols = matrix.col[mask]
        
        # Normalize by dividing by the smallest non-zero value
        min_non_zero = np.min(filtered_data[filtered_data > 0])
        normalized_data = filtered_data / min_non_zero
        
        # Apply ceiling function and convert to integers
        normalized_data = np.ceil(normalized_data).astype(int)
        normalized_data = np.where(normalized_data < 0, 0, normalized_data)
        
        # Return the new sparse matrix with the normalized values
        return coo_matrix((normalized_data, (filtered_rows, filtered_cols)), shape=matrix.shape)

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
        matrix = X.T.dot(_orig.dot(X))
        return matrix 

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
            glm_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(alpha=alpha))
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

            glm_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(alpha=alpha))
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

def generating_bin_information(contig_info, contact_matrix, remove_unclassified_contigs=False, remove_host_host=False):
    dense_matrix = contact_matrix.toarray()

    # Handle unclassified contigs
    if remove_unclassified_contigs:
        unclassified_contigs = contig_info[contig_info['Category'] == "unclassified"].index.tolist()
        contig_info = contig_info.drop(unclassified_contigs).reset_index(drop=True)
        
        # Mask for rows/columns to keep
        keep_mask = np.ones(dense_matrix.shape[0], dtype=bool)
        keep_mask[unclassified_contigs] = False
        dense_matrix = dense_matrix[keep_mask, :][:, keep_mask]

    # Identify columns for aggregation
    known_agg = {
        'Contig ID': lambda x: ', '.join(x),
        'The number of restriction sites': 'sum',
        'Contig length': 'sum',
        'Contig coverage': lambda x: (
            (contig_info.loc[x.index, 'Contig length'] * x).sum() / contig_info.loc[x.index, 'Contig length'].sum()
        )
    }
    unknown_columns = [col for col in contig_info.columns if col not in known_agg]
    
    # Apply 'first' aggregation for unknown columns
    for col in unknown_columns:
        known_agg[col] = 'first'

    # Aggregate bin data
    bin_info = contig_info.groupby('Bin ID', as_index=False).agg(known_agg)
    bin_info['Contig coverage'] = bin_info['Contig coverage'].astype(float).map("{:.2f}".format)

    # Create a mapping for temporary renaming
    rename_map = {
        'virus': 'a_virus',
        'plasmid': 'b_plasmid',
        'chromosome': 'c_chromosome'
    }
    reverse_map = {v: k for k, v in rename_map.items()}
    
    bin_info['Category'] = bin_info['Category'].replace(rename_map)
    bin_info = bin_info.sort_values(by='Category', ascending=True)
    bin_info['Category'] = bin_info['Category'].replace(reverse_map)

    unique_bins = bin_info['Bin ID']
    bin_indexes_dict = get_indexes(unique_bins, contig_info, 'Bin ID')
    host_bin = bin_info[bin_info['Category'] == 'chromosome']['Bin ID'].tolist()
    non_host_bin = bin_info[~bin_info['Category'].isin(['chromosome'])]['Bin ID'].tolist()
    
    # Combine pairs of all necessary interactions
    if remove_host_host:
        bin_non_host_pairs = list(combinations(non_host_bin, 2))
        bin_host_non_host_pairs = list(product(host_bin, non_host_bin))
        bin_all_pairs = bin_non_host_pairs + bin_host_non_host_pairs
    else:
        bin_all_pairs = list(combinations(unique_bins, 2))

    results = Parallel(n_jobs=-1)( 
        delayed(calculate_submatrix_sum)(pair, bin_indexes_dict, dense_matrix) for pair in bin_all_pairs
    )

    # Initialize lists to store row indices, column indices, and values
    rows = []
    cols = []
    data = []
    
    for annotation_i, annotation_j, value in results:
        if annotation_i == annotation_j:
            value = 0  # Set value to 0 for diagonal elements (self-connections)
        
        # Add entries to COO format lists
        rows.append(annotation_i)
        cols.append(annotation_j)
        data.append(value)
        
        rows.append(annotation_j)
        cols.append(annotation_i)
        data.append(value)
    
    # Map annotations to numerical indices if necessary
    bin_index_to_position = {bin_index: idx for idx, bin_index in enumerate(unique_bins)}
    row_indices = [bin_index_to_position[annotation] for annotation in rows]
    col_indices = [bin_index_to_position[annotation] for annotation in cols]
    
    # Create the COO sparse matrix
    bin_contact_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(len(unique_bins), len(unique_bins)))
    
    bin_info=bin_info.reset_index(drop=True)
    
    bin_info['Connected bins'] = np.count_nonzero(bin_contact_matrix.toarray(), axis=1)
    bin_info['Visibility'] = 1

    return bin_info, bin_contact_matrix

def create_normalization_layout():
    methods = [
        {
            "Method": "Raw",
            "Model/Algorithm": "Threshold-based spurious contact removal",
            "What is this method?": "Does not remove effects of any factors from input Hi-C contact matrix; suitable if matrix is already normalized or normalization is unnecessary.",
            "How does it work?": "Removes values below a specified threshold percentile to denoise the data."
        },
        {
            "Method": "normCC [1]",
            "Model/Algorithm": "Negative Binomial Regression",
            "What is this method?": "Normalization module in MetaCC framework eliminating systematic biases like contig length, restriction sites, and coverage.",
            "How does it work?": "Employs negative binomial regression to adjust for systematic biases; spurious normalized contacts are removed based on a threshold."
        },
        {
            "Method": "HiCzin [2]",
            "Model/Algorithm": "Zero-Inflated Negative Binomial Regression",
            "What is this method?": "Specifically designed for metagenomic Hi-C data, addressing explicit and implicit biases using a zero-inflated model.",
            "How does it work?": "Combines negative binomial regression for explicit biases and a zero-inflated component for unobserved interactions; spurious contacts are removed based on a threshold."
        },
        {
            "Method": "bin3C [3]",
            "Model/Algorithm": "Cut-Site Normalization & Knight-Ruiz Algorithm",
            "What is this method?": "Pipeline for genome binning; ensures uniform signals in Hi-C contact maps by removing experimental biases.",
            "How does it work?": "Normalizes data via restriction site normalization and bistochastic matrix balancing; spurious contacts are removed based on a threshold."
        },
        {
            "Method": "MetaTOR [4]",
            "Model/Algorithm": "Coverage-based Normalization",
            "What is this method?": "Computational pipeline for metagenomic binning; corrects biases introduced by contig coverage variations.",
            "How does it work?": "Normalizes interaction counts by dividing Hi-C contact count by geometric mean of contig coverage; spurious contacts are removed based on a threshold."
        }
    ]

    methods = pd.DataFrame(methods)
    
    normalization_methods = [
        {'label': 'Raw', 'value': 'Raw'},
        {'label': 'normCC', 'value': 'normCC'},
        {'label': 'HiCzin', 'value': 'HiCzin'},
        {'label': 'bin3C', 'value': 'bin3C'},
        {'label': 'MetaTOR', 'value': 'MetaTOR'}
    ]

    layout = html.Div([
        html.H2("Normalization Setups", className="mt-4"),
        html.Div([
            html.Label("Select Normalization Method:"),
            dcc.Dropdown(
                id='normalization-method',
                options=normalization_methods,
                value='Raw',
                style={'width': '100%'}
            )
        ], className="my-3"),
        
        html.Div([
            html.Div([
                html.Label("Threshold Percentage for Spurious Contact Removal (default: 5%)"),
                dcc.Input(
                    id='thres-input',
                    type='number',
                    value=5,
                    placeholder="Threshold percentage (0-100)%",
                    style={'width': '100%'}
                )
            ], id='thres-container', className="my-3"),
            
            # alpha input
            html.Div([
                html.Label("Overdispersion Parameter for Negative Binomial Regression (default: 1)"),
                dcc.Input(
                    id='alpha-input',
                    type='number',
                    value=1,
                    placeholder="Overdispersion parameter for negative binomial regression",
                    style={'width': '100%'}
                )
            ], id='alpha-container', className="my-3"),

            # Max iterations input
            html.Div([
                html.Label("Maximum Iterations (default: 1000)"),
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
                html.Label("Tolerance for Convergence (default: 1e-6)"),
                dcc.Input(
                    id='tol-input',
                    type='number',
                    value=1e-6,
                    placeholder="Tolerance for convergence",
                    style={'width': '100%'}
                )
            ], id='tol-container', className="my-3")

        ], id='normalization-parameters', className="my-3"),
        
        html.H4("Comparison of Hi-C Normalization Methods", className="mt-2"),

        dash_table.DataTable(
            id='methods-table',
            columns=[
                {"name": "Method", "id": "Method"},
                {"name": "Model/Algorithm", "id": "Model/Algorithm"},
                {"name": "What is this method?", "id": "What is this method?"},
                {"name": "How does it work?", "id": "How does it work?"}
            ],
            data=methods.to_dict('records'),
            style_table={'overflowY': 'auto'},
            style_header={'backgroundColor': 'rgb(210, 210, 210)', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Arial'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            style_cell_conditional=[{"if": {"column_id": "Method"}, "width": "120px"}]
        ),
        
        html.Div([
            html.H5("References:", style={'font-weight': 'bold'}),
            html.Ul([
                html.Li("[1] Du, Y., & Sun, F. (2023). MetaCC allows scalable and integrative analyses of both long-read and short-read metagenomic Hi-C data. Nature Communications, 14, 6231."),
                html.Li("[2] Du, Y., Laperriere, S. M, et al. (2022). Normalizing metagenomic Hi-C data and detecting spurious contacts using zero-inflated negative binomial regression. Journal of Computational Biology, 29 (2), 106-120."),
                html.Li("[3] DeMaere, M. Z., & Darling, A. E. (2019). bin3C: exploiting Hi-C sequencing data to accurately resolve metagenome-assembled genomes. Genome biology, 20, 46."),
                html.Li("[4] Baudry, L., Foutel-Rodier, T., et al. (2019). MetaTOR: a computational pipeline to recover high-quality metagenomic bins from mammalian gut proximity-ligation (meta3C) libraries. Frontiers in genetics, 10, 753.")
            ], style={'font-size': 'small', 'line-height': '1.5'}),
        ], style={'font-style': 'italic', 'marginTop': '20px'}),

        html.Hr(),
        html.H2("Additional Options", className="mt-4", style={'marginTop': '40px'}),
        
        html.P([
            html.Strong("Important Note:"), " Please do not enable the the following options if you have not provided the Binning Information File and Taxonomy Information File during the data upload process. These files are required to process these options correctly."
        ], className="mt-3"),  
        
        html.Div([
            html.Div([
                dcc.Checklist(
                    id='remove-unclassified-contigs',
                    options=[{'label': '  Remove Unclassified Contigs', 'value': 'remove_unclassified'}],
                    value=['remove_unclassified'],
                    style={'margin-bottom': '10px'}
                ),
                html.P([
                    "Check this box to exclude contigs or bins that are not classified in any taxonomic levels. "
                    "Enabling this option helps reduce the size of your dataset and speeds up processing."
                ], className="mb-3")
            ], style={'margin-bottom': '20px'}),  # First row spacing
        
            html.Div([
                dcc.Checklist(
                    id='remove-host-host',
                    options=[{'label': '  Remove Host-Host Interactions', 'value': 'remove_host'}],
                    value=['remove_host'],
                    style={'margin-bottom': '10px'}
                ),
                html.P([
                    "Check this box to remove all interactions between contigs or bins labeled as chromosomes. "
                    "Enabling this option can significantly reduce the amount of Hi-C contacts and accelerate processing."
                ], className="mb-3")
            ], style={'margin-bottom': '20px'})  # Second row spacing
        ], style={'display': 'block', 'margin-bottom': '10px'})
    ])

    return layout

def register_normalization_callbacks(app):
    @app.callback(
        [Output('thres-container', 'style'),
         Output('alpha-container', 'style'),
         Output('max-iter-container', 'style'),
         Output('tol-container', 'style')],
        Input('normalization-method', 'value')
    )

    def update_parameters(normalization_method):
        # Determine styles based on the selected normalization method
        thres_style = {'display': 'block'} if normalization_method in ['Raw', 'normCC', 'HiCzin', 'bin3C', 'MetaTOR'] else {'display': 'none'}
        alpha_style = {'display': 'block'} if normalization_method in ['normCC', 'HiCzin'] else {'display': 'none'}
        max_iter_style = {'display': 'block'} if normalization_method == 'bin3C' else {'display': 'none'}
        tol_style = {'display': 'block'} if normalization_method == 'bin3C' else {'display': 'none'}
        
        return thres_style, alpha_style, max_iter_style, tol_style

    @app.callback(
        [Output('normalization-status', 'data'),
         Output('blank-element', 'children', allow_duplicate=True)],
        [Input('execute-button', 'n_clicks')],
        [State('normalization-method', 'value'),
         State('thres-input', 'value'),
         State('alpha-input', 'value'),
         State('max-iter-input', 'value'),
         State('tol-input', 'value'),
         State('remove-unclassified-contigs', 'value'),
         State('remove-host-host', 'value'),
         State('user-folder', 'data'),
         State('current-method', 'data'),
         State('current-stage', 'data')],
        prevent_initial_call=True
    )

    def execute_normalization(n_clicks, normalization_method, threshold, alpha, max_iter, tolerance,
                              remove_unclassified_contigs, remove_host_host, user_folder, selected_method, current_stage):
        # Only trigger if in the 'Normalization' stage for the selected methods
        if not n_clicks or selected_method not in ['method1', 'method2'] or current_stage != 'Normalization':
            raise PreventUpdate
        
        logger.info(f"Running normalization for {selected_method} using {normalization_method}...")
    
        # Set default values if states are missing
        threshold = threshold if threshold is not None else 5
        max_iter = max_iter if max_iter is not None else 1000
        tolerance = tolerance if tolerance is not None else 1e-6
    
        # Convert checkbox values to booleans
        remove_unclassified_contigs = 'remove_unclassified' in remove_unclassified_contigs
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
            "alpha": alpha,
            "epsilon": 1,
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
        bin_info, bin_contact_matrix = generating_bin_information(
            contig_info,
            normalized_matrix,
            remove_unclassified_contigs,
            remove_host_host
        )
        logger.info("Bin information generation completed successfully.")
        
        # Define the output path and save the files individually
        user_output_path = f'output/{user_folder}'
        os.makedirs(user_output_path, exist_ok=True)
    
        bin_info_final_path = os.path.join(user_output_path, 'bin_info_final.csv')
        bin_contact_matrix_path = os.path.join(user_output_path, 'normalized_bin_matrix.npz')
        contig_info_path = os.path.join(user_output_path, 'contig_info_final.csv')
        normalized_matrix_path = os.path.join(user_output_path, 'normalized_contig_matrix.npz')
        unnormalized_matrix_path = os.path.join(user_output_path, 'unnormalized_contig_matrix.npz')
    
        # Save each file
        bin_info.to_csv(bin_info_final_path, index=False)
        save_npz(bin_contact_matrix_path, bin_contact_matrix)
        contig_info.to_csv(contig_info_path, index=False)
        save_npz(normalized_matrix_path, normalized_matrix)

        # Compress saved files into normalized_information.7z
        normalized_archive_path = os.path.join(user_output_path, 'normalized_information.7z')
        with py7zr.SevenZipFile(normalized_archive_path, 'w') as archive:
            archive.write(bin_info_final_path, 'bin_info_final.csv')
            archive.write(bin_contact_matrix_path, 'normalized_bin_matrix.npz')
            archive.write(contig_info_path, 'contig_info_final.csv')
            archive.write(normalized_matrix_path, 'normalized_contig_matrix.npz')
            archive.write(unnormalized_matrix_path, 'unnormalized_contig_matrix.npz')
    
        logger.info("File saving completed successfully.")
    
        # Save the loaded data to Redis with keys specific to the user folder
        bin_info_key = f'{user_folder}:bin-information'
        bin_matrix_key = f'{user_folder}:bin-dense-matrix'
        
        save_to_redis(bin_info_key, bin_info)       
        save_to_redis(bin_matrix_key, bin_contact_matrix)
        
        logger.info("Data loaded and saved to Redis successfully.")
        
        return True, ""