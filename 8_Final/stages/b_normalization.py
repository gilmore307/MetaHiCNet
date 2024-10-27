from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import os
import plotly.express as px
from scipy.sparse import coo_matrix, load_npz, spdiags
import statsmodels.api as sm
import logging

# Set up logging
logger = logging.getLogger("app_logger")

# Function to create the layout for normalization dynamically
def create_normalization_layout(method_id):
    normalization_methods = ['Raw', 'normCC', 'HiCzin', 'bin3C', 'MetaTOR']

    layout = html.Div([
        html.Div([
            html.Label("Select Normalization Method:"),
            dcc.Dropdown(
                id=f'normalization-method-{method_id}',
                options=[{'label': method, 'value': method} for method in normalization_methods],
                value='Raw',  # Default value
                style={'width': '100%'}
            )
        ], className="my-3"),

        html.Div(id=f'normalization-parameters-{method_id}', className="my-3"),  # Placeholder for dynamic input fields

        html.Div(id=f'dynamic-heatmap-method-{method_id}', className="my-3")  # Heatmap placeholder
    ])

    return layout

def preprocess(user_folder, folder_name='unnormalized_information', assets_folder='output'):
    try:
        # Locate the folder path
        folder_path = os.path.join('assets', assets_folder, user_folder, folder_name)

        # Define paths for the files within the folder
        contig_info_path = os.path.join(folder_path, 'contig_info_final.csv')
        contact_matrix_path = os.path.join(folder_path, 'raw_contact_matrix.npz')

        # Read the contig information file as a pandas DataFrame
        contig_info = pd.read_csv(contig_info_path)

        # Read the contact matrix as a sparse matrix
        contact_matrix = load_npz(contact_matrix_path)

        return contig_info, contact_matrix

    except Exception as e:
        logging.error(f"Error reading files from folder: {e}")
        return None, None

def denoise(matrix, threshold):
    threshold_value = np.percentile(matrix.data, threshold)
    mask = matrix.data > threshold_value
    return coo_matrix((matrix.data[mask], (matrix.row[mask], matrix.col[mask])), shape=matrix.shape)

def run_normalization(method, contig_info, contact_matrix, epsilon, threshold, max_iter):
    try:
        logging.info(f"Running {method} normalization...")

        if method == 'Raw':
            return denoise(contact_matrix, threshold)

        elif method == 'normCC':
            signal = contact_matrix.max(axis=1).toarray().ravel()
            site = contig_info['sites'].values
            length = contig_info['length'].values
            covcc = contact_matrix.diagonal()
            contact_matrix.setdiag(0)

            df = pd.DataFrame({
                'site': site,
                'length': length,
                'covcc': covcc,
                'signal': signal
            })

            df['log_site'] = np.log(df['site'] + epsilon)
            df['log_len'] = np.log(df['length'])
            df['log_covcc'] = np.log(df['covcc'] + epsilon)

            exog = df[['log_site', 'log_len', 'log_covcc']]
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
            contact_matrix.setdiag(0)
            site = contig_info['sites'].values
            length = contig_info['length'].values
            coverage = contig_info['coverage'].replace(0, epsilon).values

            map_x = contact_matrix.row
            map_y = contact_matrix.col
            map_data = contact_matrix.data
            index = map_x < map_y
            map_x, map_y, map_data = map_x[index], map_y[index], map_data[index]

            sample_site = np.log(site[map_x] * site[map_y])
            sample_len = np.log(length[map_x] * length[map_y])
            sample_cov = np.log(coverage[map_x] * coverage[map_y])

            sample_site = standardize(sample_site)
            sample_len = standardize(sample_len)
            sample_cov = standardize(sample_cov)

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
            num_sites = contig_info['sites'].values + epsilon
            normalized_data = [v / (num_sites[i] * num_sites[j])
                               for i, j, v in zip(contact_matrix.row, contact_matrix.col, contact_matrix.data)]

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (contact_matrix.row, contact_matrix.col)), shape=contact_matrix.shape
            )

            bistochastic_matrix, _ = _bisto_seq(normalized_contact_matrix, max_iter, 1e-6)
            return denoise(bistochastic_matrix, threshold)

        elif method == 'MetaTOR':
            signal = contact_matrix.diagonal() + epsilon
            normalized_data = [v / np.sqrt(signal[i] * signal[j])
                               for i, j, v in zip(contact_matrix.row, contact_matrix.col, contact_matrix.data)]

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (contact_matrix.row, contact_matrix.col)), shape=contact_matrix.shape
            )

            return denoise(normalized_contact_matrix, threshold)

    except Exception as e:
        logging.error(f"Error during {method} normalization: {e}")
        return None

def standardize(array):
    std = np.std(array)
    return np.zeros_like(array) if std == 0 else (array - np.mean(array)) / std

def _bisto_seq(m, max_iter, tol, x0=None, delta=0.1, Delta=3):
    logging.info("Starting bistochastic matrix balancing.")
    _orig = m.copy()
    m = m.tolil()
    is_zero_diag = m.diagonal() == 0
    if np.any(is_zero_diag):
        m.setdiag(np.where(is_zero_diag, 1, m.diagonal()))

    m = m.tocsr()
    n = m.shape[0]
    e = np.ones(n)
    x = x0.copy() if x0 is not None else e.copy()
    for _ in range(max_iter):
        x_new = 1 / (m @ (x ** 2))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    X = spdiags(x, 0, n, n, format='csr')
    return X.T @ _orig @ X, x

def register_normalization_callbacks(app, method_id):
    @app.callback(
        [Output(f'normalization-parameters-{method_id}', 'children')],
        [Input(f'normalization-method-{method_id}', 'value')]
    )
    def update_parameters(selected_method):
        # Default input components for different methods
        if selected_method == 'Raw':
            return html.Div([
                html.Small("No additional parameters required for Raw normalization.", className="text-muted")
            ])
        elif selected_method == 'normCC':
            return html.Div([
                html.Label("Epsilon (default: 1):"),
                dcc.Input(
                    id=f'epsilon-input-{method_id}',
                    type='number',
                    value=1,
                    placeholder="Epsilon value",
                    style={'width': '100%'}
                ),
                html.Small("Epsilon is used to avoid zero values in log transformations.", className="form-text text-muted")
            ])
        elif selected_method == 'HiCzin':
            return html.Div([
                html.Label("Epsilon (default: 1):"),
                dcc.Input(
                    id=f'epsilon-input-{method_id}',
                    type='number',
                    value=1,
                    placeholder="Epsilon value",
                    style={'width': '100%'}
                ),
                html.Small("Epsilon is used to avoid zero values in log transformations.", className="form-text text-muted"),

                html.Label("Threshold Percentage (default: 5):"),
                dcc.Input(
                    id=f'threshold-input-{method_id}',
                    type='number',
                    value=5,
                    placeholder="Threshold percentage for denoising",
                    style={'width': '100%'}
                ),
                html.Small("Percentage to define the threshold for contact filtering (0-100).", className="form-text text-muted")
            ])
        elif selected_method == 'bin3C':
            return html.Div([
                html.Label("Epsilon (default: 1):"),
                dcc.Input(
                    id=f'epsilon-input-{method_id}',
                    type='number',
                    value=1,
                    placeholder="Epsilon value",
                    style={'width': '100%'}
                ),
                html.Small("Epsilon is used to avoid zero values in log transformations.", className="form-text text-muted"),

                html.Label("Max Iterations (default: 1000):"),
                dcc.Input(
                    id=f'max-iter-input-{method_id}',
                    type='number',
                    value=1000,
                    placeholder="Maximum iterations for convergence",
                    style={'width': '100%'}
                ),
                html.Small("Used in bin3C normalization for the Sinkhorn-Knopp algorithm.", className="form-text text-muted")
            ])
        elif selected_method == 'MetaTOR':
            return html.Div([
                html.Label("Epsilon (default: 1):"),
                dcc.Input(
                    id=f'epsilon-input-{method_id}',
                    type='number',
                    value=1,
                    placeholder="Epsilon value",
                    style={'width': '100%'}
                ),
                html.Small("Epsilon is used to avoid zero values in log transformations.", className="form-text text-muted")
            ])
        else:
            return no_update

    @app.callback(
        [Output(f'normalization-status-method-{method_id}', 'data'),
         Output(f'dynamic-heatmap-method-{method_id}', 'children')],
        [Input(f'execute-normalization-button-{method_id}', 'n_clicks')],
        [State(f'normalization-method-{method_id}', 'value'),
         State(f'epsilon-input-{method_id}', 'value'),
         State(f'threshold-input-{method_id}', 'value'),
         State(f'max-iter-input-{method_id}', 'value'),
         State('user-folder', 'data')],
        prevent_initial_call=True
    )
    def execute_normalization(n_clicks, method, epsilon, threshold, max_iter, user_folder):
        if n_clicks is None:
            return no_update, no_update
    
        # Use the preprocess function to read data from the folder
        contig_info, contact_matrix = preprocess(user_folder)
    
        if contig_info is None or contact_matrix is None:
            return False, html.Div("Error reading files from folder. Please check the uploaded data.")
    
        # Run the normalization
        normalized_matrix = run_normalization(method, contig_info, contact_matrix, epsilon, threshold, max_iter)
    
        if normalized_matrix is None or normalized_matrix.nnz == 0:
            return False, html.Div("Normalization failed or produced an empty matrix.")
    
        dense_matrix = normalized_matrix.toarray()
        fig = px.imshow(dense_matrix, color_continuous_scale='Viridis')
        heatmap = dcc.Graph(figure=fig)
    
        return True, heatmap
