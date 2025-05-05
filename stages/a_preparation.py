import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import io
import os
import py7zr
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz, coo_matrix, csr_matrix, csc_matrix
import logging
from stages.helper import (
    save_file_to_user_folder,
    save_to_redis)

# Initialize logger
logger = logging.getLogger("app_logger")

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    if 'csv' in filename:
        # Load CSV file
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    elif 'txt' in filename:
        # Convert txt to DataFrame assuming each line represents a row and values are space/comma separated
        text = decoded.decode('utf-8')
        data = [line.split() for line in text.splitlines()]
        return pd.DataFrame(data)
    
    elif 'npz' in filename:
        # Load the npz file
        sparse_matrix = load_npz(io.BytesIO(decoded))
        return sparse_matrix.tocoo() 
    
    elif '7z' in filename:
        # Load and extract .7z archive
        with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as z:
            return z.getnames()
    
    else:
        raise ValueError("Unsupported file format.")

def get_file_size(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    size_in_bytes = len(decoded)
    size_in_kb = size_in_bytes / 1024
    return f"{size_in_kb:.2f} KB"

def validate_csv(df, required_columns, optional_columns=[]):
    all_columns = required_columns + optional_columns
    if not all(column in df.columns for column in all_columns):
        missing_cols = set(required_columns) - set(df.columns)
        logger.error(f"Missing columns in the CSV file: {missing_cols}")
        raise ValueError(f"Missing columns in the file: {missing_cols}")
    for col in required_columns:
        if df[col].isnull().any():
            logger.error(f"Required column '{col}' has missing values.")
            raise ValueError(f"Required column '{col}' has missing values.")
    return True

def validate_contig_matrix(contig_data, contact_matrix):
    num_contigs = len(contig_data)

    # Check if contact_matrix is an NPZ file containing COO matrix components
    if isinstance(contact_matrix, np.lib.npyio.NpzFile):
        if all(key in contact_matrix for key in ['data', 'row', 'col', 'shape']):
            data = contact_matrix['data']
            row = contact_matrix['row']
            col = contact_matrix['col']
            shape = tuple(contact_matrix['shape'])
            contact_matrix = coo_matrix((data, (row, col)), shape=shape)
        else:
            logger.error("The contact matrix file does not contain the expected COO matrix keys.")
            raise ValueError("The contact matrix file does not contain the expected COO matrix keys.")
    
    # Validate matrix shape
    matrix_shape = contact_matrix.shape
    if matrix_shape[0] != matrix_shape[1]:
        logger.error("The contact matrix is not square.")
        raise ValueError("The contact matrix is not square.")
    if matrix_shape[0] != num_contigs:
        logger.error(f"The contact matrix dimensions {matrix_shape} do not match the number of contigs.")
        raise ValueError(f"The contact matrix dimensions {matrix_shape} do not match the number of contigs.")
    
    return True

def list_files_in_7z(decoded):
    with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as z:
        file_list = z.getnames()
    return file_list

def adjust_taxonomy(row, taxonomy_columns):
    # Define prefixes for taxonomy tiers
    prefixes = {col: f"{col[0].lower()}_" for col in taxonomy_columns}

    if all(pd.isna(row[col]) for col in taxonomy_columns):
        row['Category'] = 'unclassified'
        
    if row['Category'] == 'virus':
        for tier in taxonomy_columns:
            if not pd.isna(row[tier]):
                   row[tier] = row[tier] + '_v'

    elif row['Category'] == 'plasmid':
        for tier in taxonomy_columns:
            if not pd.isna(row[tier]):
                row[tier] = row[tier] + '_p'
        
    for tier, prefix in prefixes.items():
        if not pd.isna(row[tier]):
            row[tier] = f"{prefix}{row[tier]}"
        else:
            row[tier] = f"{prefix}"
                
    return row

def process_data(contig_data, binning_data, taxonomy_data, contig_matrix, taxonomy_columns):
    try:
        logger.info("Starting data preparation...")
        
        if isinstance(contig_matrix, str):
            logger.error("contig_matrix is a string, which is unexpected. Please check the source.")
            raise ValueError("contact_matrix should be a sparse matrix, not a string.")
        
        # Ensure the contact_matrix is in the correct sparse format if it's not already
        if not isinstance(contig_matrix, coo_matrix):
            logger.error("contig_matrix is not a COO sparse matrix.")
            raise ValueError("contig_matrix must be a COO sparse matrix.")

        # Merge contig, binning, and taxonomy data
        combined_data = pd.merge(contig_data, binning_data, on='Contig ID', how="left")
        combined_data['Bin ID'] = combined_data.apply(
            lambda row: row['Contig ID'] if pd.isna(row['Bin ID']) else row['Bin ID'], axis=1)
        
        combined_data = pd.merge(combined_data, taxonomy_data, on='Bin ID', how="left")

        # Apply taxonomy adjustments
        combined_data = combined_data.apply(lambda row: adjust_taxonomy(row, taxonomy_columns), axis=1)

        # Fill missing bins with Contig index
        combined_data['Bin ID'] = combined_data['Bin ID'].fillna(combined_data['Contig ID'])
        
        # Remove duplicated Contig IDs, keep only the first occurrence
        combined_data = combined_data.drop_duplicates(subset='Contig ID', keep='first')

        # Return the processed combined data directly
        logger.info("Data processed successfully.")
        return combined_data

    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        return None

def create_upload_component(component_id, text, example_url, instructions):
    return dbc.Card(
        [            
            dcc.Upload(
                id=component_id,
                children=dbc.Button(text, color="primary", className="me-2", style={"width": "100%"}),
                multiple=False,
                style={'textAlign': 'center'}
            ),
            dbc.Row(
                [
                    dbc.Col(html.A('Download Example File', href=example_url, target='_blank', style={'textAlign': 'center'})),
                ],
                style={'padding': '5px'}
            ),
            dbc.CardBody([
                html.H6("Instructions:"),
                dcc.Markdown(instructions, style={'fontSize': '0.9rem', 'color': '#555'})
            ]),
            html.Div(id=f'overview-{component_id}', style={'padding': '10px'}),
            dbc.Button("Remove File", id=f'remove-{component_id}', color="danger", style={'display': 'none'}),
            dcc.Store(id=f'store-{component_id}')
        ],
        body=True,
        className="my-3"
    )

# Layouts for different methods
def create_upload_layout_method1():
    return html.Div([
        dbc.Row([
            dbc.Col(create_upload_component(
                'raw-contig-info', 
                'Upload Contig Information File (.csv)', 
                'assets/examples/contig_information.csv',
                """
                The contig information file includes the following columns:
                - **‘Contig ID**, **‘Number of restriction sites’**, and **‘Contig length’** (required).
                - **‘Contig coverage’** (optional): If not provided, it will be estimated by dividing the diagonal value in the raw Hi-C contact matrix by the ‘Contig length’.
        
                This file can be directly generated from common Meta Hi-C analysis pipelines, such as MetaCC and HiCBin.
                """
            )),
            dbc.Col(create_upload_component(
                'raw-contig-matrix', 
                'Upload Raw Hi-C Contact Matrix File (.txt, .csv or .npz)', 
                'assets/examples/raw_contact_matrix.npz',
                """
                The contact matrix can be provided in one of the following formats: .txt, .csv, or .npz.
                
                - **In .txt or .csv format**: The file should contain the columns **‘Contig ID1’**, **‘Contig ID2’**, and **‘Contacts’**.
                - **In .npz format**: The file should be either a NumPy dense matrix or a SciPy sparse matrix.
                
                This file can be directly generated from common Meta Hi-C analysis pipelines, such as MetaCC and HiCBin.
                
                **Note**: The row and column indices of the Hi-C Contact Matrix must match the row indices of the Contig Information File.
                """
            ))
        ]),
        dbc.Row([
            dbc.Col(create_upload_component(
                'raw-binning-info', 
                'Upload Binning Information File (.csv) (Optional)', 
                'assets/examples/binning_information.csv',
                """
                The binning information file is optional. You can skip it if your goal is solely to normalize the Hi-C contact matrix.
        
                It contains the following columns:
                - **‘Contig ID’** and **‘Bin ID’** (specifying the bin to which each contig belongs).
        
                This file can be directly obtained from the binning results of Meta Hi-C analysis pipelines or any other binners users select.
                """
            )),
            dbc.Col(create_upload_component(
                'raw-bin-taxonomy', 
                'Upload Taxonomy Information File (.csv) (Optional)', 
                'assets/examples/taxonomy_information.csv',
                """
                The taxonomy information file is optional. You can skip it if your goal is solely to normalize the Hi-C contact matrix.
        
                It contains the following columns:
                - **‘ID’** 
                - **‘Category’**: The taxonomic category of each bin, which can be one of the following: **‘chromosome’**, **‘virus’**, or **‘plasmid’**. If a bin cannot be assigned to a specific category, it should be marked as **‘Unclassified’** or left blank.
                - Any number of additional **Taxonimc Classification Columns** providing further taxonomic information for each bin or contig (e.g., family, genus, species).
                """
            ))
        ])
    ])

def create_upload_layout_method2():
    return html.Div([
        dbc.Row([
            dbc.Col(create_upload_component(
                'unnormalized-data-folder', 
                'Upload Unnormalized Data Folder (.7z)', 
                'assets/examples/unnormalized_information.7z',
                "This is a compresssed folder containing the following files: 'contig_info_final.csv' and 'unnormalized_contig_matrix.npz'.  \n"
                "Download this file in Hi-C Contact Normalization Results page."
            ))
        ])
    ])

def create_upload_layout_method3():
    return html.Div([
        dbc.Row([
            dbc.Col(create_upload_component(
                'normalized-data-folder', 
                'Upload Visualization Data Folder (.7z)', 
                'assets/examples/normalized_information.7z',
                "This is a compresssed folder containing the following files: 'bin_info_final.csv', 'contig_info_final.csv', 'unnormalized_contig_matrix.npz.npz', 'normalized_contig_matrix.npz.npz', 'normalized_bin_matrix.npz'.  \n"
                "Download this file in Hi-C Contact Normalization Results page."
            ))
        ])
    ])

def register_preparation_callbacks(app):
    # Callback for handling raw contig info upload with logging
    @app.callback(
        [Output('overview-raw-contig-info', 'children'),
         Output('remove-raw-contig-info', 'style'),
         Output('raw-contig-info', 'contents')],
        [Input('raw-contig-info', 'contents'),
         Input('remove-raw-contig-info', 'n_clicks')],
        [State('raw-contig-info', 'filename')],
        prevent_initial_call=True
    )
    def handle_contig_info_upload(contents, remove_click, filename):
        ctx = dash.callback_context
        if not contents:
            raise PreventUpdate
        
        if remove_click and ctx.triggered_id == 'remove-raw-contig-info':
            logger.info("Contig Info file removed.")
            return '', {'display': 'none'}, None
        
        file_size = get_file_size(contents)
        if 'csv' in filename:
            df = parse_contents(contents, filename)
            logger.info(f"Uploaded Contig Info: {filename} with size {file_size}.")
            return [
                dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                html.P(f"File Size: {file_size}")
            ], {'display': 'block'}, contents
        
        logger.error("Unsupported file format for Contig Info.")
        return "Unsupported file format", {'display': 'block'}, contents

    @app.callback(
        [Output('overview-raw-contig-matrix', 'children'),
         Output('remove-raw-contig-matrix', 'style'),
         Output('raw-contig-matrix', 'contents')],
        [Input('raw-contig-matrix', 'contents'),
         Input('remove-raw-contig-matrix', 'n_clicks')],
        [State('raw-contig-matrix', 'filename')],
        prevent_initial_call=True
    )
    def handle_raw_matrix_upload(contents, remove_click, filename):
        ctx = dash.callback_context
        if not contents:
            raise PreventUpdate
    
        if remove_click and ctx.triggered_id == 'remove-raw-contig-matrix':
            logger.info("Raw Contact Matrix file removed.")
            return '', {'display': 'none'}, None
    
        file_size = get_file_size(contents)
    
        try:
            # Parse contents using the updated parse_contents function
            parsed_data = parse_contents(contents, filename)
    
            if isinstance(parsed_data, coo_matrix):
                # Display COO matrix keys and their information
                matrix_info = [
                    html.Li(f"data: Array with shape {parsed_data.data.shape}"),
                    html.Li(f"row: Array with shape {parsed_data.row.shape}"),
                    html.Li(f"col: Array with shape {parsed_data.col.shape}"),
                    html.Li(f"shape: {parsed_data.shape}")
                ]
    
                overview = html.Ul(matrix_info)
                logger.info(f"Uploaded Raw Matrix: {filename} with size {file_size}.")
                return [overview, html.P(f"File Size: {file_size}")], {'display': 'block'}, contents
    
            elif isinstance(parsed_data, pd.DataFrame):
                # Display the first few rows as an overview
                overview = [
                    dbc.Table.from_dataframe(parsed_data.head(), striped=True, bordered=True, hover=True),
                    html.P(f"File Size: {file_size}")
                ]
                logger.info(f"Uploaded Raw Matrix: {filename} with size {file_size}.")
                return overview, {'display': 'block'}, contents
    
            else:
                logger.error("Unsupported data type parsed from the file.")
                return "Unsupported file format", {'display': 'block'}, None
    
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return f"Error processing file: {e}", {'display': 'block'}, None

    # Callback for handling binning info upload with logging
    @app.callback(
        [Output('overview-raw-binning-info', 'children'),
         Output('remove-raw-binning-info', 'style'),
         Output('raw-binning-info', 'contents')],
        [Input('raw-binning-info', 'contents'),
         Input('remove-raw-binning-info', 'n_clicks')],
        [State('raw-binning-info', 'filename')],
        prevent_initial_call=True
    )
    def handle_binning_info_upload(contents, remove_click, filename):
        ctx = dash.callback_context
        if not contents:
            raise PreventUpdate
        
        if remove_click and ctx.triggered_id == 'remove-raw-binning-info':
            logger.info("Binning Info file removed.")
            return '', {'display': 'none'}, None
        
        file_size = get_file_size(contents)
        if 'csv' in filename:
            df = parse_contents(contents, filename)
            logger.info(f"Uploaded Binning Info: {filename} with size {file_size}.")
            return [
                dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                html.P(f"File Size: {file_size}")
            ], {'display': 'block'}, contents
        
        logger.error("Unsupported file format for Binning Info.")
        return "Unsupported file format", {'display': 'block'}, None

    # Callback for handling bin taxonomy upload with logging
    @app.callback(
        [Output('overview-raw-bin-taxonomy', 'children'),
         Output('remove-raw-bin-taxonomy', 'style'),
         Output('raw-bin-taxonomy', 'contents')],
        [Input('raw-bin-taxonomy', 'contents'),
         Input('remove-raw-bin-taxonomy', 'n_clicks')],
        [State('raw-bin-taxonomy', 'filename')],
        prevent_initial_call=True
    )
    def handle_bin_taxonomy_upload(contents, remove_click, filename):
        ctx = dash.callback_context
        if not contents:
            raise PreventUpdate
        
        if remove_click and ctx.triggered_id == 'remove-raw-bin-taxonomy':
            logger.info("Bin Taxonomy file removed.")
            return '', {'display': 'none'}, None
        
        file_size = get_file_size(contents)
        if 'csv' in filename:
            df = parse_contents(contents, filename)
            logger.info(f"Uploaded Bin Taxonomy: {filename} with size {file_size}.")
            return [
                dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                html.P(f"File Size: {file_size}")
            ], {'display': 'block'}, contents
        
        logger.error("Unsupported file format for Bin Taxonomy.")
        return "Unsupported file format", {'display': 'block'}, None

    # Callback for the 'Prepare Data' button (Method 1)
    @app.callback(
        [Output('preparation-status-method1', 'data'),
         Output('blank-element', 'children')],
        [Input('execute-button', 'n_clicks'),
         Input('load-button', 'n_clicks')],
        [State('raw-contig-info', 'contents'),
         State('raw-contig-matrix', 'contents'),
         State('raw-binning-info', 'contents'),
         State('raw-bin-taxonomy', 'contents'),
         State('raw-contig-info', 'filename'),
         State('raw-contig-matrix', 'filename'),
         State('raw-binning-info', 'filename'),
         State('raw-bin-taxonomy', 'filename'),
         State('user-folder', 'data'),
         State('current-method', 'data'),
         State('current-stage', 'data')],
        prevent_initial_call=True
    )
    def prepare_data_method_1(n_clicks_execute, n_clicks_load,
                              contig_info, contig_matrix, binning_info, bin_taxonomy,
                              contig_info_name, contig_matrix_name, binning_info_name, bin_taxonomy_name,
                              user_folder, selected_method, current_stage):
        
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
    
        triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

        # Check if the triggered button is the load button (to load files from folder) or execute button (to process uploaded files)
        if selected_method != 'method1' or current_stage != 'Preparation':
            raise PreventUpdate
    
        if triggered_input == 'load-button':  # Load files from folder
            logger.info("Loading data from user folder...")
    
            # Paths to the files in the user folder
            contig_info_file = os.path.join('assets', 'examples', 'contig_information.csv')
            contig_matrix_file = os.path.join('assets', 'examples', 'raw_contact_matrix.npz')
            binning_info_file = os.path.join('assets', 'examples', 'binning_information.csv')
            taxonomy_info_file = os.path.join('assets', 'examples', 'taxonomy_information.csv')
    
            # Load files from the folder
            contig_data = pd.read_csv(contig_info_file)
            
            # Load the .npz file for contig matrix using load_npz
            contig_matrix_data = load_npz(contig_matrix_file).tocoo()
            
            # Convert the COO matrix to base64-encoded .npz format
            buffer = io.BytesIO()
            save_npz(buffer, contig_matrix_data)
            buffer.seek(0)  # Rewind the buffer
            encoded_contig_matrix = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Create the contig_matrix variable in the format of an uploaded file
            contig_matrix = f"data:application/x-npz;base64,{encoded_contig_matrix}"
            
            # Load binning and taxonomy data
            binning_data = pd.read_csv(binning_info_file)
            taxonomy_data = pd.read_csv(taxonomy_info_file)
    
        elif triggered_input == 'execute-button':
            try:
                if binning_info:
                    binning_data = parse_contents(binning_info, binning_info_name)
                else:
                    logger.info("No files uploaded for binning_info. Using default file.")
                    binning_info_file = os.path.join('assets', 'examples', 'empty_binning_information.csv')
                    binning_data = pd.read_csv(binning_info_file)
                    
                if bin_taxonomy:
                    taxonomy_data = parse_contents(bin_taxonomy, bin_taxonomy_name)
                else:
                    logger.info("No files uploaded for taxonomy_info. Using default file.")
                    taxonomy_info_file = os.path.join('assets', 'examples', 'empty_taxonomy_information.csv')
                    taxonomy_data = pd.read_csv(taxonomy_info_file)
                
                if not all([contig_info, contig_matrix]):
                    logger.error("Validation failed: Missing required files.")
                    return False, ""
            
                contig_data = parse_contents(contig_info, contig_info_name)
                contig_matrix_data = parse_contents(contig_matrix, contig_matrix_name)
    
            except Exception as e:
                logger.error(f"Error parsing uploaded files: {e}")
                return False, ""
    
        try:
            validate_csv(contig_data, ['Contig ID', 'The number of restriction sites', 'Contig length'], ['Contig coverage'])
            
            if isinstance(contig_matrix_data, (coo_matrix, csc_matrix, csr_matrix)):
                validate_contig_matrix(contig_data, contig_matrix_data)
            else:
                row = contig_matrix_data['Contig ID1'].values
                col = contig_matrix_data['Contig ID2'].values
                data = contig_matrix_data['Contacts'].values
    
                # Calculate the shape based on max row and column index + 1 (since indices are 0-based)
                num_rows = contig_matrix_data['row'].max() + 1
                num_cols = contig_matrix_data['column'].max() + 1
                shape = (num_rows, num_cols)
    
                # Create the COO matrix with the calculated shape
                contig_matrix_data = coo_matrix((data, (row, col)), shape=shape) 
                validate_contig_matrix(contig_data, contig_matrix_data)
    
            diagonal_values = contig_matrix_data.diagonal()
    
            contig_data['Within-contig Hi-C contacts'] = diagonal_values
    
            contig_data['Contig coverage'] = contig_data.apply(
                lambda row: row['Within-contig Hi-C contacts'] / row['Contig length'] 
                if pd.isna(row['Contig coverage']) else row['Contig coverage'], axis=1)
            
            binning_data = binning_data.drop_duplicates(subset='Contig ID', keep='first')
            taxonomy_data = taxonomy_data.drop_duplicates(subset='ID', keep='first')
            
            taxonomy_data.rename(columns={'ID': 'Bin ID'}, inplace=True)
            taxonomy_data.replace("Unclassified", None, inplace=True)

            taxonomy_columns = np.array([col for col in taxonomy_data.columns if col not in ['Bin ID', 'Category']])
            save_to_redis(f'{user_folder}:taxonomy-levels', taxonomy_columns)
    
            # Process data
            combined_data = process_data(contig_data, binning_data, taxonomy_data, contig_matrix_data, taxonomy_columns)
            combined_data['Category'] = combined_data['Category'].fillna('chromosome')

            # Save files to user folder
            user_output_folder = os.path.join('output', user_folder)
            os.makedirs(user_output_folder, exist_ok=True)

            save_npz(os.path.join('output', user_folder, 'unnormalized_contig_matrix.npz'), contig_matrix_data)
            encoded_csv_content = base64.b64encode(combined_data.to_csv(index=False).encode()).decode()
            save_file_to_user_folder(f"data:text/csv;base64,{encoded_csv_content}", 'contig_info_final.csv', user_folder)
            # Compress into a 7z archive
            user_output_folder = os.path.join('output', user_folder)
            unnormalized_archive_path = os.path.join(user_output_folder, 'unnormalized_information.7z')
            with py7zr.SevenZipFile(unnormalized_archive_path, 'w') as archive:
                archive.write(os.path.join(user_output_folder, 'unnormalized_contig_matrix.npz'), 'unnormalized_contig_matrix.npz')
                archive.write(os.path.join(user_output_folder, 'contig_info_final.csv'), 'contig_info_final.csv')
            return True, ""
    
        except Exception as e:
            logger.error(f"Error during preparation: {e}")
            return False, ""


    @app.callback(
        [Output('overview-unnormalized-data-folder', 'children'),
         Output('remove-unnormalized-data-folder', 'style'),
         Output('unnormalized-data-folder', 'contents')],
        [Input('unnormalized-data-folder', 'contents'),
         Input('remove-unnormalized-data-folder', 'n_clicks')],
        [State('unnormalized-data-folder', 'filename')],
        prevent_initial_call=True
    )
    def handle_method_2(contents, remove_click, filename):
        ctx = dash.callback_context
        if not contents:
            raise PreventUpdate

        if remove_click and ctx.triggered_id == 'remove-unnormalized-data-folder':
            logger.info("Unnormalized Data Folder file removed.")
            return '', {'display': 'none'}, None

        file_size = get_file_size(contents)
        decoded = base64.b64decode(contents.split(',')[1])
        file_list = list_files_in_7z(decoded)
        overview = html.Ul([html.Li(file) for file in file_list])
        logger.info(f"Uploaded Unnormalized Data Folder: {filename} with size {file_size}. Files: {file_list}")
        return [overview, html.P(f"File uploaded: {filename} ({file_size})")], {'display': 'block'}, contents

    @app.callback(
        Output('preparation-status-method2', 'data'),
        [Input('execute-button', 'n_clicks')],
        [State('unnormalized-data-folder', 'contents'),
         State('unnormalized-data-folder', 'filename'),
         State('user-folder', 'data'),
         State('current-method', 'data'),
         State('current-stage', 'data')],
        prevent_initial_call=True
    )
    def prepare_data_method_2(n_clicks, contents, filename, user_folder, selected_method, current_stage):
        # Ensure the selected method is 'method2' and the current stage is 'Preparation'
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'execute-button':
            raise PreventUpdate
        if selected_method != 'method2' or current_stage != 'Preparation':
            raise PreventUpdate
        if n_clicks is None or contents is None:
            logger.error("Validation failed for Method 2: Missing file upload or click event.")
            return False  # Validation failed

        try:
            # Decode the uploaded file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # List files in the 7z archive
            file_list = list_files_in_7z(decoded)
            logger.info(f"Files in the uploaded archive: {file_list}")
    
            # Define the extraction path
            user_folder_path = f'output/{user_folder}'
            os.makedirs(user_folder_path, exist_ok=True)
    
            # Extract the 7z file to the user's folder
            with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as archive:
                archive.extractall(path=user_folder_path)
                
            contig_info_path = os.path.join(user_folder_path, 'contig_info_final.csv')    
            contig_information = pd.read_csv(contig_info_path)
            excluded_columns = [
                'Contig ID', 
                'The number of restriction sites', 
                'Contig length', 
                'Contig coverage', 
                'Within-contig Hi-C contacts', 
                'Bin ID', 
                'Category'
            ]
            taxonomy_levels = np.array([col for col in contig_information.columns if col not in excluded_columns])
            taxonomy_levels_key = f'{user_folder}:taxonomy-levels'
            save_to_redis(taxonomy_levels_key, taxonomy_levels)
                
            logger.info("Validation and extraction successful for Method 2.")
            return True  # Validation and extraction succeeded

        except Exception as e:
            logger.error(f"Validation and preparation failed for Method 2: {e}")
            return False  # Validation failed

    @app.callback(
        [Output('overview-normalized-data-folder', 'children'),
         Output('remove-normalized-data-folder', 'style'),
         Output('normalized-data-folder', 'contents')],
        [Input('normalized-data-folder', 'contents'),
         Input('remove-normalized-data-folder', 'n_clicks')],
        [State('normalized-data-folder', 'filename')],
        prevent_initial_call=True
    )
    def handle_method_3(contents, remove_click, filename):
        ctx = dash.callback_context
        if not contents:
            raise PreventUpdate

        if remove_click and ctx.triggered_id == 'remove-normalized-data-folder':
            logger.info("Normalized Data Folder file removed.")
            return '', {'display': 'none'}, None

        file_size = get_file_size(contents)
        decoded = base64.b64decode(contents.split(',')[1])
        file_list = list_files_in_7z(decoded)
        overview = html.Ul([html.Li(file) for file in file_list])
        logger.info(f"Uploaded Normalized Data Folder: {filename} with size {file_size}. Files: {file_list}")
        return [overview, html.P(f"File uploaded: {filename} ({file_size})")], {'display': 'block'}, contents

    # Updated Method 3 Callback to Extract and Save .7z Files
    @app.callback(
        Output('preparation-status-method3', 'data'),
        [Input('execute-button', 'n_clicks')],
        [State('normalized-data-folder', 'contents'),
         State('normalized-data-folder', 'filename'),
         State('user-folder', 'data'),
         State('current-method', 'data'),
         State('current-stage', 'data')],
        prevent_initial_call=True
    )
    def prepare_data_method_3(n_clicks, contents, filename, user_folder, selected_method, current_stage):
        # Ensure the selected method is 'method3' and the current stage is 'Preparation'
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'execute-button':
            raise PreventUpdate
        if selected_method != 'method3' or current_stage != 'Preparation':
            raise PreventUpdate
        if n_clicks is None or contents is None:
            logger.error("Validation failed for Method 3: Missing file upload or click event.")
            return False  # Validation failed

        try:
            # Decode the uploaded file
            logger.info(f"Decoding file: {filename}")
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # List files in the 7z archive
            file_list = list_files_in_7z(decoded)
            logger.info(f"Files in the uploaded archive: {file_list}")
    
            # Define the extraction path
            user_output_path = f'output/{user_folder}'
            os.makedirs(user_output_path, exist_ok=True)
    
            # Extract the 7z file to the user's folder
            with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as archive:
                archive.extractall(path=user_output_path)
        
            bin_info_path = os.path.join(user_output_path, 'bin_info_final.csv')
            bin_matrix_path = os.path.join(user_output_path, 'normalized_bin_matrix.npz')
        
            # Redis keys specific to each user folder
            bin_info_key = f'{user_folder}:bin-information'
            bin_matrix_key = f'{user_folder}:bin-dense-matrix'
            taxonomy_levels_key = f'{user_folder}:taxonomy-levels'
        
            try:
                bin_information = pd.read_csv(bin_info_path)
                excluded_columns = [
                    'Contig ID', 
                    'The number of restriction sites', 
                    'Contig length', 
                    'Contig coverage', 
                    'Within-contig Hi-C contacts', 
                    'Connected bins',
                    'Bin ID', 
                    'Category',
                    'Visibility'
                ]
                taxonomy_levels = np.array([col for col in bin_information.columns if col not in excluded_columns])
                
                # Load matrix data using load_npz
                bin_dense_matrix = load_npz(bin_matrix_path).tocoo()
                
            except Exception as e:
                logger.error(f"Error loading data from files: {e}")
                return False

            # Save the loaded data to Redis with keys specific to the user folder
            save_to_redis(bin_info_key, bin_information)       
            save_to_redis(bin_matrix_key, bin_dense_matrix)
            save_to_redis(taxonomy_levels_key, taxonomy_levels)
            
            logger.info("Data loaded and saved to Redis successfully.")
            return True  # Validation and extraction succeeded

        except Exception as e:
            logger.error(f"Validation and preparation failed for Method 3: {e}")
            return False  # Validation failed