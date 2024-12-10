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
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
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
        npzfile = np.load(io.BytesIO(decoded))
        
        # Check for COO matrix keys
        if all(key in npzfile for key in ['data', 'row', 'col', 'shape']):
            data = npzfile['data']
            row = npzfile['row']
            col = npzfile['col']
            shape = tuple(npzfile['shape'])
            
            # Reconstruct the sparse matrix in COO format
            contact_matrix = coo_matrix((data, (row, col)), shape=shape)
            return contact_matrix
        
        # Check for CSR or CSC matrix keys
        elif all(key in npzfile for key in ['data', 'indices', 'indptr', 'shape']):
            data = npzfile['data']
            indices = npzfile['indices']
            indptr = npzfile['indptr']
            shape = tuple(npzfile['shape'])
            
            # Reconstruct and convert to COO
            contact_matrix = csr_matrix((data, indices, indptr), shape=shape).tocoo()
            return contact_matrix
    
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

def validate_unnormalized_folder(folder):
    expected_files = ['contig_info_final.csv', 'raw_contact_matrix.npz']
    missing_files = [file for file in expected_files if file not in folder]
    if missing_files:
        logger.error(f"Missing files in unnormalized folder: {', '.join(missing_files)}")
        raise ValueError(f"Missing files in unnormalized folder: {', '.join(missing_files)}")
    return True

def validate_normalized_folder(folder):
    expected_files = ['bin_info_final.csv', 'contig_info_final.csv', 'contig_contact_matrix.npz', 'bin_contact_matrix.npz']
    missing_files = [file for file in expected_files if file not in folder]
    if missing_files:
        logger.error(f"Missing files in normalized folder: {', '.join(missing_files)}")
        raise ValueError(f"Missing files in normalized folder: {', '.join(missing_files)}")
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
        row['Contig index'] = row['Contig index'] + "_v"
        row['Bin index'] = row['Bin index'] + "_v"

    elif row['Category'] == 'plasmid':
        for tier in taxonomy_columns:
            if not pd.isna(row[tier]):
                row[tier] = row[tier] + '_p'
        row['Contig index'] = row['Contig index'] + "_p"
        row['Bin index'] = row['Bin index'] + "_p"
        
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
        combined_data = pd.merge(contig_data, binning_data, on='Contig index', how="left")
        combined_data = pd.merge(combined_data, taxonomy_data, on='Bin index', how="left")

        # Apply taxonomy adjustments
        combined_data = combined_data.apply(lambda row: adjust_taxonomy(row, taxonomy_columns), axis=1)

        # Fill missing bins with Contig index
        combined_data['Bin index'] = combined_data['Bin index'].fillna(combined_data['Contig index'])

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
                "This file includes the following columns: 'Contig index', 'The number of restriction sites', 'Contig length', 'Contig coverage', and 'Within-contig Hi-C contacts'.  \n\n"
                "'Within-contig Hi-C contacts' is an optional column used to ensure that the contig sequence in the Contig Information aligns with the contig sequence in the Contact Matrix; leave it blank if not applicable.  \n\n"
                "'Contig coverage' is an optional column that can be automatically calculated by dividing 'Within-contig Hi-C contacts' by the 'Contig length' if not provided."
            )),
            dbc.Col(create_upload_component(
                'raw-contig-matrix', 
                'Upload Raw Contact Matrix File (.txt, .csv or .npz)', 
                'assets/examples/raw_contact_matrix.npz',
                "The contact matrix can be provided in one of the following formats:  .txt, .csv or .npz  \n\n"
                "The .txt and .csv format file should contain 3 columns: 'Contig_name1', 'Contig_name2', 'Contacts'.  \n\n"
                "The .npz Format file should be a sparse or dense matrix which can be obtained from recent metaHi-C pipelines such as MetaCC or HiCBin.  \n"
                "The matrix could be in COO, CSC or CSR format.  \n"
                "The row and column indices of the Contact Matrix must match the row indices of the Contig Information File."
            ))
        ]),
        dbc.Row([
            dbc.Col(create_upload_component(
                'raw-binning-info', 
                'Upload Binning Information File (.csv)', 
                'assets/examples/binning_information.csv',
                "This file includes the following columns: 'Contig index', 'Bin index', and 'Category'.  \n\n"
                "'Bin index' is an optional column; if left blank (contigs are not binned), the value from the 'Contig index' column will be used as the bin index.  \n\n"
                "Please indicate the category of bin in the 'Category' column if it is a 'virus' or 'plasmid'.  \n"
                "Feel free to leave the colunn blank if it is 'chromosome' or 'unclassified' (cannot be assigned to any known reference or taxonomy)."
            )),
            dbc.Col(create_upload_component(
                'raw-bin-taxonomy', 
                'Upload Taxonomy Information File (.csv)', 
                'assets/examples/taxonomy_information.csv',
                "This file includes the following columns: 'Bin index', 'Customized annotation' and taxonomy columns.  \n\n"
                "Users have the flexibility to define their taxonomy levels based on their specific needs.  \n"
                "Users can include any number of taxonomy columns, up to a maximum of 8 levels (e.g., Realm, Phylum, Order, Genus, etc.).  \n"
                "Taxonomy column names can be customized. For example, Domain or Superkingdom could be used instead of Realm.  \n\n"
                "If the Customized annotation column is filled, its values will be used to populate any missing entries across the specified taxonomy levels.  \n"
                "The 'Customized annotation' should only contain letters (a-z, A-Z), numbers (0-9), or the underscore (_) character.  \n"
                "This feature ensures that unclassified bins or missing taxonomy details are assigned meaningful labels.  \n"
                "This feature is particularly useful for annotating bins or contigs that do not align with traditional taxonomy systems,  \n"
                "such as plasmid GenBank Accession Numbers (NZ_CP123456) or viruse Zoonotic Virus Classification (SARS-CoV). \n"
                "It allows users to provide a specific annotation for such bins or contigs.  \n\n"
                "If all the taxonomy columns are blank, the bin will be marked as 'unclassified'."
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
                "The folder must include the following files: 'contig_info_final.csv' and 'raw_contact_matrix.npz'."
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
                "This folder must include the following files: 'bin_info_final.csv', 'contig_info_final.csv', 'contig_contact_matrix.npz', 'bin_contact_matrix.npz'."
            ))
        ])
    ])

def register_preparation_callbacks(app):
    # Callback for handling raw contig info upload with logging
    @app.callback(
        [Output('overview-raw-contig-info', 'children'),
         Output('remove-raw-contig-info', 'style')],
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
            return '', {'display': 'none'}
        
        file_size = get_file_size(contents)
        if 'csv' in filename:
            df = parse_contents(contents, filename)
            logger.info(f"Uploaded Contig Info: {filename} with size {file_size}.")
            return [
                dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                html.P(f"File Size: {file_size}")
            ], {'display': 'block'}
        
        logger.error("Unsupported file format for Contig Info.")
        return "Unsupported file format", {'display': 'block'}
    
    @app.callback(
        [Output('overview-raw-contig-matrix', 'children'),
         Output('remove-raw-contig-matrix', 'style')],
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
            return '', {'display': 'none'}
    
        file_size = get_file_size(contents)
        if 'npz' in filename:
            try:
                contig_matrix = parse_contents(contents, filename)
                
                if not isinstance(contig_matrix, coo_matrix):
                    logger.error("Parsed matrix is not a COO sparse matrix.")
                    raise ValueError("Parsed matrix must be a COO sparse matrix.")
    
                # Open the npz file again to extract keys and their values
                decoded = base64.b64decode(contents.split(',')[1])
                npzfile = np.load(io.BytesIO(decoded))
    
                # Display COO matrix keys and their information
                matrix_info = []
                for key in npzfile.files:
                    if key == 'shape':
                        # Display the shape of the matrix as a tuple
                        shape_value = tuple(npzfile[key])
                        matrix_info.append(html.Li(f"{key}: {shape_value}"))
                    elif key in ['data', 'row', 'col']:
                        # For COO matrix components, show the array's shape
                        value = npzfile[key]
                        value_str = f"Array with shape {value.shape}"
                        matrix_info.append(html.Li(f"{key}: {value_str}"))
                    else:
                        # For other keys, display the value directly
                        value = npzfile[key]
                        value_str = str(value) if not isinstance(value, np.ndarray) else f"Array with shape {value.shape}"
                        matrix_info.append(html.Li(f"{key}: {value_str}"))
    
                overview = html.Ul(matrix_info)
                logger.info(f"Uploaded Raw Matrix: {filename} with size {file_size}.")
                return [overview, html.P(f"File Size: {file_size}")], {'display': 'block'}
    
            except Exception as e:
                logger.error(f"Error processing matrix file: {e}")
                return "Error processing matrix file", {'display': 'block'}
            
        elif 'csv' in filename or 'txt' in filename:
            try:
                # Parse the file into a DataFrame
                df = parse_contents(contents, filename)
                
                # Display the first few rows as an overview
                overview =[
                    dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                    html.P(f"File Size: {file_size}")
                ]
                logger.info(f"Uploaded Raw Matrix: {filename} with size {file_size}.")
                return overview, {'display': 'block'}
            
            except Exception as e:
                logger.error(f"Error processing CSV/TXT file: {e}")
                return "Error processing CSV/TXT file", {'display': 'block'}
    
        logger.error("Unsupported file format for Raw Matrix.")
        return "Unsupported file format", {'display': 'block'}
    
    # Callback for handling binning info upload with logging
    @app.callback(
        [Output('overview-raw-binning-info', 'children'),
         Output('remove-raw-binning-info', 'style')],
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
            return '', {'display': 'none'}
        
        file_size = get_file_size(contents)
        if 'csv' in filename:
            df = parse_contents(contents, filename)
            logger.info(f"Uploaded Binning Info: {filename} with size {file_size}.")
            return [
                dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                html.P(f"File Size: {file_size}")
            ], {'display': 'block'}
        
        logger.error("Unsupported file format for Binning Info.")
        return "Unsupported file format", {'display': 'block'}

    # Callback for handling bin taxonomy upload with logging
    @app.callback(
        [Output('overview-raw-bin-taxonomy', 'children'),
         Output('remove-raw-bin-taxonomy', 'style')],
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
            return '', {'display': 'none'}
        
        file_size = get_file_size(contents)
        if 'csv' in filename:
            df = parse_contents(contents, filename)
            logger.info(f"Uploaded Bin Taxonomy: {filename} with size {file_size}.")
            return [
                dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                html.P(f"File Size: {file_size}")
            ], {'display': 'block'}
        
        logger.error("Unsupported file format for Bin Taxonomy.")
        return "Unsupported file format", {'display': 'block'}

    # Callback for the 'Prepare Data' button (Method 1)
    @app.callback(
        [Output('preparation-status-method1', 'data'),
         Output('blank-element', 'children')],
        [Input('execute-button', 'n_clicks')],
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
    def prepare_data_method_1(n_clicks, contig_info, contig_matrix, binning_info, bin_taxonomy,
                              contig_info_name, contig_matrix_name, binning_info_name, bin_taxonomy_name,
                              user_folder, selected_method, current_stage):
        # Ensure the selected method is 'method1' and the current stage is 'Preparation'
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'execute-button':
            raise PreventUpdate
        if selected_method != 'method1' or current_stage != 'Preparation':
            raise PreventUpdate
        if n_clicks is None:
            raise PreventUpdate

        logger.info("Validating and preparing data for Method 1...")
    
        if not all([contig_info, contig_matrix, binning_info, bin_taxonomy]):
            logger.error("Validation failed: Missing required files.")
            return False, ""
    
        try:
            # Parse and validate the contents directly
            contig_data = parse_contents(contig_info, contig_info_name)
            validate_csv(contig_data, ['Contig index', 'The number of restriction sites', 'Contig length'], ['Contig coverage'])
        except Exception as e:
            logger.error(f"Validation failed for Contig Information file: {e}")
            return False, ""
        
        try:
            contig_matrix_data = parse_contents(contig_matrix, contig_matrix_name)
            # Verify that contig_matrix_data is of the expected type
            if isinstance(contig_matrix_data, (coo_matrix, csc_matrix, csr_matrix)):
                validate_contig_matrix(contig_data, contig_matrix_data)
            else:
                row = contig_matrix_data['row'].values
                col = contig_matrix_data['column'].values
                data = contig_matrix_data['data'].values
                
                # Calculate the shape based on max row and column index + 1 (since indices are 0-based)
                num_rows = contig_matrix_data['row'].max() + 1
                num_cols = contig_matrix_data['column'].max() + 1
                shape = (num_rows, num_cols)
                
                # Create the COO matrix with the calculated shape
                contig_matrix_data = coo_matrix((data, (row, col)), shape=shape) 
                validate_contig_matrix(contig_data, contig_matrix_data)
            
            diagonal_values = contig_matrix_data.diagonal()
            
            contig_data['Within-contig Hi-C contacts'] = contig_data.apply(
                lambda row: diagonal_values[contig_data.index.get_loc(row.name)], axis=1)
            
            contig_data['Contig coverage'] = contig_data.apply(
                lambda row: row['Within-contig Hi-C contacts'] / row['Contig length'] 
                if pd.isna(row['Contig coverage']) else row['Contig coverage'], axis=1)
        except Exception as e:
            logger.error(f"Validation failed for Contact Matrix file: {e}")
            return False, ""
        
        try:
            binning_data = parse_contents(binning_info, binning_info_name)
            validate_csv(binning_data, ['Contig index'], ['Bin index'])
            binning_data['Bin index'] = binning_data.apply(
                lambda row: row['Contig index'] if pd.isna(row['Bin index']) else row['Bin index'], axis=1)

        except Exception as e:
            logger.error(f"Validation failed for Bining Information file: {e}")
            return False, ""
        
        try:
            taxonomy_data = parse_contents(bin_taxonomy, bin_taxonomy_name)
            taxonomy_columns = np.array([col for col in taxonomy_data.columns if col not in ['Bin index', 'Category']])
            taxonomy_data.replace("Unclassified", None, inplace=True)

            save_to_redis(f'{user_folder}:taxonomy-levels', taxonomy_columns)
        except Exception as e:
            logger.error(f"Validation failed for Taxonomy Information file: {e}")
            return False, ""
    
        logger.info("Validation successful. Proceeding with data preparation...")
            
        try:    
            # Process data using parsed dataframes directly
            combined_data = process_data(contig_data, binning_data, taxonomy_data, contig_matrix_data, taxonomy_columns)
            combined_data['Category'] = combined_data['Category'].fillna('chromosome')
    
            # Save `raw_contact_matrix.npz` using `save_file_to_user_folder`
            save_file_to_user_folder(contig_matrix, 'raw_contact_matrix.npz', user_folder)

            # Save `contig_info_final.csv`
            encoded_csv_content = base64.b64encode(combined_data.to_csv(index=False).encode()).decode()
            save_file_to_user_folder(f"data:text/csv;base64,{encoded_csv_content}", 'contig_info_final.csv', user_folder)
            
            # Compress into `unnormalized_information.7z` including the two files
            user_output_folder = os.path.join('output', user_folder)
            unnormalized_archive_path = os.path.join(user_output_folder, 'unnormalized_information.7z')
            with py7zr.SevenZipFile(unnormalized_archive_path, 'w') as archive:
                archive.write(os.path.join(user_output_folder, 'raw_contact_matrix.npz'), 'raw_contact_matrix.npz')
                archive.write(os.path.join(user_output_folder, 'contig_info_final.csv'), 'contig_info_final.csv')
            return True, ""
        except Exception as e:
            logger.error(f"Preparation failed for Method 1: {e}")
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
            logger.info(f"Decoding file: {filename}")
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # List files in the 7z archive
            file_list = list_files_in_7z(decoded)
            logger.info(f"Files in the uploaded archive: {file_list}")
            
            # Validate the folder contents
            logger.info("Validating unnormalized folder contents...")
            validate_unnormalized_folder(file_list)
    
            # Define the extraction path
            user_folder_path = f'output/{user_folder}'
            os.makedirs(user_folder_path, exist_ok=True)
    
            # Extract the 7z file to the user's folder
            with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as archive:
                archive.extractall(path=user_folder_path)
            
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
            
            # Validate the folder contents
            logger.info("Validating and save normalized folder contents...")
            validate_normalized_folder(file_list)
    
            # Define the extraction path
            user_output_path = f'output/{user_folder}'
            os.makedirs(user_output_path, exist_ok=True)
    
            # Extract the 7z file to the user's folder
            with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as archive:
                archive.extractall(path=user_output_path)
        
            bin_info_path = os.path.join(user_output_path, 'bin_info_final.csv')
            bin_matrix_path = os.path.join(user_output_path, 'bin_contact_matrix.npz')
            contig_info_path = os.path.join(user_output_path, 'contig_info_final.csv')
            contig_matrix_path = os.path.join(user_output_path, 'contig_contact_matrix.npz')
        
            # Redis keys specific to each user folder
            bin_info_key = f'{user_folder}:bin-information'
            bin_matrix_key = f'{user_folder}:bin-dense-matrix'
            contig_info_key = f'{user_folder}:contig-information'
            contig_matrix_key = f'{user_folder}:contig-dense-matrix'
            taxonomy_levels_key = f'{user_folder}:taxonomy-levels'
        
            try:
                bin_information = pd.read_csv(bin_info_path)
                excluded_columns = [
                    'Contig index', 
                    'The number of restriction sites', 
                    'Contig length', 
                    'Contig coverage', 
                    'Within-contig Hi-C contacts', 
                    'Bin index', 
                    'Category'
                ]
                taxonomy_levels = np.array([col for col in bin_information.columns if col not in excluded_columns])
                
                bin_matrix_data = np.load(bin_matrix_path)
                bin_dense_matrix = coo_matrix(
                    (bin_matrix_data['data'], (bin_matrix_data['row'], bin_matrix_data['col'])),
                    shape=tuple(bin_matrix_data['shape'])
                )
        
                contig_information = pd.read_csv(contig_info_path)
                
                contig_matrix_data = np.load(contig_matrix_path)
                contig_dense_matrix = coo_matrix(
                    (contig_matrix_data['data'], (contig_matrix_data['row'], contig_matrix_data['col'])),
                    shape=tuple(contig_matrix_data['shape'])
                )
            except Exception as e:
                logger.error(f"Error loading data from files: {e}")
                return False

            # Save the loaded data to Redis with keys specific to the user folder
            save_to_redis(bin_info_key, bin_information)       
            save_to_redis(bin_matrix_key, bin_dense_matrix)
            save_to_redis(contig_info_key, contig_information)
            save_to_redis(contig_matrix_key, contig_dense_matrix)
            save_to_redis(taxonomy_levels_key, taxonomy_levels)
            
            logger.info("Data loaded and saved to Redis successfully.")
            return True  # Validation and extraction succeeded

        except Exception as e:
            logger.error(f"Validation and preparation failed for Method 3: {e}")
            return False  # Validation failed