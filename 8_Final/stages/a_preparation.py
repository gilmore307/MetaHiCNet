import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import io
import os
import py7zr
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import logging
from helper import save_file_to_user_folder, query_plasmid_id

# Initialize logger
logger = logging.getLogger("app_logger")

# Helper functions
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    if 'csv' in filename:
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif 'npz' in filename:
        # Load the npz file
        npzfile = np.load(io.BytesIO(decoded))
        
        # Extract the required arrays to reconstruct the sparse matrix
        if all(key in npzfile for key in ['data', 'indices', 'indptr', 'shape']):
            data = npzfile['data']
            indices = npzfile['indices']
            indptr = npzfile['indptr']
            shape = tuple(npzfile['shape'])
            
            # Reconstruct the sparse matrix
            contact_matrix = csr_matrix((data, indices, indptr), shape=shape)
            return contact_matrix
        else:
            raise ValueError("The contact matrix file does not contain the expected sparse matrix keys.")
    elif '7z' in filename:
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

    if isinstance(contact_matrix, np.lib.npyio.NpzFile):
        if all(key in contact_matrix for key in ['data', 'indices', 'indptr', 'shape']):
            data = contact_matrix['data']
            indices = contact_matrix['indices']
            indptr = contact_matrix['indptr']
            shape = tuple(contact_matrix['shape'])
            contact_matrix = csr_matrix((data, indices, indptr), shape=shape)
        else:
            logger.error("The contact matrix file does not contain the expected sparse matrix keys.")
            raise ValueError("The contact matrix file does not contain the expected sparse matrix keys.")
    
    matrix_shape = contact_matrix.shape
    if matrix_shape[0] != matrix_shape[1]:
        logger.error("The contact matrix is not square.")
        raise ValueError("The contact matrix is not square.")
    if matrix_shape[0] != num_contigs:
        logger.error(f"The contact matrix dimensions {matrix_shape} do not match the number of contigs.")
        raise ValueError(f"The contact matrix dimensions {matrix_shape} do not match the number of contigs.")
    
    if 'Self-contact' in contig_data.columns:
        diagonal_values = np.diag(contact_matrix.toarray())
        self_contact = contig_data['Self-contact'].dropna()
        if not np.allclose(self_contact, diagonal_values[:len(self_contact)]):
            logger.error("The 'Self-contact' column values do not match the diagonal of the contact matrix.")
            raise ValueError("The 'Self-contact' column values do not match the diagonal of the contact matrix.")
    
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

def adjust_taxonomy(row, taxonomy_columns, prefixes):
    last_non_blank = ""
    
    for tier in taxonomy_columns:
        row[tier] = str(row[tier]) if pd.notna(row[tier]) else ""

    if row['Type'] != 'unmapped':
        for tier in taxonomy_columns:
            if row[tier]:
                last_non_blank = row[tier]
            else:
                row[tier] = f"Unspecified {last_non_blank}"
    else:
        for tier in taxonomy_columns:
            row[tier] = "unmapped"

    if row['Type'] == 'phage':
        row['Domain'] = 'Virus'
        row['Phylum'] = 'Virus'
        row['Class'] = 'Virus'
        for tier in taxonomy_columns:
            row[tier] = row[tier] + '_v'
        row['Contig'] = row['Contig'] + "_v"
        row['Bin'] = row['Bin'] + "_v"

    if row['Type'] == 'plasmid':
        for tier in taxonomy_columns:
            row[tier] = row[tier] + '_p'
        row['Contig'] = row['Contig'] + "_p"
        row['Bin'] = row['Bin'] + "_p"

    for tier, prefix in prefixes.items():
        row[tier] = f"{prefix}{row[tier]}" if row[tier] else "N/A"

    return row

def process_data(contig_data, binning_data, taxonomy_data, contig_matrix, user_folder):
    try:
        logger.info("Starting data preparation...")
        
                # Debugging print to check the format of contact_matrix
        logger.debug(f"Type of contact_matrix: {type(contig_matrix)}")
        logger.debug(f"Content of contact_matrix: {contig_matrix}")
        
        if isinstance(contig_matrix, str):
            logger.error("contig_matrix is a string, which is unexpected. Please check the source.")
            raise ValueError("contact_matrix should be a sparse matrix, not a string.")
        
        # Ensure the contact_matrix is in the correct sparse format if it's not already
        if not isinstance(contig_matrix, csr_matrix):
            logger.error("contig_matrix is not a CSR sparse matrix.")
            raise ValueError("contig_matrix must be a CSR sparse matrix.")

        # Query plasmid classification for any available plasmid IDs
        logger.info("Querying plasmid IDs for classification...")
        plasmid_ids = taxonomy_data['Plasmid ID'].dropna().unique().tolist()
        plasmid_classification_df = query_plasmid_id(plasmid_ids)

        # Ensure plasmid_classification_df has the expected columns
        expected_columns = [
            'NUCCORE_ACC', 
            'TAXONOMY_superkingdom', 
            'TAXONOMY_phylum', 
            'TAXONOMY_class', 
            'TAXONOMY_order', 
            'TAXONOMY_family', 
            'TAXONOMY_genus', 
            'TAXONOMY_species'
        ]

        # Check if the returned dataframe contains all expected columns
        if not all(col in plasmid_classification_df.columns for col in expected_columns):
            logger.error("The plasmid classification data does not contain the expected columns.")
            raise ValueError("The plasmid classification data does not contain the expected columns.")

        # Rename columns to match internal structure
        plasmid_classification_df.rename(columns={
            'NUCCORE_ACC': 'Plasmid ID',
            'TAXONOMY_superkingdom': 'Kingdom',
            'TAXONOMY_phylum': 'Phylum',
            'TAXONOMY_class': 'Class',
            'TAXONOMY_order': 'Order',
            'TAXONOMY_family': 'Family',
            'TAXONOMY_genus': 'Genus',
            'TAXONOMY_species': 'Species'
        }, inplace=True)

        # Define prefixes for taxonomy tiers
        prefixes = {
            'Domain': 'd_',
            'Kingdom': 'k_',
            'Phylum': 'p_',
            'Class': 'c_',
            'Order': 'o_',
            'Family': 'f_',
            'Genus': 'g_',
            'Species': 's_'
        }

        # Replace certain text in the classification dataframe
        plasmid_classification_df = plasmid_classification_df.replace(r"\s*\(.*\)", "", regex=True)
        plasmid_classification_df['Domain'] = plasmid_classification_df['Kingdom']

        # Merge plasmid classification with taxonomy data
        taxonomy_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        taxonomy_data = taxonomy_data.merge(
            plasmid_classification_df[['Plasmid ID'] + taxonomy_columns],
            on='Plasmid ID',
            how='left',
            suffixes=('', '_new')
        )

        # Fill taxonomy columns with new classification where available
        for column in taxonomy_columns:
            taxonomy_data[column] = taxonomy_data[column + '_new'].combine_first(taxonomy_data[column])

        # Drop unnecessary columns after merge
        taxonomy_data = taxonomy_data.drop(columns=['Plasmid ID'] + [col + '_new' for col in taxonomy_columns])

        # Merge contig, binning, and taxonomy data
        combined_data = pd.merge(contig_data, binning_data, on="Contig", how="left")
        combined_data = pd.merge(combined_data, taxonomy_data, on="Bin", how="left")

        # Apply taxonomy adjustments
        combined_data = combined_data.apply(lambda row: adjust_taxonomy(row, taxonomy_columns, prefixes), axis=1)

        # Fill missing bins with 'Unbinned MAG'
        combined_data['Bin'] = combined_data['Bin'].fillna('Unbinned MAG')
        
        # Set the 'Signal' column in combined_data using the diagonal values from the contact matrix
        diagonal_values = contig_matrix.diagonal()
        print(len(diagonal_values), max(diagonal_values), diagonal_values)
        combined_data['Signal'] = diagonal_values

        # Return the processed combined data directly
        logger.info("Data processed successfully.")
        return combined_data

    except Exception as e:
        logger.error(f"Error during data preparation: {e}; no preview will be generated.")
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
                "This file must include columns like 'Contig', 'Restriction sites', 'Length', 'Coverage', and 'Signal'."
            )),
            dbc.Col(create_upload_component(
                'raw-contig-matrix', 
                'Upload Raw Contact Matrix File (.npz)', 
                'assets/examples/raw_contact_matrix.npz',
                "Matrix file must include keys such as 'indices', 'indptr', 'format', 'shape', 'data'."
            ))
        ]),
        dbc.Row([
            dbc.Col(create_upload_component(
                'raw-binning-info', 
                'Upload Binning Information File (.csv)', 
                'assets/examples/binning_information.csv',
                "This file must include columns like 'Contig', 'Bin', and 'Type'."
            )),
            dbc.Col(create_upload_component(
                'raw-bin-taxonomy', 
                'Upload Bin Taxonomy File (.csv)', 
                'assets/examples/taxonomy.csv',
                "This file must include columns like 'Bin', 'Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Plasmid ID'."
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
                "The folder must include files like 'contig_info_final.csv' and 'raw_contact_matrix.npz'."
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
                "This folder should include files like 'bin_info_final.csv', 'contig_info_final.csv', 'contig_contact_matrix.npz', 'bin_contact_matrix.npz'."
            ))
        ])
    ])

def create_processed_data_preview(combined_data):
    if not combined_data.empty:
        preview_table = dbc.Table.from_dataframe(combined_data.head(), striped=True, bordered=True, hover=True)
        return html.Div([
            html.H5('Processed Data Preview'),
            preview_table
        ])
    return None

# Registering callbacks for uploads with logging
def register_preparation_callbacks(app):
    # Callback for raw contig info upload (Method 1) with logging
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
            return [dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                    html.P(f"File Size: {file_size}")], {'display': 'block'}, contents
        
        logger.error("Unsupported file format for Contig Info.")
        return "Unsupported file format", {'display': 'block'}, contents

    # Callback for raw contact matrix upload (Method 1) with logging
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
        if 'npz' in filename:
            try:
                # Parse the contents to ensure it is a sparse matrix
                contig_matrix = parse_contents(contents, filename)
                
                if not isinstance(contig_matrix, csr_matrix):
                    logger.error("Parsed matrix is not a CSR sparse matrix.")
                    raise ValueError("Parsed matrix must be a CSR sparse matrix.")
                
                # Open the npz file again to extract keys and their values
                decoded = base64.b64decode(contents.split(',')[1])
                npzfile = np.load(io.BytesIO(decoded))
    
                # Display keys and their values or types
                matrix_info = []
                for key in npzfile.files:
                    if key == 'format':
                        # Directly display the format value
                        format_value = npzfile[key].item()  # Retrieve the scalar value
                        matrix_info.append(html.Li(f"{key}: {format_value}"))
                    elif key == 'shape':
                        # Directly display the shape as a tuple
                        shape_value = tuple(npzfile[key])
                        matrix_info.append(html.Li(f"{key}: {shape_value}"))
                    else:
                        # For other keys, show the value
                        value = npzfile[key]
                        # Convert the value to a string representation (e.g., shape for arrays)
                        if isinstance(value, np.ndarray):
                            value_str = f"Array with shape {value.shape}"
                        else:
                            value_str = str(value)
                        matrix_info.append(html.Li(f"{key}: {value_str}"))
    
                overview = html.Ul(matrix_info)
                logger.info(f"Uploaded Raw Matrix: {filename} with size {file_size}.")
                return [overview, html.P(f"File Size: {file_size}")], {'display': 'block'}, contents
    
            except Exception as e:
                logger.error(f"Error processing matrix file: {e}")
                return "Error processing matrix file", {'display': 'block'}, contents
    
        logger.error("Unsupported file format for Raw Matrix.")
        return "Unsupported file format", {'display': 'block'}, contents

    # Callback for binning info upload (Method 1) with logging
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
            return [dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                    html.P(f"File Size: {file_size}")], {'display': 'block'}, contents
        
        logger.error("Unsupported file format for Binning Info.")
        return "Unsupported file format", {'display': 'block'}, contents

    # Callback for bin taxonomy upload (Method 1) with logging
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
            return [dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                    html.P(f"File Size: {file_size}")], {'display': 'block'}, contents
        
        logger.error("Unsupported file format for Bin Taxonomy.")
        return "Unsupported file format", {'display': 'block'}, contents

    # Callback for the 'Prepare Data' button (Method 1)
    @app.callback(
        [Output('preparation-status-method1', 'data'),
         Output('dynamic-content-method1', 'children', allow_duplicate=True)],
        [Input('execute-button-method1', 'n_clicks')],
        [State('raw-contig-info', 'contents'),
         State('raw-contig-matrix', 'contents'),
         State('raw-binning-info', 'contents'),
         State('raw-bin-taxonomy', 'contents'),
         State('raw-contig-info', 'filename'),
         State('raw-contig-matrix', 'filename'),
         State('raw-binning-info', 'filename'),
         State('raw-bin-taxonomy', 'filename'),
         State('user-folder', 'data')],
        prevent_initial_call=True
    )
    def prepare_data_method_1(n_clicks, contig_info, contig_matrix, binning_info, bin_taxonomy,
                              contig_info_name, contig_matrix_name, binning_info_name, bin_taxonomy_name,
                              user_folder):
        logger.info("Validating and preparing data for Method 1...")
        if n_clicks is None:
            raise PreventUpdate
    
        if not all([contig_info, contig_matrix, binning_info, bin_taxonomy]):
            logger.error("Validation failed: Missing required files.")
            return False, no_update
    
        try:
            # Parse and validate the contents directly
            contig_data = parse_contents(contig_info, contig_info_name)
            required_columns = ['Contig', 'Restriction sites', 'Length', 'Coverage']
            validate_csv(contig_data, required_columns, optional_columns=['Signal'])
    
            contig_matrix_data = parse_contents(contig_matrix, contig_matrix_name)
            
            # Verify that contig_matrix_data is of the expected type
            if not isinstance(contig_matrix_data, csr_matrix):
                logger.error("contig_matrix_data is not a CSR sparse matrix.")
                raise ValueError("contig_matrix must be a CSR sparse matrix.")
            
            validate_contig_matrix(contig_data, contig_matrix_data)
    
            binning_data = parse_contents(binning_info, binning_info_name)
            required_columns = ['Contig', 'Bin', 'Type']
            validate_csv(binning_data, required_columns)
    
            taxonomy_data = parse_contents(bin_taxonomy, bin_taxonomy_name)
            required_columns = ['Bin']
            optional_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Plasmid ID']
            validate_csv(taxonomy_data, required_columns, optional_columns)
    
            logger.info("Validation successful. Proceeding with data preparation...")
            
            # Process data using parsed dataframes directly
            combined_data = process_data(contig_data, binning_data, taxonomy_data, contig_matrix_data, user_folder)
    
            if combined_data is not None:
                # Save only the matrix and the combined data
                save_file_to_user_folder(contig_matrix, 'raw_contact_matrix.npz', user_folder)
    
                csv_buffer = io.StringIO()
                combined_data.to_csv(csv_buffer, index=False)
                csv_content = csv_buffer.getvalue()
                
                # Convert csv_content to base64-encoded string with required prefix
                encoded_csv_content = base64.b64encode(csv_content.encode()).decode()
                encoded_csv_content = f"data:text/csv;base64,{encoded_csv_content}"
                
                # Use save_file_to_user_folder to save the processed CSV with encoded content
                save_file_to_user_folder(encoded_csv_content, 'contig_info_final.csv', user_folder)
    
                # Create the processed data preview component
                preview_component = create_processed_data_preview(combined_data)
    
                logger.info("Data preparation successful, and files saved.")
                return True, preview_component
            else:
                logger.error("Data preparation failed.")
                return False, no_update
    
        except Exception as e:
            logger.error(f"Validation and preparation failed for Method 1: {e}")
            return False, no_update

    # Callback for Unnormalized Folder Upload (Method 2) with logging
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
        [Input('execute-button-method2', 'n_clicks')],
        [State('unnormalized-data-folder', 'contents'),
         State('unnormalized-data-folder', 'filename'),
         State('user-folder', 'data')],
        prevent_initial_call=True
    )
    def prepare_data_method_2(n_clicks, contents, filename, user_folder):
        logger.info("Validating and preparing Method 2")
        if n_clicks is None:
            raise PreventUpdate
        
        if contents is None:
            logger.error("Validation failed for Method 2: No file uploaded.")
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
            user_folder_path = f'assets/output/{user_folder}'
            os.makedirs(user_folder_path, exist_ok=True)
    
            # Extract the 7z file to the user's folder
            with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as archive:
                archive.extractall(path=user_folder_path)
            
            logger.info("Validating successful for Method 2.")
            return True  # Validation and extraction succeeded
    
        except Exception as e:
            logger.error(f"Validation and preparation failed for Method 2: {e}")
            return False  # Validation failed

    # Callback for Normalized Folder Upload (Method 3) with logging
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
        [Input('execute-button-method3', 'n_clicks')],
        [State('normalized-data-folder', 'contents'),
         State('normalized-data-folder', 'filename'),
         State('user-folder', 'data')],
        prevent_initial_call=True
    )
    def prepare_data_method_3(n_clicks, contents, filename, user_folder):
        logger.info("Validating and preparing Method 3")
        if n_clicks is None:
            raise PreventUpdate
        
        if contents is None:
            logger.error("Validation failed for Method 3: No file uploaded.")
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
            user_folder_path = f'assets/output/{user_folder}'
            os.makedirs(user_folder_path, exist_ok=True)
    
            # Extract the 7z file to the user's folder
            with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as archive:
                archive.extractall(path=user_folder_path)
            
            logger.info("Validating successful for Method 3.")
            return True  # Validation and extraction succeeded
    
        except Exception as e:
            logger.error(f"Validation and preparation failed for Method 3: {e}")
            return False  # Validation failed