import dash
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import io
import os
import py7zr
import numpy as np
from scipy.sparse import csr_matrix
import logging
from helper import (
    save_file_to_user_folder, 
    parse_contents, 
    get_file_size, 
    validate_csv, 
    validate_contig_matrix, 
    validate_unnormalized_folder, 
    validate_normalized_folder,
    list_files_in_7z,
    process_data)

# Initialize logger
logger = logging.getLogger("app_logger")


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
                "This file must include the following columns: 'Contig', 'Restriction sites', 'Length', 'Coverage', and 'Signal'."
            )),
            dbc.Col(create_upload_component(
                'raw-contig-matrix', 
                'Upload Raw Contact Matrix File (.npz)', 
                'assets/examples/raw_contact_matrix.npz',
                "Matrix file must include the following keys: 'indices', 'indptr', 'format', 'shape', 'data'."
            ))
        ]),
        dbc.Row([
            dbc.Col(create_upload_component(
                'raw-binning-info', 
                'Upload Binning Information File (.csv)', 
                'assets/examples/binning_information.csv',
                "This file must include the following columns: 'Contig', 'Bin', and 'Type'."
            )),
            dbc.Col(create_upload_component(
                'raw-bin-taxonomy', 
                'Upload Bin Taxonomy File (.csv)', 
                'assets/examples/taxonomy.csv',
                "This file must include the following columns: 'Bin', 'Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Plasmid ID'."
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

def create_processed_data_preview(combined_data):
    if not combined_data.empty:
        preview_table = dbc.Table.from_dataframe(combined_data.head(), striped=True, bordered=True, hover=True)
        return html.Div([
            html.H5('Processed Data Preview'),
            preview_table
        ], style={'margin': '20px 0'})
    return html.Div("No data available for preview.", style={'color': 'red', 'margin': '20px 0'})


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
                return [overview, html.P(f"File Size: {file_size}")], {'display': 'block'}
    
            except Exception as e:
                logger.error(f"Error processing matrix file: {e}")
                return "Error processing matrix file", {'display': 'block'}
    
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
         Output('dynamic-content', 'children', allow_duplicate=True)],
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
            validate_csv(binning_data, ['Contig', 'Bin', 'Type'])
    
            taxonomy_data = parse_contents(bin_taxonomy, bin_taxonomy_name)
            validate_csv(taxonomy_data, ['Bin'], ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Plasmid ID'])
    
            logger.info("Validation successful. Proceeding with data preparation...")
            
            # Process data using parsed dataframes directly
            combined_data = process_data(contig_data, binning_data, taxonomy_data, contig_matrix_data, user_folder)
    
            if combined_data is not None:
                save_file_to_user_folder(contig_matrix, 'raw_contact_matrix.npz', user_folder)
                encoded_csv_content = base64.b64encode(combined_data.to_csv(index=False).encode()).decode()
                save_file_to_user_folder(f"data:text/csv;base64,{encoded_csv_content}", 'contig_info_final.csv', user_folder)
                
                preview_component = create_processed_data_preview(combined_data)
    
                logger.info("Data preparation successful, and files saved.")
                return True, preview_component
            else:
                logger.error("Data preparation failed.")
                return False, no_update
    
        except Exception as e:
            logger.error(f"Validation and preparation failed for Method 1: {e}")
            return False, no_update

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
            user_folder_path = f'assets/output/{user_folder}'
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
            logger.info("Validating normalized folder contents...")
            validate_normalized_folder(file_list)
    
            # Define the extraction path
            user_folder_path = f'assets/output/{user_folder}'
            os.makedirs(user_folder_path, exist_ok=True)
    
            # Extract the 7z file to the user's folder
            with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as archive:
                archive.extractall(path=user_folder_path)
            
            logger.info("Validation and extraction successful for Method 3.")
            return True  # Validation and extraction succeeded

        except Exception as e:
            logger.error(f"Validation and preparation failed for Method 3: {e}")
            return False  # Validation failed