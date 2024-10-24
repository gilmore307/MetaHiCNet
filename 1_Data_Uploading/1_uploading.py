import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io
import base64
import py7zr
import zipfile
from scipy.sparse import csr_matrix

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

stages_mapping = {
    'method1': ['File Uploading', 'Data Processing', 'Normalization', 'Spurious Contact Removal', 'Visualization'],
    'method2': ['File Uploading', 'Normalization', 'Spurious Contact Removal', 'Visualization'],
    'method3': ['File Uploading', 'Visualization']
}

# Function to generate the flowchart using buttons for stages_mapping
def create_flowchart(stage_highlight, method='method1'):

    method_stages_mapping = stages_mapping.get(method, stages_mapping['method1'])

    buttons = []
    for idx, stage in enumerate(method_stages_mapping):
        color = 'primary' if stage == stage_highlight else 'light'
        buttons.append(
            dbc.Button(stage, color=color, disabled=True, className="mx-2 my-2")
        )
        if idx < len(method_stages_mapping) - 1:
            # Add an arrow between stages_mapping
            buttons.append(html.Span("→", style={'font-size': '24px', 'margin': '0 10px'}))

    return html.Div(buttons, style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})

# Helper function to get the next stage in the sequence for the selected method
def get_next_stage(current_stage, method='method1'):
    method_stages_mapping = stages_mapping.get(method, stages_mapping['method1'])  # Get the stages for the selected method
    if current_stage in method_stages_mapping:
        current_index = method_stages_mapping.index(current_stage)
        if current_index < len(method_stages_mapping) - 1:
            return method_stages_mapping[current_index + 1]
    return current_stage  # Return the same stage if already at the last stage

# Function to parse contents of the uploaded files
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df
    elif 'npz' in filename:
        npzfile = np.load(io.BytesIO(decoded))
        return npzfile
    elif '7z' in filename:
        with zipfile.ZipFile(io.BytesIO(decoded)) as z:
            return z.namelist()  # Listing files for simplicity
    else:
        return None

# Function to get file size in a readable format
def get_file_size(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    size_in_bytes = len(decoded)
    size_in_kb = size_in_bytes / 1024
    return f"{size_in_kb:.2f} KB"

# Function to validate file formats based on method
def validate_file(df, required_columns, optional_columns=[]):
    all_columns = required_columns + optional_columns
    assert all(column in df.columns for column in all_columns), "Missing columns in the file."
    for col in required_columns:
        assert not df[col].isnull().any(), f"Required column '{col}' has missing values."
    return True

# Combined validation for matrix shape and self-contact
def validate_contig_matrix(contig_data, contact_matrix):
    num_contigs = len(contig_data)

    # Handle the case where the contact_matrix is an NpzFile in sparse format
    if isinstance(contact_matrix, np.lib.npyio.NpzFile):
        if all(key in contact_matrix for key in ['data', 'indices', 'indptr', 'shape']):
            # Reconstruct the sparse matrix correctly
            data = contact_matrix['data']
            indices = contact_matrix['indices']
            indptr = contact_matrix['indptr']
            shape = tuple(contact_matrix['shape'])  # Ensure shape is a tuple
            contact_matrix = csr_matrix((data, indices, indptr), shape=shape)
        else:
            raise ValueError("The contact matrix file does not contain the expected sparse matrix keys.")

    matrix_shape = contact_matrix.shape

    assert matrix_shape[0] == matrix_shape[1], "The contact matrix is not square."
    assert matrix_shape[0] == num_contigs, \
        f"The contact matrix dimensions {matrix_shape} do not match the number of contigs in the information table."

    if 'Self-contact' in contig_data.columns:
        diagonal_values = np.diag(contact_matrix.toarray())  # Convert sparse to dense for diagonal extraction
        self_contact = contig_data['Self-contact'].dropna()
        assert np.allclose(self_contact, diagonal_values[:len(self_contact)]), \
            "The 'Self-contact' column values do not match the diagonal of the contact matrix."

    return True

# Function to validate the contents of the unnormalized folder
def validate_unnormalized_folder(folder):
    expected_files = ['bin_info_final.csv', 'contig_info_final.csv', 'raw_contact_matrix.npz']
    missing_files = [file for file in expected_files if file not in folder]
    if missing_files:
        raise ValueError(f"Missing files in unnormalized folder: {', '.join(missing_files)}")
    return True

# Function to validate the contents of the normalized folder
def validate_normalized_folder(folder):
    expected_files = ['bin_info_final.csv', 'contig_info_final.csv', 'contig_contact_matrix.npz', 'bin_contact_matrix.npz']
    missing_files = [file for file in expected_files if file not in folder]
    if missing_files:
        raise ValueError(f"Missing files in normalized folder: {', '.join(missing_files)}")
    return True

# Function to extract and list files from a .7z archive
def list_files_in_7z(decoded):
    with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as z:
        file_list = z.getnames()
    return file_list

# Define a helper function for creating upload components
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
            # File overview container
            html.Div(id=f'overview-{component_id}', style={'padding': '10px'}),
            # Remove file button
            dbc.Button("Remove File", id=f'remove-{component_id}', color="danger", style={'display': 'none'}),
            # Hidden div to reset file input
            dcc.Store(id=f'store-{component_id}')
        ],
        body=True,
        className="my-3"
    )

# Define the layout of the app
app.layout = dbc.Container([
    html.H1("Meta Hi-C Visualization", className="my-4 text-center"),

    # Store to hold the current stage for each method
    dcc.Store(id='current-stage-method1', data='File Uploading'),
    dcc.Store(id='current-stage-method2', data='File Uploading'),
    dcc.Store(id='current-stage-method3', data='File Uploading'),

    dcc.Tabs(id='tabs-method', value='method1', children=[
        # Method 1: Raw Data Uploads
        dcc.Tab(label='First-time users: Upload raw data', value='method1', children=[
            html.Div(id='flowchart-container-method1'),
            dbc.Row([
                dbc.Col(create_upload_component(
                    'raw-contig-info',
                    'Upload Contig Information File (.csv)', 
                    '/examples/contig_information.csv',
                    "This file must include the following columns: 'Contig', 'Restriction sites', 'Length', 'Coverage', and 'Self-contact'."
                )),
                dbc.Col(create_upload_component(
                    'raw-contig-matrix',
                    'Upload Raw Contact Matrix File (.npz)', 
                    '/examples/raw_contact_matrix.npz',
                    "The Unnormalized Contact Matrix must include the following keys: 'indices', 'indptr', 'format', 'shape', 'data'."
                )),
            ]),
            dbc.Row([
                dbc.Col(create_upload_component(
                    'raw-binning-info',
                    'Upload Binning Information File (.csv)', 
                    '/examples/binning_information.csv',
                    "This file must include the following columns: 'Contig', 'Bin', and 'Type'."
                )),
                dbc.Col(create_upload_component(
                    'raw-bin-taxonomy',
                    'Upload Bin Taxonomy File (.csv)', 
                    '/examples/taxonomy.csv',
                    "This file must include the following columns: 'Bin', 'Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Plasmid ID'."
                )),
            ]),
            dbc.Button("Validate All Files", id="validate-button", color="success", className="mt-3"),
            html.Div(id="validation-output", style={'padding': '0px', 'color': 'green'}),
        ]),

        # Method 2: Unnormalized Data Uploads
        dcc.Tab(label='Change normalization method: Upload unnormalized data', value='method2', children=[
            html.Div(id='flowchart-container-method2'),
            dbc.Row([
                dbc.Col(create_upload_component(
                    'unnormalized-data-folder',
                    'Upload Unnormalized Data Folder (.7z)', 
                    '/examples/unnormalized_information.7z',
                    "Please upload the 'unnormalized_information' folder generated from your previous visualization.  \n"
                    "It must include the following files: 'bin_info_final.csv', 'contig_info_final.csv', 'raw_contact_matrix.npz'."
                )),
            ]),
            dbc.Button("Validate All Files", id="validate-button-unnormalized", color="success", className="mt-3"),
            html.Div(id="validation-output-unnormalized", style={'padding': '0px', 'color': 'green'})
        ]),

        # Method 3: Normalized Data Uploads
        dcc.Tab(label='Continue previous visualization: Upload normalized data', value='method3', children=[
            html.Div(id='flowchart-container-method3'),
            dbc.Row([ 
                dbc.Col(create_upload_component(
                    'normalized-data-folder',
                    'Upload Visualization Data Folder (.7z)', 
                    '/examples/normalized_information.7z',
                    "Please upload the 'normalized_information' folder generated from your previous visualization.  \n"
                    "It must include the following files: 'bin_info_final.csv', 'contig_info_final.csv', 'contig_contact_matrix.npz', 'bin_contact_matrix.npz'."
                )),
            ]),
            dbc.Button("Validate All Files", id="validate-button-normalized", color="success", className="mt-3"),
            html.Div(id="validation-output-normalized", style={'padding': '0px', 'color': 'green'})
        ]),
    ]),
], fluid=True)

# Callback to update the flowchart based on the selected tab and current stage
@app.callback(
    [Output('flowchart-container-method1', 'children'),
     Output('flowchart-container-method2', 'children'),
     Output('flowchart-container-method3', 'children')],
    [Input('tabs-method', 'value'),
     Input('current-stage-method1', 'data'),
     Input('current-stage-method2', 'data'),
     Input('current-stage-method3', 'data')]
)
def update_flowchart(selected_method, stage_method1, stage_method2, stage_method3):
    if selected_method == 'method1':
        return create_flowchart(stage_method1, method='method1'), None, None
    elif selected_method == 'method2':
        return None, create_flowchart(stage_method2, method='method2'), None
    elif selected_method == 'method3':
        return None, None, create_flowchart(stage_method3, method='method3')

# Callback to store file contents and show overview with remove button for contig information
@app.callback(
    [Output('overview-raw-contig-info', 'children'),
     Output('remove-raw-contig-info', 'style'),
     Output('raw-contig-info', 'contents')],
    [Input('raw-contig-info', 'contents'),
     Input('remove-raw-contig-info', 'n_clicks')],
    [State('raw-contig-info', 'filename')]
)
def handle_contig_info_upload(contents, remove_click, filename):
    ctx = dash.callback_context
    if not contents:
        return '', {'display': 'none'}, None
    
    # If remove button is clicked, clear the overview and reset the input
    if remove_click and ctx.triggered_id == 'remove-raw-contig-info':
        return '', {'display': 'none'}, None
    
    # Otherwise, show the file overview
    file_size = get_file_size(contents)
    if 'csv' in filename:
        df = parse_contents(contents, filename)
        return [dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
                html.P(f"File Size: {file_size}")], {'display': 'block'}, contents
    
    elif 'npz' in filename:
        npzfile = np.load(io.BytesIO(base64.b64decode(contents.split(',')[1])))
        overview = html.Ul([html.Li(file) for file in npzfile.files])
        return [overview, html.P(f"File Size: {file_size}")], {'display': 'block'}, contents
    
    elif '7z' in filename:
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(contents.split(',')[1]))) as z:
            overview = html.Ul([html.Li(f) for f in z.namelist()])
        return [overview, html.P(f"File Size: {file_size}")], {'display': 'block'}, contents
    
    return "Unsupported file format", {'display': 'block'}, contents

# Callbacks for other file uploads (re-upload mechanism added)
@app.callback(
    [Output('overview-raw-contig-matrix', 'children'),
     Output('remove-raw-contig-matrix', 'style'),
     Output('raw-contig-matrix', 'contents')],
    [Input('raw-contig-matrix', 'contents'),
     Input('remove-raw-contig-matrix', 'n_clicks')],
    [State('raw-contig-matrix', 'filename')]
)
def handle_raw_matrix_uploadw(contents, remove_click, filename):
    ctx = dash.callback_context
    if not contents:
        return '', {'display': 'none'}, None
    
    if remove_click and ctx.triggered_id == 'remove-raw-contig-matrix':
        return '', {'display': 'none'}, None
    
    file_size = get_file_size(contents)
    npzfile = np.load(io.BytesIO(base64.b64decode(contents.split(',')[1])))
    overview = html.Ul([html.Li(file) for file in npzfile.files])
    return [overview, html.P(f"File Size: {file_size}")], {'display': 'block'}, contents

# Callback for binning information
@app.callback(
    [Output('overview-raw-binning-info', 'children'),
     Output('remove-raw-binning-info', 'style'),
     Output('raw-binning-info', 'contents')],
    [Input('raw-binning-info', 'contents'),
     Input('remove-raw-binning-info', 'n_clicks')],
    [State('raw-binning-info', 'filename')]
)
def handle_binning_info_upload(contents, remove_click, filename):
    ctx = dash.callback_context
    if not contents:
        return '', {'display': 'none'}, None
    
    if remove_click and ctx.triggered_id == 'remove-raw-binning-info':
        return '', {'display': 'none'}, None
    
    file_size = get_file_size(contents)
    df = parse_contents(contents, filename)
    return [dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
            html.P(f"File Size: {file_size}")], {'display': 'block'}, contents

# Callback for taxonomy information
@app.callback(
    [Output('overview-raw-bin-taxonomy', 'children'),
     Output('remove-raw-bin-taxonomy', 'style'),
     Output('raw-bin-taxonomy', 'contents')],
    [Input('raw-bin-taxonomy', 'contents'),
     Input('remove-raw-bin-taxonomy', 'n_clicks')],
    [State('raw-bin-taxonomy', 'filename')]
)
def handle_bin_taxonomy_upload(contents, remove_click, filename):
    ctx = dash.callback_context
    if not contents:
        return '', {'display': 'none'}, None
    
    if remove_click and ctx.triggered_id == 'remove-raw-bin-taxonomy':
        return '', {'display': 'none'}, None
    
    file_size = get_file_size(contents)
    df = parse_contents(contents, filename)
    return [dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
            html.P(f"File Size: {file_size}")], {'display': 'block'}, contents

# Validation logic for Method 1
@app.callback(
    [Output('current-stage-method1', 'data'),
     Output('validation-output', 'children')],
    [Input('validate-button', 'n_clicks')],
    [State('raw-contig-info', 'contents'),
     State('raw-contig-matrix', 'contents'),
     State('raw-binning-info', 'contents'),
     State('raw-bin-taxonomy', 'contents'),
     State('raw-contig-info', 'filename'),
     State('raw-contig-matrix', 'filename'),
     State('raw-binning-info', 'filename'),
     State('raw-bin-taxonomy', 'filename'),
     State('current-stage-method1', 'data')]
)
def validate_method_1(n_clicks, contig_info, contig_matrix, binning_info, bin_taxonomy, contig_info_name, contig_matrix_name, binning_info_name, bin_taxonomy_name, current_stage):
    if n_clicks is None or not all([contig_info, contig_matrix, binning_info, bin_taxonomy]):
        return current_stage, "Please upload all required files to validate."

    try:
        # Validate contig information file
        contig_data = parse_contents(contig_info, contig_info_name)
        required_columns = ['Contig', 'Restriction sites', 'Length', 'Coverage']
        validate_file(contig_data, required_columns, optional_columns=['Self-contact'])

        # Validate contig matrix
        contig_matrix_data = parse_contents(contig_matrix, contig_matrix_name)
        validate_contig_matrix(contig_data, contig_matrix_data)

        # Validate binning information file
        binning_data = parse_contents(binning_info, binning_info_name)
        required_columns = ['Contig', 'Bin', 'Type']
        validate_file(binning_data, required_columns)

        # Validate bin taxonomy file
        taxonomy_data = parse_contents(bin_taxonomy, bin_taxonomy_name)
        required_columns = ['Bin']
        optional_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Plasmid ID']
        validate_file(taxonomy_data, required_columns, optional_columns)

        # If validation passes, move to the next stage
        return get_next_stage(current_stage, method='method1'), "All files successfully validated!"

    except Exception as e:
        # If validation fails, remain in the current stage and show an error message
        return current_stage, f"Validation failed: {str(e)}"

# Callback for Method 2 file uploads (unnormalized folder) with validation and file overview
@app.callback(
    [Output('overview-unnormalized-data-folder', 'children'),
     Output('remove-unnormalized-data-folder', 'style'),
     Output('unnormalized-data-folder', 'contents')],
    [Input('unnormalized-data-folder', 'contents'),
     Input('remove-unnormalized-data-folder', 'n_clicks')],
    [State('unnormalized-data-folder', 'filename')]
)
def handle_unnormalized_folder_upload(contents, remove_click, filename):
    ctx = dash.callback_context
    if not contents:
        return '', {'display': 'none'}, None
    
    if remove_click and ctx.triggered_id == 'remove-unnormalized-data-folder':
        return '', {'display': 'none'}, None

    # Validate the contents of the .7z file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        file_list = list_files_in_7z(decoded)
        validate_unnormalized_folder(file_list)
    except ValueError as e:
        return f"Validation failed: {str(e)}", {'display': 'none'}, None
    except Exception as e:
        return f"Error opening .7z file: {str(e)}", {'display': 'none'}, None

    # Create an overview of the files inside the folder
    overview = html.Ul([html.Li(file) for file in file_list])
    
    file_size = get_file_size(contents)
    return [overview, html.P(f"File uploaded: {filename} ({file_size})")], {'display': 'block'}, contents

# Validation logic for Method 2
@app.callback(
    [Output('current-stage-method2', 'data'),
     Output('validation-output-unnormalized', 'children')],
    [Input('validate-button-unnormalized', 'n_clicks')],
    [State('unnormalized-data-folder', 'contents'),
     State('unnormalized-data-folder', 'filename'),
     State('current-stage-method2', 'data')]
)
def validate_method_2(n_clicks, contents, filename, current_stage):
    if n_clicks is None or contents is None:
        return current_stage, "No file uploaded. Please upload a file to validate."

    # Validate the contents of the .7z file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        file_list = list_files_in_7z(decoded)
        validate_unnormalized_folder(file_list)
        # If validation passes, move to the next stage
        return get_next_stage(current_stage, method='method2'), "Unnormalized folder successfully validated!"
    
    except Exception as e:
        # If validation fails, remain in the current stage and show an error message
        return current_stage, f"Validation failed: {str(e)}"


# Callback for Method 3 file uploads (normalized folder) with validation and file overview
@app.callback(
    [Output('overview-normalized-data-folder', 'children'),
     Output('remove-normalized-data-folder', 'style'),
     Output('normalized-data-folder', 'contents')],
    [Input('normalized-data-folder', 'contents'),
     Input('remove-normalized-data-folder', 'n_clicks')],
    [State('normalized-data-folder', 'filename')]
)
def handle_normalized_upload(contents, remove_click, filename):
    ctx = dash.callback_context
    if not contents:
        return '', {'display': 'none'}, None
    
    if remove_click and ctx.triggered_id == 'remove-normalized-data-folder':
        return '', {'display': 'none'}, None

    # Validate the contents of the .7z file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        file_list = list_files_in_7z(decoded)
        validate_normalized_folder(file_list)
    except ValueError as e:
        return f"Validation failed: {str(e)}", {'display': 'none'}, None
    except Exception as e:
        return f"Error opening .7z file: {str(e)}", {'display': 'none'}, None

    # Create an overview of the files inside the folder
    overview = html.Ul([html.Li(file) for file in file_list])
    
    file_size = get_file_size(contents)
    return [overview, html.P(f"File uploaded: {filename} ({file_size})")], {'display': 'block'}, contents

# Validation logic for Method 3
@app.callback(
    [Output('current-stage-method3', 'data'),
     Output('validation-output-normalized', 'children')],
    [Input('validate-button-normalized', 'n_clicks')],
    [State('normalized-data-folder', 'contents'),
     State('normalized-data-folder', 'filename'),
     State('current-stage-method3', 'data')]
)
def validate_method_3(n_clicks, contents, filename, current_stage):
    if n_clicks is None or contents is None:
        return current_stage, "No file uploaded. Please upload a file to validate."

    # Validate the contents of the .7z file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        # Extract the list of files in the uploaded .7z file
        file_list = list_files_in_7z(decoded)

        # Validate that the expected files are present
        validate_normalized_folder(file_list)

        # If validation passes, move to the next stage
        return get_next_stage(current_stage, method='method3'), "Normalized folder successfully validated!"

    except Exception as e:
        # If validation fails, remain in the current stage and return the error message
        return current_stage, f"Validation failed: {str(e)}"

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)