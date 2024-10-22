import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io
import base64
import zipfile

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
        # Assuming it's a compressed 7z folder and we will list files (can modify this for extraction later)
        with zipfile.ZipFile(io.BytesIO(decoded)) as z:
            return z.namelist()  # Listing files for simplicity
    else:
        return None

# Function to validate file formats based on method
def validate_file(df, required_columns, optional_columns=[]):
    # Combine required and optional columns, marking required ones for stricter validation
    all_columns = required_columns + optional_columns

    # Check that all columns (required and optional) exist in the DataFrame
    assert all(column in df.columns for column in all_columns), "Missing columns in the file."

    # Check that required columns do not have missing values
    for col in required_columns:
        assert not df[col].isnull().any(), f"Required column '{col}' has missing values."

    return True

# Combined validation for matrix shape and self-contact
def validate_contig_matrix(contig_data, contact_matrix):
    num_contigs = len(contig_data)
    matrix_shape = contact_matrix.shape

    assert matrix_shape[0] == matrix_shape[1], "The contact matrix is not square."
    assert matrix_shape[0] == num_contigs, \
        f"The contact matrix dimensions {matrix_shape} do not match the number of contigs in the information table."

    if 'Self-contact' in contig_data.columns:
        diagonal_values = np.diag(contact_matrix)
        self_contact = contig_data['Self-contact'].dropna()
        assert np.allclose(self_contact, diagonal_values[:len(self_contact)]), \
            "The 'Self-contact' column values do not match the diagonal of the contact matrix."

    return True

# Custom upload component with download link and specific instructions
def create_upload_component(id, text, example_url, instructions):
    return dbc.Card(
        [
            dcc.Upload(
                id=id,
                children=dbc.Button(text, color="primary", className="me-2", style={"width": "100%"}),
                multiple=False,
                style={'textAlign': 'center'}
            ),
            dbc.Row(
                [
                    dbc.Col(html.A('Download Example File', href=example_url, target="_blank", style={'textAlign': 'center'})),
                ],
                style={'padding': '5px'}
            ),
            dbc.CardBody([
                html.H6("Instructions:"),
                dcc.Markdown(instructions, style={'fontSize': '0.9rem', 'color': '#555'})
            ]),
            html.Div(id=f'output-{id}', style={'padding': '10px'})
        ],
        body=True,
        className="my-3"
    )

# Dash layout
app.layout = dbc.Container([
    html.H1("File Upload System", className="my-4 text-center"),
    
    dcc.Tabs([

        # Method 1: Raw Data Uploads
        dcc.Tab(label='First-time users: Upload raw data', children=[
            dbc.Row([
                dbc.Col(create_upload_component(
                    'raw-contig-info',
                    'Upload Contig Information File (.csv)', 
                    '/examples/contig_information.csv',
                    "This file must include the following columns: 'Contig', 'Restriction sites', 'Length', 'Coverage', and 'Self-contact'.  \n"
                    "'Self-contact' is optional, leave it blank if it is unavailable."
                )),
                dbc.Col(create_upload_component(
                    'raw-contig-matrix',
                    'Upload Raw Contact Matrix File (.npz)', 
                    '/examples/raw_contact_matrix.npz',
                    "Please upload the Unnormalized Contact Matrix File."
                )),
            ]),
            dbc.Row([
                dbc.Col(create_upload_component(
                    'raw-binning-info',
                    'Upload Binning Information File (.csv)', 
                    '/examples/binning_information.csv',
                    "This file must include the following columns: 'Contig', 'Bin', and 'Type'.  \n"
                    "Sequence of rows does not matter.  \n"
                    "Use the contig name as the Bin if binning is not performed."
                )),
                dbc.Col(create_upload_component(
                    'raw-bin-taxonomy',
                    'Upload Bin Taxonomy File (.csv)', 
                    '/examples/taxonomy.csv',
                    "This file must include the following columns: 'Bin', 'Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Plasmid ID'.  \n"
                    "Leave taxonomy columns blank if data is unavailable or unmapped.  \n"
                    "Taxonomy columns are unnecessary if there is a plasmid ID available."
                )),
            ])
        ]),

        # Method 2: Unnormalized Data Uploads (Changed to folder-based upload)
        dcc.Tab(label='Change normalization method: Upload unnormalized data', children=[
            dbc.Row([
                dbc.Col(create_upload_component(
                    'unnormalized-data-folder',
                    'Upload Unnormalized Data Folder (.7z)', 
                    '/examples/unnormalized_information.7z',
                    "Please upload the 'unnormalized_information' folder generated from your previous visualization."
                )),
            ])
        ]),

        # Method 3: Normalized Data Uploads (Already folder-based)
        dcc.Tab(label='Continue previous visualization: Upload normalized data', children=[
            dbc.Row([
                dbc.Col(create_upload_component(
                    'normalized-data-folder',
                    'Upload Visualization Data Folder (.7z)', 
                    '/examples/normalized_information.7z',
                    "Please upload the 'normalized_information' folder generated from your previous visualization."
                )),
            ])
        ]),
    ]),
], fluid=True)

# Callback for Method 1 file uploads
@app.callback(
    [Output('output-raw-contig-info', 'children'),
     Output('output-raw-contig-matrix', 'children'),
     Output('output-raw-binning-info', 'children'),
     Output('output-raw-bin-taxonomy', 'children')],
    [Input('raw-contig-info', 'contents'),
     Input('raw-contig-matrix', 'contents'),
     Input('raw-binning-info', 'contents'),
     Input('raw-bin-taxonomy', 'contents')],
    [State('raw-contig-info', 'filename'),
     State('raw-contig-matrix', 'filename'),
     State('raw-binning-info', 'filename'),
     State('raw-bin-taxonomy', 'filename')]
)
def handle_raw_upload(contig_info, contig_matrix, binning_info, bin_taxonomy, contig_info_name, contig_matrix_name, binning_info_name, bin_taxonomy_name):
    try:
        # Handle the contig information file
        contig_matrix_output = 'No contig matrix file uploaded.'
        contig_matrix_data = None
        if contig_matrix and contig_matrix_name:
            contig_matrix_data = parse_contents(contig_matrix, contig_matrix_name)
            contig_matrix_output = 'Contig matrix file uploaded.'

        if contig_info and contig_info_name:
            contig_data = parse_contents(contig_info, contig_info_name)
            required_columns = ['Contig', 'Restriction sites', 'Length', 'Coverage']
            validate_file(contig_data, required_columns, optional_columns=['Self-contact'])
            
            if contig_matrix_data is not None:
                validate_contig_matrix(contig_data, contig_matrix_data)
            
            contig_info_output = 'Contig information file uploaded and validated.'
        else:
            contig_info_output = 'No contig information file uploaded.'

        # Handle the binning information file
        if binning_info and binning_info_name:
            binning_data = parse_contents(binning_info, binning_info_name)
            required_columns = ['Contig', 'Bin', 'Type']
            validate_file(binning_data, required_columns)
            binning_info_output = 'Binning information file uploaded and validated.'
        else:
            binning_info_output = 'No binning information file uploaded.'

        # Handle the bin taxonomy file
        if bin_taxonomy and bin_taxonomy_name:
            taxonomy_data = parse_contents(bin_taxonomy, bin_taxonomy_name)
            required_columns = ['Bin']  # Only 'Bin' is required
            optional_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Plasmid ID']
            validate_file(taxonomy_data, required_columns, optional_columns)
            bin_taxonomy_output = 'Bin taxonomy file uploaded and validated.'
        else:
            bin_taxonomy_output = 'No bin taxonomy file uploaded.'

        return contig_info_output, contig_matrix_output, binning_info_output, bin_taxonomy_output
    except Exception as e:
        return str(e), str(e), str(e), str(e)

# Callback for Method 2 file uploads (now folder-based)
@app.callback(
    Output('output-unnormalized-data-folder', 'children'),
    [Input('unnormalized-data-folder', 'contents')],
    [State('unnormalized-data-folder', 'filename')]
)
def handle_unnormalized_folder_upload(folder, folder_name):
    try:
        if folder and folder_name:
            folder_data = parse_contents(folder, folder_name)
            folder_output = 'Unnormalized data folder uploaded.'
        else:
            folder_output = 'No unnormalized data folder uploaded.'
        return folder_output
    except Exception as e:
        return str(e)

# Callback for Method 3 file uploads (remains folder-based with .7z)
@app.callback(
    Output('output-normalized-data-folder', 'children'),
    [Input('normalized-data-folder', 'contents')],
    [State('normalized-data-folder', 'filename')]
)
def handle_normalized_upload(folder, folder_name):
    try:
        if folder and folder_name:
            folder_data = parse_contents(folder, folder_name)
            folder_output = 'Normalized data folder uploaded.'
        else:
            folder_output = 'No normalized data folder uploaded.'
        return folder_output
    except Exception as e:
        return str(e)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
