import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csc_matrix
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import plotly.graph_objects as go
from dash import callback_context

# File paths for the current environment
contig_info_path = '../0_Documents/contig_information.csv'
raw_contact_matrix_path = '../0_Documents/raw_contact_matrix.npz'

# Load the data
contact_matrix_data = np.load(raw_contact_matrix_path)
data = contact_matrix_data['data']
indices = contact_matrix_data['indices']
indptr = contact_matrix_data['indptr']
shape = contact_matrix_data['shape']
sparse_matrix = csc_matrix((data, indices, indptr), shape=shape)
dense_matrix = sparse_matrix.toarray()

contig_information = pd.read_csv(contig_info_path)

# Calculate the total intra-species contacts
intra_species_contacts = contig_information.groupby('Contig annotation').apply(
    lambda group: dense_matrix[group.index][:, group.index].sum()
)

# Calculate the total inter-species contacts between annotations
unique_annotations = contig_information['Contig annotation'].unique()
inter_species_contacts = pd.DataFrame(0, index=unique_annotations, columns=unique_annotations)

for annotation_i in unique_annotations:
    for annotation_j in unique_annotations:
        if annotation_i != annotation_j:
            contacts = dense_matrix[
                contig_information[contig_information['Contig annotation'] == annotation_i].index
            ][:,
                contig_information[contig_information['Contig annotation'] == annotation_j].index
            ].sum()
            inter_species_contacts.at[annotation_i, annotation_j] = contacts

# Function for basic visualization setup
def basic_visualization():
    # Create a graph for the Arc Diagram
    G = nx.Graph()

    # Add nodes with size based on intra-species contacts
    max_intra_species_contacts = intra_species_contacts.max()
    is_viral_colors = {'True': '#F4B084', 'False': '#8EA9DB'}  # Red for viral, blue for non-viral
    for annotation, contacts in intra_species_contacts.items():
        is_viral = contig_information.loc[
            contig_information['Contig annotation'] == annotation, 'Is Viral'
        ].values[0]
        color = is_viral_colors[str(is_viral)]
        size = (contacts / max_intra_species_contacts) * 100  # Scale node size for plotly
        G.add_node(annotation, size=size, color=color, contacts=contacts)

    # Add edges with weight based on inter-species contacts
    for annotation_i in unique_annotations:
        for annotation_j in unique_annotations:
            if annotation_i != annotation_j:
                if inter_species_contacts.at[annotation_i, annotation_j] > 0:
                    G.add_edge(annotation_i, annotation_j, contacts=inter_species_contacts.at[annotation_i, annotation_j])

    # Initial node positions using a force-directed layout with increased dispersion
    pos = nx.spring_layout(G, dim=2, k=2, iterations=50)

    cyto_elements = nx_to_cyto_elements(G, pos)
    cyto_stylesheet = base_stylesheet.copy()
    return cyto_elements, cyto_stylesheet, G

# Function to convert NetworkX graph to Cytoscape elements
def nx_to_cyto_elements(G, pos):
    elements = []
    for node in G.nodes:
        elements.append({
            'data': {
                'id': node,
                'label': node,
                'size': G.nodes[node]['size'],
                'color': G.nodes[node]['color']
            },
            'position': {
                'x': pos[node][0] * 500,
                'y': pos[node][1] * 500
            }
        })
    for edge in G.edges(data=True):
        elements.append({
            'data': {
                'source': edge[0],
                'target': edge[1],
                'weight': edge[2]['contacts']
            }
        })
    return elements

# Base stylesheet for Cytoscape
base_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'width': 'data(size)',
            'height': 'data(size)',
            'background-color': 'data(color)',
            'label': 'data(label)',
        }
    },
    {
        'selector': 'edge',
        'style': {
            'width': 2,
            'line-color': '#ccc',
        }
    }
]

# Function to create a bar chart
def create_bar_chart(row_contig, col_contig=None):
    bar_data = inter_species_contacts.loc[row_contig].sort_values()  # Sort the data
    bar_colors = []
    for col in bar_data.index:
        if col == col_contig:
            bar_colors.append('rgba(128,0,128,0.8)')  # Highlight color (purple) for selected edge
        elif col == row_contig:
            bar_colors.append('rgba(255,165,0,0.8)')  # Row species color
        else:
            bar_colors.append('rgba(0,128,0,0.8)')  # Default edge color
    bar_trace = go.Bar(x=bar_data.index, y=bar_data.values, marker_color=bar_colors)
    bar_layout = go.Layout(
        title=f"Inter-species Contacts for {row_contig}",
        xaxis=dict(showticklabels=False),  # Remove x-axis tick labels
        yaxis=dict(title="Contact Count", showticklabels=False),  # Remove y-axis tick labels
        height=720,  # Adjusted height
        margin=dict(t=30, b=0, l=0, r=0)  # Adjusted margin
    )
    bar_fig = go.Figure(data=[bar_trace], layout=bar_layout)
    return bar_fig

# Function to get the matrix of intra- and inter-species contacts
def get_combined_matrix(annotation_i, annotation_j=None):
    if annotation_j is None:
        annotation_j = annotation_i
    indices_i = contig_information[contig_information['Contig annotation'] == annotation_i].index
    indices_j = contig_information[contig_information['Contig annotation'] == annotation_j].index
    combined_indices = indices_i.union(indices_j)
    matrix = dense_matrix[combined_indices][:, combined_indices]
    contig_names = contig_information.loc[combined_indices, 'Contig name']
    matrix_df = pd.DataFrame(matrix, index=contig_names, columns=contig_names)
    return matrix_df

# Function to create conditional styles for the DataTable
def create_conditional_styles(matrix_df):
    styles = []
    numeric_df = matrix_df.select_dtypes(include=[np.number])
    log_max_value = np.log1p(numeric_df.values.max())
    for i in range(len(numeric_df)):
        for j in range(len(numeric_df.columns)):
            value = numeric_df.iloc[i, j]
            log_value = np.log1p(value)
            opacity = 0.6  # Set a fixed opacity for transparency
            styles.append({
                'if': {
                    'row_index': i,
                    'column_id': numeric_df.columns[j]
                },
                'backgroundColor': f'rgba({255 - int(log_value / log_max_value * 255)}, {255 - int(log_value / log_max_value * 255)}, 255, {opacity})' # Set background color for the contact matrix.
            })
    return styles

# Function to arrange contigs
def arrange_contigs(contigs, inter_contig_edges, radius=0.6, jitter=0.05):
    # Identify contigs that connect to other species
    connecting_contigs = [contig for contig in contigs if contig in inter_contig_edges]
    other_contigs = [contig for contig in contigs if contig not in connecting_contigs]

    # Arrange connecting contigs in a circle with jitter
    num_connecting = len(connecting_contigs)
    circle_positions = {}
    angle_step = 2 * np.pi / num_connecting if num_connecting > 0 else 0
    for i, contig in enumerate(connecting_contigs):
        angle = i * angle_step
        x = radius * np.cos(angle) + np.random.uniform(-jitter, jitter)
        y = radius * np.sin(angle) + np.random.uniform(-jitter, jitter)
        circle_positions[contig] = (x, y)

    # Arrange other contigs randomly inside the circle
    inner_positions = {}
    for contig in other_contigs:
        r = radius * 0.8 * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        inner_positions[contig] = (x, y)

    return {**circle_positions, **inner_positions}

# Convert inter_species_contacts and intra_species_contacts to a DataFrame
inter_intra_species_df = inter_species_contacts.copy()
for annotation in unique_annotations:
    inter_intra_species_df.at[annotation, annotation] = intra_species_contacts.get(annotation, 0)

inter_intra_species_df.insert(0, 'Species', inter_intra_species_df.index)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Set the global graph G
cyto_elements, cyto_stylesheet, G = basic_visualization()

common_style = {
    'height': '38px',  # Adjusted height to match both elements
    'display': 'inline-block',
    'margin-right': '10px',
    'vertical-align': 'middle'
}

# Help page (empty for now)
help_modal = html.Div([
    dbc.Modal([
        dbc.ModalHeader("Help"),
        dbc.ModalBody([
            # help content will be added soon
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-help", className="ml-auto")
        )
    ], id="modal", size="lg", is_open=False)
])

# Layout for the Dash app
app.layout = html.Div([
    html.Div([
        html.Button("Download Selected Item", id="download-btn", style={**common_style}),
        html.Button("Reset Selection", id="reset-btn", style={**common_style}),
        html.Button("Help", id="open-help", style={**common_style}),
        dcc.Download(id="download-dataframe-csv"),
        dcc.Dropdown(
            id='visualization-selector',
            options=[
                {'label': 'Species', 'value': 'species'},
                {'label': 'Species Contact', 'value': 'species_contact'},
                {'label': 'Contig', 'value': 'contig'}
            ],
            value='species',
            style={'width': '300px', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='species-selector',
            options=[{'label': species, 'value': species} for species in unique_annotations],
            value=None,
            placeholder="Select a species",
            style={'width': '300px', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='secondary-species-selector',
            options=[{'label': species, 'value': species} for species in unique_annotations],
            value=None,
            placeholder="Select a secondary species",
            style={'width': '300px', 'display': 'none'}  # Hide by default
        ),
        dcc.Dropdown(
            id='contig-selector',
            options=[],
            value=None,
            placeholder="Select a contig",
            style={'width': '300px', 'display': 'none'}  # Hide by default
        )
    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin': '20px'}),
    html.Div([
        html.Div([
            dcc.Graph(id='bar-chart', style={'height': '80vh', 'width': '30vw', 'display': 'inline-block'})
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            cyto.Cytoscape(
                id='cyto-graph',
                elements=cyto_elements,
                style={'height': '80vh', 'width': '60vw', 'display': 'inline-block'},
                layout={'name': 'preset'},  # Use preset to keep the initial positions
                stylesheet=cyto_stylesheet,
                userZoomingEnabled=False,  # Disable zooming
                userPanningEnabled=False,  # Disable panning
                autolock=True  # Disable node movement
            )
        ], style={'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'width': '100%', 'display': 'flex'}),
    html.Div([
        dash_table.DataTable(
            id='contact-table',
            columns=[{"name": col, "id": col} for col in inter_intra_species_df.columns],
            data=inter_intra_species_df.to_dict('records'),
            style_table={'height': 'auto', 'overflowY': 'auto', 'overflowX': 'auto', 'width': '99vw', 'minWidth': '100%'},
            style_data_conditional=create_conditional_styles(inter_intra_species_df),
            style_cell={'textAlign': 'left', 'minWidth': '120px', 'width': '120px', 'maxWidth': '180px'},
            fixed_rows={'headers': True},  # Freeze the first row
            fixed_columns={'headers': True, 'data': 1}  # Freeze the first column
        )
    ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
    help_modal
], style={'height': '100vh', 'overflowY': 'auto', 'width': '100%'})


def get_contig_nodes(species):
    contigs = contig_information[contig_information['Contig annotation'] == species]['Contig name']
    return contigs

# Visualization functions
def species_visualization(active_cell, selected_species, table_data):
    row_contig = table_data[active_cell['row']]['Species'] if active_cell else selected_species
    col_contig = active_cell['column_id'] if active_cell else None

    G_copy = G.copy()

    row_contacts = inter_species_contacts.loc[row_contig]

    min_node_size = 100
    max_node_size = 500

    # Calculate node sizes proportionally
    max_contigs = max([len(contig_information[contig_information['Contig annotation'] == node]) for node in G_copy.nodes])
    min_contigs = min([len(contig_information[contig_information['Contig annotation'] == node]) for node in G_copy.nodes])

    nodes_to_remove = []
    selected_node_size = None
    for node in G_copy.nodes:
        num_contigs = len(contig_information[contig_information['Contig annotation'] == node])
        scaled_size = min_node_size + (max_node_size - min_node_size) * (num_contigs - min_contigs) / (max_contigs - min_contigs)
        
        if node == row_contig:
            G_copy.nodes[node]['size'] = scaled_size
            selected_node_size = scaled_size
            G_copy.nodes[node]['color'] = '#A9D08E'  # Green for selected node
        else:
            num_connected_contigs = len(contig_information[(contig_information['Contig annotation'] == node) & (dense_matrix[:, contig_information[contig_information['Contig annotation'] == row_contig].index].sum(axis=1) > 0)])
            if num_connected_contigs == 0:
                nodes_to_remove.append(node)
            else:
                scaled_size = min_node_size + (max_node_size - min_node_size) * (num_connected_contigs - min_contigs) / (max_contigs - min_contigs)
                G_copy.nodes[node]['size'] = scaled_size
                is_viral = contig_information.loc[
                    contig_information['Contig annotation'] == node, 'Is Viral'
                ].values[0]
                G_copy.nodes[node]['color'] = '#F4B084' if is_viral else '#8EA9DB'  # Red for viral, blue for non-viral

    for node in nodes_to_remove:
        G_copy.remove_node(node)

    edges_to_remove = []
    for edge in G_copy.edges(data=True):
        if edge[0] == row_contig or edge[1] == row_contig:
            G_copy.edges[edge[0], edge[1]]['weight'] = row_contacts[edge[1]] if edge[0] == row_contig else row_contacts[edge[0]]
            G_copy.edges[edge[0], edge[1]]['width'] = 30  # Set edge width to 30
        else:
            edges_to_remove.append((edge[0], edge[1]))

    for edge in edges_to_remove:
        G_copy.remove_edge(edge[0], edge[1])

    if selected_node_size is not None:
        min_edge_length = selected_node_size / 2
    else:
        min_edge_length = 50  # Fallback in case the selected node size is not set

    new_pos = nx.spring_layout(G_copy, pos={row_contig: (0, 0)}, fixed=[row_contig], k=min_edge_length / (len(G_copy.nodes) ** 0.5), iterations=50)

    # Get and arrange contigs within the selected species node
    contigs = get_contig_nodes(row_contig)
    indices = contig_information[contig_information['Contig annotation'] == row_contig].index
    inter_contig_edges = set()

    for i in indices:
        for j in range(dense_matrix.shape[0]):
            if dense_matrix[i, j] != 0 and contig_information.at[j, 'Contig annotation'] != row_contig:
                inter_contig_edges.add(contig_information.at[i, 'Contig name'])
                inter_contig_edges.add(contig_information.at[j, 'Contig name'])

    contig_positions = arrange_contigs(contigs, inter_contig_edges)

    cyto_elements = nx_to_cyto_elements(G_copy, new_pos)
    
    # Add contig nodes and edges to the Cytoscape elements
    for contig, (x, y) in contig_positions.items():
        cyto_elements.append({
            'data': {
                'id': contig,
                'size': 5,  # Adjust size as needed
                'color': 'red' if contig in inter_contig_edges else 'blue',  # Red for inter-contig edges, blue otherwise
                'parent': row_contig  # Indicate that this node is within the selected species node
            },
            'position': {
                'x': new_pos[row_contig][0] + x * 100,  # Scale position appropriately
                'y': new_pos[row_contig][1] + y * 100
            }
        })
    
    # Make a copy of the base stylesheet and customize it for this visualization
    cyto_stylesheet = base_stylesheet.copy()

    bar_fig = create_bar_chart(row_contig, col_contig)

    return cyto_elements, cyto_stylesheet, bar_fig, row_contig

def species_contact_visualization(active_cell, selected_species, secondary_species, table_data):
    if not selected_species or not secondary_species:
        return basic_visualization()[0], basic_visualization()[1], go.Figure(), selected_species

    row_contig = selected_species
    col_contig = secondary_species

    G_copy = nx.Graph()
    G_copy.add_node(row_contig, size=2000, color='blue')
    G_copy.add_node(col_contig, size=2000, color='red')

    new_pos = {row_contig: (-1, 0), col_contig: (1, 0)}

    row_indices = contig_information[contig_information['Contig annotation'] == row_contig].index
    col_indices = contig_information[contig_information['Contig annotation'] == col_contig].index
    inter_contigs_row = set()
    inter_contigs_col = set()

    for i in row_indices:
        for j in col_indices:
            if dense_matrix[i, j] != 0:
                inter_contigs_row.add(contig_information.at[i, 'Contig name'])
                inter_contigs_col.add(contig_information.at[j, 'Contig name'])

    all_contigs = inter_contigs_row.union(inter_contigs_col)
    contig_positions = arrange_contigs(all_contigs, list(), radius=0.1, jitter=0.05)

    for contig, (x, y) in contig_positions.items():
        color = 'green' if contig in inter_contigs_row else 'yellow'
        G_copy.add_node(contig, size=2, color=color)
        new_pos[contig] = (new_pos[row_contig][0] + x, new_pos[row_contig][1] + y) if contig in inter_contigs_row else (new_pos[col_contig][0] + x, new_pos[col_contig][1] + y)

    cyto_elements = nx_to_cyto_elements(G_copy, new_pos)
    cyto_stylesheet = base_stylesheet.copy()  # Use a unique stylesheet for this visualization

    bar_fig = create_bar_chart(row_contig, col_contig)

    return cyto_elements, cyto_stylesheet, bar_fig, row_contig

def contig_visualization(active_cell, selected_species, selected_contig, table_data):
    # Empty function for now
    return basic_visualization()[0], basic_visualization()[1], go.Figure(), selected_species

@app.callback(
    [Output('cyto-graph', 'elements'),
     Output('cyto-graph', 'stylesheet'),
     Output('bar-chart', 'figure'),
     Output('species-selector', 'value'),
     Output('visualization-selector', 'value'),
     Output('secondary-species-selector', 'value'),
     Output('secondary-species-selector', 'style'),
     Output('contig-selector', 'style')],
    [Input('contact-table', 'active_cell'),
     Input('reset-btn', 'n_clicks'),
     Input('species-selector', 'value'),
     Input('secondary-species-selector', 'value'),
     Input('contig-selector', 'value'),
     Input('visualization-selector', 'value')],
    [State('contact-table', 'data')]
)
def update_figure(active_cell, reset_clicks, selected_species, secondary_species, selected_contig, visualization_type, table_data):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initialize default values for cyto_elements and bar_fig
    cyto_elements, cyto_stylesheet, _ = basic_visualization()
    bar_fig = go.Figure()
    secondary_species_style = {'display': 'none'}
    contig_selector_style = {'display': 'none'}

    if triggered_id == 'reset-btn' or not active_cell:
        # Reset all selections to show the original plot
        return cyto_elements, cyto_stylesheet, bar_fig, None, 'species', None, secondary_species_style, contig_selector_style

    row_contig = table_data[active_cell['row']]['Species']
    col_contig = active_cell['column_id'] if active_cell['column_id'] is not None else None

    # Determine the visualization mode and selectors based on the selected cell
    if col_contig and col_contig != 'Species' and row_contig != col_contig:
        visualization_type = 'species_contact'
        selected_species = row_contig
        secondary_species = col_contig
        secondary_species_style = {'width': '300px', 'display': 'inline-block'}
    elif col_contig and col_contig == 'Species':
        visualization_type = 'species'
        selected_species = row_contig
        secondary_species = None
        secondary_species_style = {'display': 'none'}
    else:
        visualization_type = visualization_type
        secondary_species_style = {'display': 'none'}
        if visualization_type == 'contig':
            contig_selector_style = {'width': '300px', 'display': 'inline-block'}

    if visualization_type == 'species':
        cyto_elements, cyto_stylesheet, bar_fig, selected_species = species_visualization(active_cell, selected_species, table_data)
        return cyto_elements, cyto_stylesheet, bar_fig, selected_species, visualization_type, None, secondary_species_style, contig_selector_style
    elif visualization_type == 'species_contact':
        cyto_elements, cyto_stylesheet, bar_fig, selected_species = species_contact_visualization(active_cell, selected_species, secondary_species, table_data)
        return cyto_elements, cyto_stylesheet, bar_fig, selected_species, visualization_type, secondary_species, secondary_species_style, contig_selector_style
    elif visualization_type == 'contig':
        cyto_elements, cyto_stylesheet, bar_fig, selected_species = contig_visualization(active_cell, selected_species, selected_contig, table_data)
        contig_selector_style = {'width': '300px', 'display': 'inline-block'}
        return cyto_elements, cyto_stylesheet, bar_fig, selected_species, visualization_type, secondary_species, secondary_species_style, contig_selector_style

@app.callback(
    Output('contig-selector', 'options'),
    [Input('species-selector', 'value')]
)
def update_contig_options(selected_species):
    if selected_species:
        contigs = get_contig_nodes(selected_species)
        return [{'label': contig, 'value': contig} for contig in contigs]
    return []

if __name__ == '__main__':
    app.run_server(debug=True)
