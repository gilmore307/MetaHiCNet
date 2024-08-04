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
import plotly.colors as pcolors
from math import sqrt, sin, cos, pi

# File paths for the current environment
contig_info_path = '../0_Documents/contig_information.csv'
raw_contact_matrix_path = '../0_Documents/raw_contact_matrix.npz'

# Load the data
contig_information = pd.read_csv(contig_info_path)
contact_matrix_data = np.load(raw_contact_matrix_path)
data = contact_matrix_data['data']
indices = contact_matrix_data['indices']
indptr = contact_matrix_data['indptr']
shape = contact_matrix_data['shape']
sparse_matrix = csc_matrix((data, indices, indptr), shape=shape)
dense_matrix = sparse_matrix.toarray()

# Remove contigs with annotation "Unmapped"
unmapped_contigs = contig_information[contig_information['Contig annotation'] == "Unmapped"].index
contig_information = contig_information.drop(unmapped_contigs).reset_index(drop=True)
dense_matrix = np.delete(dense_matrix, unmapped_contigs, axis=0)
dense_matrix = np.delete(dense_matrix, unmapped_contigs, axis=1)

# Function to get contig indexes based on species
def get_contig_indexes(species):
    contigs = contig_information[contig_information['Contig annotation'] == species].index
    return contigs

# Function to generate gradient values in a range [A, B]
def generate_gradient_values(input_array, range_A, range_B):
    min_val = np.min(input_array)
    max_val = np.max(input_array)
    scaled_values = range_A + ((input_array - min_val) / (max_val - min_val)) * (range_B - range_A)
    return scaled_values

# Calculate the total intra-species contacts and inter-species contacts
unique_annotations = contig_information['Contig annotation'].unique()
species_contact_matrix = pd.DataFrame(0.0, index=unique_annotations, columns=unique_annotations)

for annotation_i in unique_annotations:
    for annotation_j in unique_annotations:
        contacts = dense_matrix[
            get_contig_indexes(annotation_i)
        ][:,
            get_contig_indexes(annotation_j)
        ].sum()
        species_contact_matrix.at[annotation_i, annotation_j] = contacts

# Copy species_contact_matrix and add a column for species names for display
species_contact_matrix_display = species_contact_matrix.copy()
species_contact_matrix_display.insert(0, 'Species', species_contact_matrix_display.index)

# Extract and rename the necessary columns for the new matrix
matrix_columns = {
    'Contig name': 'Contig',
    'Contig annotation': 'Species',
    'Number of restriction sites': 'Restriction sites',
    'Contig length': 'Contig length',
    'Contig coverage': 'Contig coverage',
    'Hi-C contacts mapped to the same contigs': 'Intra-contig contact'
}

contig_matrix_display = contig_information[list(matrix_columns.keys())].rename(columns=matrix_columns)

# Function to calculate contacts
def calculate_contacts(annotation, annotation_type='species'):
    if annotation_type == 'species':
        intra_contacts = species_contact_matrix.at[annotation, annotation]
        inter_contacts = species_contact_matrix.loc[annotation].sum() - intra_contacts
    else:
        contig_index = contig_information[contig_information['Contig name'] == annotation].index[0]
        species = contig_information.loc[contig_index, 'Contig annotation']
        intra_contacts = dense_matrix[contig_index, get_contig_indexes(species)].sum()
        inter_contacts = dense_matrix[contig_index, :].sum() - intra_contacts
    
    return intra_contacts, inter_contacts

def basic_visualization():
    # Create a graph for the Arc Diagram
    G = nx.Graph()

    # Add nodes with size based on total contig coverage
    is_viral_colors = {'True': '#F4B084', 'False': '#8EA9DB'}  # Red for viral, blue for non-viral
    total_contig_coverage = contig_information.groupby('Contig annotation')['Contig coverage'].sum().reindex(unique_annotations)
    node_sizes = generate_gradient_values(total_contig_coverage.values, 50, 200)  # Example range from 50 to 200

    node_colors = {}
    for annotation, size in zip(total_contig_coverage.index, node_sizes):
        is_viral = contig_information.loc[
            contig_information['Contig annotation'] == annotation, 'Is Viral'
        ].values[0]
        color = is_viral_colors[str(is_viral)]
        node_colors[annotation] = color
        G.add_node(annotation, size=size, color=color)

    # Add edges with weight based on inter-species contacts
    inter_species_contacts = []
    for annotation_i in unique_annotations:
        for annotation_j in unique_annotations:
            if annotation_i != annotation_j and species_contact_matrix.at[annotation_i, annotation_j] > 0:
                weight = species_contact_matrix.at[annotation_i, annotation_j]
                G.add_edge(annotation_i, annotation_j, weight=weight)
                inter_species_contacts.append(weight)

    # Generate gradient values for the edge weights
    edge_weights = generate_gradient_values(np.array(inter_species_contacts), 10, 100)  # Example range from 10 to 100

    # Assign the gradient values as edge weights
    for (u, v, d), weight in zip(G.edges(data=True), edge_weights):
        d['weight'] = weight

    # Initial node positions using a force-directed layout with increased dispersion
    pos = nx.spring_layout(G, dim=2, k=1, iterations=50, weight='weight')

    cyto_elements = nx_to_cyto_elements(G, pos)
    cyto_stylesheet = base_stylesheet.copy()

    # Prepare data for bar chart with 3 traces
    inter_species_contact_sum = species_contact_matrix.sum(axis=1) - np.diag(species_contact_matrix.values)
    total_contig_coverage_sum = total_contig_coverage.values
    contig_counts = contig_information['Contig annotation'].value_counts()

    data_dict = {
        'Total Inter-Species Contact': pd.DataFrame({'name': unique_annotations, 'value': inter_species_contact_sum, 'color': [node_colors.get(species, 'rgba(0,128,0,0.8)') for species in unique_annotations]}),
        'Total Coverage': pd.DataFrame({'name': unique_annotations, 'value': total_contig_coverage_sum, 'color': [node_colors.get(species, 'rgba(0,128,0,0.8)') for species in unique_annotations]}),
        'Contig Number': pd.DataFrame({'name': unique_annotations, 'value': contig_counts, 'color': [node_colors.get(species, 'rgba(0,128,0,0.8)') for species in unique_annotations]})
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, cyto_stylesheet, G, bar_fig

# Function to convert NetworkX graph to Cytoscape elements
def nx_to_cyto_elements(G, pos):
    elements = []
    node_ids = set(G.nodes)  # Keep track of added node IDs
    for node in G.nodes:
        data = {
            'id': node,
            'size': G.nodes[node]['size'],
            'color': G.nodes[node]['color'],
            'label': node if 'parent' not in G.nodes[node] else ''  # Add label for species nodes only
        }
        if 'parent' in G.nodes[node]:
            data['parent'] = G.nodes[node]['parent']
        elements.append({
            'data': data,
            'position': {
                'x': pos[node][0] * 500,
                'y': pos[node][1] * 500
            }
        })
    for edge in G.edges(data=True):
        if edge[0] in node_ids and edge[1] in node_ids:  # Ensure both nodes exist
            elements.append({
                'data': {
                    'source': edge[0],
                    'target': edge[1],
                    'weight': edge[2]['weight']
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
            'label': 'data(label)'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'width': 2,
            'line-color': '#ccc',
        }
    },
    {
        'selector': 'node:selected',
        'style': {
            'border-width': 8,
            'border-color': 'black'  # black color for selected node border
        }
    },
    {
        'selector': 'edge:selected',
        'style': {
            'width': 5,  # set selected edge width to 5
            'line-color': 'black'  # black color for selected edge
        }
    }
]

def create_bar_chart(data_dict):
    traces = []

    for idx, (trace_name, data_frame) in enumerate(data_dict.items()):
        # Sort the data in descending order by 'value'
        bar_data = data_frame.sort_values(by='value', ascending=False)

        bar_colors = bar_data['color']  # Use the color column for bar colors
        bar_trace = go.Bar(
            x=bar_data['name'], 
            y=bar_data['value'], 
            name=trace_name, 
            marker_color=bar_colors,
            visible=True if idx == 0 else 'legendonly'
        )
        traces.append(bar_trace)

    bar_layout = go.Layout(
        xaxis=dict(
            title="",
            tickangle=-45,
            tickfont=dict(size=12),
            rangeslider=dict(
               visible=True,
               thickness=0.05  # 5% of the plot area height
           )
        ),
        yaxis=dict(title="Value", tickfont=dict(size=15)),
        height=400,  # Adjusted height
        margin=dict(t=0, b=0, l=0, r=0),  # Adjusted margin
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)  # Move legend above plot and center align
    )

    bar_fig = go.Figure(data=traces, layout=bar_layout)
    return bar_fig

# Function to style species contact table using Blugrn color scheme
def styling_species_table(matrix_df):
    columns = species_contact_matrix_display.columns[1:]
    styles = []
    numeric_df = matrix_df[columns].select_dtypes(include=[np.number])
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
                'backgroundColor': f'rgba({255 - int(log_value / log_max_value * 255)}, {255 - int(log_value / log_max_value * 255)}, 255, {opacity})'  # Set background color for the contact matrix.
            })
    return styles

# Function to style contig info table
def styling_contig_table(matrix_df):
    styles = []
    columns = ['Restriction sites', 'Contig length', 'Contig coverage', 'Intra-contig contact']

    for col in columns:
        numeric_column = matrix_df[col].astype(float)
        log_max_value = np.log1p(numeric_column.max())
        for i, value in enumerate(numeric_column):
            if not pd.isnull(value):
                log_value = np.log1p(value)
                opacity = 0.6  # Set a fixed opacity for transparency
                styles.append({
                    'if': {
                        'row_index': i,
                        'column_id': col
                    },
                    'backgroundColor': f'rgba({255 - int(log_value / log_max_value * 255)}, {255 - int(log_value / log_max_value * 255)}, 255, {opacity})'  # Set background color for the contact matrix.
                })
    
    return styles

# Function to arrange contigs
def arrange_contigs(contigs, inter_contig_edges, distance, selected_contig=None, center_position=(0, 0)):
    phi = (1 + sqrt(5)) / 2  # golden ratio

    # Identify contigs that connect to other species
    connecting_contigs = [contig for contig in contigs if contig in inter_contig_edges and contig != selected_contig]
    other_contigs = [contig for contig in contigs if contig not in inter_contig_edges and contig != selected_contig]

    # Arrange inner contigs in a sunflower pattern
    inner_positions = {}
    angle_stride = 2 * pi / phi ** 2

    max_inner_radius = 0  # To keep track of the maximum radius used for inner nodes

    for k, contig in enumerate(other_contigs, start=1):
        r = distance * sqrt(k)  # Distance increases with sqrt(k) to maintain spacing
        theta = k * angle_stride
        x = center_position[0] + r * cos(theta)
        y = center_position[1] + r * sin(theta)
        inner_positions[contig] = (x, y)
        if r > max_inner_radius:
            max_inner_radius = r

    # Place selected contig in the center
    if selected_contig:
        inner_positions[selected_contig] = center_position

    # Arrange connecting contigs in concentric circles starting from the boundary of inner nodes
    distance *= 2
    outer_positions = {}
    layer_radius = max_inner_radius + distance  # Start from the boundary of inner nodes
    current_layer = 1
    nodes_in_layer = int(2 * pi * layer_radius / distance)
    angle_step = 2 * pi / nodes_in_layer

    for i, contig in enumerate(connecting_contigs):
        if i >= nodes_in_layer * current_layer:
            current_layer += 1
            layer_radius = max_inner_radius + distance * current_layer
            nodes_in_layer = int(2 * pi * layer_radius / distance)
            angle_step = 2 * pi / nodes_in_layer

        angle = (i % nodes_in_layer) * angle_step
        x = center_position[0] + layer_radius * cos(angle)
        y = center_position[1] + layer_radius * sin(angle)
        outer_positions[contig] = (x, y)

    return {**inner_positions, **outer_positions}

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Set the global graph G
cyto_elements, cyto_stylesheet, G, bar_fig = basic_visualization()

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

# Use the styling functions in the Dash layout
app.layout = html.Div([
    html.Div([
        html.Button("Download Selected Item", id="download-btn", style={**common_style}),
        html.Button("Reset Selection", id="reset-btn", style={**common_style}),
        html.Button("Help", id="open-help", style={**common_style}),
        dcc.Download(id="download-dataframe-csv"),
        dcc.Dropdown(
            id='visualization-selector',
            options=[
                {'label': 'Intra-species', 'value': 'intra_species'},
                {'label': 'Inter-species Contact', 'value': 'inter_species'},
                {'label': 'Contig', 'value': 'contig'}
            ],
            value='intra_species',
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
        ),
        html.Button("Confirm Selection", id="confirm-btn", style={**common_style}),
    ], style={
        'display': 'flex',
        'justify-content': 'space-between',
        'align-items': 'center',
        'margin': '0px',
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'width': '100%',
        'z-index': '1000',
        'background-color': 'llite',
        'padding': '10px',  # Add padding to ensure content does not overlap with page content
        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'  # Optional: Add a shadow for better visibility
    }),
    html.Div(style={'height': '60px'}),  # Add a placeholder div to account for the fixed header height
    html.Div([
        html.Div([
            dcc.Graph(id='bar-chart', config={'displayModeBar': False}, figure=bar_fig, style={'height': '40vh', 'width': '30vw', 'display': 'inline-block'}),
            html.Div(id='row-count', style={'margin': '0px','height': '2vh'}),  # Show row number of contig matrix
            dash_table.DataTable(
                id='contig-info-table',
                columns=[{"name": col, "id": col} for col in contig_matrix_display.columns],
                data=contig_matrix_display.to_dict('records'),
                style_table={'height': '40vh', 'overflowY': 'auto', 'overflowX': 'auto', 'width': '30vw', 'minWidth': '100%'},
                style_cell={'textAlign': 'left', 'minWidth': '120px', 'width': '120px', 'maxWidth': '180px'},
                style_header={'whiteSpace': 'normal', 'height': 'auto'},  # Allow headers to wrap
                style_data_conditional=styling_contig_table(contig_matrix_display),
                fixed_rows={'headers': True},  # Freeze the first row
                fixed_columns={'headers': True, 'data': 2}  # Freeze the first 2 columns
            )
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            cyto.Cytoscape(
                id='cyto-graph',
                elements=cyto_elements,
                style={'height': '80vh', 'width': '60vw', 'display': 'inline-block'},
                layout={'name': 'preset'},  # Use preset to keep the initial positions
                stylesheet=cyto_stylesheet,
                zoom=1,
                userZoomingEnabled=True,
                wheelSensitivity=0.1  # Reduce the wheel sensitivity (default is 1)

            )
        ], style={'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'width': '100%', 'display': 'flex'}),
    html.Div([
        dash_table.DataTable(
            id='contact-table',
            columns=[{"name": col, "id": col} for col in species_contact_matrix_display.columns],
            data=species_contact_matrix_display.to_dict('records'),
            style_table={'height': 'auto', 'overflowY': 'auto', 'overflowX': 'auto', 'width': '99vw', 'minWidth': '100%'},
            style_data_conditional=styling_species_table(species_contact_matrix_display),
            style_cell={'textAlign': 'left', 'minWidth': '120px', 'width': '120px', 'maxWidth': '180px'},
            style_header={'whiteSpace': 'normal', 'height': 'auto'},  # Allow headers to wrap
            fixed_rows={'headers': True},  # Freeze the first row
            fixed_columns={'headers': True, 'data': 1}  # Freeze the first column
        )
    ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
    help_modal
], style={'height': '100vh', 'overflowY': 'auto', 'width': '100%'})
        
# Visualization functions
def species_visualization(active_cell, selected_species, table_data):
    row_contig = table_data[active_cell['row']]['Species'] if active_cell else selected_species

    G_copy = G.copy()

    # Generate node sizes using generate_gradient_values
    contig_counts = [len(contig_information[contig_information['Contig annotation'] == node]) for node in G_copy.nodes]
    node_sizes = generate_gradient_values(np.array(contig_counts), 100, 500)  # Example range from 100 to 500

    nodes_to_remove = []
    for node, size in zip(G_copy.nodes, node_sizes):
        if node == row_contig:
            G_copy.nodes[node]['size'] = size
            G_copy.nodes[node]['color'] = '#FFFFFF'  # White for selected node
        else:
            num_connected_contigs = len(contig_information[(contig_information['Contig annotation'] == node) & (dense_matrix[:, get_contig_indexes(row_contig)].sum(axis=1) > 0)])
            if num_connected_contigs == 0:
                nodes_to_remove.append(node)
            else:
                G_copy.nodes[node]['size'] = size
                is_viral = contig_information.loc[
                    contig_information['Contig annotation'] == node, 'Is Viral'
                ].values[0]
                G_copy.nodes[node]['color'] = '#F4B084' if is_viral else '#8EA9DB'  # Red for viral, blue for non-viral

    for node in nodes_to_remove:
        G_copy.remove_node(node)

    edges_to_remove = []
    inter_species_contacts = []

    # Collect edge weights and identify edges to remove
    for edge in G_copy.edges(data=True):
        if edge[0] == row_contig or edge[1] == row_contig:
            weight = species_contact_matrix.at[row_contig, edge[1]] if edge[0] == row_contig else species_contact_matrix.at[edge[0], row_contig]
            inter_species_contacts.append(weight)
        else:
            edges_to_remove.append((edge[0], edge[1]))

    # Remove edges not connected to row_contig
    for edge in edges_to_remove:
        G_copy.remove_edge(edge[0], edge[1])

    # Generate gradient values for the edge weights
    normalized_weights = generate_gradient_values(np.array(inter_species_contacts), 10, 100)  # Example range from 10 to 100

    # Assign the normalized weights as edge weights
    for edge, weight in zip(G_copy.edges(data=True), normalized_weights):
        if edge[0] == row_contig or edge[1] == row_contig:
            G_copy.edges[edge[0], edge[1]]['weight'] = weight

    # Calculate min_edge_length based on the largest node size to avoid node overlap
    max_node_size = max(node_sizes)
    min_edge_length = max_node_size / 2  # Node size is diameter, so radius is size / 2

    # Set k parameter based on the number of nodes and their sizes to avoid overlap
    k_value = min_edge_length / sqrt(len(G_copy.nodes))

    new_pos = nx.spring_layout(G_copy, pos={row_contig: (0, 0)}, fixed=[row_contig], k=k_value/2, iterations=50, weight='weight')
    
    # Get and arrange contigs within the selected species node
    indices = get_contig_indexes(row_contig)
    contigs = contig_information.loc[indices, 'Contig name']
    inter_contig_edges = set()

    for i in indices:
        for j in range(dense_matrix.shape[0]):
            if dense_matrix[i, j] != 0 and contig_information.at[j, 'Contig annotation'] != row_contig:
                inter_contig_edges.add(contig_information.at[i, 'Contig name'])
                inter_contig_edges.add(contig_information.at[j, 'Contig name'])

    contig_positions = arrange_contigs(contigs, inter_contig_edges, distance=0.05, center_position=new_pos[row_contig])

    cyto_elements = nx_to_cyto_elements(G_copy, new_pos)
    
    # Add contig nodes and edges to the Cytoscape elements
    for contig, (x, y) in contig_positions.items():
        cyto_elements.append({
            'data': {
                'id': contig,
                'size': 5,  # Adjust size as needed
                'color': '#7030A0' if contig in inter_contig_edges else '#00B050',  # Purple for inter-contig edges, green otherwise
                'parent': row_contig  # Indicate that this node is within the selected species node
            },
            'position': {
                'x': new_pos[row_contig][0] + x * 100,  # Scale position appropriately
                'y': new_pos[row_contig][1] + y * 100
            }
        })
    
    # Make a copy of the base stylesheet and customize it for this visualization
    cyto_stylesheet = base_stylesheet.copy()

    # Customize the stylesheet for the selected species node
    cyto_stylesheet.append({
        'selector': f'node[id="{row_contig}"]',
        'style': {
            'background-color': '#FFFFFF',
            'border-color': 'black',
            'border-width': 5,
            'label': 'data(label)',
            'font-size': '50px'
        }
    })
    
    # Customize the stylesheet for other species nodes
    for node in G_copy.nodes:
        if node != row_contig:
            cyto_stylesheet.append({
                'selector': f'node[id="{node}"]',
                'style': {
                    'font-size': '40px'  # Adjust label size for other species nodes
                }
            })
            
    # Prepare data for bar chart
    contig_contact_counts = contig_information[contig_information['Contig annotation'] != row_contig]['Contig annotation'].value_counts()
    inter_species_contacts = species_contact_matrix.loc[row_contig].drop(row_contig)
    
    data_dict = {
        'Contig Number': pd.DataFrame({'name': contig_contact_counts.index, 'value': contig_contact_counts.values, 'color': [G.nodes[species]['color'] for species in contig_contact_counts.index]}),
        'Inter-Species Contacts': pd.DataFrame({'name': inter_species_contacts.index, 'value': inter_species_contacts.values, 'color': [G.nodes[species]['color'] for species in inter_species_contacts.index]})
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, cyto_stylesheet, bar_fig, row_contig

def species_contact_visualization(active_cell, selected_species, secondary_species, table_data):
    if not selected_species or not secondary_species:
        return basic_visualization()[0], basic_visualization()[1], basic_visualization()[3], selected_species

    row_contig = selected_species
    col_contig = secondary_species

    G_copy = nx.Graph()
    G_copy.add_node(row_contig, size=1, color='#A9D08E', label=row_contig)  # Green for primary species
    G_copy.add_node(col_contig, size=1, color='#FFD966', label=col_contig)  # Yellow for secondary species

    new_pos = {row_contig: (-1, 0), col_contig: (1, 0)}

    row_indices = get_contig_indexes(row_contig)
    col_indices = get_contig_indexes(col_contig)
    inter_contigs_row = set()
    inter_contigs_col = set()

    interspecies_contacts = []

    for i in row_indices:
        for j in col_indices:
            contact_value = dense_matrix[i, j]
            if contact_value != 0:
                inter_contigs_row.add(contig_information.at[i, 'Contig name'])
                inter_contigs_col.add(contig_information.at[j, 'Contig name'])
                interspecies_contacts.append({
                    'name': f"{contig_information.at[i, 'Contig name']} - {contig_information.at[j, 'Contig name']}",
                    'value': contact_value,
                    'color': 'blue'  # Set blue color for the bars
                })

    contig_positions_row = arrange_contigs(inter_contigs_row, list(), distance=0.05, center_position=new_pos[row_contig])
    contig_positions_col = arrange_contigs(inter_contigs_col, list(), distance=0.05, center_position=new_pos[col_contig])

    for contig, (x, y) in contig_positions_row.items():
        G_copy.add_node(contig, size=5, color='red', parent=row_contig)  # Red for primary
        new_pos[contig] = (x, y)

    for contig, (x, y) in contig_positions_col.items():
        G_copy.add_node(contig, size=5, color='blue', parent=col_contig)  # Blue for secondary
        new_pos[contig] = (x, y)

    cyto_elements = []
    for node, data in G_copy.nodes(data=True):
        cyto_elements.append({
            'data': {
                'id': node,
                'size': data['size'],
                'color': data['color'],
                'label': data['label'] if 'label' in data else '',
                'parent': data['parent'] if 'parent' in data else ''
            },
            'position': {
                'x': new_pos[node][0] * 100,
                'y': new_pos[node][1] * 100
            }
        })

    cyto_stylesheet = base_stylesheet.copy()  # Use a unique stylesheet for this visualization

    # Customize the stylesheet for parent nodes
    cyto_stylesheet.append({
        'selector': f'node[id="{row_contig}"]',
        'style': {
            'background-color': '#FFFFFF',
            'border-color': 'black',
            'border-width': 5,
            'label': 'data(label)',
            'font-size': '20px'
        }
    })
    cyto_stylesheet.append({
        'selector': f'node[id="{col_contig}"]',
        'style': {
            'background-color': '#FFFFFF',
            'border-color': 'black',
            'border-width': 5,
            'label': 'data(label)',
            'font-size': '20px'
        }
    })

    # Prepare data for bar chart
    interspecies_contacts_df = pd.DataFrame(interspecies_contacts)

    data_dict = {
        'Interspecies Contacts': interspecies_contacts_df
    }

    bar_fig = create_bar_chart(data_dict)

    # Filter contigs involved in interspecies contacts and retain all columns
    involved_contigs = contig_information[contig_information['Contig name'].isin(inter_contigs_row | inter_contigs_col)].index

    return cyto_elements, cyto_stylesheet, bar_fig, involved_contigs

def contig_visualization(active_cell, selected_species, selected_contig, table_data):
    if not selected_contig:
        return basic_visualization()[0], basic_visualization()[1], basic_visualization()[3], selected_species

    # Find the index of the selected contig
    selected_contig_index = contig_information[contig_information['Contig name'] == selected_contig].index[0]
    selected_species = contig_information.loc[selected_contig_index, 'Contig annotation']

    # Get all indices that have contact with the selected contig
    contacts_indices = dense_matrix[selected_contig_index].nonzero()[0]
    
    # Remove self-contact
    contacts_indices = contacts_indices[contacts_indices != selected_contig_index]
    
    contacts_species = contig_information.loc[contacts_indices, 'Contig annotation']
    contacts_contigs = contig_information.loc[contacts_indices, 'Contig name']

    # Create the graph
    G_copy = nx.Graph()

    # Use a red-to-blue color scale
    color_scale = pcolors.diverging.Portland

    # Rank species based on the number of contigs with contact to the selected contig
    species_contact_counts = contacts_species.value_counts()
    species_contact_ranks = species_contact_counts.rank(method='first').astype(int)
    max_rank = species_contact_ranks.max()

    # Add species nodes and their positions
    for species in contacts_species.unique():
        species_rank = species_contact_ranks[species]
        gradient_color = color_scale[int((species_rank / max_rank) * (len(color_scale) - 1))]
        G_copy.add_node(species, size=50, color=gradient_color)  # Gradient color for species nodes

    # Set k value to avoid overlap and generate positions for the graph nodes
    k_value = 1 / sqrt(len(G_copy.nodes))
    pos = nx.spring_layout(G_copy, k=k_value, iterations=50, weight='weight')

    # Add contig nodes to the graph
    for species in contacts_species.unique():
        species_contigs = contacts_contigs[contacts_species == species]
        contig_positions = arrange_contigs(
            species_contigs,
            [],
            0.05,
            center_position=pos[species],
            selected_contig=selected_contig if species == selected_species else None
        )
        for contig, (x, y) in contig_positions.items():
            G_copy.add_node(contig, size=10 if contig != selected_contig else 50, color='black' if contig == selected_contig else G_copy.nodes[species]['color'], parent=species)  # Same color as species, red for selected contig
            if contig != selected_contig:
                G_copy.add_edge(selected_contig, contig, weight=dense_matrix[selected_contig_index, contig_information[contig_information['Contig name'] == contig].index[0]])
            pos[contig] = (x, y)  # Use positions directly from arrange_contigs

    # Ensure the selected contig node is positioned above all other contigs
    pos[selected_contig] = pos[selected_species]

    cyto_elements = nx_to_cyto_elements(G_copy, pos)
    cyto_stylesheet = base_stylesheet.copy()

    # Customize the stylesheet for species nodes
    for species in contacts_species.unique():
        cyto_stylesheet.append({
            'selector': f'node[id="{species}"]',
            'style': {
                'background-color': '#FFFFFF',
                'border-color': G_copy.nodes[species]['color'],
                'border-width': 10,
                'label': 'data(label)'
            }
        })

    # Prepare data for bar chart
    contig_contact_values = dense_matrix[selected_contig_index, contacts_indices]
    contig_data = pd.DataFrame({'name': contacts_contigs, 'value': contig_contact_values, 'color': [G_copy.nodes[contig]['color'] for contig in contacts_contigs]})

    species_contact_values = []
    contig_contact_counts_per_species = []  # For the new trace
    for species in contacts_species.unique():
        species_indexes = get_contig_indexes(species)
        contact_value = dense_matrix[selected_contig_index, species_indexes].sum()
        species_contact_values.append(contact_value)
        contig_contact_counts_per_species.append(len(species_indexes))  # Count the number of contigs per species

    species_data = pd.DataFrame({'name': contacts_species.unique(), 'value': species_contact_values, 'color': [G_copy.nodes[species]['color'] for species in contacts_species.unique()]})
    contig_contact_counts_data = pd.DataFrame({'name': contacts_species.unique(), 'value': contig_contact_counts_per_species, 'color': [G_copy.nodes[species]['color'] for species in contacts_species.unique()]})

    data_dict = {
        'Contig Contacts': contig_data,
        'Species Contacts': species_data,
        'Contig Contact Counts': contig_contact_counts_data  # New trace
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, cyto_stylesheet, bar_fig, selected_species

def synchronize_selections(triggered_id, selected_node_data, selected_edge_data, contig_info_active_cell, contact_table_active_cell, table_data, contig_info_table_data):
    # If a node in the network is selected
    if triggered_id == 'cyto-graph' and selected_node_data:
        selected_node_id = selected_node_data[0]['id'] if selected_node_data else None
        print(f"Selected node ID: {selected_node_id}")  # Debug print
        # Check if the selected node is a contig or a species
        if selected_node_id in contig_information['Contig name'].values:
            contig_info = contig_information[contig_information['Contig name'] == selected_node_id].iloc[0]
            print(f"Contig info: {contig_info}")  # Debug print
            if 'Contig annotation' in contig_info and 'Contig name' in contig_info:
                selected_species = contig_info['Contig annotation']
                selected_contig = contig_info['Contig name']
                return selected_species, selected_contig, None, None
        else:
            selected_species = selected_node_id
            return selected_species, None, None, None

    # If an edge in the network is selected
    if triggered_id == 'cyto-graph' and selected_edge_data:
        source_species = selected_edge_data[0]['source']
        target_species = selected_edge_data[0]['target']
        return source_species, None, target_species, None

    # If a cell in the contig-info-table is selected
    if triggered_id == 'contig-info-table' and contig_info_active_cell:
        row_contig_info = contig_info_table_data[contig_info_active_cell['row']]
        print(f"Row contig info: {row_contig_info}")  # Debug print
        if 'Species' in row_contig_info and 'Contig' in row_contig_info:
            selected_species = row_contig_info['Species']
            selected_contig = row_contig_info['Contig']
            return selected_species, selected_contig, None, None

    # If a cell in the contact-table is selected
    if triggered_id == 'contact-table' and contact_table_active_cell:
        row_contig = table_data[contact_table_active_cell['row']]['Species']
        col_contig = contact_table_active_cell['column_id'] if contact_table_active_cell['column_id'] != 'Species' else None
        return row_contig, None, col_contig, None

    return None, None, None, None

@app.callback(
    [Output('species-selector', 'value'),
     Output('secondary-species-selector', 'value'),
     Output('secondary-species-selector', 'style'),
     Output('contig-selector', 'style'),
     Output('visualization-selector', 'value'),
     Output('contig-selector', 'value'),
     Output('contact-table', 'active_cell'),
     Output('contig-info-table', 'active_cell')],
    [Input('contact-table', 'active_cell'),
     Input('visualization-selector', 'value'),
     Input('contig-info-table', 'active_cell'),
     Input('cyto-graph', 'selectedNodeData'),
     Input('cyto-graph', 'selectedEdgeData')],
    [State('contact-table', 'data'),
     State('contig-info-table', 'data')]
)
def sync_selectors(contact_table_active_cell, visualization_type, contig_info_active_cell, selected_node_data, selected_edge_data, contact_table_data, contig_info_table_data):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    selected_species, selected_contig, secondary_species, _ = synchronize_selections(
        triggered_id, selected_node_data, selected_edge_data, contig_info_active_cell, contact_table_active_cell, contact_table_data, contig_info_table_data
    )

    # If a species is selected in the network or species table
    if selected_species and not selected_contig:
        if secondary_species:
            visualization_type = 'inter_species'
            secondary_species_style = {'width': '300px', 'display': 'inline-block'}
            contig_selector_style = {'display': 'none'}
        else:
            visualization_type = 'intra_species'
            secondary_species_style = {'display': 'none'}
            contig_selector_style = {'display': 'none'}
        return selected_species, secondary_species, secondary_species_style, contig_selector_style, visualization_type, None, None, None

    # If a contig is selected in the network or contig table
    if selected_contig:
        visualization_type = 'contig'
        return selected_species, None, {'display': 'none'}, {'width': '300px', 'display': 'inline-block'}, visualization_type, selected_contig, None, None

    # Default cases based on visualization_type
    if visualization_type == 'inter_species':
        return None, None, {'width': '300px', 'display': 'inline-block'}, {'display': 'none'}, 'inter_species', None, None, None
    elif visualization_type == 'contig':
        return None, None, {'display': 'none'}, {'width': '300px', 'display': 'inline-block'}, 'contig', None, None, None
    else:
        return None, None, {'display': 'none'}, {'display': 'none'}, 'intra_species', None, None, None

@app.callback(
    [Output('cyto-graph', 'elements'),
     Output('cyto-graph', 'stylesheet'),
     Output('bar-chart', 'figure'),
     Output('contig-info-table', 'data')],
    [Input('reset-btn', 'n_clicks'),
     Input('confirm-btn', 'n_clicks')],
    [State('species-selector', 'value'),
     State('secondary-species-selector', 'value'),
     State('contig-selector', 'value'),
     State('visualization-selector', 'value'),
     State('contact-table', 'data')]
)
def update_visualization(reset_clicks, confirm_clicks, selected_species, secondary_species, selected_contig, visualization_type, table_data):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initialize default values for cyto_elements, bar_fig, and filtered_data
    cyto_elements, cyto_stylesheet, _, bar_fig = basic_visualization()
    filtered_data = contig_matrix_display

    if triggered_id == 'reset-btn' or not selected_species:
        # Reset all selections to show the original plot and all contigs
        return cyto_elements, cyto_stylesheet, bar_fig, filtered_data.to_dict('records')

    if triggered_id == 'confirm-btn':
        if visualization_type == 'inter_species':
            if not selected_species or not secondary_species:
                cyto_elements, cyto_stylesheet, _, bar_fig = basic_visualization()
                filtered_data = contig_matrix_display
                return cyto_elements, cyto_stylesheet, bar_fig, filtered_data.to_dict('records')

            cyto_elements, cyto_stylesheet, bar_fig, involved_contigs = species_contact_visualization(None, selected_species, secondary_species, table_data)
            filtered_data = contig_matrix_display.loc[involved_contigs]
            return cyto_elements, cyto_stylesheet, bar_fig, filtered_data.to_dict('records')

        elif visualization_type == 'intra_species':
            cyto_elements, cyto_stylesheet, bar_fig, _ = species_visualization(None, selected_species, table_data)
            species_indexes = get_contig_indexes(selected_species)
            filtered_data = contig_matrix_display.loc[species_indexes]
            return cyto_elements, cyto_stylesheet, bar_fig, filtered_data.to_dict('records')

        elif visualization_type == 'contig':
            cyto_elements, cyto_stylesheet, bar_fig, _ = contig_visualization(None, selected_species, selected_contig, table_data)
            selected_contig_index = contig_information[contig_information['Contig name'] == selected_contig].index[0]
            contacts_indices = dense_matrix[selected_contig_index].nonzero()[0]
            contacts_indices = contacts_indices[contacts_indices != selected_contig_index]
            filtered_data = contig_matrix_display.loc[contacts_indices]
            return cyto_elements, cyto_stylesheet, bar_fig, filtered_data.to_dict('records')

    return cyto_elements, cyto_stylesheet, bar_fig, filtered_data.to_dict('records')

@app.callback(
    Output('row-count', 'children'),
    [Input('contig-info-table', 'data')]
)
def update_row_count(data):
    return f"Total Number of Rows: {len(data)}"

@app.callback(
    Output('contig-selector', 'options'),
    [Input('species-selector', 'value')],
    [State('visualization-selector', 'value')]
)
def populate_contig_selector(selected_species, visualization_type):
    if visualization_type == 'contig' and selected_species:
        contigs = contig_information.loc[get_contig_indexes(selected_species), 'Contig name']
        return [{'label': contig, 'value': contig} for contig in contigs]
    return []

if __name__ == '__main__':
    app.run_server(debug=True)