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

def get_contig_indexes(species):
    contigs = contig_information[contig_information['Contig annotation'] == species].index
    return contigs

# Calculate the total intra-species contacts and inter-species contacts
unique_annotations = contig_information['Contig annotation'].unique()
species_contact_matrix = pd.DataFrame(0, index=unique_annotations, columns=unique_annotations)

for annotation_i in unique_annotations:
    for annotation_j in unique_annotations:
        contacts = dense_matrix[
            contig_information[contig_information['Contig annotation'] == annotation_i].index
        ][:,
            contig_information[contig_information['Contig annotation'] == annotation_j].index
        ].sum()
        if annotation_i == annotation_j:
            contacts /= 2  # Divide by 2 to account for symmetry in inter-species contacts
        species_contact_matrix.at[annotation_i, annotation_j] = contacts

# Copy species_contact_matrix and add a column for species names for display
species_contact_matrix_display = species_contact_matrix.copy()
species_contact_matrix_display.insert(0, 'Species', species_contact_matrix_display.index)

def calculate_contacts(annotation, annotation_type='species'):
    if annotation_type == 'species':
        intra_contacts = species_contact_matrix.at[annotation, annotation]
        inter_contacts = species_contact_matrix.loc[annotation].sum() - intra_contacts
    elif annotation_type == 'contig':
        contig_index = contig_information[contig_information['Contig name'] == annotation].index[0]
        species = contig_information.loc[contig_index, 'Contig annotation']
        intra_contacts = dense_matrix[contig_index, contig_information[contig_information['Contig annotation'] == species].index].sum()
        inter_contacts = dense_matrix[contig_index, :].sum() - intra_contacts
    else:
        raise ValueError("annotation_type must be either 'species' or 'contig'")
    
    return intra_contacts, inter_contacts

# Function for basic visualization setup
def basic_visualization():
    # Create a graph for the Arc Diagram
    G = nx.Graph()

    # Add nodes with size based on intra-species contacts
    max_intra_species_contacts = species_contact_matrix.max().max()
    is_viral_colors = {'True': '#F4B084', 'False': '#8EA9DB'}  # Red for viral, blue for non-viral
    for annotation in unique_annotations:
        is_viral = contig_information.loc[
            contig_information['Contig annotation'] == annotation, 'Is Viral'
        ].values[0]
        color = is_viral_colors[str(is_viral)]
        size = (species_contact_matrix.at[annotation, annotation] / max_intra_species_contacts) * 100  # Scale node size for plotly
        G.add_node(annotation, size=size, color=color, contacts=species_contact_matrix.at[annotation, annotation])

    # Add edges with weight based on inter-species contacts
    for annotation_i in unique_annotations:
        for annotation_j in unique_annotations:
            if annotation_i != annotation_j:
                if species_contact_matrix.at[annotation_i, annotation_j] > 0:
                    G.add_edge(annotation_i, annotation_j, contacts=species_contact_matrix.at[annotation_i, annotation_j])

    # Initial node positions using a force-directed layout with increased dispersion
    pos = nx.spring_layout(G, dim=2, k=2, iterations=50)

    cyto_elements = nx_to_cyto_elements(G, pos)
    cyto_stylesheet = base_stylesheet.copy()
    return cyto_elements, cyto_stylesheet, G

# Function to convert NetworkX graph to Cytoscape elements
def nx_to_cyto_elements(G, pos):
    elements = []
    for node in G.nodes:
        data = {
            'id': node,
            'size': G.nodes[node]['size'],
            'color': G.nodes[node]['color'],
        }
        if 'parent' in G.nodes[node]:
            data['parent'] = G.nodes[node]['parent']
        else:
            data['label'] = node  # Only add label for species nodes
        elements.append({
            'data': data,
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
def create_bar_chart(trace='inter_species', slider_value=0):
    species = []
    inter_species_vals = []
    intra_species_vals = []
    contig_counts = []

    for annotation in unique_annotations:
        species.append(annotation)
        intra_contacts, inter_contacts = calculate_contacts(annotation, annotation_type='species')
        inter_species_vals.append(inter_contacts)
        intra_species_vals.append(intra_contacts)
        contig_counts.append(len(get_contig_indexes(annotation)))

    # Create a DataFrame to sort the values
    df = pd.DataFrame({
        'Species': species,
        'Inter-species Contact': inter_species_vals,
        'Intra-species Contact': intra_species_vals,
        'Contig Count': contig_counts
    })

    # Sort the DataFrame by the selected trace
    if trace == 'inter_species':
        df = df.sort_values(by='Inter-species Contact', ascending=False)
    elif trace == 'intra_species':
        df = df.sort_values(by='Intra-species Contact', ascending=False)
    elif trace == 'contig_count':
        df = df.sort_values(by='Contig Count', ascending=False)

    # Limit to the range specified by the slider
    df = df.iloc[slider_value:slider_value+20]

    trace1 = go.Bar(
        x=df['Species'],
        y=df['Inter-species Contact'],
        name='Inter-species Contact',
        visible=(trace == 'inter_species')
    )
    trace2 = go.Bar(
        x=df['Species'],
        y=df['Intra-species Contact'],
        name='Intra-species Contact',
        visible=(trace == 'intra_species')
    )
    trace3 = go.Bar(
        x=df['Species'],
        y=df['Contig Count'],
        name='Contig Count',
        visible=(trace == 'contig_count')
    )

    combined_bar_fig = go.Figure(data=[trace1, trace2, trace3])
    combined_bar_fig.update_layout(
        barmode='group',
        xaxis=dict(title='Species'),
        yaxis=dict(title='Value'),
        height=720,
        margin=dict(t=30, b=0, l=0, r=0),
        showlegend=False,  # Hide the legend
        dragmode=False  # Disable the toolbar
    )

    return combined_bar_fig

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
def arrange_contigs(contigs, inter_contig_edges, radius, jitter, selected_contig=None):
    # Identify contigs that connect to other species
    connecting_contigs = [contig for contig in contigs if contig in inter_contig_edges and contig != selected_contig]
    other_contigs = [contig for contig in contigs if contig not in inter_contig_edges and contig != selected_contig]

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

    # Place selected contig in the center
    if selected_contig:
        inner_positions[selected_contig] = (0, 0)

    return {**circle_positions, **inner_positions}

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
        ),
        html.Button("Confirm Selection", id="confirm-btn", style={**common_style}),
    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin': '20px'}),
    html.Div([
        html.Div([
            dcc.RadioItems(
                id='trace-selector',
                options=[
                    {'label': 'Inter-species Contact', 'value': 'inter_species'},
                    {'label': 'Intra-species Contact', 'value': 'intra_species'},
                    {'label': 'Contig Count', 'value': 'contig_count'}
                ],
                value='inter_species',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                style={'textAlign': 'center'}
            ),
            dcc.Graph(id='bar-chart', style={'height': '80vh', 'width': '30vw', 'display': 'inline-block'}),
            html.Div([
                dcc.Slider(
                    id='bar-slider',
                    min=0,
                    max=len(unique_annotations) - 20,
                    step=1,
                    value=0,
                    marks={i: f'{i+1}' for i in range(0, len(unique_annotations), 20)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                )
            ], id='slider-div', style={'display': 'none'})  # Initially hidden
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            cyto.Cytoscape(
                id='cyto-graph',
                elements=cyto_elements,
                style={'height': '80vh', 'width': '60vw', 'display': 'inline-block'},
                layout={'name': 'preset'},  # Use preset to keep the initial positions
                stylesheet=cyto_stylesheet
            )
        ], style={'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'width': '100%', 'display': 'flex'}),
    html.Div([
        dash_table.DataTable(
            id='contact-table',
            columns=[{"name": col, "id": col} for col in species_contact_matrix_display.columns],
            data=species_contact_matrix_display.reset_index(drop=True).to_dict('records'),
            style_table={'height': 'auto', 'overflowY': 'auto', 'overflowX': 'auto', 'width': '99vw', 'minWidth': '100%'},
            style_data_conditional=create_conditional_styles(species_contact_matrix),
            style_cell={'textAlign': 'left', 'minWidth': '120px', 'width': '120px', 'maxWidth': '180px'},
            fixed_rows={'headers': True},  # Freeze the first row
            fixed_columns={'headers': True, 'data': 1}  # Freeze the first column
        )
    ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
    help_modal
], style={'height': '100vh', 'overflowY': 'auto', 'width': '100%'})


@app.callback(
    [Output('cyto-graph', 'elements'),
     Output('cyto-graph', 'stylesheet'),
     Output('bar-chart', 'figure'),
     Output('slider-div', 'style')],
    [Input('reset-btn', 'n_clicks'),
     Input('confirm-btn', 'n_clicks'),
     Input('trace-selector', 'value'),
     Input('bar-slider', 'value')],
    [State('species-selector', 'value'),
     State('secondary-species-selector', 'value'),
     State('contig-selector', 'value'),
     State('visualization-selector', 'value'),
     State('contact-table', 'data')]
)
def update_visualization(reset_clicks, confirm_clicks, trace, slider_value, selected_species, secondary_species, selected_contig, visualization_type, table_data):
    # Initialize default values for cyto_elements and bar_fig
    cyto_elements, cyto_stylesheet, _ = basic_visualization()

    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'reset-btn' or not selected_species:
        # Reset all selections to show the original plot
        return cyto_elements, cyto_stylesheet, create_bar_chart(trace, slider_value), {'display': 'none'}  # Hide slider

    if triggered_id == 'confirm-btn':
        if visualization_type == 'species_contact':
            if not selected_species or not secondary_species:
                return basic_visualization()[0], basic_visualization()[1], create_bar_chart(trace, slider_value), {'display': 'none'}

            cyto_elements, cyto_stylesheet, bar_fig, selected_species = species_contact_visualization(None, selected_species, secondary_species, table_data, trace, slider_value)
            return cyto_elements, cyto_stylesheet, bar_fig, {'display': 'block' if len(unique_annotations) > 20 else 'none'}
        elif visualization_type == 'species':
            cyto_elements, cyto_stylesheet, bar_fig, selected_species = species_visualization(None, selected_species, table_data, trace, slider_value)
            return cyto_elements, cyto_stylesheet, bar_fig, {'display': 'block' if len(unique_annotations) > 20 else 'none'}
        elif visualization_type == 'contig':
            cyto_elements, cyto_stylesheet, bar_fig, selected_species = contig_visualization(None, selected_species, selected_contig, table_data, trace, slider_value)
            return cyto_elements, cyto_stylesheet, bar_fig, {'display': 'block' if len(unique_annotations) > 20 else 'none'}

    # Get the number of bars for the bar chart
    num_bars = len(unique_annotations)

    # Determine if the slider should be displayed
    slider_style = {'display': 'block'} if num_bars > 20 else {'display': 'none'}
    
    # Return the updated elements, stylesheet, and bar chart figure
    return cyto_elements, cyto_stylesheet, create_bar_chart(trace, slider_value), slider_style

@app.callback(
    Output('contig-selector', 'options'),
    [Input('species-selector', 'value')],
    [State('visualization-selector', 'value')]
)
def populate_contig_selector(selected_species, visualization_type):
    if visualization_type == 'contig' and selected_species:
        contigs = get_contig_indexes(selected_species)
        return [{'label': contig_information.loc[index, 'Contig name'], 'value': contig_information.loc[index, 'Contig name']} for index in contigs]
    return []

def species_visualization(active_cell, selected_species, table_data, trace, slider_value):
    row_contig = table_data[active_cell['row']]['Species'] if active_cell else selected_species

    G_copy = G.copy()

    row_contacts = species_contact_matrix.loc[row_contig]

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
    contigs = get_contig_indexes(row_contig)
    indices = contig_information[contig_information['Contig annotation'] == row_contig].index
    inter_contig_edges = set()

    for i in indices:
        for j in range(dense_matrix.shape[0]):
            if dense_matrix[i, j] != 0 and contig_information.at[j, 'Contig annotation'] != row_contig:
                inter_contig_edges.add(contig_information.at[i, 'Contig name'])
                inter_contig_edges.add(contig_information.at[j, 'Contig name'])

    contig_positions = arrange_contigs(contigs, inter_contig_edges, radius=0.6, jitter=0.05)

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

    bar_fig = create_bar_chart(trace, slider_value)

    return cyto_elements, cyto_stylesheet, bar_fig, row_contig

def species_contact_visualization(active_cell, selected_species, secondary_species, table_data, trace, slider_value):
    if not selected_species or not secondary_species:
        return basic_visualization()[0], basic_visualization()[1], go.Figure(), selected_species

    row_contig = selected_species
    col_contig = secondary_species

    G_copy = nx.Graph()
    G_copy.add_node(row_contig, size=500, color='#A9D08E')  # Green for primary species
    G_copy.add_node(col_contig, size=500, color='#FFD966')  # Yellow for secondary species

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

    contig_positions_row = arrange_contigs(inter_contigs_row, list(), radius=0.1, jitter=0.05)
    contig_positions_col = arrange_contigs(inter_contigs_col, list(), radius=0.1, jitter=0.05)

    for contig, (x, y) in contig_positions_row.items():
        G_copy.add_node(contig, size=20, color='#00FF00', parent=row_contig)  # Green for primary
        new_pos[contig] = (new_pos[row_contig][0] + x * 20, new_pos[row_contig][1] + y * 20)

    for contig, (x, y) in contig_positions_col.items():
        G_copy.add_node(contig, size=20, color='blue', parent=col_contig)  # Blue for secondary
        new_pos[contig] = (new_pos[col_contig][0] + x * 20, new_pos[col_contig][1] + y * 20)

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
                'x': new_pos[node][0] * 500,
                'y': new_pos[node][1] * 500
            }
        })

    cyto_stylesheet = base_stylesheet.copy()  # Use a unique stylesheet for this visualization

    # Customize the stylesheet for parent nodes
    cyto_stylesheet.append({
        'selector': f'node[id="{row_contig}"]',
        'style': {
            'background-color': '#FFFFFF',
            'border-color': '#A9D08E',
            'border-width': 5,
            'label': 'data(label)'
        }
    })
    cyto_stylesheet.append({
        'selector': f'node[id="{col_contig}"]',
        'style': {
            'background-color': '#FFFFFF',
            'border-color': '#8EA9DB',
            'border-width': 5,
            'label': 'data(label)'
        }
    })

    bar_fig = create_bar_chart(trace, slider_value)

    return cyto_elements, cyto_stylesheet, bar_fig, row_contig

def contig_visualization(active_cell, selected_species, selected_contig, table_data):
    if not selected_contig:
        return basic_visualization()[0], basic_visualization()[1], go.Figure(), selected_species

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
    color_scale = pcolors.diverging.RdBu

    # Rank species based on the number of contigs with contact to the selected contig
    species_contact_counts = contacts_species.value_counts()
    species_contact_ranks = species_contact_counts.rank(method='first').astype(int)
    max_rank = species_contact_ranks.max()

    # Add species nodes and contig nodes
    species_positions = {}
    for species in contacts_species.unique():
        species_rank = species_contact_ranks[species]
        gradient_color = color_scale[int((species_rank / max_rank) * (len(color_scale) - 1))]
        G_copy.add_node(species, size=50, color=gradient_color)  # Gradient color for species nodes
        species_contigs = contacts_contigs[contacts_species == species]
        
        # Arrange contig nodes from the same species closer
        contig_positions = arrange_contigs(species_contigs, [], 0.5, 0.05, selected_contig=selected_contig if species == selected_species else None)
        for contig, (x, y) in contig_positions.items():
            G_copy.add_node(contig, size=10 if contig != selected_contig else 50, color='red' if contig == selected_contig else gradient_color, parent=species)  # Same color as species, red for selected contig
            if contig != selected_contig:
                G_copy.add_edge(selected_contig, contig, contacts=dense_matrix[selected_contig_index, contig_information[contig_information['Contig name'] == contig].index[0]])
            species_positions[contig] = (x, y)

    # Ensure the selected contig node is positioned above all other contigs
    species_positions[selected_contig] = (0, 0)  # Place it at the center of its species node

    # Generate positions for the graph nodes
    pos = nx.spring_layout(G_copy, k=0.5, iterations=50)
    
    # Update positions for contig nodes to be closer to their species node
    for contig, (x, y) in species_positions.items():
        pos[contig] = (pos[selected_species][0] + x, pos[selected_species][1] + y)

    # Explicitly set the position of the selected contig to ensure it is above all others
    pos[selected_contig] = (pos[selected_species][0], pos[selected_species][1])

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

    # Create a bar chart to show contacts of the selected contig
    contact_counts = dense_matrix[selected_contig_index, contacts_indices]
    bar_data = pd.Series(contact_counts, index=contacts_contigs).sort_values()
    bar_colors = ['rgba(0,0,255,0.8)' if contig != selected_contig else 'rgba(255,0,0,0.8)' for contig in bar_data.index]  # Blue for contigs, Red for selected contig
    bar_trace = go.Bar(x=bar_data.index, y=bar_data.values, marker_color=bar_colors)
    bar_layout = go.Layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(title="Contact Count", showticklabels=False),
        height=720,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    bar_fig = go.Figure(data=[bar_trace], layout=bar_layout)

    return cyto_elements, cyto_stylesheet, bar_fig, selected_species

if __name__ == '__main__':
    app.run_server(debug=True)
