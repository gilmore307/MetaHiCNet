import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csc_matrix
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import dash_ag_grid as dag
import plotly.graph_objects as go
from dash import callback_context
import plotly.express as px
from math import sqrt, sin, cos, pi
from openai import OpenAI

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

# Function to calculate contacts
def calculate_contacts(annotation, annotation_type='species'):
    if annotation_type == 'species':
        intra_contacts = species_matrix.at[annotation, annotation]
        inter_contacts = species_matrix.loc[annotation].sum() - intra_contacts
    else:
        contig_index = contig_information[contig_information['Contig name'] == annotation].index[0]
        species = contig_information.loc[contig_index, 'Contig annotation']
        intra_contacts = dense_matrix[contig_index, get_contig_indexes(species)].sum()
        inter_contacts = dense_matrix[contig_index, :].sum() - intra_contacts
    
    return intra_contacts, inter_contacts

# Function to convert NetworkX graph to Cytoscape elements with sizes and colors
def nx_to_cyto_elements(G, pos, invisible_nodes=set(), invisible_edges=set()):
    elements = []
    for node in G.nodes:
        elements.append({
            'data': {
                'id': node,
                'label': node if G.nodes[node].get('parent') is None else '',  # Add label for species nodes only
                'label_size': G.nodes[node].get('label_size', 6), # Default size to 6
                'size': G.nodes[node].get('size', 1),  # Default size to 1
                'color': G.nodes[node].get('color', '#000'),  # Default color
                'border_color': G.nodes[node].get('border_color', None),  # Default to None
                'border_width': G.nodes[node].get('border_width', None),  # Default to None
                'parent': G.nodes[node].get('parent', None),  # Default to None
                'visible': 'none' if node in invisible_nodes else 'element'  # Set visibility
            },
            'position': {
                'x': pos[node][0] * 100,
                'y': pos[node][1] * 100
            }
        })
    for edge in G.edges(data=True):
        elements.append({
            'data': {
                'source': edge[0],
                'target': edge[1],
                'width': edge[2].get('width', 1),  # Default width
                'color': edge[2].get('color', '#ccc'),  # Default color
                'visible': 'none' if (edge[0], edge[1]) in invisible_edges or (edge[1], edge[0]) in invisible_edges else 'element'  # Set visibility
            }
        })
    return elements


def add_selection_styles(selected_nodes=None, selected_edges=None):
    cyto_stylesheet = base_stylesheet.copy()

    # Define the new styles to be added
    if selected_nodes:
        for node in selected_nodes:
            node_style = {
                'selector': f'node[id="{node}"]',
                'style': {
                    'border-width': 2,
                    'border-color': 'black'
                }
            }
            cyto_stylesheet.append(node_style)

    if selected_edges:
        for source, target in selected_edges:
            edge_style = {
                'selector': f'edge[source="{source}"][target="{target}"], edge[source="{target}"][target="{source}"]',
                'style': {
                    'width': 2,
                    'line-color': 'black',
                    'display': 'element'  # Make the edge visible
                }
            }
            cyto_stylesheet.append(edge_style)

    return cyto_stylesheet

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

# Function to call OpenAI API using GPT-4 with the new API format
def get_chatgpt_response(prompt):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150)
    return response.choices[0].message.content

# Function to style species contact table using Blugrn color scheme
def styling_species_table(matrix_df):
    columns = species_matrix_display.columns[1:]
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
def styling_contig_table(matrix_df, contig_information, contig_colors, species_colors):
    columns = ['Restriction sites', 'Contig length', 'Contig coverage', 'Intra-contig contact']
    styles = []
    for col in columns:
        numeric_df = matrix_df[[col]].select_dtypes(include=[np.number])
        col_min = np.log1p(numeric_df.values.min())
        col_max = np.log1p(numeric_df.values.max())
        col_range = col_max - col_min
        n_bins = 10  # Number of bins for color scaling
        bounds = [i * (col_range / n_bins) + col_min for i in range(n_bins + 1)]
        opacity = 0.6  # Set a fixed opacity for transparency

        for i in range(1, len(bounds)):
            min_bound = bounds[i - 1]
            max_bound = bounds[i]
            if i == len(bounds) - 1:
                max_bound += 1

            styles.append({
                "condition": f"params.colDef.field == '{col}' && Math.log1p(params.value) >= {min_bound} && Math.log1p(params.value) < {max_bound}",
                "style": {
                    'backgroundColor': f"rgba({255 - int((min_bound - col_min) / col_range * 255)}, {255 - int((min_bound - col_min) / col_range * 255)}, 255, {opacity})",
                    'color': "white" if i > len(bounds) / 2.0 else "inherit"
                }
            })

    # Function to add opacity to a hex color
    def add_opacity_to_color(hex_color, opacity):
        if hex_color.startswith('#') and len(hex_color) == 7:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'
        else:
            # Return a default color if hex_color is invalid
            return f'rgba(255, 255, 255, {opacity})'

    # Add style conditions for the "Contig" column
    for contig in matrix_df['Contig']:
        contig_color = contig_colors.get(contig, species_colors.get(contig_information.loc[contig_information['Contig name'] == contig, 'Contig annotation'].values[0], '#FFFFFF'))
        contig_color_with_opacity = add_opacity_to_color(contig_color, 0.6)
        styles.append(
            {
                "condition": f"params.colDef.field == 'Contig' && params.value == '{contig}'",
                "style": {
                    'backgroundColor': contig_color_with_opacity,
                    'color': 'black'
                }
            }
        )

    # Add style conditions for the "Species" column
    for species in matrix_df['Species'].unique():
        species_color = species_colors.get(species, '#FFFFFF')  # Default to white if species color is not found
        species_color_with_opacity = add_opacity_to_color(species_color, 0.6)
        styles.append(
            {
                "condition": f"params.colDef.field == 'Species' && params.value == '{species}'",
                "style": {
                    'backgroundColor': species_color_with_opacity,
                    'color': 'black'
                }
            }
        )

    return styles

# Function to get contig colors from Cytoscape elements or use species color if not found
def get_contig_and_species_colors(contig_information, cyto_elements):
    contig_colors = {}
    species_colors = {}

    # Extract colors from Cytoscape elements
    for element in cyto_elements:
        if 'data' in element and 'color' in element['data'] and 'id' in element['data']:
            contig_colors[element['data']['id']] = element['data']['color']

    # Get species colors based on viral status
    for species in contig_information['Contig annotation'].unique():
        is_viral = str(contig_information.loc[contig_information['Contig annotation'] == species, 'Is Viral'].values[0])
        species_colors[species] = is_viral_colors.get(is_viral, '#FFFFFF')  # Default to white if species color is not found

    return contig_colors, species_colors

# Function to arrange contigs
def arrange_contigs(contigs, inter_contig_edges, distance, selected_contig=None, center_position=(0, 0)):
    distance /= 100 
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

# Function to visualize species relationship
def basic_visualization():
    G = nx.Graph()

    # Add nodes with size based on total contig coverage
    total_contig_coverage = contig_information.groupby('Contig annotation')['Contig coverage'].sum().reindex(unique_annotations)
    node_sizes = generate_gradient_values(total_contig_coverage.values, 10, 30)  # Example range from 10 to 30

    node_colors = {}
    for annotation, size in zip(total_contig_coverage.index, node_sizes):
        is_viral = contig_information.loc[
            contig_information['Contig annotation'] == annotation, 'Is Viral'
        ].values[0]
        color = is_viral_colors[str(is_viral)]
        node_colors[annotation] = color
        G.add_node(annotation, size=size, color=color, parent=None)  # Removed border attributes

    # Add edges with weight based on inter-species contacts
    inter_species_contacts = []
    for annotation_i in unique_annotations:
        for annotation_j in unique_annotations:
            if annotation_i != annotation_j and species_matrix.at[annotation_i, annotation_j] > 0:
                weight = species_matrix.at[annotation_i, annotation_j]
                G.add_edge(annotation_i, annotation_j, weight=weight)
                inter_species_contacts.append(weight)

    # Generate gradient values for the edge weights
    edge_weights = generate_gradient_values(np.array(inter_species_contacts), 10, 300) 

    # Assign the gradient values as edge weights and set default edge color
    for (u, v, d), weight in zip(G.edges(data=True), edge_weights):
        d['weight'] = weight

    # Initial node positions using a force-directed layout with increased dispersion
    pos = nx.spring_layout(G, dim=2, k=5, iterations=50, weight='weight')

    cyto_elements = nx_to_cyto_elements(G, pos)

    # Prepare data for bar chart with 3 traces
    inter_species_contact_sum = species_matrix.sum(axis=1) - np.diag(species_matrix.values)
    total_contig_coverage_sum = total_contig_coverage.values
    contig_counts = contig_information['Contig annotation'].value_counts()

    data_dict = {
        'Total Inter-Species Contact': pd.DataFrame({'name': unique_annotations, 'value': inter_species_contact_sum, 'color': [node_colors.get(species, 'rgba(0,128,0,0.8)') for species in unique_annotations]}),
        'Total Coverage': pd.DataFrame({'name': unique_annotations, 'value': total_contig_coverage_sum, 'color': [node_colors.get(species, 'rgba(0,128,0,0.8)') for species in unique_annotations]}),
        'Contig Number': pd.DataFrame({'name': unique_annotations, 'value': contig_counts, 'color': [node_colors.get(species, 'rgba(0,128,0,0.8)') for species in unique_annotations]})
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

# Function to visualize intra-species relationships
def intra_species_visualization(selected_species):
    G = nx.Graph()

    # Add nodes with size based on contig counts
    contig_counts = [len(contig_information[contig_information['Contig annotation'] == node]) for node in unique_annotations]
    node_sizes = generate_gradient_values(np.array(contig_counts), 10, 30)

    nodes_to_remove = []
    for annotation, size in zip(unique_annotations, node_sizes):
        is_viral = contig_information.loc[
            contig_information['Contig annotation'] == annotation, 'Is Viral'
        ].values[0]
        color = is_viral_colors[str(is_viral)]

        if annotation == selected_species:
            G.add_node(annotation, size=size, color='#FFFFFF', border_color='#000', border_width=2, parent=None)  # White for selected node
        else:
            num_connected_contigs = len(contig_information[(contig_information['Contig annotation'] == annotation) & (dense_matrix[:, get_contig_indexes(selected_species)].sum(axis=1) > 0)])
            if num_connected_contigs == 0:
                nodes_to_remove.append(annotation)
            else:
                G.add_node(annotation, size=size, color=color, parent=None)  # Red for viral, blue for non-viral

    # Add edges with weight based on inter-species contacts
    inter_species_contacts = []
    for annotation_i in unique_annotations:
        for annotation_j in unique_annotations:
            if annotation_i != annotation_j and species_matrix.at[annotation_i, annotation_j] > 0:
                weight = species_matrix.at[annotation_i, annotation_j]
                G.add_edge(annotation_i, annotation_j, weight=weight)
                inter_species_contacts.append(weight)

    # Remove nodes not connected to selected species
    for node in nodes_to_remove:
        G.remove_node(node)

    # Generate gradient values for the edge weights
    edge_weights = generate_gradient_values(np.array(inter_species_contacts), 10, 100)

    edges_to_remove = []
    inter_species_contacts = []

    # Collect edge weights and identify edges to remove
    for edge in G.edges(data=True):
        if edge[0] == selected_species or edge[1] == selected_species:
            weight = species_matrix.at[selected_species, edge[1]] if edge[0] == selected_species else species_matrix.at[edge[0], selected_species]
            inter_species_contacts.append(weight)
        else:
            edges_to_remove.append((edge[0], edge[1]))

    # Remove edges not connected to selected_species
    for edge in edges_to_remove:
        G.remove_edge(edge[0], edge[1])
    # Assign the gradient values as edge weights and set default edge color
    for (u, v, d), weight in zip(G.edges(data=True), edge_weights):
        if edge[0] == selected_species or edge[1] == selected_species:
            d['weight'] = weight

    # Calculate k_value based on the number of contigs of the selected species
    num_contigs = len(get_contig_indexes(selected_species))
    k_value = sqrt(num_contigs)

    new_pos = nx.spring_layout(G, pos={selected_species: (0, 0)}, fixed=[selected_species], k=k_value, iterations=50, weight='weight')

    # Get and arrange contigs within the selected species node
    indices = get_contig_indexes(selected_species)
    contigs = contig_information.loc[indices, 'Contig name']
    inter_contig_edges = set()

    for i in indices:
        for j in range(dense_matrix.shape[0]):
            if dense_matrix[i, j] != 0 and contig_information.at[j, 'Contig annotation'] != selected_species:
                inter_contig_edges.add(contig_information.at[i, 'Contig name'])
                inter_contig_edges.add(contig_information.at[j, 'Contig name'])

    contig_positions = arrange_contigs(contigs, inter_contig_edges, distance=1, center_position=new_pos[selected_species])

    # Add contig nodes and edges to the graph G
    for contig, (x, y) in contig_positions.items():
        G.add_node(contig, size=1, color='#7030A0' if contig in inter_contig_edges else '#00B050', parent=selected_species)
        new_pos[contig] = (new_pos[selected_species][0] + x, new_pos[selected_species][1] + y)

    cyto_elements = nx_to_cyto_elements(G, new_pos)

    # Prepare data for bar chart
    contig_contact_counts = contig_information[contig_information['Contig annotation'] != selected_species]['Contig annotation'].value_counts()
    inter_species_contacts = species_matrix.loc[selected_species].drop(selected_species)

    # Filter out contigs that are not in the graph
    filtered_contig_counts = contig_contact_counts[contig_contact_counts.index.isin(G.nodes)]
    filtered_inter_species_contacts = inter_species_contacts[inter_species_contacts.index.isin(G.nodes)]

    data_dict = {
        'Contig Number': pd.DataFrame({'name': filtered_contig_counts.index, 'value': filtered_contig_counts.values, 'color': [G.nodes[species]['color'] for species in filtered_contig_counts.index]}),
        'Inter-Species Contacts': pd.DataFrame({'name': filtered_inter_species_contacts.index, 'value': filtered_inter_species_contacts.values, 'color': [G.nodes[species]['color'] for species in filtered_inter_species_contacts.index]})
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

# Function to visualize inter-species relationships
def inter_species_visualization(selected_species, secondary_species):
    if not selected_species or not secondary_species:
        cyto_elements, bar_fig = basic_visualization()
        return cyto_elements, bar_fig

    row_contig = selected_species
    col_contig = secondary_species

    G = nx.Graph()
    G.add_node(row_contig, color='#FFFFFF', border_color='black', border_width=2, label=row_contig)
    G.add_node(col_contig, color='#FFFFFF', border_color='black', border_width=2, label=col_contig)

    new_pos = {row_contig: (-0.2, 0), col_contig: (0.2, 0)}

    row_indices = get_contig_indexes(row_contig)
    col_indices = get_contig_indexes(col_contig)
    inter_contigs_row = set()
    inter_contigs_col = set()

    interspecies_contacts = []
    contig_contact_counts = []
    inter_contig_contacts = []

    for i in row_indices:
        for j in col_indices:
            contact_value = dense_matrix[i, j]
            if contact_value != 0:
                inter_contigs_row.add(contig_information.at[i, 'Contig name'])
                inter_contigs_col.add(contig_information.at[j, 'Contig name'])
                interspecies_contacts.append({
                    'name': f"{contig_information.at[i, 'Contig name']} - {contig_information.at[j, 'Contig name']}",
                    'value': contact_value,
                    'color': 'green'  # Set green color for the bars
                })
                contig_contact_counts.append({
                    'name': contig_information.at[i, 'Contig name'],
                    'species': selected_species,
                    'count': 1,
                    'color': '#C00000'  # Set red color for the bars
                })
                contig_contact_counts.append({
                    'name': contig_information.at[j, 'Contig name'],
                    'species': secondary_species,
                    'count': 1,
                    'color': '#0070C0'  # Set blue color for the bars
                })
                inter_contig_contacts.append({
                    'name': contig_information.at[i, 'Contig name'],
                    'value': contact_value,
                    'color': '#C00000'  # Set red color for the bars
                })
                inter_contig_contacts.append({
                    'name': contig_information.at[j, 'Contig name'],
                    'value': contact_value,
                    'color': '#0070C0'  # Set blue color for the bars
                })

    contig_positions_row = arrange_contigs(inter_contigs_row, list(), distance=1, center_position=new_pos[row_contig])
    contig_positions_col = arrange_contigs(inter_contigs_col, list(), distance=1, center_position=new_pos[col_contig])

    # Add contig nodes to the graph G
    for contig, (x, y) in contig_positions_row.items():
        G.add_node(contig, color='#C00000', parent=row_contig)  # Red for primary
        new_pos[contig] = (x, y)

    for contig, (x, y) in contig_positions_col.items():
        G.add_node(contig, color='#0070C0', parent=col_contig)  # Blue for secondary
        new_pos[contig] = (x, y)

    # Add edges between contigs
    for i in row_indices:
        for j in col_indices:
            contact_value = dense_matrix[i, j]
            if contact_value != 0:
                G.add_edge(contig_information.at[i, 'Contig name'], contig_information.at[j, 'Contig name'], weight=contact_value)

    invisible_edges = [(u, v) for u, v in G.edges]  # Mark all contig edges as invisible

    cyto_elements = nx_to_cyto_elements(G, new_pos, list(), invisible_edges)

    # Prepare data for bar chart
    interspecies_contacts_df = pd.DataFrame(interspecies_contacts)

    contig_contact_counts_df = pd.DataFrame(contig_contact_counts)
    contig_contact_counts_summary = contig_contact_counts_df.groupby(['name', 'color']).size().reset_index(name='value')

    inter_contig_contacts_df = pd.DataFrame(inter_contig_contacts)
    inter_contig_contacts_summary = inter_contig_contacts_df.groupby(['name', 'color']).sum().reset_index()

    data_dict = {
        'Inter Contig Contacts': interspecies_contacts_df,
        'Contig Contacts Counts': contig_contact_counts_summary,
        'Contig Contacts Value': inter_contig_contacts_summary
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

# Function to visualize contig relationships
def contig_visualization(selected_species, selected_contig):
    if not selected_contig:
        cyto_elements, bar_fig = basic_visualization()
        return cyto_elements, bar_fig

    # Find the index of the selected contig
    selected_contig_index = contig_information[contig_information['Contig name'] == selected_contig].index[0]
    selected_species = contig_information.loc[selected_contig_index, 'Contig annotation']

    # Get all indices that have contact with the selected contig
    contacts_indices = dense_matrix[selected_contig_index].nonzero()[0]
    
    # Remove self-contact
    contacts_indices = contacts_indices[contacts_indices != selected_contig_index]
    
    contacts_species = contig_information.loc[contacts_indices, 'Contig annotation']
    contacts_contigs = contig_information.loc[contacts_indices, 'Contig name']
    
    G = nx.Graph()

    # Use a categorical color scale
    color_scale = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

    # Rank species based on the number of contigs with contact to the selected contig
    species_contact_counts = contacts_species.value_counts()
    species_contact_ranks = species_contact_counts.rank(method='first').astype(int)
    max_rank = species_contact_ranks.max()

    # Add species nodes and their positions
    for species in contacts_species.unique():
        species_rank = species_contact_ranks[species]
        gradient_color = color_scale[int((species_rank / max_rank) * (len(color_scale) - 1))]
        G.add_node(species, size=1, color='#FFFFFF', border_color=gradient_color, border_width=2)  # White color for nodes, gradient color for border


    # Set k value to avoid overlap and generate positions for the graph nodes
    k_value = sqrt(len(G.nodes))
    pos = nx.spring_layout(G, k=k_value, iterations=50, weight='weight')

    # Add contig nodes to the graph
    for species in contacts_species.unique():
        species_contigs = contacts_contigs[contacts_species == species]
        contig_positions = arrange_contigs(species_contigs, [], distance=2, center_position=pos[species],selected_contig=selected_contig if species == selected_species else None)
        for contig, (x, y) in contig_positions.items():
            G.add_node(contig, size=1 if contig != selected_contig else 5, color='black' if contig == selected_contig else G.nodes[species]['border_color'], parent=species)  # Same color as species, black for selected contig
            if contig != selected_contig:
                G.add_edge(selected_contig, contig, weight=dense_matrix[selected_contig_index, contig_information[contig_information['Contig name'] == contig].index[0]])
            pos[contig] = (x, y)  # Use positions directly from arrange_contigs

    # Ensure the selected contig node is positioned above all other contigs
    pos[selected_contig] = pos[selected_species]

    cyto_elements = nx_to_cyto_elements(G, pos)
    
    # Prepare data for bar chart
    contig_contact_values = dense_matrix[selected_contig_index, contacts_indices]
    contig_data = pd.DataFrame({'name': contacts_contigs, 'value': contig_contact_values, 'color': [G.nodes[contig]['color'] for contig in contacts_contigs]})

    species_contact_values = []
    contig_contact_counts_per_species = [] 
    for species in contacts_species.unique():
        species_indexes = get_contig_indexes(species)
        contact_value = dense_matrix[selected_contig_index, species_indexes].sum()
        species_contact_values.append(contact_value)
        contig_contact_counts_per_species.append(len(species_indexes))  # Count the number of contigs per species

    species_data = pd.DataFrame({'name': contacts_species.unique(), 'value': species_contact_values, 'color': [G.nodes[species]['color'] for species in contacts_species.unique()]})
    contig_contact_counts_data = pd.DataFrame({'name': contacts_species.unique(), 'value': contig_contact_counts_per_species, 'color': [G.nodes[species]['color'] for species in contacts_species.unique()]})

    data_dict = {
        'Contig Contacts': contig_data,
        'Species Contacts': species_data,
        'Contig Contact Counts': contig_contact_counts_data  # New trace
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

def synchronize_selections(triggered_id, selected_node_data, selected_edge_data, contig_info_selected_rows, contact_table_active_cell, table_data, contig_info_table_data):
    # Initialize the return values
    selected_species = None
    selected_contig = None
    secondary_species = None

    # If a node in the network is selected
    if triggered_id == 'cyto-graph' and selected_node_data:
        selected_node_id = selected_node_data[0]['id']
        # Check if the selected node is a contig or a species
        if selected_node_id in contig_information['Contig name'].values:
            contig_info = contig_information[contig_information['Contig name'] == selected_node_id].iloc[0]
            selected_species = contig_info['Contig annotation']
            selected_contig = contig_info['Contig name']
        else:
            selected_species = selected_node_id

    # If an edge in the network is selected
    elif triggered_id == 'cyto-graph' and selected_edge_data:
        source_species = selected_edge_data[0]['source']
        target_species = selected_edge_data[0]['target']
        selected_species = source_species
        secondary_species = target_species

    # If a row in the contig-info-table is selected
    elif triggered_id == 'contig-info-table' and contig_info_selected_rows:
        selected_row = contig_info_selected_rows[0]
        if 'Species' in selected_row and 'Contig' in selected_row:
            selected_species = selected_row['Species']
            selected_contig = selected_row['Contig']

    # If a cell in the contact-table is selected
    elif triggered_id == 'contact-table' and contact_table_active_cell:
        row_species = table_data[contact_table_active_cell['row']]['Species']
        col_species = contact_table_active_cell['column_id'] if contact_table_active_cell['column_id'] != 'Species' else None
        selected_species = row_species
        secondary_species = col_species

    return selected_species, selected_contig, secondary_species

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Set OpenAI API key
client = OpenAI(api_key='')

# File paths for the current environment
contig_info_path = '../0_Documents/contig_information.csv'
raw_contact_matrix_path= '../0_Documents/raw_contact_matrix.npz'

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

# Calculate the total intra-species contacts and inter-species contacts
unique_annotations = contig_information['Contig annotation'].unique()
is_viral_colors = {'True': '#F4B084', 'False': '#8EA9DB'}  # Red for viral, blue for non-viral
species_matrix = pd.DataFrame(0.0, index=unique_annotations, columns=unique_annotations)

for annotation_i in unique_annotations:
    for annotation_j in unique_annotations:
        contacts = dense_matrix[
            get_contig_indexes(annotation_i)
        ][:,
            get_contig_indexes(annotation_j)
        ].sum()
        species_matrix.at[annotation_i, annotation_j] = contacts

# Copy species_matrix and add a column for species names for display
species_matrix_display = species_matrix.copy()
species_matrix_display.insert(0, 'Species', species_matrix_display.index)

# Extract and rename the necessary columns for the new matrix
matrix_columns = {
    'Contig name': 'Contig',
    'Contig annotation': 'Species',
    'Number of restriction sites': 'Restriction sites',
    'Contig length': 'Contig length',
    'Contig coverage': 'Contig coverage',
    'Hi-C contacts mapped to the same contigs': 'Intra-contig contact'
}

# Set the global graph G
cyto_elements, bar_fig = basic_visualization()

contig_matrix_display = contig_information[list(matrix_columns.keys())].rename(columns=matrix_columns)

# Add a "Visibility" column to the contig_matrix_display DataFrame
contig_matrix_display['Visibility'] = 1  # Default value to 1 (visible)

# Extract colors for contigs and species
contig_colors, species_colors = get_contig_and_species_colors(contig_information, cyto_elements)

# Apply the updated function in your Dash layout
styleConditions = styling_contig_table(contig_matrix_display, contig_information, contig_colors, species_colors)

# Convert your DataFrame to a list of dictionaries
contig_data_dict = contig_matrix_display.to_dict('records')

# Define the column definitions for AG Grid
column_defs = [
    {"headerName": "Contig", "field": "Contig", "pinned": 'left', "width": 120},
    {"headerName": "Species", "field": "Species", "pinned": 'left', "width": 140},
    {"headerName": "Restriction sites", "field": "Restriction sites", "width": 140, "wrapHeaderText": True},
    {"headerName": "Contig length", "field": "Contig length", "width": 140, "wrapHeaderText": True},
    {"headerName": "Contig coverage", "field": "Contig coverage", "width": 140, "wrapHeaderText": True},
    {"headerName": "Intra-contig contact", "field": "Intra-contig contact", "width": 140, "wrapHeaderText": True},
    {"headerName": "Visibility", "field": "Visibility",  "hide": True}
]

# Define the default column definitions
default_col_def = {
    "sortable": True,
    "filter": True,
    "resizable": True,
    "cellStyle": {
        "styleConditions": styleConditions
    }
}

# Base stylesheet for Cytoscape
base_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'width': 'data(size)',
            'height': 'data(size)',
            'background-color': 'data(color)',
            'label': 'data(label)',
            'font-size': 'data(label_size)',
            'border-color': 'data(border_color)',
            'border-width': 'data(border_width)',
            'parent': 'data(parent)',
            'display': 'data(visible)'  # Use the visibility attribute
        }
    },
    {
        'selector': 'edge',
        'style': {
            'width': 'data(width)',
            'line-color': 'data(color)',
            'opacity': 0.6,
            'display': 'data(visible)'  # Use the visibility attribute
        }
    }
]

current_visualization_mode = {
    'visualization_type': None,
    'selected_species': None,
    'secondary_species': None,
    'selected_contig': None
}


common_style = {
    'height': '38px',
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
        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'  # Add a shadow
    }),
    html.Div(style={'height': '60px'}),  # Add a placeholder div to account for the fixed header height
    html.Div([
        html.Div([
            dcc.Graph(id='bar-chart', config={'displayModeBar': False}, figure=bar_fig, style={'height': '40vh', 'width': '30vw', 'display': 'inline-block'}),
            html.Div(id='row-count', style={'margin': '0px', 'height': '2vh', 'display': 'inline-block'}),
            dcc.Checklist(
                id='visibility-filter',
                options=[{'label': '  Only show contigs in the map', 'value': 'filter'}],
                value=['filter'],
                style={'display': 'inline-block', 'margin-right': '10px', 'float': 'right'}
            ),
            dag.AgGrid(
                id='contig-info-table',
                columnDefs=column_defs,
                rowData=contig_data_dict,
                defaultColDef=default_col_def,
                style={'height': '40vh', 'width': '30vw', 'display': 'inline-block'},
                dashGridOptions={
                    'headerPinned': 'top',
                    'rowSelection': 'single'  # Enable single row selection
                }
            )
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            cyto.Cytoscape(
                id='cyto-graph',
                elements=cyto_elements,
                stylesheet=base_stylesheet,
                style={'height': '80vh', 'width': '48vw', 'display': 'inline-block'},
                layout={'name': 'preset'},  # Use preset to keep the initial positions
                zoom=1,
                userZoomingEnabled=True,
                wheelSensitivity=0.1  # Reduce the wheel sensitivity
            )
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
    html.Div([
        html.Div(id='hover-info', style={'height': '50vh', 'width': '20vw', 'background-color': 'white', 'padding': '5px', 'border': '1px solid #ccc', 'margin-top': '3px'}),
            html.Div([
                dcc.Textarea(
                    id='chatgpt-input',
                    placeholder='Enter your query here...',
                    style={'width': '100%', 'height': '15vh', 'display': 'inline-block'}
                ),
                html.Button('Interpret Data', id='interpret-button', n_clicks=0, style={'width': '100%', 'display': 'inline-block'})
            ], style={'width': '20vw', 'display': 'inline-block'}),
            html.Div(id='gpt-answer', style={'height': '15vh', 'width': '20vw', 'background-color': 'white', 'padding': '5px', 'border': '1px solid #ccc', 'margin-top': '3px'})
        ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px'}),
    ], style={'width': '100%', 'display': 'flex'}),
    html.Div([
        dash_table.DataTable(
            id='contact-table',
            columns=[{"name": col, "id": col} for col in species_matrix_display.columns],
            data=species_matrix_display.to_dict('records'),
            style_table={'height': 'auto', 'overflowY': 'auto', 'overflowX': 'auto', 'width': '99vw', 'minWidth': '100%'},
            style_data_conditional=styling_species_table(species_matrix_display),
            style_cell={'textAlign': 'left', 'minWidth': '120px', 'width': '120px', 'maxWidth': '180px'},
            style_header={'whiteSpace': 'normal', 'height': 'auto'},  # Allow headers to wrap
            fixed_rows={'headers': True},  # Freeze the first row
            fixed_columns={'headers': True, 'data': 1}  # Freeze the first column
        )
    ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
    help_modal
], style={'height': '100vh', 'overflowY': 'auto', 'width': '100%'})

@app.callback(
    [Output('visualization-selector', 'value'),
     Output('species-selector', 'value'),
     Output('secondary-species-selector', 'value'),
     Output('secondary-species-selector', 'style'),
     Output('contig-selector', 'value'),
     Output('contig-selector', 'style'),
     Output('contact-table', 'active_cell'),
     Output('contig-info-table', 'selectedRows')],
    [Input('visualization-selector', 'value'),
     Input('contact-table', 'active_cell'),
     Input('contig-info-table', 'selectedRows'),
     Input('cyto-graph', 'selectedNodeData'),
     Input('cyto-graph', 'selectedEdgeData')],
    [State('contact-table', 'data'),
     State('contig-info-table', 'rowData')],
    prevent_initial_call=True
)
def sync_selectors(visualization_type, contact_table_active_cell, contig_info_selected_rows, selected_node_data, selected_edge_data, contact_table_data, contig_info_table_data):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    selected_species, selected_contig, secondary_species = synchronize_selections(
        triggered_id, selected_node_data, selected_edge_data, contig_info_selected_rows, contact_table_active_cell, contact_table_data, contig_info_table_data
    )

    # If a contig is selected in the network or contig table
    if selected_contig:
        visualization_type = 'contig'
        secondary_species_style = {'display': 'none'}
        contig_selector_style = {'width': '300px', 'display': 'inline-block'}
        return visualization_type, selected_species, None, secondary_species_style, selected_contig, contig_selector_style, None, []

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
        return visualization_type, selected_species, secondary_species, secondary_species_style, None, contig_selector_style, None, []

    # Default cases based on visualization_type
    if visualization_type == 'inter_species':
        secondary_species_style = {'width': '300px', 'display': 'inline-block'}
        contig_selector_style = {'display': 'none'}
        return visualization_type, None, None, secondary_species_style, None, contig_selector_style, None, []
    elif visualization_type == 'contig':
        secondary_species_style = {'display': 'none'}
        contig_selector_style = {'width': '300px', 'display': 'inline-block'}
        return visualization_type, None, None, secondary_species_style, None, contig_selector_style, None, []
    else:
        secondary_species_style = {'display': 'none'}
        contig_selector_style = {'display': 'none'}
        return visualization_type, None, None, secondary_species_style, None, contig_selector_style, None, []

# Callback to update the visualization
@app.callback(
    [Output('cyto-graph', 'elements'),
     Output('bar-chart', 'figure'),
     Output('contig-info-table', 'columnDefs')],
    [Input('reset-btn', 'n_clicks'),
     Input('confirm-btn', 'n_clicks')],
    [State('visualization-selector', 'value'),
     State('species-selector', 'value'),
     State('secondary-species-selector', 'value'),
     State('contig-selector', 'value'),
     State('contact-table', 'data')],
    prevent_initial_call=True
)
def update_visualization(reset_clicks, confirm_clicks, visualization_type, selected_species, secondary_species, selected_contig, table_data):
    global current_visualization_mode
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initialize default values for cyto_elements and bar_fig
    cyto_elements, bar_fig = basic_visualization()

    if triggered_id == 'reset-btn' or not selected_species:
        # Reset all selections to show the original plot and all contigs
        current_visualization_mode = {
            'visualization_type': None,
            'selected_species': None,
            'secondary_species': None,
            'selected_contig': None
        }
    elif triggered_id == 'confirm-btn':
        # Update the current visualization mode with selected values
        current_visualization_mode['visualization_type'] = visualization_type
        current_visualization_mode['selected_species'] = selected_species
        current_visualization_mode['secondary_species'] = secondary_species
        current_visualization_mode['selected_contig'] = selected_contig

        if visualization_type == 'inter_species':
            if selected_species and secondary_species:
                cyto_elements, bar_fig = inter_species_visualization(selected_species, secondary_species)

        elif visualization_type == 'intra_species':
            cyto_elements, bar_fig = intra_species_visualization(selected_species)

        elif visualization_type == 'contig':
            cyto_elements, bar_fig = contig_visualization(selected_species, selected_contig)

    # Update column definitions with style conditions
    contig_colors, species_colors = get_contig_and_species_colors(contig_information, cyto_elements)
    styleConditions = styling_contig_table(contig_matrix_display, contig_information, contig_colors, species_colors)
    column_defs_updated = column_defs.copy()
    for col_def in column_defs_updated:
        if 'cellStyle' not in col_def:
            col_def['cellStyle'] = {}
        col_def['cellStyle'].update({"styleConditions": styleConditions})

    return cyto_elements, bar_fig, column_defs_updated

@app.callback(
    [Output('cyto-graph', 'stylesheet'),
     Output('hover-info', 'children')],
    [Input('species-selector', 'value'),
     Input('secondary-species-selector', 'value'),
     Input('contig-selector', 'value')],
    prevent_initial_call=True
)
def update_selected_styles(selected_species, secondary_species, selected_contig):
    selected_nodes = []
    selected_edges = []
    hover_info = "No selection"

    if selected_species and secondary_species:
        selected_edges.append((selected_species, secondary_species))
        selected_nodes.append(selected_species)
        selected_nodes.append(secondary_species)
        hover_info = f"Edge between {selected_species} and {secondary_species}"
    elif selected_contig:
        selected_nodes.append(selected_contig)
        contig_info = contig_information[contig_information['Contig name'] == selected_contig].iloc[0]
        hover_info = f"Contig: {selected_contig}<br>Species: {contig_info['Contig annotation']}"

        if current_visualization_mode['visualization_type'] == 'inter_species':
            if contig_info['Contig annotation'] == current_visualization_mode['secondary_species']:
                # Find contigs from the selected species that have contact with the selected contig
                selected_contig_index = contig_information[contig_information['Contig name'] == selected_contig].index[0]
                selected_species_indices = get_contig_indexes(current_visualization_mode['selected_species'])
                connected_contigs = []

                for j in selected_species_indices:
                    if dense_matrix[j, selected_contig_index] != 0:  # Check contact from selected species to the selected contig
                        connected_contig = contig_information.at[j, 'Contig name']
                        connected_contigs.append(connected_contig)

                # Add the connected contigs and edges to the lists
                selected_nodes.extend(connected_contigs)
                for contig in connected_contigs:
                    selected_edges.append((contig, selected_contig))  # Edge goes from the connected contig to the selected contig

            else:
                # Find the contigs in the secondary species that have contact with the selected contig
                selected_contig_index = contig_information[contig_information['Contig name'] == selected_contig].index[0]
                secondary_species_indices = get_contig_indexes(current_visualization_mode['secondary_species'])
                connected_contigs = []

                for j in secondary_species_indices:
                    if dense_matrix[selected_contig_index, j] != 0:  # Check contact from selected contig to secondary species
                        connected_contig = contig_information.at[j, 'Contig name']
                        connected_contigs.append(connected_contig)

                # Add the connected contigs and edges to the lists
                selected_nodes.extend(connected_contigs)
                for contig in connected_contigs:
                    selected_edges.append((selected_contig, contig))  # Edge goes from selected contig to the connected contig

    elif selected_species:
        selected_nodes.append(selected_species)
        hover_info = f"Species: {selected_species}"

    # Add selection styles for the selected nodes and edges
    stylesheet = add_selection_styles(selected_nodes, selected_edges)

    return stylesheet, hover_info

@app.callback(
    [Output('contig-info-table', 'rowData'), 
     Output('contig-info-table', 'filterModel'),
     Output('row-count', 'children')],
    [Input('species-selector', 'value'),
     Input('secondary-species-selector', 'value'),
     Input('visibility-filter', 'value')],
    [State('contig-info-table', 'rowData')]
)
def update_filter_model_and_row_count(selected_species, secondary_species, filter_value, contig_data):
    filter_model = {}
    filtered_data = contig_data
    
    # Set the default visibility to 1
    for row in filtered_data:
        row['Visibility'] = 1
        
    # Update the filter model based on selected species and secondary species
    if selected_species and not secondary_species:
        filter_model['Species'] = {
            "filterType": "text",
            "operator": "OR",
            "conditions": [
                {
                    "filter": selected_species,
                    "filterType": "text",
                    "type": "contains",
                }
            ]
        }
        for row in filtered_data:
            if row['Species'] != selected_species:
                row['Visibility'] = 2

    elif selected_species and secondary_species:
        filter_model['Species'] = {
            "filterType": "text",
            "operator": "OR",
            "conditions": [
                {
                    "filter": selected_species,
                    "filterType": "text",
                    "type": "contains",
                },
                {
                    "filter": secondary_species,
                    "filterType": "text",
                    "type": "contains",
                }
            ]
        }
        for row in filtered_data:
            if row['Species'] not in [selected_species, secondary_species]:
                row['Visibility'] = 2
    else:
        filter_model = {}

    # Set visibility based on the current visualization mode
    if current_visualization_mode['visualization_type'] == 'intra_species':
        if current_visualization_mode['selected_species']:
            for row in filtered_data:
                if row['Species'] != current_visualization_mode['selected_species']:
                    row['Visibility'] = 0

    elif current_visualization_mode['visualization_type'] == 'inter_species':
        if current_visualization_mode['selected_species'] and current_visualization_mode['secondary_species']:
            row_indices = get_contig_indexes(current_visualization_mode['selected_species'])
            col_indices = get_contig_indexes(current_visualization_mode['secondary_species'])
            inter_contigs_row = set()
            inter_contigs_col = set()

            for i in row_indices:
                for j in col_indices:
                    contact_value = dense_matrix[i, j]
                    if contact_value != 0:
                        inter_contigs_row.add(contig_information.at[i, 'Contig name'])
                        inter_contigs_col.add(contig_information.at[j, 'Contig name'])

            inter_contigs = inter_contigs_row.union(inter_contigs_col)

            for row in filtered_data:
                if row['Contig'] not in inter_contigs:
                    row['Visibility'] = 0

    elif current_visualization_mode['visualization_type'] == 'contig':
        if current_visualization_mode['selected_contig']:
            selected_contig_index = contig_information[contig_information['Contig name'] == current_visualization_mode['selected_contig']].index[0]

            connected_contigs = set()
            for j in range(dense_matrix.shape[0]):
                if dense_matrix[selected_contig_index, j] != 0:
                    connected_contigs.add(contig_information.at[j, 'Contig name'])

            for row in filtered_data:
                if row['Contig'] not in connected_contigs and row['Contig'] != current_visualization_mode['selected_contig']:
                    row['Visibility'] = 0

    # Apply filter if the checkbox is checked
    if 'filter' in filter_value:
        filter_model['Visibility'] = {
            "filterType": "number",
            "operator": "OR",
            "conditions": [
                {
                    "filter": 0,
                    "filterType": "number",
                    "type": "notEqual",
                }
            ]
        }

    row_count_text = f"Total Number of Rows: {len([row for row in filtered_data if row['Visibility'] == 1])}"
    return filtered_data, filter_model, row_count_text

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

# Dash callback to use ChatGPT
@app.callback(
    Output('gpt-answer', 'children'),
    [Input('interpret-button', 'n_clicks')],
    [State('chatgpt-input', 'value')],
    prevent_initial_call=True
)
def interpret_data(n_clicks, query):
    if n_clicks > 0 and query:
        try:
            interpretation = get_chatgpt_response(query)
            return f"Interpretation:\n{interpretation}"
        except Exception as e:
            return f"Error: {str(e)}"
    return "No query provided."

if __name__ == '__main__':
    app.run_server(debug=True)