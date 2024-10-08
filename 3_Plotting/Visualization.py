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
import os
from concurrent.futures import ThreadPoolExecutor
import logging

class DashLoggerHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []  # Store logs in a list

    def emit(self, record):
        # This is the method that is called for each log message
        log_entry = self.format(record)
        self.logs.append(log_entry)  # Add log message to the list

    def get_logs(self):
        # Return the logs as a string (joining the list)
        return '\n'.join(self.logs)

    def clear_logs(self):
        # Clear the log list
        self.logs.clear()


# Function to get bin indexes based on annotation in a specific part of the dataframe
def get_bin_indexes(annotations, contig_information):
    # Number of threads to use: 4 * CPU core count
    num_threads = 4 * os.cpu_count()
    
    # Ensure annotations is a list even if a single annotation is provided
    if isinstance(annotations, str):
        annotations = [annotations]
    
    def fetch_indexes(annotation):
        return annotation, contig_information[contig_information['Bin annotation'] == annotation].index

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(fetch_indexes, annotation): annotation for annotation in annotations}
        
        bin_indexes = {}
        for future in futures:
            annotation = futures[future]
            try:
                annotation, indexes = future.result()
                bin_indexes[annotation] = indexes
            except Exception as e:
                logger.error(f'Error fetching bin indexes for annotation: {annotation}, error: {e}')
        
    # If only one annotation was given as input, return its indexes directly
    if len(bin_indexes) == 1:
        return list(bin_indexes.values())[0]

    return bin_indexes

# Function to generate gradient values in a range [A, B]
def generate_gradient_values(input_array, range_A, range_B):
    min_val = np.min(input_array)
    max_val = np.max(input_array)
    scaled_values = range_A + ((input_array - min_val) / (max_val - min_val)) * (range_B - range_A)
    return scaled_values

# Function to convert NetworkX graph to Cytoscape elements with sizes and colors
def nx_to_cyto_elements(G, pos, invisible_nodes=set(), invisible_edges=set()):
    elements = []
    for node in G.nodes:
        elements.append({
            'data': {
                'id': node,
                'label': node if G.nodes[node].get('parent') is None else '',  # Add label for annotation nodes only
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

# Function to handle empty DataFrames and log when no contact is found
def create_bar_chart(data_dict):
    logger.info("Starting to create bar chart with data_dict")
    traces = []

    for idx, (trace_name, data_frame) in enumerate(data_dict.items()):
        logger.info(f"Creating bar trace for {trace_name}")
        bar_data = data_frame.sort_values(by='value', ascending=False)
        bar_colors = bar_data['color']

        # Check if 'hover' column exists
        if 'hover' in bar_data.columns:
            hover_text = bar_data['hover']
            bar_trace = go.Bar(
                x=bar_data['name'], 
                y=bar_data['value'], 
                name=trace_name, 
                marker_color=bar_colors,
                visible=True if idx == 0 else 'legendonly',
                hovertext=hover_text,
                hoverinfo='text'
            )
        else:
            bar_trace = go.Bar(
                x=bar_data['name'], 
                y=bar_data['value'], 
                name=trace_name, 
                marker_color=bar_colors,
                visible=True if idx == 0 else 'legendonly'
            )

        traces.append(bar_trace)

    logger.info("Bar traces created, now creating layout")
    bar_layout = go.Layout(
        xaxis=dict(
            title="",
            tickangle=-45,
            tickfont=dict(size=12),
            rangeslider=dict(visible=True, thickness=0.05)
        ),
        yaxis=dict(title="Value", tickfont=dict(size=15)),
        height=400,
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
    )

    bar_fig = go.Figure(data=traces, layout=bar_layout)
    logger.info("Bar chart created successfully")
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

# Function to style annotation contact table using Blugrn color scheme
def styling_annotation_table(contact_matrix_display):
    styles = []
    numeric_df = contact_matrix_display.select_dtypes(include=[np.number])
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

# Function to style bin info table
def styling_bin_table(bin_colors, annotation_colors, unique_annotations):
    taxonomy_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    columns = ['Restriction sites', 'Bin length', 'Bin coverage', 'Intra-bin contact']
    styles = []
    
    for col in columns:
        numeric_df = contig_information_display[[col]].select_dtypes(include=[np.number])
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

    # Add style conditions for the "Bin" column
    for bin in contig_information_display['Bin']:
        bin_color = bin_colors.get(bin, annotation_colors.get(contig_information.loc[contig_information['Bin'] == bin, 'Bin annotation'].values[0], '#FFFFFF'))
        bin_color_with_opacity = add_opacity_to_color(bin_color, 0.6)
        styles.append(
            {
                "condition": f"params.colDef.field == 'Bin' && params.value == '{bin}'",
                "style": {
                    'backgroundColor': bin_color_with_opacity,
                    'color': 'black'
                }
            }
        )

    # Add style conditions for the "Annotation" column

    for annotation in unique_annotations:
        annotation_color = annotation_colors.get(annotation, '#FFFFFF')
        annotation_color_with_opacity = add_opacity_to_color(annotation_color, 0.6)
        for taxonomy_column in taxonomy_columns:
            styles.append({
                "condition": f"params.colDef.field == '{taxonomy_column}' && params.value == '{annotation}'",
                "style": {
                    'backgroundColor': annotation_color_with_opacity,
                    'color': 'black'
                }
            })

    return styles

# Function to get bin colors from Cytoscape elements or use annotation color if not found
def get_bin_and_annotation_colors(cyto_elements, contig_information):
    bin_colors = {}
    annotation_colors = {}

    # Extract colors from Cytoscape elements
    if cyto_elements:
        for element in cyto_elements:
            if 'data' in element and 'color' in element['data'] and 'id' in element['data']:
                bin_colors[element['data']['id']] = element['data']['color']

    # Get annotation colors based on viral status
    for annotation in contig_information['Bin annotation'].unique():
        bin_type = contig_information[contig_information['Bin annotation'] == annotation]['type'].values[0]
        annotation_colors[annotation] = type_colors.get(bin_type, default_color)
        
    return bin_colors, annotation_colors

# Function to arrange bins
def arrange_bins(bins, inter_bin_edges, distance, selected_bin=None, center_position=(0, 0)):
    distance /= 100 
    phi = (1 + sqrt(5)) / 2  # golden ratio

    # Identify bins that connect to other annotation
    connecting_bins = [bin for bin in bins if bin in inter_bin_edges and bin != selected_bin]
    other_bins = [bin for bin in bins if bin not in inter_bin_edges and bin != selected_bin]

    # Arrange inner bins in a sunflower pattern
    inner_positions = {}
    angle_stride = 2 * pi / phi ** 2

    max_inner_radius = 0  # To keep track of the maximum radius used for inner nodes

    for k, bin in enumerate(other_bins, start=1):
        r = distance * sqrt(k)  # Distance increases with sqrt(k) to maintain spacing
        theta = k * angle_stride
        x = center_position[0] + r * cos(theta)
        y = center_position[1] + r * sin(theta)
        inner_positions[bin] = (x, y)
        if r > max_inner_radius:
            max_inner_radius = r

    # Place selected bin in the center
    if selected_bin:
        inner_positions[selected_bin] = center_position

    # Arrange connecting bins in concentric circles starting from the boundary of inner nodes
    distance *= 2
    outer_positions = {}
    layer_radius = max_inner_radius + distance  # Start from the boundary of inner nodes
    current_layer = 1
    nodes_in_layer = int(2 * pi * layer_radius / distance)
    angle_step = 2 * pi / nodes_in_layer

    for i, bin in enumerate(connecting_bins):
        if i >= nodes_in_layer * current_layer:
            current_layer += 1
            layer_radius = max_inner_radius + distance * current_layer
            nodes_in_layer = int(2 * pi * layer_radius / distance)
            angle_step = 2 * pi / nodes_in_layer

        angle = (i % nodes_in_layer) * angle_step
        x = center_position[0] + layer_radius * cos(angle)
        y = center_position[1] + layer_radius * sin(angle)
        outer_positions[bin] = (x, y)

    return {**inner_positions, **outer_positions}

def taxonomy_visualization():
    logger.info('Creating taxonomy visualization')

    host_data = contig_information_intact[contig_information_intact['type'].isin(['chromosome', 'plasmid'])]
    virus_data = contig_information_intact[contig_information_intact['type'] == 'phage']

    level_mapping = {
        'Community': 9,
        'Domain': 8,
        'Kingdom': 7,
        'Phylum': 6,
        'Class': 5,
        'Order': 4,
        'Family': 3,
        'Genus': 2,
        'Species': 1
    }

    records = []
    existing_annotations = set()

    records.append({
        "annotation": "Community",
        "parent": "",
        "level": 9,
        "level_name": "Community",
        "type": "Community",
        "total coverage": 0,
        "border_color": "black",
        "Bin": contig_information_intact['Bin'].unique()
    })
    existing_annotations.add("Community")

    for _, row in host_data.iterrows():
        for level, level_num in level_mapping.items():
            if level == 'Community':
                continue

            annotation = row[level]
            parent = "Community" if level_num == 8 else row[list(level_mapping.keys())[list(level_mapping.values()).index(level_num + 1)]]

            if annotation not in existing_annotations:
                records.append({
                    "annotation": annotation,
                    "parent": parent,
                    "level": level_num,
                    "level_name": level.capitalize(),
                    "type": row["type"],
                    "total coverage": row['Bin coverage'],
                    "border_color": type_colors.get(row["type"], "gray"),
                    "Bin": [row["Bin"]]
                })
                existing_annotations.add(annotation)
            else:
                for rec in records:
                    if rec['annotation'] == annotation:
                        rec['total coverage'] += row['Bin coverage']
                        rec['Bin'].append(row['Bin'])
                        break

    for _, row in virus_data.iterrows():
        for level, level_num in level_mapping.items():
            if level not in ['Class', 'Order', 'Family']:
                continue

            annotation = 'Virus' if level_num == 5 else row[level]
            parent = (
                "Community" if level_num == 5
                else 'Virus' if level_num == 4
                else row[list(level_mapping.keys())[list(level_mapping.values()).index(level_num + 1)]]
            )

            if annotation not in existing_annotations:
                records.append({
                    "annotation": annotation,
                    "parent": parent,
                    "level": level_num,
                    "level_name": level.capitalize(),
                    "type": row["type"],
                    "total coverage": row['Bin coverage'],
                    "border_color": type_colors.get(row["type"], "gray"),
                    "Bin": [row["Bin"]]
                })
                existing_annotations.add(annotation)
            else:
                for rec in records:
                    if rec['annotation'] == annotation:
                        rec['total coverage'] += row['Bin coverage']
                        rec['Bin'].append(row['Bin'])
                        break

    hierarchy_df = pd.DataFrame(records)
    hierarchy_df['scaled_coverage'] = generate_gradient_values(hierarchy_df['total coverage'], 10, 30)
    hierarchy_df.loc[hierarchy_df['type'] == 'phage', 'scaled_coverage'] *= 20

    # Limit the number of bins shown in hover to avoid long hover boxes
    def format_bins(bin_list, max_bins=5):
        if len(bin_list) > max_bins:
            return ', '.join(bin_list[:max_bins]) + f"... (+{len(bin_list) - max_bins} more)"
        return ', '.join(bin_list)

    hierarchy_df['limited_bins'] = hierarchy_df['Bin'].apply(lambda bins: format_bins(bins))

    fig = px.treemap(
        hierarchy_df,
        names='annotation',
        parents='parent',
        values='scaled_coverage',
        color='level',
        color_continuous_scale='Sunset'
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=2, color=hierarchy_df['border_color'])
        ),
        customdata=hierarchy_df[['level_name', 'limited_bins']],  # Use limited bins for hover
        hovertemplate='<b>%{label}</b><br>Level: %{customdata[0]}<br>Coverage: %{value}<br>Bins: %{customdata[1]}'
    )

    fig.update_layout(
        coloraxis_showscale=False,
        font_size=20,
        autosize=True,
        margin=dict(t=30, b=0, l=0, r=0)
    )

    # Prepare data for bar chart with 3 traces (keep existing logic)
    total_bin_coverage = contig_information.groupby('Bin annotation')['Bin coverage'].sum().reindex(unique_annotations)
    inter_annotation_contact_sum = contact_matrix.sum(axis=1) - np.diag(contact_matrix.values)
    total_bin_coverage_sum = total_bin_coverage.values
    bin_counts = contig_information['Bin annotation'].value_counts()

    node_colors = {}
    for annotation in total_bin_coverage.index:
        bin_type = contig_information.loc[contig_information['Bin annotation'] == annotation, 'type'].values[0]
        color = type_colors.get(bin_type, default_color)
        node_colors[annotation] = color

    data_dict = {
        'Total Inter-Annotation Contact': pd.DataFrame({'name': unique_annotations, 'value': inter_annotation_contact_sum, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]}),
        'Total Coverage': pd.DataFrame({'name': unique_annotations, 'value': total_bin_coverage_sum, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]}),
        'Bin Number': pd.DataFrame({'name': unique_annotations, 'value': bin_counts, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]})
    }

    bar_fig = create_bar_chart(data_dict)

    return fig, bar_fig

#Function to visualize annotation relationship
def basic_visualization(contig_information, unique_annotations, contact_matrix):
    G = nx.Graph()

    # Add nodes with size based on total bin coverage
    total_bin_coverage = contig_information.groupby('Bin annotation')['Bin coverage'].sum().reindex(unique_annotations)
    node_sizes = generate_gradient_values(total_bin_coverage.values, 10, 30)  # Example range from 10 to 30

    node_colors = {}
    for annotation, size in zip(total_bin_coverage.index, node_sizes):
        bin_type = contig_information.loc[
            contig_information['Bin annotation'] == annotation, 'type'
        ].values[0]
        color = type_colors.get(bin_type, default_color)
        node_colors[annotation] = color
        G.add_node(annotation, size=size, color=color, parent=None)

    # Collect all edge weights
    edge_weights = []
    invisible_edges = set()
    
    for i, annotation_i in enumerate(unique_annotations):
        for j, annotation_j in enumerate(unique_annotations[i + 1:], start=i + 1):
            weight = contact_matrix.at[annotation_i, annotation_j]
            if weight > 0:
                G.add_edge(annotation_i, annotation_j, weight=weight)
                invisible_edges.add((annotation_i, annotation_j))
                edge_weights.append(weight)

    # Normalize edge weights using generate_gradient_values
    if edge_weights:
        normalized_weights = generate_gradient_values(np.array(edge_weights), 1, 3)

        # Assign normalized weights back to edges
        for (i, (u, v)) in enumerate(G.edges()):
            G[u][v]['weight'] = normalized_weights[i]

    # Initial node positions using a force-directed layout with increased dispersion
    n = len(G)
    k = 1 /sqrt(n)
    pos = nx.spring_layout(G, dim=2, k=k, iterations=200, weight='weight',scale=5.0)

    # Convert to Cytoscape elements
    cyto_elements = nx_to_cyto_elements(G, pos, invisible_edges=invisible_edges)

    # Prepare data for bar chart with 3 traces
    inter_annotation_contact_sum = contact_matrix.sum(axis=1) - np.diag(contact_matrix.values)
    total_bin_coverage_sum = total_bin_coverage.values
    bin_counts = contig_information['Bin annotation'].value_counts()

    data_dict = {
        'Total Inter-Annotation Contact': pd.DataFrame({'name': unique_annotations, 'value': inter_annotation_contact_sum, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]}),
        'Total Coverage': pd.DataFrame({'name': unique_annotations, 'value': total_bin_coverage_sum, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]}),
        'Bin Number': pd.DataFrame({'name': unique_annotations, 'value': bin_counts, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]})
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

# Function to visualize intra-annotation relationships
def intra_annotation_visualization(selected_annotation, contig_information, unique_annotations, contact_matrix):
    G = nx.Graph()

    # Add nodes with size based on bin counts
    bin_counts = [len(contig_information[contig_information['Bin annotation'] == node]) for node in unique_annotations]
    node_sizes = generate_gradient_values(np.array(bin_counts), 10, 30)
    indices = get_bin_indexes(selected_annotation, contig_information)

    nodes_to_remove = []
    for annotation, size in zip(unique_annotations, node_sizes):
        bin_type = contig_information.loc[
            contig_information['Bin annotation'] == annotation, 'type'
        ].values[0]
        
        color = type_colors.get(bin_type, default_color)

        if annotation == selected_annotation:
            G.add_node(annotation, size=size, color='#FFFFFF', border_color='#000', border_width=2, parent=None)  # White for selected node
        else:
            num_connected_bins = len(contig_information[(contig_information['Bin annotation'] == annotation) & (bin_dense_matrix[:, indices].sum(axis=1) > 0)])
            if num_connected_bins == 0:
                nodes_to_remove.append(annotation)
            else:
                G.add_node(annotation, size=size, color=color, parent=None)  # Red for viral, blue for non-viral

    # Add edges with weight based on inter-annotation contacts
    inter_annotation_contacts = []
    for annotation_i in unique_annotations:
        for annotation_j in unique_annotations:
            if annotation_i != annotation_j and contact_matrix.at[annotation_i, annotation_j] > 0:
                weight = contact_matrix.at[annotation_i, annotation_j]
                G.add_edge(annotation_i, annotation_j, weight=weight)
                inter_annotation_contacts.append(weight)

    # Remove nodes not connected to selected annotation
    for node in nodes_to_remove:
        if node in G:
            G.remove_node(node)

    # Generate gradient values for the edge weights
    edge_weights = generate_gradient_values(np.array(inter_annotation_contacts), 10, 100)

    edges_to_remove = []
    inter_annotation_contacts = []

    # Collect edge weights and identify edges to remove
    for edge in G.edges(data=True):
        if edge[0] == selected_annotation or edge[1] == selected_annotation:
            weight = contact_matrix.at[selected_annotation, edge[1]] if edge[0] == selected_annotation else contact_matrix.at[edge[0], selected_annotation]
            inter_annotation_contacts.append(weight)
        else:
            edges_to_remove.append((edge[0], edge[1]))

    # Remove edges not connected to selected_annotation
    for edge in edges_to_remove:
        G.remove_edge(edge[0], edge[1])
    # Assign the gradient values as edge weights and set default edge color
    for (u, v, d), weight in zip(G.edges(data=True), edge_weights):
        if edge[0] == selected_annotation or edge[1] == selected_annotation:
            d['weight'] = weight

    # Calculate k_value based on the number of bins of the selected annotation
    num_bins = len(indices)
    k_value = sqrt(num_bins)

    new_pos = nx.spring_layout(G, pos={selected_annotation: (0, 0)}, fixed=[selected_annotation], k=k_value, iterations=50, weight='weight')

    # Get and arrange bins within the selected annotation node
    bins = contig_information.loc[indices, 'Bin']
    # Initialize required variables
    bin_contact_values = []
    bin_contact_colors = []
    bin_contact_hovers = [] 
    inter_bin_edges = set()
    
    # Unified loop
    for i in indices:
        bin_i = contig_information.at[i, 'Bin']
        for j in range(bin_dense_matrix.shape[0]):
            if bin_dense_matrix[i, j] != 0 and contig_information.at[j, 'Bin annotation'] != selected_annotation:
                bin_j = contig_information.at[j, 'Bin']
                contact_value = bin_dense_matrix[i, j]
    
                # Add to bin contact values and colors
                bin_contact_values.append((bin_i, bin_j, contact_value))
    
                # Get the color of annotation B (where B is the annotation of bin_j)
                annotation_b = contig_information.at[j, 'Bin annotation']
                bin_j_color = G.nodes[annotation_b]['color']
                bin_contact_colors.append(bin_j_color)
    
                # Add hover information (annotation B and contact value)
                hover_info = f"{annotation_b} - {bin_j}, Contact: {int(contact_value)}"
                bin_contact_hovers.append(hover_info)
    
                # Add bins to inter_bin_edges
                inter_bin_edges.add(bin_i)
                inter_bin_edges.add(bin_j)

    bin_positions = arrange_bins(bins, inter_bin_edges, distance=1, center_position=new_pos[selected_annotation])

    # Add bin nodes and edges to the graph G
    for bin, (x, y) in bin_positions.items():
        G.add_node(bin, size=1, color='#7030A0' if bin in inter_bin_edges else '#00B050', parent=selected_annotation)
        new_pos[bin] = (new_pos[selected_annotation][0] + x, new_pos[selected_annotation][1] + y)

    cyto_elements = nx_to_cyto_elements(G, new_pos)

    # Prepare data for bar chart
    bin_contact_counts = contig_information[contig_information['Bin annotation'] != selected_annotation]['Bin annotation'].value_counts()
    inter_annotation_contacts = contact_matrix.loc[selected_annotation].drop(selected_annotation)

    # Filter out bins that are not in the graph
    filtered_bin_counts = bin_contact_counts[bin_contact_counts.index.isin(G.nodes)]
    filtered_inter_annotation_contacts = inter_annotation_contacts[inter_annotation_contacts.index.isin(G.nodes)]

    # Convert bin contact data to a DataFrame for visualization
    bin_contact_df = pd.DataFrame(bin_contact_values, columns=['Bin 1', 'Bin 2', 'Contact Value'])

    # Prepare data for the bar chart
    data_dict = {
        'Bin Number': pd.DataFrame({
            'name': filtered_bin_counts.index, 
            'value': filtered_bin_counts.values, 
            'color': [G.nodes[annotation]['color'] for annotation in filtered_bin_counts.index]
        }),
        'Inter-Annotation Contacts': pd.DataFrame({
            'name': filtered_inter_annotation_contacts.index, 
            'value': filtered_inter_annotation_contacts.values, 
            'color': [G.nodes[annotation]['color'] for annotation in filtered_inter_annotation_contacts.index]
        }),
        'Bin-to-Bin Contacts': pd.DataFrame({
            'name': bin_contact_df['Bin 1'] + ' to ' + bin_contact_df['Bin 2'], 
            'value': bin_contact_df['Contact Value'], 
            'color': bin_contact_colors,  # Use annotation colors for bin-to-bin contacts
            'hover': bin_contact_hovers  # Adding hover data here
        })
    }

    # Create the bar chart with the additional bin-to-bin contact trace
    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

# Function to visualize inter-annotation relationships
def inter_annotation_visualization(selected_annotation, secondary_annotation, contig_information):

    row_bin = selected_annotation
    col_bin = secondary_annotation

    G = nx.Graph()
    G.add_node(row_bin, color='#FFFFFF', border_color='black', border_width=2, label=row_bin)
    G.add_node(col_bin, color='#FFFFFF', border_color='black', border_width=2, label=col_bin)

    new_pos = {row_bin: (-0.2, 0), col_bin: (0.2, 0)}

    row_indices = get_bin_indexes(row_bin, contig_information)
    col_indices = get_bin_indexes(col_bin, contig_information)
    inter_bins_row = set()
    inter_bins_col = set()

    interannotation_contacts = []
    bin_contact_counts = []
    inter_bin_contacts = []

    for i in row_indices:
        for j in col_indices:
            contact_value = bin_dense_matrix[i, j]
            if contact_value != 0:
                inter_bins_row.add(contig_information.at[i, 'Bin'])
                inter_bins_col.add(contig_information.at[j, 'Bin'])
                interannotation_contacts.append({
                    'name': f"{contig_information.at[i, 'Bin']} - {contig_information.at[j, 'Bin']}",
                    'value': contact_value,
                    'color': 'green'  # Set green color for the bars
                })
                bin_contact_counts.append({
                    'name': contig_information.at[i, 'Bin'],
                    'annotation': selected_annotation,
                    'count': 1,
                    'color': '#C00000'  # Set red color for the bars
                })
                bin_contact_counts.append({
                    'name': contig_information.at[j, 'Bin'],
                    'annotation': secondary_annotation,
                    'count': 1,
                    'color': '#0070C0'  # Set blue color for the bars
                })
                inter_bin_contacts.append({
                    'name': contig_information.at[i, 'Bin'],
                    'value': contact_value,
                    'color': '#C00000'  # Set red color for the bars
                })
                inter_bin_contacts.append({
                    'name': contig_information.at[j, 'Bin'],
                    'value': contact_value,
                    'color': '#0070C0'  # Set blue color for the bars
                })

    bin_positions_row = arrange_bins(inter_bins_row, list(), distance=1, center_position=new_pos[row_bin])
    bin_positions_col = arrange_bins(inter_bins_col, list(), distance=1, center_position=new_pos[col_bin])

    # Add bin nodes to the graph G
    for bin, (x, y) in bin_positions_row.items():
        G.add_node(bin, color='#C00000', parent=row_bin)  # Red for primary
        new_pos[bin] = (x, y)

    for bin, (x, y) in bin_positions_col.items():
        G.add_node(bin, color='#0070C0', parent=col_bin)  # Blue for secondary
        new_pos[bin] = (x, y)

    # Add edges between bins
    for i in row_indices:
        for j in col_indices:
            contact_value = bin_dense_matrix[i, j]
            if contact_value != 0:
                G.add_edge(contig_information.at[i, 'Bin'], contig_information.at[j, 'Bin'], weight=contact_value)

    invisible_edges = [(u, v) for u, v in G.edges]  # Mark all bin edges as invisible

    cyto_elements = nx_to_cyto_elements(G, new_pos, list(), invisible_edges)

    # Prepare data for bar chart
    interannotation_contacts_df = pd.DataFrame(interannotation_contacts)

    bin_contact_counts_df = pd.DataFrame(bin_contact_counts)
    
    # Log if the DataFrame is empty
    if bin_contact_counts_df.empty:
        logger.warning(f"bin_contact_counts_df is empty. No data available between {selected_annotation} and {secondary_annotation}")
        # Return empty chart or handle accordingly
        return [], go.Figure()  # Return empty elements and figure

    # Try grouping by 'name' and 'color' with error handling
    try:
        bin_contact_counts_summary = bin_contact_counts_df.groupby(['name', 'color']).size().reset_index(name='value')
    except KeyError as e:
        logger.error(f"KeyError in bin_contact_counts_df: {e}. DataFrame columns: {bin_contact_counts_df.columns}")
        # Return empty chart or handle accordingly
        return [], go.Figure()
    
    inter_bin_contacts_df = pd.DataFrame(inter_bin_contacts)

    try:
        inter_bin_contacts_summary = inter_bin_contacts_df.groupby(['name', 'color']).sum().reset_index()
    except KeyError as e:
        logger.error(f"KeyError in inter_bin_contacts_df: {e}. DataFrame columns: {inter_bin_contacts_df.columns}")
        # Return empty chart or handle accordingly
        return [], go.Figure()
    
    data_dict = {
        'Inter Bin Contacts': interannotation_contacts_df,
        'Bin Contacts Counts': bin_contact_counts_summary,
        'Bin Contacts Value': inter_bin_contacts_summary
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

# Function to visualize bin relationships
def bin_visualization(selected_annotation, selected_bin, contig_information, unique_annotations):

    # Find the index of the selected bin
    selected_bin_index = contig_information[contig_information['Bin'] == selected_bin].index[0]
    selected_annotation = contig_information.loc[selected_bin_index, 'Bin annotation']

    # Get all indices that have contact with the selected bin
    contacts_indices = bin_dense_matrix[selected_bin_index].nonzero()[0]
    
    # Remove self-contact
    contacts_indices = contacts_indices[contacts_indices != selected_bin_index]
    
    contacts_annotation = contig_information.loc[contacts_indices, 'Bin annotation']
    contacts_bins = contig_information.loc[contacts_indices, 'Bin']
    
    G = nx.Graph()

    # Use a categorical color scale
    color_scale = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

    # Rank annotation based on the number of bins with contact to the selected bin
    annotation_contact_counts = contacts_annotation.value_counts()
    annotation_contact_ranks = annotation_contact_counts.rank(method='first').astype(int)
    max_rank = annotation_contact_ranks.max()
    
    # Fetch bin indexes for all unique annotations at once
    unique_annotations = contacts_annotation.unique().tolist()
    annotation_indexes_dict = get_bin_indexes(unique_annotations, contig_information)

    # Add annotation nodes and their positions
    for annotation in contacts_annotation.unique():
        annotation_rank = annotation_contact_ranks[annotation]
        gradient_color = color_scale[int((annotation_rank / max_rank) * (len(color_scale) - 1))]
        G.add_node(annotation, size=1, color='#FFFFFF', border_color=gradient_color, border_width=2)  # White color for nodes, gradient color for border


    # Set k value to avoid overlap and generate positions for the graph nodes
    k_value = sqrt(len(G.nodes))
    pos = nx.spring_layout(G, k=k_value, iterations=50, weight='weight')

    # Add bin nodes to the graph
    for annotation in contacts_annotation.unique():
        annotation_bins = contacts_bins[contacts_annotation == annotation]
        bin_positions = arrange_bins(annotation_bins, [], distance=2, center_position=pos[annotation],selected_bin=selected_bin if annotation == selected_annotation else None)
        for bin, (x, y) in bin_positions.items():
            G.add_node(bin, size=1 if bin != selected_bin else 5, color='black' if bin == selected_bin else G.nodes[annotation]['border_color'], parent=annotation)  # Same color as annotation, black for selected bin
            if bin != selected_bin:
                G.add_edge(selected_bin, bin, weight=bin_dense_matrix[selected_bin_index, contig_information[contig_information['Bin'] == bin].index[0]])
            pos[bin] = (x, y)  # Use positions directly from arrange_bins
            
    if selected_annotation not in pos:
        logger.warning(f"Selected annotation '{selected_annotation}' not found in position dictionary. Assigning default position.")
        pos[selected_annotation] = (0, 0)  # Assign a default or placeholder position

    if selected_bin not in pos:
        logger.warning(f"Selected bin '{selected_bin}' not found in position dictionary. Using annotation position.")
        pos[selected_bin] = pos[selected_annotation]

    # Ensure the selected bin node is positioned above all other bins
    pos[selected_bin] = pos[selected_annotation]

    cyto_elements = nx_to_cyto_elements(G, pos)
    
    # Prepare data for bar chart
    bin_contact_values = bin_dense_matrix[selected_bin_index, contacts_indices]
    bin_data = pd.DataFrame({'name': contacts_bins, 'value': bin_contact_values, 'color': [G.nodes[bin]['color'] for bin in contacts_bins]})

    annotation_contact_values = []
    bin_contact_counts_per_annotation = [] 
    for annotation in unique_annotations:
        annotation_indexes = annotation_indexes_dict[annotation]  # Use the pre-fetched indexes
        contact_value = bin_dense_matrix[selected_bin_index, annotation_indexes].sum()
        annotation_contact_values.append(contact_value)
        bin_contact_counts_per_annotation.append(len(annotation_indexes))

    annotation_data = pd.DataFrame({'name': contacts_annotation.unique(), 'value': annotation_contact_values, 'color': [G.nodes[annotation]['color'] for annotation in contacts_annotation.unique()]})
    bin_contact_counts_data = pd.DataFrame({'name': contacts_annotation.unique(), 'value': bin_contact_counts_per_annotation, 'color': [G.nodes[annotation]['color'] for annotation in contacts_annotation.unique()]})

    data_dict = {
        'Bin Contacts': bin_data,
        'Annotation Contacts': annotation_data,
        'Bin Contact Counts': bin_contact_counts_data  # New trace
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

def prepare_data(contig_information_intact, bin_dense_matrix, taxonomy_level = 'Family'):
    global contig_information
    global unique_annotations
    global contact_matrix
    global contact_matrix_display
    
    if taxonomy_level is None:
        taxonomy_level = 'Family'

    contig_information = contig_information_intact.copy()
    contig_information['Bin annotation'] = contig_information[taxonomy_level]
    taxonomy_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    contig_information = contig_information.drop(columns=taxonomy_columns)
    
    unique_annotations = contig_information['Bin annotation'].unique()

    contact_matrix = pd.DataFrame(0.0, index=unique_annotations, columns=unique_annotations)
    bin_indexes_dict = get_bin_indexes(unique_annotations, contig_information)

    # Use the pre-fetched indexes for calculating contacts
    for annotation_i in unique_annotations:
        for annotation_j in unique_annotations:   
            indexes_i = bin_indexes_dict[annotation_i]
            indexes_j = bin_indexes_dict[annotation_j]
            sub_matrix = bin_dense_matrix[np.ix_(indexes_i, indexes_j)]
            
            contact_matrix.at[annotation_i, annotation_j] = sub_matrix.sum()

    contact_matrix_display = contact_matrix.astype(int).copy()  # Convert to int for display
    contact_matrix_display.insert(0, 'Annotation', contact_matrix_display.index)  # Add the 'Annotation' column

    return contig_information, unique_annotations, contact_matrix, contact_matrix_display

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create a logger and add the custom Dash logger handler
logger = logging.getLogger(__name__)
dash_logger_handler = DashLoggerHandler()
dash_logger_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
dash_logger_handler.setFormatter(formatter)
logger.addHandler(dash_logger_handler)
logger.setLevel(logging.INFO)

# Example of logging messages
logger.info("App started")

# Set OpenAI API key
client = OpenAI(api_key='')

# File paths for the current environment
bin_info_path = '../1_Data_Processing/output/bin_info_final.csv'
bin_contact_matrix_path= '../1_Data_Processing/output/bin_contact_matrix.npz'

# Load the data
logger.info('Loading data')
contig_information_intact = pd.read_csv(bin_info_path)
bin_contact_matrix_data = np.load(bin_contact_matrix_path)
data = bin_contact_matrix_data['data']
indices = bin_contact_matrix_data['indices']
indptr = bin_contact_matrix_data['indptr']
shape = bin_contact_matrix_data['shape']
bin_sparse_matrix = csc_matrix((data, indices, indptr), shape=shape)
bin_dense_matrix = bin_sparse_matrix.toarray()

matrix_columns = {
    'Binning information': 'Bin',
    'Domain': 'Domain',
    'Kingdom': 'Kingdom',
    'Phylum': 'Phylum',
    'Class': 'Class',
    'Order': 'Order',
    'Family': 'Family',
    'Genus': 'Genus',
    'Species': 'Species',
    'Restriction sites': 'Restriction sites',
    'Contig length': 'Bin length',
    'Contig coverage': 'Bin coverage',
    'Intra-contig contact': 'Intra-bin contact'
}
contig_information_intact = contig_information_intact.rename(columns=matrix_columns)

# Add a "Visibility" column to the contig_information_display DataFrame
contig_information_display = contig_information_intact[list(matrix_columns.values())]
contig_information_display.loc[:, 'Visibility'] = 1  # Default value to 1 (visible)

logger.info('Data loading completed')  

contig_information, unique_annotations, contact_matrix, contact_matrix_display = prepare_data(contig_information_intact, bin_dense_matrix)
  
type_colors = {
    'chromosome': '#0000FF',
    'phage': '#FF0000',
    'plasmid': '#00FF00'
}
default_color = '#808080' 

# Define the column definitions for AG Grid
column_defs = [
    {
        "headerName": "Bin",
        "children": [
            {"headerName": "Bin", "field": "Bin", "pinned": 'left', "width": 120}
        ]
    },
    {        
         "headerName": "Taxonomy",
         "children": [   
            {"headerName": "Species", "field": "Species", "width": 140, "wrapHeaderText": True},
            {"headerName": "Genus", "field": "Genus", "width": 140, "wrapHeaderText": True},
            {"headerName": "Family", "field": "Family", "width": 140, "wrapHeaderText": True},
            {"headerName": "Order", "field": "Order", "width": 140, "wrapHeaderText": True},
            {"headerName": "Class", "field": "Class", "width": 140, "wrapHeaderText": True},
            {"headerName": "Phylum", "field": "Phylum", "width": 140, "wrapHeaderText": True},
            {"headerName": "Kingdom", "field": "Kingdom", "width": 140, "wrapHeaderText": True},
            {"headerName": "Domain", "field": "Domain", "width": 140, "wrapHeaderText": True}
        ]
    },
    {
        "headerName": "Contact Information",
        "children": [
            {"headerName": "Restriction sites", "field": "Restriction sites", "width": 140, "wrapHeaderText": True},
            {"headerName": "Bin length", "field": "Bin length", "width": 140, "wrapHeaderText": True},
            {"headerName": "Bin coverage", "field": "Bin coverage", "width": 140, "wrapHeaderText": True},
            {"headerName": "Intra-bin contact", "field": "Intra-bin contact", "width": 140, "wrapHeaderText": True},
            {"headerName": "Visibility", "field": "Visibility", "hide": True} 
        ]
    }
]

# Define the default column definitions

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
    'taxonomy_level': None,
    'selected_annotation': None,
    'secondary_annotation': None,
    'selected_bin': None
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

logger.info('Generating default visulization') 
treemap_fig, bar_fig = taxonomy_visualization()

# Use the styling functions in the Dash layout
app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds (every 1 second)
        n_intervals=0
        ),
    html.Div([
        html.Button("Download Current View", id="download-btn", style={**common_style}),
        html.Button("Reset", id="reset-btn", style={**common_style}),
        html.Button("Help", id="open-help", style={**common_style}),
        dcc.Download(id="download-dataframe-csv"),
        dcc.Dropdown(
            id='visualization-selector',
            options=[
                {'label': 'Taxonomy Hierarchy', 'value': 'taxonomy_hierarchy'},
                {'label': 'Community', 'value': 'basic'},
                {'label': 'Intra-annotation', 'value': 'intra_annotation'},
                {'label': 'Inter-annotation', 'value': 'inter_annotation'},
                {'label': 'Bin', 'value': 'bin'}
            ],
            value='taxonomy_hierarchy',
            style={'width': '300px', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='taxonomy-level-selector',
            options=[
                {'label': 'Domain', 'value': 'Domain'},
                {'label': 'Kingdom', 'value': 'Kingdom'},
                {'label': 'Phylum', 'value': 'Phylum'},
                {'label': 'Class', 'value': 'Class'},
                {'label': 'Order', 'value': 'Order'},
                {'label': 'Family', 'value': 'Family'},
                {'label': 'Genus', 'value': 'Genus'},
                {'label': 'Species', 'value': 'Species'},
            ],
            value='Family',
            placeholder="Select Taxonomy Level",
            style={'width': '300px', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='annotation-selector',
            options=[],
            value=None,
            placeholder="Select a annotation",
            style={}
        ),
        dcc.Dropdown(
            id='secondary-annotation-selector',
            options=[],
            value=None,
            placeholder="Select a secondary annotation",
            style={}
        ),
        dcc.Dropdown(
            id='bin-selector',
            options=[],
            value=None,
            placeholder="Select a bin",
            style={}
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
                options=[{'label': '  Only show bins in the map', 'value': 'filter'}],
                value=['filter'],
                style={'display': 'inline-block', 'margin-right': '10px', 'float': 'right'}
            ),
            dag.AgGrid(
                id='bin-info-table',
                columnDefs=column_defs,
                rowData=contig_information_display.to_dict('records'),
                defaultColDef={
                        "sortable": True,
                        "filter": True,
                        "resizable": True
                    },
                style={'height': '40vh', 'width': '30vw', 'display': 'inline-block'},
                dashGridOptions={
                    'headerPinned': 'top',
                    'rowSelection': 'single'  # Enable single row selection
                }
            )
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            dcc.Graph(id='treemap-graph', figure=treemap_fig, config={'displayModeBar': False}, style={'height': '80vh', 'width': '48vw', 'display': 'inline-block'}),
            cyto.Cytoscape(
                id='cyto-graph',
                elements=[],
                stylesheet=base_stylesheet,
                style={},
                layout={'name': 'preset'},  # Use preset to keep the initial positions
                zoom=1,
                userZoomingEnabled=True,
                wheelSensitivity=0.1  # Reduce the wheel sensitivity
            )
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
    html.Div([
        html.Div(id='hover-info', style={'height': '20vh', 'width': '19vw', 'background-color': 'white', 'padding': '5px', 'border': '1px solid #ccc', 'margin-top': '3px'}),
            html.Div([
                dcc.Textarea(
                    id='chatgpt-input',
                    placeholder='Enter your query here...',
                    style={'width': '100%', 'height': '15vh', 'display': 'inline-block'}
                ),
                html.Button('Interpret Data', id='interpret-button', n_clicks=0, style={'width': '100%', 'display': 'inline-block'})
            ], style={'width': '19vw', 'display': 'inline-block'}),
            html.Div(id='gpt-answer', 
                     style={'height': '45vh', 
                            'width': '19vw', 
                            'whiteSpace': 'pre-wrap',
                            'font-size': '8px',
                            'background-color': 'white', 
                            'padding': '5px', 
                            'border': '1px solid #ccc', 
                            'margin-top': '3px',
                            'overflowY': 'auto' }
            ),
            dcc.Store(id='scroll-trigger', data=False)
        ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px'}),
    ], style={'width': '100%', 'display': 'flex'}),
    html.Div([
        dash_table.DataTable(
            id='contact-table',
            columns=[],
            data=[],
            style_table={'height': 'auto', 'overflowY': 'auto', 'overflowX': 'auto', 'width': '99vw', 'minWidth': '100%'},
            style_data_conditional=[],
            style_cell={'textAlign': 'left', 'minWidth': '120px', 'width': '120px', 'maxWidth': '180px'},
            style_header={'whiteSpace': 'normal', 'height': 'auto'},  # Allow headers to wrap
            fixed_rows={'headers': True},  # Freeze the first row
            fixed_columns={'headers': True, 'data': 1}  # Freeze the first column
        )
    ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
    help_modal
], style={'height': '100vh', 'overflowY': 'auto', 'width': '100%'})

@app.callback(
    [Output('contact-table', 'columns'),
     Output('contact-table', 'data'),
     Output('contact-table', 'style_data_conditional')],
    [Input('taxonomy-level-selector', 'value')]
)
def update_data(taxonomy_level):
    
    contig_information, unique_annotations, contact_matrix, contact_matrix_display = prepare_data(contig_information_intact, bin_dense_matrix, taxonomy_level)    
    
    global current_visualization_mode
    
    current_visualization_mode = {
        'visualization_type': 'taxonomy_hierarchy',
        'taxonomy_level': taxonomy_level,
        'selected_annotation': None,
        'secondary_annotation': None,
        'selected_bin': None
    }
    
    # Generate table columns based on the DataFrame's columns
    table_columns = [{"name": col, "id": col} for col in contact_matrix_display.columns]
    # Convert the DataFrame into a list of dictionaries (format required by Dash tables)
    table_data = contact_matrix_display.to_dict('records')
    # Generate the conditional styling based on the stored data
    style_conditions = styling_annotation_table(contact_matrix_display)
        
    return table_columns, table_data, style_conditions

@app.callback(
    [Output('visualization-selector', 'value'),
     Output('annotation-selector', 'value'),
     Output('annotation-selector', 'style'),
     Output('secondary-annotation-selector', 'value'),
     Output('secondary-annotation-selector', 'style'),
     Output('bin-selector', 'value'),
     Output('bin-selector', 'style'),
     Output('contact-table', 'active_cell'),
     Output('bin-info-table', 'selectedRows')],
    [Input('reset-btn', 'n_clicks'),
     Input('visualization-selector', 'value'),
     Input('contact-table', 'active_cell'),
     Input('bin-info-table', 'selectedRows'),
     Input('cyto-graph', 'selectedNodeData'),
     Input('cyto-graph', 'selectedEdgeData')],
     State('taxonomy-level-selector', 'value'),
     prevent_initial_call=True
)
def sync_selectors(reset_clicks,visualization_type, contact_table_active_cell, bin_info_selected_rows, selected_node_data, selected_edge_data, taxonomy_level):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    selected_annotation, secondary_annotation, selected_bin = synchronize_selections(
        triggered_id, selected_node_data, selected_edge_data, bin_info_selected_rows, contact_table_active_cell, taxonomy_level )
    
    annotation_selector_style = {'display': 'none'}
    secondary_annotation_style = {'display': 'none'}
    bin_selector_style = {'display': 'none'}
    
    # Reset all the selections
    if triggered_id == 'reset-btn':
        visualization_type = 'taxonomy_hierarchy'
        selected_annotation = None
        secondary_annotation = None
        selected_bin = None
        

    # If a bin is selected in the network or bin table
    if selected_bin:
        visualization_type = 'bin'
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
        bin_selector_style = {'width': '300px', 'display': 'inline-block'}

    # If a annotation is selected in the network or annotation table
    if selected_annotation and not selected_bin:
        if secondary_annotation:
            visualization_type = 'inter_annotation'
            annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
            secondary_annotation_style = {'width': '300px', 'display': 'inline-block'}
        else:
            visualization_type = 'intra_annotation'
            annotation_selector_style = {'width': '300px', 'display': 'inline-block'}

    # Default cases based on visualization_type
    if visualization_type == 'intra_annotation':
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
    elif visualization_type == 'inter_annotation':
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
        secondary_annotation_style = {'width': '300px', 'display': 'inline-block'}
    elif visualization_type == 'bin':
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
        bin_selector_style = {'width': '300px', 'display': 'inline-block'}

    return visualization_type, selected_annotation, annotation_selector_style, secondary_annotation, secondary_annotation_style, selected_bin, bin_selector_style, None, []
 
def synchronize_selections(triggered_id, selected_node_data, selected_edge_data, bin_info_selected_rows, contact_table_active_cell, taxonomy_level ):
    # Initialize the return values
    selected_annotation = None
    secondary_annotation = None
    selected_bin = None
    

    # If a node in the network is selected
    if triggered_id == 'cyto-graph' and selected_node_data:
        selected_node_id = selected_node_data[0]['id']
        # Check if the selected node is a bin or a annotation
        if selected_node_id in contig_information['Bin'].values:
            bin_info = contig_information[contig_information['Bin'] == selected_node_id].iloc[0]
            selected_annotation = bin_info['Bin annotation']
            selected_bin = bin_info['Bin']
        else:
            selected_annotation = selected_node_id

    # If an edge in the network is selected
    elif triggered_id == 'cyto-graph' and selected_edge_data:
        source_annotation = selected_edge_data[0]['source']
        target_annotation = selected_edge_data[0]['target']
        selected_annotation = source_annotation
        secondary_annotation = target_annotation

    # If a row in the bin-info-table is selected
    elif triggered_id == 'bin-info-table' and bin_info_selected_rows:
        print(bin_info_selected_rows)
        selected_row = bin_info_selected_rows[0]
        if taxonomy_level in selected_row and 'Bin' in selected_row:
            selected_annotation = selected_row[taxonomy_level]
            selected_bin = selected_row['Bin']

    # If a cell in the contact-table is selected
    elif triggered_id == 'contact-table' and contact_table_active_cell:
        row_id = contact_table_active_cell['row']
        column_id = contact_table_active_cell['column_id']
        row_annotation = contact_matrix_display.iloc[row_id]['Annotation']
        col_annotation = column_id if column_id != 'Annotation' else None
        selected_annotation = row_annotation
        secondary_annotation = col_annotation
    
    logger.info(f'Current selected: \n{selected_annotation}, \n{secondary_annotation}, \n{selected_bin}')

    return selected_annotation, secondary_annotation, selected_bin

# Callback to update the visualizationI want to 
@app.callback(
    [Output('cyto-graph', 'elements'),
     Output('cyto-graph', 'style'),
     Output('bar-chart', 'figure'),
     Output('treemap-graph', 'figure'),
     Output('treemap-graph', 'style'),
     Output('bin-info-table', 'defaultColDef')],
    [Input('reset-btn', 'n_clicks'),
     Input('confirm-btn', 'n_clicks')],
    [State('visualization-selector', 'value'),
     State('annotation-selector', 'value'),
     State('secondary-annotation-selector', 'value'),
     State('bin-selector', 'value'),
     State('contact-table', 'data')],
     prevent_initial_call=True
)
def update_visualization(reset_clicks, confirm_clicks, visualization_type, selected_annotation, secondary_annotation, selected_bin, table_data):
    logger.info('update_visualization triggered')
    logger.info(f'Visualization type: {visualization_type}')
    logger.info(f'Selected annotation: {selected_annotation}')
    logger.info(f'Secondary annotation: {secondary_annotation}')
    logger.info(f'Selected bin: {selected_bin}')
    
    global current_visualization_mode
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'reset-btn':
        # Reset to show the original plot
        logger.info('reset to show taxonomy_hierarchy visulization')
        current_visualization_mode = {
            'visualization_type': 'taxonomy_hierarchy',
            'selected_annotation': None,
            'secondary_annotation': None,
            'selected_bin': None
        }
        treemap_fig, bar_fig = taxonomy_visualization()
        treemap_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}
        cyto_elements = []
        cyto_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}

    elif triggered_id == 'confirm-btn':
        # Update the current visualization mode with selected values
        current_visualization_mode['visualization_type'] = visualization_type
        current_visualization_mode['selected_annotation'] = selected_annotation
        current_visualization_mode['secondary_annotation'] = secondary_annotation
        current_visualization_mode['selected_bin'] = selected_bin

        if visualization_type == 'taxonomy_hierarchy':
            logger.info('show taxonomy_hierarchy visulization')
            treemap_fig, bar_fig = taxonomy_visualization()
            treemap_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}
            cyto_elements = []
            cyto_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
        elif visualization_type == 'basic' or not selected_annotation:
            logger.info('show basic visulization')
            cyto_elements, bar_fig = basic_visualization(contig_information, unique_annotations, contact_matrix)
            treemap_fig = go.Figure()
            treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            cyto_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}
        elif visualization_type == 'intra_annotation':
            logger.info('show intra_annotation visulization')
            cyto_elements, bar_fig = intra_annotation_visualization(selected_annotation, contig_information, unique_annotations, contact_matrix)
            treemap_fig = go.Figure()
            treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            cyto_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}
        elif visualization_type == 'inter_annotation':
            logger.info('show inter_annotation visulization')
            if selected_annotation and secondary_annotation:
                cyto_elements, bar_fig = inter_annotation_visualization(selected_annotation, secondary_annotation, contig_information)
                treemap_fig = go.Figure()
                treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
                cyto_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}
        elif visualization_type == 'bin':
            logger.info('show bin visulization')
            cyto_elements, bar_fig = bin_visualization(selected_annotation, selected_bin, contig_information, unique_annotations)
            treemap_fig = go.Figure()
            treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            cyto_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}

    # Update column definitions with style conditions
    bin_colors, annotation_colors = get_bin_and_annotation_colors(cyto_elements, contig_information)
    styleConditions = styling_bin_table(bin_colors, annotation_colors, unique_annotations)
    default_col_def = {
        "sortable": True,
        "filter": True,
        "resizable": True,
        "cellStyle": {
            "styleConditions": styleConditions
        }
    }
    return cyto_elements, cyto_style, bar_fig, treemap_fig, treemap_style, default_col_def

@app.callback(
    [Output('cyto-graph', 'stylesheet'),
     Output('hover-info', 'children')],
    [Input('annotation-selector', 'value'),
     Input('secondary-annotation-selector', 'value'),
     Input('bin-selector', 'value')],
     prevent_initial_call=True
)
def update_selected_styles(selected_annotation, secondary_annotation, selected_bin):
    selected_nodes = []
    selected_edges = []
    hover_info = "No selection"

    if selected_annotation and secondary_annotation:
        selected_edges.append((selected_annotation, secondary_annotation))
        selected_nodes.append(selected_annotation)
        selected_nodes.append(secondary_annotation)
        hover_info = f"Edge between {selected_annotation} and {secondary_annotation}"
    elif selected_bin:
        selected_nodes.append(selected_bin)
        bin_info = contig_information[contig_information['Bin'] == selected_bin].iloc[0]
        hover_info = f"Bin: {selected_bin}<br>Annotation: {bin_info['Bin annotation']}"

        if current_visualization_mode['visualization_type'] == 'inter_annotation':
            if bin_info['Bin annotation'] == current_visualization_mode['secondary_annotation']:
                # Find bins from the selected annotation that have contact with the selected bin
                selected_bin_index = contig_information[contig_information['Bin'] == selected_bin].index[0]
                selected_annotation_indices = get_bin_indexes(current_visualization_mode['selected_annotation'], contig_information)
                connected_bins = []

                for j in selected_annotation_indices:
                    if bin_dense_matrix[j, selected_bin_index] != 0:  # Check contact from selected annotation to the selected bin
                        connected_bin = contig_information.at[j, 'Bin']
                        connected_bins.append(connected_bin)

                # Add the connected bins and edges to the lists
                selected_nodes.extend(connected_bins)
                for bin in connected_bins:
                    selected_edges.append((bin, selected_bin))  # Edge goes from the connected bin to the selected bin

            else:
                # Find the bins in the secondary annotation that have contact with the selected bin
                selected_bin_index = contig_information[contig_information['Bin'] == selected_bin].index[0]
                secondary_annotation_indices = get_bin_indexes(current_visualization_mode['secondary_annotation'], contig_information)
                connected_bins = []

                for j in secondary_annotation_indices:
                    if bin_dense_matrix[selected_bin_index, j] != 0:  # Check contact from selected bin to secondary annotation
                        connected_bin = contig_information.at[j, 'Bin']
                        connected_bins.append(connected_bin)

                # Add the connected bins and edges to the lists
                selected_nodes.extend(connected_bins)
                for bin in connected_bins:
                    selected_edges.append((selected_bin, bin))  # Edge goes from selected bin to the connected bin

    elif selected_annotation:
        selected_nodes.append(selected_annotation)
        hover_info = f"Annotation: {selected_annotation}"

        if current_visualization_mode['visualization_type'] == 'basic':
            # Only show edges connected to the selected node using the contact matrix
            annotation_index = unique_annotations.tolist().index(selected_annotation)
            for i, contact_value in enumerate(contact_matrix.iloc[annotation_index]):
                if contact_value > 0:
                    connected_annotation = unique_annotations[i]
                    selected_edges.append((selected_annotation, connected_annotation))

    # Add selection styles for the selected nodes and edges
    stylesheet = add_selection_styles(selected_nodes, selected_edges)

    return stylesheet, hover_info

@app.callback(
    [Output('bin-info-table', 'rowData'), 
     Output('bin-info-table', 'filterModel'),
     Output('row-count', 'children')],
    [Input('annotation-selector', 'value'),
     Input('secondary-annotation-selector', 'value'),
     Input('visibility-filter', 'value')],
    [State('taxonomy-level-selector', 'value'),
     State('bin-info-table', 'rowData')]
)
def update_filter_model_and_row_count(selected_annotation, secondary_annotation, filter_value, taxonomy_level, bin_data):
    filter_model = {}
    filtered_data = bin_data
    
    # Set the default visibility to 1
    for row in filtered_data:
        row['Visibility'] = 1
        
    # Update the filter model based on selected annotation and secondary annotation
    if selected_annotation and not secondary_annotation:
        filter_model[taxonomy_level] = {
            "filterType": "text",
            "operator": "OR",
            "conditions": [
                {
                    "filter": selected_annotation,
                    "filterType": "text",
                    "type": "contains",
                }
            ]
        }
        for row in filtered_data:
            if row[taxonomy_level] != selected_annotation:
                row['Visibility'] = 2

    elif selected_annotation and secondary_annotation:
        filter_model[taxonomy_level] = {
            "filterType": "text",
            "operator": "OR",
            "conditions": [
                {
                    "filter": selected_annotation,
                    "filterType": "text",
                    "type": "contains",
                },
                {
                    "filter": secondary_annotation,
                    "filterType": "text",
                    "type": "contains",
                }
            ]
        }
        for row in filtered_data:
            if row[taxonomy_level] not in [selected_annotation, secondary_annotation]:
                row['Visibility'] = 2
    else:
        filter_model = {}

    # Set visibility based on the current visualization mode
    if current_visualization_mode['visualization_type'] == 'intra_annotation':
        if current_visualization_mode['selected_annotation']:
            for row in filtered_data:
                if row[taxonomy_level] != current_visualization_mode['selected_annotation']:
                    row['Visibility'] = 0

    elif current_visualization_mode['visualization_type'] == 'inter_annotation':
        if current_visualization_mode['selected_annotation'] and current_visualization_mode['secondary_annotation']:
            row_indices = get_bin_indexes(current_visualization_mode['selected_annotation'], contig_information)
            col_indices = get_bin_indexes(current_visualization_mode['secondary_annotation'], contig_information)
            inter_bins_row = set()
            inter_bins_col = set()

            for i in row_indices:
                for j in col_indices:
                    contact_value = bin_dense_matrix[i, j]
                    if contact_value != 0:
                        inter_bins_row.add(contig_information.at[i, 'Bin'])
                        inter_bins_col.add(contig_information.at[j, 'Bin'])

            inter_bins = inter_bins_row.union(inter_bins_col)

            for row in filtered_data:
                if row['Bin'] not in inter_bins:
                    row['Visibility'] = 0

    elif current_visualization_mode['visualization_type'] == 'bin':
        if current_visualization_mode['selected_bin']:
            selected_bin_index = contig_information[contig_information['Bin'] == current_visualization_mode['selected_bin']].index[0]

            connected_bins = set()
            for j in range(bin_dense_matrix.shape[0]):
                if bin_dense_matrix[selected_bin_index, j] != 0:
                    connected_bins.add(contig_information.at[j, 'Bin'])

            for row in filtered_data:
                if row['Bin'] not in connected_bins and row['Bin'] != current_visualization_mode['selected_bin']:
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
    [Output('annotation-selector', 'options'),
     Output('secondary-annotation-selector', 'options'),
     Output('bin-selector', 'options')],
    [Input('annotation-selector', 'value')],
    [State('visualization-selector', 'value')],
    prevent_initial_call=True
)
def update_dropdowns(selected_annotation, visualization_type):
    # Initialize empty lists for options
    annotation_options = []
    secondary_annotation_options = []
    bin_options = []

    annotation_options = [{'label': annotation, 'value': annotation} for annotation in unique_annotations]
        
    # Only show secondary annotation options if visualization type is 'inter_annotation'
    if visualization_type == 'inter_annotation':
        secondary_annotation_options = annotation_options  # Same options for secondary annotation dropdown
    else:
        secondary_annotation_options = []  # Hide secondary annotation dropdown if not in inter_annotation mode

    # Only show bin options if visualization type is 'bin'
    if visualization_type == 'bin' and selected_annotation:
        bins = contig_information.loc[get_bin_indexes(selected_annotation, contig_information), 'Bin']
        bin_options = [{'label': bin, 'value': bin} for bin in bins]

    return annotation_options, secondary_annotation_options, bin_options

# Dash callback to use ChatGPT
@app.callback(
    Output('gpt-answer', 'children'),
    [Input('interval-component', 'n_intervals'), 
     Input('interpret-button', 'n_clicks')],
    [State('chatgpt-input', 'value')]
)
def update_logs_and_gpt(n, interpret_clicks, gpt_query):
    # Capture logs
    logs = dash_logger_handler.get_logs()

    # If the interpret button is clicked, handle the GPT query
    if interpret_clicks > 0 and gpt_query:
        gpt_response = get_chatgpt_response(gpt_query)
        logger.info(f"GPT Answer: {gpt_response}")  # Log GPT's answer

    return logs if logs else "No logs yet"

# Client-side callback to scroll to the bottom when 'scroll-trigger' is True
app.clientside_callback(
    """
    function(scrollTrigger) {
        if (scrollTrigger) {
            var gptAnswerDiv = document.getElementById('gpt-answer');
            gptAnswerDiv.scrollTop = gptAnswerDiv.scrollHeight;
        }
        return false;  // Reset the scroll-trigger
    }
    """,
    Output('scroll-trigger', 'data'),
    Input('scroll-trigger', 'data')
)
if __name__ == '__main__':
    app.run_server(debug=True)