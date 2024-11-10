import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import dash_ag_grid as dag
import plotly.graph_objects as go
from dash import callback_context
import plotly.express as px
import plotly.colors as colors 
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
def get_indexes(annotations, bin_information):
    # Number of threads to use: 4 * CPU core count
    num_threads = 2 * os.cpu_count()
    bin_indexes = {}
    
    # Ensure annotations is a list even if a single annotation is provided
    if isinstance(annotations, str):
        annotations = [annotations]
    
    def fetch_indexes(annotation):
        return annotation, bin_information[bin_information['Annotation'] == annotation].index

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(fetch_indexes, annotation): annotation for annotation in annotations}
        
        for future in futures:
            annotation = futures[future]
            try:
                annotation, indexes = future.result()
                bin_indexes[annotation] = indexes
            except Exception as e:
                logger.error(f'Error fetching bin indexes for annotation: {annotation}, error: {e}')

    return bin_indexes

# Function to generate gradient values in a range [A, B]
def generate_gradient_values(input_array, range_A, range_B):
    if len(input_array) == 1:
        # If there's only one value, return the midpoint of the range
        return np.array([0.5 * (range_A + range_B)])
    
    min_val = np.min(input_array)
    max_val = np.max(input_array)

    if min_val == max_val:
        # If all values are the same, return the midpoint for all values
        return np.full_like(input_array, 0.5 * (range_A + range_B))

    # For multiple values, scale them between range_A and range_B
    scaled_values = range_A + ((input_array - min_val) / (max_val - min_val)) * (range_B - range_A)
    return scaled_values

# Function to convert NetworkX graph to Cytoscape elements with sizes and colors
def nx_to_cyto_elements(G, pos, invisible_nodes=set(), invisible_edges=set()):
    logger.info('Converting to cyto element.')
    elements = []
    for node in G.nodes:
        elements.append({
            'data': {
                'id': node,
                'label': node,  
                'label_size': 6 if G.nodes[node].get('parent') is None else 3, # Default size to 6
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
        if data_frame.empty or 'value' not in data_frame.columns:
            logger.warning(f"No data or 'value' column missing for {trace_name}, skipping trace.")
            continue  # Skip if no 'value' column or data is empty
            
        logger.info(f"Creating bar trace for {trace_name}")
        bar_data = data_frame.sort_values(by='value', ascending=False)
        bar_colors = bar_data['color']

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

    if not traces:
        logger.warning("No valid traces created, returning empty figure.")
        return go.Figure()  # Return an empty figure if no valid traces are created
    
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

    # Function to add opacity to a hex color
def add_opacity_to_color(hex_color, opacity):
    if hex_color.startswith('#') and len(hex_color) == 7:
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'
    else:
        # Return a default color if hex_color is invalid
        return f'rgba(255, 255, 255, {opacity})'

# Function to style annotation contact table using Blugrn color scheme
def styling_annotation_table(contact_matrix_display):
    styles = []
    numeric_df = contact_matrix_display.select_dtypes(include=[np.number])
    log_max_value = np.log1p(numeric_df.values.max())
    opacity = 0.6  # Set a fixed opacity for transparency

    # Styling for numeric values in the table (excluding the "Annotation" column)
    for i in range(len(numeric_df)):
        for j in range(len(numeric_df.columns)):
            value = numeric_df.iloc[i, j]
            log_value = np.log1p(value)
            styles.append({
                'if': {
                    'row_index': i,
                    'column_id': numeric_df.columns[j]
                },
                'backgroundColor': f'rgba({255 - int(log_value / log_max_value * 255)}, {255 - int(log_value / log_max_value * 255)}, 255, {opacity})'  # Set background color for the contact matrix.
            })

    # Styling for the first column ("Annotation") based on type_colors
    for i, annotation in enumerate(contact_matrix_display['Annotation']):
        bin_type = bin_information.loc[bin_information['Annotation'] == annotation, 'Type'].values[0]
        annotation_color = type_colors.get(bin_type, default_color)
        annotation_color_with_opacity = add_opacity_to_color(annotation_color, opacity)

        styles.append({
            'if': {
                'row_index': i,
                'column_id': 'Annotation'
            },
            'backgroundColor': annotation_color_with_opacity,
        })

    return styles


# Function to style bin info table
def styling_information_table(display_data, information_data, bin_colors, annotation_colors, unique_annotations, table_type='bin'):
    # Define taxonomy columns common to both tables
    taxonomy_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    # Define other columns based on table type
    columns = ['Restriction sites', 'Length', 'Coverage']

    styles = []
    
    # Style numeric columns (shared by both table types)
    for col in columns:
        numeric_df = display_data[[col]].select_dtypes(include=[np.number])
        numeric_df += 1
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

    # Add style conditions for the "Bin" or "Contig" column based on table type
    id_column = 'Bin' if table_type == 'bin' else 'Contig'
    for item in display_data[id_column]:
        color = bin_colors.get(item, annotation_colors.get(information_data.loc[information_data[id_column] == item, 'Annotation'].values[0], '#FFFFFF'))
        color_with_opacity = add_opacity_to_color(color, 0.6)
        styles.append(
            {
                "condition": f"params.colDef.field == '{id_column}' && params.value == '{item}'",
                "style": {
                    'backgroundColor': color_with_opacity,
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
def get_node_colors(cyto_elements, information_data):
    bin_colors = {}
    annotation_colors = {}

    # Extract colors from Cytoscape elements
    if cyto_elements:
        for element in cyto_elements:
            if 'data' in element and 'color' in element['data'] and 'id' in element['data']:
                bin_colors[element['data']['id']] = element['data']['color']

    # Get annotation colors based on the information data (either bin or contig)
    for annotation in information_data['Annotation'].unique():
        bin_type = information_data[information_data['Annotation'] == annotation]['Type'].values[0]
        annotation_colors[annotation] = type_colors.get(bin_type, default_color)
        
    return bin_colors, annotation_colors

# Function to arrange bins
def arrange_nodes(bins, inter_bin_edges, distance, selected_element=None, center_position=(0, 0)):
    distance /= 100 
    phi = (1 + sqrt(5)) / 2  # golden ratio

    # Identify bins that connect to other annotation
    connecting_bins = [bin for bin in bins if bin in inter_bin_edges and bin != selected_element]
    other_bins = [bin for bin in bins if bin not in inter_bin_edges and bin != selected_element]

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
    if selected_element:
        inner_positions[selected_element] = center_position

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

    host_data = bin_information_intact[bin_information_intact['Type'].isin(['chromosome', 'plasmid'])]
    virus_data = bin_information_intact[bin_information_intact['Type'] == 'phage']

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
        "Bin": bin_information_intact['Bin'].unique()
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
                    "type": row['Type'],
                    "total coverage": row['Coverage'],
                    "border_color": type_colors.get(row['Type'], "gray"),
                    "Bin": [row["Bin"]]
                })
                existing_annotations.add(annotation)
            else:
                for rec in records:
                    if rec['annotation'] == annotation:
                        rec['total coverage'] += row['Coverage']
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
                    "type": row['Type'],
                    "total coverage": row['Coverage'],
                    "border_color": type_colors.get(row['Type'], "gray"),
                    "Bin": [row["Bin"]]
                })
                existing_annotations.add(annotation)
            else:
                for rec in records:
                    if rec['annotation'] == annotation:
                        rec['total coverage'] += row['Coverage']
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
    total_bin_coverage = bin_information.groupby('Annotation')['Coverage'].sum().reindex(unique_annotations)
    inter_annotation_contact_sum = contact_matrix.sum(axis=1) - np.diag(contact_matrix.values)
    total_bin_coverage_sum = total_bin_coverage.values
    bin_counts = bin_information['Annotation'].value_counts()

    node_colors = {}
    for annotation in total_bin_coverage.index:
        bin_type = bin_information.loc[bin_information['Annotation'] == annotation, 'Type'].values[0]
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
def annotation_visualization(bin_information, unique_annotations, contact_matrix, selected_node=None):
    G = nx.Graph()
    
    if selected_node:
        connected_nodes = [selected_node]
    
        # Only iterate over nodes connected to the selected node (i == selected_index)
        for annotation_j in unique_annotations:
            if annotation_j != selected_node:  # Skip the selected node itself
                weight = contact_matrix.at[selected_node, annotation_j]
                if weight > 0:
                    connected_nodes.append(annotation_j)

    else:
        connected_nodes = unique_annotations
        
    # Add nodes with size based on total bin coverage
    total_bin_coverage = bin_information.groupby('Annotation')['Coverage'].sum().reindex(connected_nodes)
    node_sizes = generate_gradient_values(total_bin_coverage.values, 10, 30)  # Example range from 10 to 30

    node_colors = {}
    for annotation, size in zip(total_bin_coverage.index, node_sizes):
        bin_type = bin_information.loc[
            bin_information['Annotation'] == annotation, 'Type'
        ].values[0]
        color = type_colors.get(bin_type, default_color)
        node_colors[annotation] = color
        G.add_node(annotation, size=size, color=color, parent=None)

    # Collect all edge weights
    edge_weights = []
    invisible_edges = set()
    
    for i, annotation_i in enumerate(connected_nodes):
        for j, annotation_j in enumerate(connected_nodes[i + 1:], start=i + 1):
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
    if selected_node:
        pos = nx.spring_layout(G, dim=2, k=1, iterations=200, weight='weight', scale=5.0, fixed=[selected_node], pos={selected_node: (0, 0)})
    else:
        pos = nx.spring_layout(G, dim=2, k=1, iterations=200, weight='weight', scale=5.0)

    # Convert to Cytoscape elements
    cyto_elements = nx_to_cyto_elements(G, pos, invisible_edges=invisible_edges)
    
    # Enable animation in the Cytoscape layout
    cyto_style = {
        'name': 'preset',  # Use 'preset' to keep the initial positions
        'animate': True,  # Enable animation
        'animationDuration': 500,  # Set duration of the animation (in ms)
        'fit': False  # Automatically zoom to fit the nodes
    }
    
    if not selected_node:
        # Prepare data for bar chart with 3 traces
        inter_annotation_contact_sum = contact_matrix.sum(axis=1) - np.diag(contact_matrix.values)
        total_bin_coverage_sum = total_bin_coverage.values
        bin_counts = bin_information['Annotation'].value_counts()
    
        data_dict = {
            'Total Inter-Annotation Contact': pd.DataFrame({'name': unique_annotations, 'value': inter_annotation_contact_sum, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]}),
            'Total Coverage': pd.DataFrame({'name': unique_annotations, 'value': total_bin_coverage_sum, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]}),
            'Bin Number': pd.DataFrame({'name': unique_annotations, 'value': bin_counts, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]})
        }
    
        bar_fig = create_bar_chart(data_dict)
        return cyto_elements, bar_fig, cyto_style

    # If a node is selected, don't return the bar chart
    return cyto_elements, None, cyto_style

# Function to visualize bin relationships
def bin_visualization(selected_annotation, selected_bin, bin_information, bin_dense_matrix, unique_annotations):
    
    # Find the index of the selected bin
    selected_bin_index = bin_information[bin_information['Bin'] == selected_bin].index[0]
    selected_annotation = bin_information.loc[selected_bin_index, 'Annotation']
    
    logger.info(f"Selected Bin Index: {selected_bin_index}")
    logger.info(f"Selected Annotation: {selected_annotation}")

    # Get all indices that have contact with the selected bin
    contacts_indices = bin_dense_matrix[selected_bin_index].nonzero()[0]
    contacts_indices = contacts_indices[contacts_indices != selected_bin_index]
    
    logger.info(f"Contacts Indices: {contacts_indices}")

    # If no contacts found, raise a warning
    if len(contacts_indices) == 0:
        logger.warning(f'No contacts found for the selected bin: {selected_bin}')
        contacts_annotation = pd.Series([selected_annotation])
        contacts_bins = pd.Series([])
    else:
        contacts_annotation = bin_information.loc[contacts_indices, 'Annotation']
        contacts_bins = bin_information.loc[contacts_indices, 'Bin']
    
    logger.info(f"Contacts Annotation: {contacts_annotation}")
    logger.info(f"Contacts Bins: {contacts_bins}")

    G = nx.Graph()

    # Use a categorical color scale
    color_scale_mapping = {
        'phage': colors.sequential.Reds,
        'plasmid': colors.sequential.Greens,
        'chromosome': colors.sequential.Blues
    }

    # Rank annotation based on the number of bins with contact to the selected bin
    annotation_contact_counts = contacts_annotation.value_counts()
    annotation_contact_ranks = annotation_contact_counts.rank(method='first').astype(int)
    max_rank = annotation_contact_ranks.max()

    # Add annotation nodes
    for annotation in contacts_annotation.unique():
        annotation_type = bin_information.loc[bin_information['Annotation'] == annotation, 'Type'].values[0]
        color_scale = color_scale_mapping.get(annotation_type, [default_color])  
        
        # Assign the gradient color based on rank
        annotation_rank = annotation_contact_ranks.get(annotation, int(max_rank/2))
        gradient_color = color_scale[int((annotation_rank / max_rank) * (len(color_scale) - 1))]
        
        G.add_node(annotation, size=1, color='#FFFFFF', border_color=gradient_color, border_width=2)

    # Ensure selected_annotation is added
    if selected_annotation not in G.nodes:
        annotation_type = bin_information.loc[bin_information['Annotation'] == selected_annotation, 'Type'].values[0]
        color_scale = color_scale_mapping.get(annotation_type, ['#00FF00'])
        gradient_color = color_scale[len(color_scale) - 1]
    
        G.add_node(selected_annotation, size=1, color='#FFFFFF', border_color=gradient_color, border_width=2)

    # Collect all contact values between selected_annotation and other annotations
    contact_values = []
    for annotation in contacts_annotation.unique():
        annotation_bins = bin_information[bin_information['Annotation'] == annotation].index
        annotation_bins = annotation_bins[annotation_bins < bin_dense_matrix.shape[1]]  # Filter indices
        contact_value = bin_dense_matrix[selected_bin_index, annotation_bins].sum()

        if contact_value > 0:
            contact_values.append(contact_value)
    
    # Normalize contact values using generate_gradient_values
    scaled_weights = generate_gradient_values(contact_values, 1, 3) if contact_values else [1] * len(contacts_annotation.unique())

    # Add edges between selected_annotation and other annotations with scaled weights
    for i, annotation in enumerate(contacts_annotation.unique()):
        annotation_bins = bin_information[bin_information['Annotation'] == annotation].index
        annotation_bins = annotation_bins[annotation_bins < bin_dense_matrix.shape[1]]  # Filter indices
        contact_value = bin_dense_matrix[selected_bin_index, annotation_bins].sum()
        
        if contact_value > 0:
            G.add_edge(selected_annotation, annotation, weight=scaled_weights[i])

    # Pin selected_bin to position (0, 0) during spring layout calculation
    fixed_positions = {selected_annotation: (0, 0)}

    # Calculate the layout using spring_layout, where weight affects node distance
    pos = nx.spring_layout(G, k=1, iterations=200, fixed=[selected_annotation], pos=fixed_positions, weight='weight')

    # Remove the edges after positioning
    G.remove_edges_from(list(G.edges()))

    # Handle selected_bin node
    G.add_node(selected_bin, size=5, color='white', border_color='black', border_width=2, parent=selected_annotation)
    pos[selected_bin] = (0, 0)

    # Add bin nodes to the graph
    for annotation in contacts_annotation.unique():
        annotation_bins = contacts_bins[contacts_annotation == annotation]
        bin_positions = arrange_nodes(annotation_bins, [], distance=10, center_position=pos[annotation], selected_element=None)
        for bin, (x, y) in bin_positions.items():
            G.add_node(bin, 
                       size=3, 
                       color=G.nodes[annotation]['border_color'],  # Use the color from the annotation
                       parent=annotation)
            G.add_edge(selected_bin, bin)
            pos[bin] = (x, y)
    
    cyto_elements = nx_to_cyto_elements(G, pos)
    
    # Prepare data for bar chart
    bin_contact_values = bin_dense_matrix[selected_bin_index, contacts_indices]
    
    try:
        bin_data = pd.DataFrame({
            'name': contacts_bins, 
            'value': bin_contact_values, 
            'color': [G.nodes[bin]['color'] for bin in contacts_bins],
            'hover': [f"({bin_information.loc[bin_information['Bin'] == bin, 'Annotation'].values[0]}, {value})" for bin, value in zip(contacts_bins, bin_contact_values)]
        })
    except Exception as e:
        logger.error(f"Error creating bin_data DataFrame: {e}")
        bin_data = pd.DataFrame()  # or assign None if that fits better with your app
    
    # Attempt to create annotation_data DataFrame
    try:
        annotation_data = pd.DataFrame({
            'name': contacts_annotation.unique(),
            'value': contact_values,
            'color': [G.nodes[annotation]['border_color'] for annotation in contacts_annotation.unique()]
        })
    except Exception as e:
        logger.error(f"Error creating annotation_data DataFrame: {e}")
        annotation_data = pd.DataFrame()  # or assign None if that fits better with your app

    data_dict = {
        'Bin Contacts': bin_data,
        'Annotation Contacts': annotation_data
    }

    bar_fig = create_bar_chart(data_dict)

    logger.info("Finished bin_visualization function")

    return cyto_elements, bar_fig

def contig_visualization(selected_annotation, selected_contig, contig_information, contig_dense_matrix, unique_annotations):

    # Find the index of the selected contig
    selected_contig_index = contig_information[contig_information['Contig'] == selected_contig].index[0]
    selected_annotation = contig_information.loc[selected_contig_index, 'Annotation']
    
    logger.info(f"Selected Contig Index: {selected_contig_index}")
    logger.info(f"Selected Annotation: {selected_annotation}")

    # Get all indices that have contact with the selected contig
    contacts_indices = contig_dense_matrix[selected_contig_index].nonzero()[0]
    contacts_indices = contacts_indices[contacts_indices != selected_contig_index]
    
    logger.info(f"Contacts Indices: {contacts_indices}")

    # If no contacts found, raise a warning
    if len(contacts_indices) == 0:
        logger.warning(f'No contacts found for the selected contig: {selected_contig}')
        contacts_annotation = pd.Series([selected_annotation])
        contacts_contigs = pd.Series([])
    else:
        contacts_annotation = contig_information.loc[contacts_indices, 'Annotation']
        contacts_contigs = contig_information.loc[contacts_indices, 'Contig']
    
    logger.info(f"Contacts Annotation: {contacts_annotation}")
    logger.info(f"Contacts Contigs: {contacts_contigs}")

    G = nx.Graph()

    # Use a categorical color scale
    color_scale_mapping = {
        'phage': colors.sequential.Reds,
        'plasmid': colors.sequential.Greens,
        'chromosome': colors.sequential.Blues
    }

    # Rank annotation based on the number of contigs with contact to the selected contig
    annotation_contact_counts = contacts_annotation.value_counts()
    annotation_contact_ranks = annotation_contact_counts.rank(method='first').astype(int)
    max_rank = annotation_contact_ranks.max()

    # Add annotation nodes
    for annotation in contacts_annotation.unique():
        annotation_type = contig_information.loc[contig_information['Annotation'] == annotation, 'Type'].values[0]
        color_scale = color_scale_mapping.get(annotation_type, [default_color])
        
        # Assign the gradient color based on rank
        annotation_rank = annotation_contact_ranks.get(annotation, int(max_rank/2))
        gradient_color = color_scale[int((annotation_rank / max_rank) * (len(color_scale) - 1))]

        G.add_node(annotation, size=1, color='#FFFFFF', border_color=gradient_color, border_width=2)

    # Ensure selected_annotation is added
    if selected_annotation not in G.nodes:
        annotation_type = contig_information.loc[contig_information['Annotation'] == selected_annotation, 'Type'].values[0]
        color_scale = color_scale_mapping.get(annotation_type, ['#00FF00'])
        gradient_color = color_scale[len(color_scale) - 1]

        G.add_node(selected_annotation, size=1, color='#FFFFFF', border_color=gradient_color, border_width=2)

    # Collect all contact values between selected_annotation and other annotations
    contact_values = []
    for annotation in contacts_annotation.unique():
        annotation_contigs = contig_information[contig_information['Annotation'] == annotation].index
        annotation_contigs = annotation_contigs[annotation_contigs < contig_dense_matrix.shape[1]]  # Filter indices
        contact_value = contig_dense_matrix[selected_contig_index, annotation_contigs].sum()

        if contact_value > 0:
            contact_values.append(contact_value)

    # Normalize contact values using generate_gradient_values
    scaled_weights = generate_gradient_values(contact_values, 1, 3) if contact_values else [1] * len(contacts_annotation.unique())

    # Add edges between selected_annotation and other annotations with scaled weights
    for i, annotation in enumerate(contacts_annotation.unique()):
        annotation_contigs = contig_information[contig_information['Annotation'] == annotation].index
        annotation_contigs = annotation_contigs[annotation_contigs < contig_dense_matrix.shape[1]]  # Filter indices
        contact_value = contig_dense_matrix[selected_contig_index, annotation_contigs].sum()
        
        if contact_value > 0:
            G.add_edge(selected_annotation, annotation, weight=scaled_weights[i])

    # Pin selected_contig to position (0, 0) during spring layout calculation
    fixed_positions = {selected_annotation: (0, 0)}

    # Calculate the layout using spring_layout, where weight affects node distance
    pos = nx.spring_layout(G, k=1, iterations=200, fixed=[selected_annotation], pos=fixed_positions, weight='weight')

    # Remove the edges after positioning
    G.remove_edges_from(list(G.edges()))

    # Handle selected_contig node
    G.add_node(selected_contig, size=5, color='white', border_color='black', border_width=2, parent=selected_annotation)
    pos[selected_contig] = (0, 0)

    # Add contig nodes to the graph
    for annotation in contacts_annotation.unique():
        annotation_contigs = contacts_contigs[contacts_annotation == annotation]
        contig_positions = arrange_nodes(annotation_contigs, [], distance=10, center_position=pos[annotation], selected_element=None)
        for contig, (x, y) in contig_positions.items():
            G.add_node(contig, 
                       size=3, 
                       color=G.nodes[annotation]['border_color'],  # Use the color from the annotation
                       parent=annotation)
            G.add_edge(selected_contig, contig)
            pos[contig] = (x, y)

    cyto_elements = nx_to_cyto_elements(G, pos)
    
    # Prepare data for bar chart
    contig_contact_values = contig_dense_matrix[selected_contig_index, contacts_indices]

    try:
        contig_data = pd.DataFrame({
            'name': contacts_contigs, 
            'value': contig_contact_values, 
            'color': [G.nodes[contig]['color'] for contig in contacts_contigs],
            'hover': [f"({contig_information.loc[contig_information['Contig'] == contig, 'Annotation'].values[0]}, {value})" for contig, value in zip(contacts_contigs, contig_contact_values)]
        })
    except Exception as e:
        logger.error(f"Error creating contig_data DataFrame: {e}")
        contig_data = pd.DataFrame()  # or assign None if appropriate for your app
    
    # Attempt to create annotation_data DataFrame
    try:
        annotation_data = pd.DataFrame({
            'name': contacts_annotation.unique(),
            'value': contact_values,
            'color': [G.nodes[annotation]['border_color'] for annotation in contacts_annotation.unique()]
        })
    except Exception as e:
        logger.error(f"Error creating annotation_data DataFrame: {e}")
        annotation_data = pd.DataFrame()  # or assign None if appropriate for your app


    data_dict = {
        'contig Contacts': contig_data,
        'Annotation Contacts': annotation_data
    }

    bar_fig = create_bar_chart(data_dict)

    logger.info("Finished contig_visualization function")

    return cyto_elements, bar_fig

def prepare_data(bin_information_intact, contig_information_intact, bin_dense_matrix, taxonomy_level = 'Family'):
    global bin_information
    global contig_information
    global unique_annotations
    global contact_matrix
    global contact_matrix_display
    
    if taxonomy_level is None:
        taxonomy_level = 'Family'
        
    taxonomy_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    bin_information = bin_information_intact.copy()
    bin_information['Annotation'] = bin_information[taxonomy_level]
    bin_information = bin_information.drop(columns=taxonomy_columns)
    
    contig_information = contig_information_intact.copy()
    contig_information['Annotation'] = contig_information[taxonomy_level]
    contig_information = contig_information.drop(columns=taxonomy_columns)
    
    unique_annotations = bin_information['Annotation'].unique()

    contact_matrix = pd.DataFrame(0.0, index=unique_annotations, columns=unique_annotations)
    bin_indexes_dict = get_indexes(unique_annotations, bin_information)

    # Use the pre-fetched indexes for calculating contacts
    for annotation_i in unique_annotations:
        for annotation_j in unique_annotations:   
            indexes_i = bin_indexes_dict[annotation_i]
            indexes_j = bin_indexes_dict[annotation_j]
            sub_matrix = bin_dense_matrix[np.ix_(indexes_i, indexes_j)]
            
            contact_matrix.at[annotation_i, annotation_j] = sub_matrix.sum()

    contact_matrix_display = contact_matrix.astype(int).copy()  # Convert to int for display
    contact_matrix_display.insert(0, 'Annotation', contact_matrix_display.index)  # Add the 'Annotation' column

    return bin_information, contig_information, unique_annotations, contact_matrix, contact_matrix_display

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

# Load the data
logger.info('Loading data')

bin_info_path = '../assets/examples/output/bin_info_final.csv'
bin_matrix_path= '../assets/examples/output/bin_contact_matrix.npz'

bin_information_intact = pd.read_csv(bin_info_path)
bin_matrix_data = np.load(bin_matrix_path)
data = bin_matrix_data['data']
row = bin_matrix_data['row']
col = bin_matrix_data['col']
shape = tuple(bin_matrix_data['shape'])
bin_sparse_matrix = coo_matrix((data, (row, col)), shape=shape)
bin_dense_matrix = bin_sparse_matrix.toarray()

contig_info_path = '../assets/examples/output/contig_info_final.csv'
contig_matrix_path= '../assets/examples/output/contig_contact_matrix.npz'

contig_information_intact = pd.read_csv(contig_info_path)
contig_matrix_data = np.load(contig_matrix_path)
data = contig_matrix_data['data']
row = contig_matrix_data['row']
col = contig_matrix_data['col']
shape = tuple(contig_matrix_data['shape'])
contig_sparse_matrix = coo_matrix((data, (row, col)), shape=shape)
contig_dense_matrix = contig_sparse_matrix.toarray()

logger.info('Data loading completed')  

bin_display_columns = ['Bin','Domain','Kingdom','Phylum','Class','Order','Family','Genus','Species','Restriction sites','Length','Coverage']
bin_information_display = bin_information_intact[bin_display_columns].copy()
bin_information_display.loc[:, 'Visibility'] = 1  # Default value to 1 (visible)

contig_display_columns = ['Contig','Domain','Kingdom','Phylum','Class','Order','Family','Genus','Species','Restriction sites','Length','Coverage']
contig_information_display = contig_information_intact[contig_display_columns].copy()
contig_information_display.loc[:, 'Visibility'] = 1  # Default value to 1 (visible)

bin_information, contig_information, unique_annotations, contact_matrix, contact_matrix_display = prepare_data(bin_information_intact, contig_information_intact, bin_dense_matrix)
  
type_colors = {
    'chromosome': '#4472C4',
    'phage': '#E83D20',
    'plasmid': '#70AD47'
}
default_color = '#808080' 

# Define the column definitions for AG Grid
bin_column_defs = [
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
            {"headerName": "Length", "field": "Length", "width": 140, "wrapHeaderText": True},
            {"headerName": "Coverage", "field": "Coverage", "width": 140, "wrapHeaderText": True},
            {"headerName": "Visibility", "field": "Visibility", "hide": True}
        ]
    }
]

contig_column_defs = [
    {
        "headerName": "Contig",
        "children": [
            {"headerName": "Contig", "field": "Contig", "pinned": 'left', "width": 120}
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
            {"headerName": "Length", "field": "Length", "width": 140, "wrapHeaderText": True},
            {"headerName": "Coverage", "field": "Coverage", "width": 140, "wrapHeaderText": True},
            {"headerName": "Visibility", "field": "Visibility", "hide": True}
        ]
    }
]

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
    'selected_bin': None,
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
app.layout = dcc.Loading(
    id="loading-spinner",
    type="default",
    fullscreen=True,
    delay_show=1000,
    children=[
        html.Div([
            dcc.Interval(
                id='interval-component',
                interval=1*1000,  # in milliseconds (every 1 second)
                n_intervals=0
            ),
            html.Div([
                html.Button("Download files", id="download-btn", style={**common_style}),
                html.Button("Reset", id="reset-btn", style={**common_style}),
                html.Button("Help", id="open-help", style={**common_style}),
                dcc.Download(id="download-dataframe-csv"),
                dcc.Dropdown(
                    id='visualization-selector',
                    options=[
                        {'label': 'Taxonomy Hierarchy', 'value': 'taxonomy_hierarchy'},
                        {'label': 'Annotation Visualization', 'value': 'basic'},
                        {'label': 'Bin Visualization', 'value': 'bin'},
                        {'label': 'Contig Visualization', 'value': 'contig'}
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
                    placeholder="Select an annotation",
                    style={}
                ),
                dcc.Dropdown(
                    id='bin-selector',
                    options=[],
                    value=None,
                    placeholder="Select a bin",
                    style={}
                ),
                dcc.Dropdown(
                    id='contig-selector',
                    options=[],
                    value=None,
                    placeholder="Select a contig",
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
                'background-color': 'lightgrey',
                'padding': '10px',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),
            html.Div(style={'height': '60px'}),  # Placeholder for header height
            html.Div([
                html.Div([
                    dcc.Graph(id='bar-chart', config={'displayModeBar': False}, figure=go.Figure(), style={'height': '40vh', 'width': '30vw', 'display': 'inline-block'}),
                    html.Div(id='row-count', style={'margin': '0px', 'height': '2vh', 'display': 'inline-block'}),
                    dcc.Checklist(
                        id='visibility-filter',
                        options=[{'label': '  Only show elements in the diagram', 'value': 'filter'}],
                        value=['filter'],
                        style={'display': 'inline-block', 'margin-right': '10px', 'float': 'right'}
                    ),
                    html.Div([
                        dcc.Tabs(id='table-tabs', value='bin', children=[
                            dcc.Tab(label='Bin Info', value='bin', className="p-0"),
                            dcc.Tab(label='Contig Info', value='contig', className="p-0")
                        ]),
                        html.Div(
                            id='table-container',
                            children=[
                                dag.AgGrid(
                                    id='bin-info-table',
                                    columnDefs=bin_column_defs,
                                    rowData=bin_information_display.to_dict('records'),
                                    defaultColDef={},
                                    style={'display': 'none'},
                                    dashGridOptions={
                                        'headerPinned': 'top',
                                        'rowSelection': 'single'
                                    }
                                ),
                                dag.AgGrid(
                                    id='contig-info-table',
                                    columnDefs=contig_column_defs,
                                    rowData=contig_information_display.to_dict('records'),
                                    defaultColDef={},
                                    style={'display': 'none'},
                                    dashGridOptions={
                                        'headerPinned': 'top',
                                        'rowSelection': 'single'
                                    }
                                )
                            ]
                        ),
                    ])
                ], style={'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    dcc.Graph(id='treemap-graph', figure=go.Figure(), config={'displayModeBar': False}, style={'height': '80vh', 'width': '48vw', 'display': 'inline-block'}),
                    cyto.Cytoscape(
                        id='cyto-graph',
                        elements=[],
                        stylesheet=base_stylesheet,
                        style={},
                        layout={'name': 'preset'},
                        zoom=1,
                        userZoomingEnabled=True,
                        wheelSensitivity=0.1
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
                                    'font-size': '10px',
                                    'background-color': 'white',
                                    'padding': '5px',
                                    'border': '1px solid #ccc',
                                    'margin-top': '3px',
                                    'overflowY': 'auto'}
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
                    style_header={'whiteSpace': 'normal', 'height': 'auto'},
                    fixed_rows={'headers': True},
                    fixed_columns={'headers': True, 'data': 1}
                )
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
            help_modal
        ], style={'height': '100vh', 'overflowY': 'auto', 'width': '100%'})
    ]
)

@app.callback(
    [Output('contact-table', 'columns'),
     Output('contact-table', 'data'),
     Output('contact-table', 'style_data_conditional'),
     Output('reset-btn', 'n_clicks')],
    [Input('taxonomy-level-selector', 'value')],
    [State('reset-btn', 'n_clicks')]
)
def update_data_and_trigger_reset(taxonomy_level, reset_clicks):
    # Update the data based on taxonomy level
    bin_information, contig_information, unique_annotations, contact_matrix, contact_matrix_display = prepare_data(bin_information_intact, contig_information_intact, bin_dense_matrix, taxonomy_level)    
    
    global current_visualization_mode
    
    # Check if the current visualization mode is not 'taxonomy_hierarchy'
    if current_visualization_mode['visualization_type'] != 'taxonomy_hierarchy':
        # Simulate the reset button click by incrementing its click count
        reset_clicks = (reset_clicks or 0) + 1
    
    # Update the visualization mode to 'taxonomy_hierarchy' based on the new input
    current_visualization_mode = {
        'visualization_type': 'taxonomy_hierarchy',
        'taxonomy_level': taxonomy_level,
        'selected_annotation': None,
        'selected_bin': None,
        'selected_contig': None
    }

    # Generate table columns based on the DataFrame's columns
    table_columns = [{"name": col, "id": col} for col in contact_matrix_display.columns]
    # Convert the DataFrame into a list of dictionaries (format required by Dash tables)
    table_data = contact_matrix_display.to_dict('records')
    # Generate the conditional styling based on the stored data
    style_conditions = styling_annotation_table(contact_matrix_display)

    return table_columns, table_data, style_conditions, reset_clicks

@app.callback(
    [Output('bin-info-table', 'rowData'),
     Output('bin-info-table', 'style'),
     Output('bin-info-table', 'filterModel'),
     Output('bin-info-table', 'defaultColDef'),
     Output('contig-info-table', 'rowData'),
     Output('contig-info-table', 'style'),
     Output('contig-info-table', 'filterModel'),
     Output('contig-info-table', 'defaultColDef'),
     Output('row-count', 'children')],
    [Input('table-tabs', 'value'),
     Input('bin-info-table', 'rowData'),
     Input('contig-info-table', 'rowData'),
     Input('annotation-selector', 'value'),
     Input('visibility-filter', 'value')],
    [State('cyto-graph', 'elements'),
     State('taxonomy-level-selector', 'value')]
)
def display_and_filter_table(selected_tab, bin_row_data, contig_row_data, selected_annotation, filter_value, cyto_elements, taxonomy_level):
    # Default styles to hide tables initially
    bin_style = {'display': 'none'}
    contig_style = {'display': 'none'}
    bin_filter_model = {}
    bin_col_def = {}
    contig_filter_model = {}
    contig_col_def = {}
    row_count_text = ""

    # Apply filtering logic for the selected table and annotation
    def apply_filter_logic(row_data):
        filter_model = {}
        # Set the default visibility to 1 for all rows initially
        for row in row_data:
            row['Visibility'] = 1

        # Apply annotation filter based on the selected annotation
        if selected_annotation:
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
            for row in row_data:
                if row[taxonomy_level] != selected_annotation:
                    row['Visibility'] = 2
        else:
            filter_model = {}

        # Set visibility based on the current visualization mode
        if selected_tab == 'bin':
            if current_visualization_mode['selected_bin']:
                selected_bin_index = bin_information[bin_information['Bin'] == current_visualization_mode['selected_bin']].index[0]
        
                connected_bins = set()
                for j in range(bin_dense_matrix.shape[0]):
                    if bin_dense_matrix[selected_bin_index, j] != 0:
                        connected_bins.add(bin_information.at[j, 'Bin'])
        
                for row in row_data:
                    if row['Bin'] not in connected_bins and row['Bin'] != current_visualization_mode['selected_bin']:
                        row['Visibility'] = 0
        elif selected_tab == 'contig':
            if current_visualization_mode['selected_contig']:
                selected_contig_index = contig_information[contig_information['Contig'] == current_visualization_mode['selected_contig']].index[0]
        
                connected_contigs = set()
                for j in range(contig_dense_matrix.shape[0]):
                    if contig_dense_matrix[selected_contig_index, j] != 0:
                        connected_contigs.add(contig_information.at[j, 'Contig'])
        
                for row in row_data:
                    if row['Contig'] not in connected_contigs and row['Contig'] != current_visualization_mode['selected_contig']:
                        row['Visibility'] = 0
                    
        # Apply visibility filter if checkbox is selected
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
        return filter_model

    # Update based on the selected tab and apply filters
    if selected_tab == 'bin':
        bin_style = {'display': 'block'}
        contig_style = {'display': 'none'}
        display_table = bin_information_display
        information_data = bin_information
        bin_colors, annotation_colors = get_node_colors(cyto_elements, information_data)
        style_conditions = styling_information_table(display_table, information_data, bin_colors, annotation_colors, unique_annotations, selected_tab)
        bin_col_def = {
            "sortable": True,
            "filter": True,
            "resizable": True,
            "cellStyle": {
                "styleConditions": style_conditions
            }
        }
        bin_filter_model = apply_filter_logic(bin_row_data)
        contig_filter_model = {}
        
    elif selected_tab == 'contig':
        bin_style = {'display': 'none'}
        contig_style = {'display': 'block'}
        display_table = contig_information_display
        information_data = contig_information
        contig_colors, annotation_colors = get_node_colors(cyto_elements, information_data)
        style_conditions = styling_information_table(display_table, information_data, contig_colors, annotation_colors, unique_annotations, selected_tab)
        contig_col_def = {
            "sortable": True,
            "filter": True,
            "resizable": True,
            "cellStyle": {
                "styleConditions": style_conditions
            }
        }
        bin_filter_model = {}
        contig_filter_model = apply_filter_logic(contig_row_data)
    
    row_count_text = f"Total Number of Rows: {sum(1 for row in contig_row_data if row['Visibility'] == 1)}"

    return (
        bin_row_data, bin_style, bin_filter_model, bin_col_def,
        contig_row_data, contig_style, contig_filter_model, contig_col_def,
        row_count_text
    )

@app.callback(
    [Output('visualization-selector', 'value'),
     Output('annotation-selector', 'value'),
     Output('annotation-selector', 'style'),
     Output('bin-selector', 'value'),
     Output('bin-selector', 'style'),
     Output('contig-selector', 'value'),
     Output('contig-selector', 'style'),
     Output('table-tabs', 'value'),  # Update the active tab based on visualization type
     Output('contact-table', 'active_cell'),
     Output('bin-info-table', 'selectedRows'),
     Output('contig-info-table', 'selectedRows')],
    [Input('reset-btn', 'n_clicks'),
     Input('visualization-selector', 'value'),
     Input('contact-table', 'active_cell'),
     Input('bin-info-table', 'selectedRows'),
     Input('contig-info-table', 'selectedRows'),
     Input('cyto-graph', 'selectedNodeData'),
     Input('table-tabs', 'value')],
    [State('taxonomy-level-selector', 'value')],  # Pass the active tab as state
    prevent_initial_call=True
)
def sync_selectors(reset_clicks, visualization_type, contact_table_active_cell, bin_info_selected_rows, contig_info_selected_rows, selected_node_data, current_tab, taxonomy_level):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    selected_annotation, selected_bin, selected_contig = synchronize_selections(
        triggered_id, selected_node_data, bin_info_selected_rows, contig_info_selected_rows, contact_table_active_cell, taxonomy_level, current_tab
    )

    # Initialize the styles as hidden
    annotation_selector_style = {'display': 'none'}
    bin_selector_style = {'display': 'none'}
    contig_selector_style = {'display': 'none'}
    tab_value = current_tab  # Default to the current tab

    # Reset selections if reset button is clicked
    if triggered_id == 'reset-btn':
        visualization_type = 'taxonomy_hierarchy'
        selected_annotation = None
        selected_bin = None
        selected_contig = None
        
    if triggered_id == 'table-tabs':
        visualization_type = current_tab
            
    # Update styles, tab, and visualization type based on selections
    if selected_bin or visualization_type == 'bin':
        visualization_type = 'bin'
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
        bin_selector_style = {'width': '300px', 'display': 'inline-block'}
        tab_value = 'bin'  # Switch to bin tab

    elif selected_contig or visualization_type == 'contig':
        visualization_type = 'contig'
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
        contig_selector_style = {'width': '300px', 'display': 'inline-block'}
        tab_value = 'contig'  # Switch to contig tab

    elif selected_annotation and visualization_type == 'basic':
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}

    return (visualization_type, selected_annotation, annotation_selector_style,
            selected_bin, bin_selector_style, selected_contig, contig_selector_style,
            tab_value, None, [], [])

def synchronize_selections(triggered_id, selected_node_data, bin_info_selected_rows, contig_info_selected_rows, contact_table_active_cell, taxonomy_level, table_tab_value):
    selected_annotation = None
    selected_bin = None
    selected_contig = None

    # Node selected in the network
    if triggered_id == 'cyto-graph' and selected_node_data:
        selected_node_id = selected_node_data[0]['id']
        
        # Check if the node exists in both bin and contig information
        is_in_bin = selected_node_id in bin_information['Bin'].values
        is_in_contig = selected_node_id in contig_information['Contig'].values

        # Decide based on the active tab
        if is_in_bin and is_in_contig:
            if table_tab_value == 'bin':
                # Use bin info if the bin tab is active
                bin_info = bin_information[bin_information['Bin'] == selected_node_id].iloc[0]
                selected_annotation = bin_info['Annotation']
                selected_bin = bin_info['Bin']
            elif table_tab_value == 'contig':
                # Use contig info if the contig tab is active
                contig_info = contig_information[contig_information['Contig'] == selected_node_id].iloc[0]
                selected_annotation = contig_info['Annotation']
                selected_contig = contig_info['Contig']
        elif is_in_bin:
            # If only in bin, select bin info
            bin_info = bin_information[bin_information['Bin'] == selected_node_id].iloc[0]
            selected_annotation = bin_info['Annotation']
            selected_bin = bin_info['Bin']
        elif is_in_contig:
            # If only in contig, select contig info
            contig_info = contig_information[contig_information['Contig'] == selected_node_id].iloc[0]
            selected_annotation = contig_info['Annotation']
            selected_contig = contig_info['Contig']
        else:
            # Default to treating it as an annotation
            selected_annotation = selected_node_id

    # Row selected in bin-info-table
    elif triggered_id == 'bin-info-table' and bin_info_selected_rows:
        selected_row = bin_info_selected_rows[0]
        if taxonomy_level in selected_row and 'Bin' in selected_row:
            selected_annotation = selected_row[taxonomy_level]
            selected_bin = selected_row['Bin']

    # Row selected in contig-info-table
    elif triggered_id == 'contig-info-table' and contig_info_selected_rows:
        selected_row = contig_info_selected_rows[0]
        if taxonomy_level in selected_row and 'Contig' in selected_row:
            selected_annotation = selected_row[taxonomy_level]
            selected_contig = selected_row['Contig']

    # Cell selected in contact-table
    elif triggered_id == 'contact-table' and contact_table_active_cell:
        row_id = contact_table_active_cell['row']
        row_annotation = contact_matrix_display.iloc[row_id]['Annotation']
        selected_annotation = row_annotation

    logger.info(f'Current selected: Annotation={selected_annotation}, Bin={selected_bin}, Contig={selected_contig}')

    return selected_annotation, selected_bin, selected_contig

# Callback to update the visualizationI want to 
@app.callback(
    [Output('cyto-graph', 'elements'),
     Output('cyto-graph', 'style'),
     Output('bar-chart', 'figure'),
     Output('treemap-graph', 'figure'),
     Output('treemap-graph', 'style')],
    [Input('reset-btn', 'n_clicks'),
     Input('confirm-btn', 'n_clicks')],
    [State('visualization-selector', 'value'),
     State('annotation-selector', 'value'),
     State('bin-selector', 'value'),
     State('contig-selector', 'value'),
     State('contact-table', 'data'),
     State('table-tabs', 'value')],  # Get the selected tab to determine bin or contig view
    prevent_initial_call=True
)
def update_visualization(reset_clicks, confirm_clicks, visualization_type, selected_annotation, selected_bin, selected_contig, table_data, selected_tab):
    logger.info('update_visualization triggered')
    logger.info(f'Visualization type: {visualization_type}')
    logger.info(f'Selected annotation: {selected_annotation}')
    logger.info(f'Selected bin: {selected_bin}')
    logger.info(f'Selected contig: {selected_contig}')
    
    global current_visualization_mode
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'reset-btn':
        # Reset to show the original plot
        logger.info('reset to show taxonomy_hierarchy visualization')
        current_visualization_mode = {
            'visualization_type': 'taxonomy_hierarchy',
            'selected_annotation': None,
            'selected_bin': None,
            'selected_contig': None
        }
        treemap_fig, bar_fig = taxonomy_visualization()
        treemap_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}
        cyto_elements = []
        cyto_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}

    elif triggered_id == 'confirm-btn':
        # Update the current visualization mode with selected values
        current_visualization_mode['visualization_type'] = visualization_type
        current_visualization_mode['selected_annotation'] = selected_annotation
        current_visualization_mode['selected_bin'] = selected_bin
        current_visualization_mode['selected_contig'] = selected_contig

        if visualization_type == 'taxonomy_hierarchy':
            logger.info('show taxonomy_hierarchy visualization')
            treemap_fig, bar_fig = taxonomy_visualization()
            treemap_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}
            cyto_elements = []
            cyto_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            
        elif visualization_type == 'basic' or (selected_annotation and not selected_bin and not selected_contig):
            logger.info('show basic annotation visualization')
            cyto_elements, bar_fig, _ = annotation_visualization(bin_information, unique_annotations, contact_matrix)
            treemap_fig = go.Figure()
            treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            cyto_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}

        elif visualization_type == 'bin' and selected_bin:
            logger.info('show bin visualization')
            cyto_elements, bar_fig = bin_visualization(selected_annotation, selected_bin, bin_information, bin_dense_matrix, unique_annotations)
            treemap_fig = go.Figure()
            treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            cyto_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}

        elif visualization_type == 'contig' and selected_contig:
            logger.info('show contig visualization')
            cyto_elements, bar_fig = contig_visualization(selected_annotation, selected_contig, contig_information, contig_dense_matrix, unique_annotations)
            treemap_fig = go.Figure()
            treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            cyto_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}

    # Choose the appropriate information dataset based on selected_tab

    
    # Return the appropriate column definition based on the selected tab
    if selected_tab == 'bin':
        return cyto_elements, cyto_style, bar_fig, treemap_fig, treemap_style
    elif selected_tab == 'contig':
        return cyto_elements, cyto_style, bar_fig, treemap_fig, treemap_style
    
@app.callback(
    [Output('cyto-graph', 'elements', allow_duplicate=True),
     Output('cyto-graph', 'stylesheet'),
     Output('hover-info', 'children'),
     Output('cyto-graph', 'layout')],
    [Input('annotation-selector', 'value'),
     Input('bin-selector', 'value'),
     Input('contig-selector', 'value')],
    prevent_initial_call=True
)
def update_selected_styles(selected_annotation, selected_bin, selected_contig):
    selected_nodes = []
    selected_edges = []
    hover_info = "No selection"
    cyto_elements = dash.no_update  # Default to no update for elements
    cyto_style = dash.no_update  # Default to no update for layout style

    # Determine which selection to use based on the active tab
    if  selected_bin:
        selected_nodes.append(selected_bin)
        bin_info = bin_information[bin_information['Bin'] == selected_bin].iloc[0]
        hover_info = f"Bin: {selected_bin}<br>Annotation: {bin_info['Annotation']}"
        # No need to update cyto_elements or layout if a bin is selected

    elif selected_contig:
        selected_nodes.append(selected_contig)
        contig_info = contig_information[contig_information['Contig'] == selected_contig].iloc[0]
        hover_info = f"Contig: {selected_contig}<br>Annotation: {contig_info['Annotation']}"
        # No need to update cyto_elements or layout if a contig is selected

    if current_visualization_mode['visualization_type'] == 'basic':
        selected_nodes.append(selected_annotation)
        hover_info = f"Annotation: {selected_annotation}"

        # Only call annotation_visualization in 'basic' mode
        if selected_annotation:
            # Show edges connected to the selected node using the contact matrix
            annotation_index = unique_annotations.tolist().index(selected_annotation)
            for i, contact_value in enumerate(contact_matrix.iloc[annotation_index]):
                if contact_value > 0:
                    connected_annotation = unique_annotations[i]
                    selected_edges.append((selected_annotation, connected_annotation))

            # Generate elements and layout with animation effect
            cyto_elements, _, cyto_style = annotation_visualization(
                bin_information, unique_annotations, contact_matrix, selected_node=selected_annotation
            )
        else:
            cyto_elements, _, cyto_style = annotation_visualization(
                bin_information, unique_annotations, contact_matrix
            )

    # Add selection styles for the selected nodes and edges
    stylesheet = add_selection_styles(selected_nodes, selected_edges)

    return cyto_elements, stylesheet, hover_info, cyto_style

@app.callback(
    [Output('annotation-selector', 'options'),
     Output('bin-selector', 'options'),
     Output('contig-selector', 'options')],
    [Input('annotation-selector', 'value')],
    [State('visualization-selector', 'value'),
     State('table-tabs', 'value')]  # Use selected tab to determine bin or contig
)
def update_dropdowns(selected_annotation, visualization_type, selected_tab):
    # Initialize empty lists for dropdown options
    annotation_options = []
    bin_options = []
    contig_options = []

    # Determine which dataset to use based on selected_tab
    if selected_tab == 'bin':
        # Populate annotation options for bin information
        annotation_options = [{'label': annotation, 'value': annotation} for annotation in bin_information['Annotation'].unique()]

        # Populate bin options if a specific annotation is selected
        if visualization_type == 'bin' and selected_annotation:
            bin_index_dict = get_indexes(selected_annotation, bin_information)
            bin_indexes = bin_index_dict[selected_annotation]
            bins = bin_information.loc[bin_indexes, 'Bin']
            bin_options = [{'label': bin, 'value': bin} for bin in bins]

    elif selected_tab == 'contig':
        # Populate annotation options for contig information
        annotation_options = [{'label': annotation, 'value': annotation} for annotation in contig_information['Annotation'].unique()]

        # Populate contig options if a specific annotation is selected
        if visualization_type == 'contig' and selected_annotation:
            contig_index_dict = get_indexes(selected_annotation, contig_information)
            contig_indexes = contig_index_dict[selected_annotation]
            contigs = contig_information.loc[contig_indexes, 'Contig']
            contig_options = [{'label': contig, 'value': contig} for contig in contigs]

    return annotation_options, bin_options, contig_options

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