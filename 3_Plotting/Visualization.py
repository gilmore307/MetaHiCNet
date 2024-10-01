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


# Function to get contig indexes based on annotation in a specific part of the dataframe
def get_contig_indexes(annotations, contig_information):
    # Number of threads to use: 4 * CPU core count
    num_threads = 4 * os.cpu_count()
    
    # Ensure annotations is a list even if a single annotation is provided
    if isinstance(annotations, str):
        annotations = [annotations]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for annotation in annotations:
            futures[executor.submit(
                lambda ann: (ann, contig_information[contig_information['Contig annotation'] == ann].index),
                annotation)] = annotation
        
        contig_indexes = {}
        for future in futures:
            annotation = futures[future]
            try:
                annotation, indexes = future.result()
                contig_indexes[annotation] = indexes
            except Exception as e:
                logger.error(f'Error fetching contig indexes for annotation: {annotation}, error: {e}')
        
    # If only one annotation was given as input, return its indexes directly
    if len(contig_indexes) == 1:
        return list(contig_indexes.values())[0]

    return contig_indexes

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

# Function to create bar chart
def create_bar_chart(data_dict):
    logger.info("Starting to create bar chart with data_dict")
    traces = []

    for idx, (trace_name, data_frame) in enumerate(data_dict.items()):
        logger.info(f"Creating bar trace for {trace_name}")
        bar_data = data_frame.sort_values(by='value', ascending=False)
        bar_colors = bar_data['color']
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

# Function to style contig info table
def styling_contig_table(contig_colors, annotation_colors, unique_annotations):
    taxonomy_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    columns = ['Restriction sites', 'Contig length', 'Contig coverage', 'Intra-contig contact']
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

    # Add style conditions for the "Contig" column
    for contig in contig_information_display['Contig']:
        contig_color = contig_colors.get(contig, annotation_colors.get(contig_information.loc[contig_information['Contig name'] == contig, 'Contig annotation'].values[0], '#FFFFFF'))
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

# Function to get contig colors from Cytoscape elements or use annotation color if not found
def get_contig_and_annotation_colors(cyto_elements, contig_information):
    contig_colors = {}
    annotation_colors = {}

    # Extract colors from Cytoscape elements
    if cyto_elements:
        for element in cyto_elements:
            if 'data' in element and 'color' in element['data'] and 'id' in element['data']:
                contig_colors[element['data']['id']] = element['data']['color']

    # Get annotation colors based on viral status
    for annotation in contig_information['Contig annotation'].unique():
        contig_type = contig_information[contig_information['Contig annotation'] == annotation]['type'].values[0]
        annotation_colors[annotation] = type_colors.get(contig_type, default_color)
        
    return contig_colors, annotation_colors

# Function to arrange contigs
def arrange_contigs(contigs, inter_contig_edges, distance, selected_contig=None, center_position=(0, 0)):
    distance /= 100 
    phi = (1 + sqrt(5)) / 2  # golden ratio

    # Identify contigs that connect to other annotation
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
        "Bin": contig_information_intact['Bin'].unique()  # Add Bin attribute as an empty list for Community
    })
    existing_annotations.add("Community")
    
    for _, row in host_data.iterrows():
        for level, level_num in level_mapping.items():
            if level == 'Community':
                continue
                
            annotation = f"{level}_{row[level]}"
            parent = "Community" if level_num == 8 else f"{list(level_mapping.keys())[list(level_mapping.values()).index(level_num+1)]}_{row[list(level_mapping.keys())[list(level_mapping.values()).index(level_num+1)]]}"
            
            if annotation not in existing_annotations:
                records.append({
                    "annotation": annotation,
                    "parent": parent,
                    "level": level_num,
                    "level_name": level.capitalize(),
                    "type": row["type"],
                    "total coverage": row['Contig coverage'],
                    "border_color": type_colors.get(row["type"], "gray"),
                    "Bin": [row["Bin"]]
                })
                existing_annotations.add(annotation)
            else:
                for rec in records:
                    if rec['annotation'] == annotation:
                        rec['total coverage'] += row['Contig coverage']
                        rec['Bin'].append(row['Bin']) 
                        break
                   
    for _, row in virus_data.iterrows():
        for level, level_num in level_mapping.items():
            if level not in ['Class', 'Order', 'Family']:
                continue
                
            annotation = 'Virus' if level_num == 5 else f"{level}_{row[level]}"
            parent = (
                "Community" if level_num == 5
                else 'Virus' if level_num == 4
                else f"{list(level_mapping.keys())[list(level_mapping.values()).index(level_num+1)]}_{row[list(level_mapping.keys())[list(level_mapping.values()).index(level_num+1)]]}"
            )
            
            if annotation not in existing_annotations:
                records.append({
                    "annotation": annotation,
                    "parent": parent,
                    "level": level_num,
                    "level_name": level.capitalize(),
                    "type": row["type"],
                    "total coverage": row['Contig coverage'],
                    "border_color": type_colors.get(row["type"], "gray"),
                    "Bin": [row["Bin"]]
                })
                existing_annotations.add(annotation)
            else:
                for rec in records:
                    if rec['annotation'] == annotation:
                        rec['total coverage'] += row['Contig coverage']
                        rec['Bin'].append(row['Bin'])
                        break

    hierarchy_df = pd.DataFrame(records)
    hierarchy_df['scaled_coverage'] = generate_gradient_values(hierarchy_df['total coverage'], 10, 30)
    hierarchy_df.loc[hierarchy_df['type'] == 'phage', 'scaled_coverage'] *= 20

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
        customdata=hierarchy_df[['level_name', 'Bin']],  # Add Bin to the hover data
        hovertemplate='<b>%{label}</b><br>Level: %{customdata[0]}<br>Coverage: %{value}<br>Bins: %{customdata[1]}'
    )

    fig.update_layout(
        coloraxis_showscale=False,
        font_size=20,
        autosize=True,
        margin=dict(t=30, b=0, l=0, r=0)
    )

    logger.info('Taxonomy visualization creation completed')
    
    # Prepare data for bar chart with 3 traces
    total_contig_coverage = contig_information.groupby('Contig annotation')['Contig coverage'].sum().reindex(unique_annotations)
    inter_annotation_contact_sum = contact_matrix.sum(axis=1) - np.diag(contact_matrix.values)
    total_contig_coverage_sum = total_contig_coverage.values
    contig_counts = contig_information['Contig annotation'].value_counts()
    
    node_colors = {}
    for annotation in total_contig_coverage.index:
        contig_type = contig_information.loc[
            contig_information['Contig annotation'] == annotation, 'type'
        ].values[0]
        color = type_colors.get(contig_type, default_color)
        node_colors[annotation] = color
        

    data_dict = {
        'Total Inter-Annotation Contact': pd.DataFrame({'name': unique_annotations, 'value': inter_annotation_contact_sum, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]}),
        'Total Coverage': pd.DataFrame({'name': unique_annotations, 'value': total_contig_coverage_sum, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]}),
        'Contig Number': pd.DataFrame({'name': unique_annotations, 'value': contig_counts, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]})
    }

    bar_fig = create_bar_chart(data_dict)
    
    return fig, bar_fig

# Function to visualize annotation relationship
def basic_visualization(contig_information, unique_annotations, contact_matrix):
    G = nx.Graph()

    # Add nodes with size based on total contig coverage
    total_contig_coverage = contig_information.groupby('Contig annotation')['Contig coverage'].sum().reindex(unique_annotations)
    node_sizes = generate_gradient_values(total_contig_coverage.values, 10, 30)  # Example range from 10 to 30

    node_colors = {}
    for annotation, size in zip(total_contig_coverage.index, node_sizes):
        contig_type = contig_information.loc[
            contig_information['Contig annotation'] == annotation, 'type'
        ].values[0]
        color = type_colors.get(contig_type, default_color)
        node_colors[annotation] = color
        G.add_node(annotation, size=size, color=color, parent=None)

    # Add edges with weight based on inter-annotation contacts
    non_zero_count = np.count_nonzero(np.triu(contact_matrix.values, k=1))  # Count non-zero values in the upper triangle

    inter_annotation_contacts = np.empty(non_zero_count)  # Preallocate the array

    index = 0  # Track the position to insert
    for i, annotation_i in enumerate(unique_annotations):
        for j, annotation_j in enumerate(unique_annotations[i + 1:], start=i + 1):
            weight = contact_matrix.at[annotation_i, annotation_j]
            if weight > 0:
                G.add_edge(annotation_i, annotation_j, weight=weight)
                inter_annotation_contacts[index] = weight
                index += 1


    # Generate gradient values for the edge weights
    edge_weights = generate_gradient_values(inter_annotation_contacts, 10, 300) 

    # Assign the gradient values as edge weights and set default edge color
    for (u, v, d), weight in zip(G.edges(data=True), edge_weights):
        d['weight'] = weight

    # Initial node positions using a force-directed layout with increased dispersion
    pos = nx.spring_layout(G, dim=2, k=10, iterations=50, weight='weight')

    cyto_elements = nx_to_cyto_elements(G, pos)

    # Prepare data for bar chart with 3 traces
    inter_annotation_contact_sum = contact_matrix.sum(axis=1) - np.diag(contact_matrix.values)
    total_contig_coverage_sum = total_contig_coverage.values
    contig_counts = contig_information['Contig annotation'].value_counts()

    data_dict = {
        'Total Inter-Annotation Contact': pd.DataFrame({'name': unique_annotations, 'value': inter_annotation_contact_sum, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]}),
        'Total Coverage': pd.DataFrame({'name': unique_annotations, 'value': total_contig_coverage_sum, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]}),
        'Contig Number': pd.DataFrame({'name': unique_annotations, 'value': contig_counts, 'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]})
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

# Function to visualize intra-annotation relationships
def intra_annotation_visualization(selected_annotation, contig_information, unique_annotations, contact_matrix):
    G = nx.Graph()

    # Add nodes with size based on contig counts
    contig_counts = [len(contig_information[contig_information['Contig annotation'] == node]) for node in unique_annotations]
    node_sizes = generate_gradient_values(np.array(contig_counts), 10, 30)
    indices = get_contig_indexes(selected_annotation, contig_information)

    nodes_to_remove = []
    for annotation, size in zip(unique_annotations, node_sizes):
        contig_type = contig_information.loc[
            contig_information['Contig annotation'] == annotation, 'type'
        ].values[0]
        
        color = type_colors.get(contig_type, default_color)

        if annotation == selected_annotation:
            G.add_node(annotation, size=size, color='#FFFFFF', border_color='#000', border_width=2, parent=None)  # White for selected node
        else:
            num_connected_contigs = len(contig_information[(contig_information['Contig annotation'] == annotation) & (dense_matrix[:, indices].sum(axis=1) > 0)])
            if num_connected_contigs == 0:
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

    # Calculate k_value based on the number of contigs of the selected annotation
    num_contigs = len(indices)
    k_value = sqrt(num_contigs)

    new_pos = nx.spring_layout(G, pos={selected_annotation: (0, 0)}, fixed=[selected_annotation], k=k_value, iterations=50, weight='weight')

    # Get and arrange contigs within the selected annotation node
    contigs = contig_information.loc[indices, 'Contig name']
    inter_contig_edges = set()

    for i in indices:
        for j in range(dense_matrix.shape[0]):
            if dense_matrix[i, j] != 0 and contig_information.at[j, 'Contig annotation'] != selected_annotation:
                inter_contig_edges.add(contig_information.at[i, 'Contig name'])
                inter_contig_edges.add(contig_information.at[j, 'Contig name'])

    contig_positions = arrange_contigs(contigs, inter_contig_edges, distance=1, center_position=new_pos[selected_annotation])

    # Add contig nodes and edges to the graph G
    for contig, (x, y) in contig_positions.items():
        G.add_node(contig, size=1, color='#7030A0' if contig in inter_contig_edges else '#00B050', parent=selected_annotation)
        new_pos[contig] = (new_pos[selected_annotation][0] + x, new_pos[selected_annotation][1] + y)

    cyto_elements = nx_to_cyto_elements(G, new_pos)

    # Prepare data for bar chart
    contig_contact_counts = contig_information[contig_information['Contig annotation'] != selected_annotation]['Contig annotation'].value_counts()
    inter_annotation_contacts = contact_matrix.loc[selected_annotation].drop(selected_annotation)

    # Filter out contigs that are not in the graph
    filtered_contig_counts = contig_contact_counts[contig_contact_counts.index.isin(G.nodes)]
    filtered_inter_annotation_contacts = inter_annotation_contacts[inter_annotation_contacts.index.isin(G.nodes)]

    data_dict = {
        'Contig Number': pd.DataFrame({'name': filtered_contig_counts.index, 'value': filtered_contig_counts.values, 'color': [G.nodes[annotation]['color'] for annotation in filtered_contig_counts.index]}),
        'Inter-Annotation Contacts': pd.DataFrame({'name': filtered_inter_annotation_contacts.index, 'value': filtered_inter_annotation_contacts.values, 'color': [G.nodes[annotation]['color'] for annotation in filtered_inter_annotation_contacts.index]})
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

# Function to visualize inter-annotation relationships
def inter_annotation_visualization(selected_annotation, secondary_annotation, contig_information):

    row_contig = selected_annotation
    col_contig = secondary_annotation

    G = nx.Graph()
    G.add_node(row_contig, color='#FFFFFF', border_color='black', border_width=2, label=row_contig)
    G.add_node(col_contig, color='#FFFFFF', border_color='black', border_width=2, label=col_contig)

    new_pos = {row_contig: (-0.2, 0), col_contig: (0.2, 0)}

    row_indices = get_contig_indexes(row_contig, contig_information)
    col_indices = get_contig_indexes(col_contig, contig_information)
    inter_contigs_row = set()
    inter_contigs_col = set()

    interannotation_contacts = []
    contig_contact_counts = []
    inter_contig_contacts = []

    for i in row_indices:
        for j in col_indices:
            contact_value = dense_matrix[i, j]
            if contact_value != 0:
                inter_contigs_row.add(contig_information.at[i, 'Contig name'])
                inter_contigs_col.add(contig_information.at[j, 'Contig name'])
                interannotation_contacts.append({
                    'name': f"{contig_information.at[i, 'Contig name']} - {contig_information.at[j, 'Contig name']}",
                    'value': contact_value,
                    'color': 'green'  # Set green color for the bars
                })
                contig_contact_counts.append({
                    'name': contig_information.at[i, 'Contig name'],
                    'annotation': selected_annotation,
                    'count': 1,
                    'color': '#C00000'  # Set red color for the bars
                })
                contig_contact_counts.append({
                    'name': contig_information.at[j, 'Contig name'],
                    'annotation': secondary_annotation,
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
    interannotation_contacts_df = pd.DataFrame(interannotation_contacts)

    contig_contact_counts_df = pd.DataFrame(contig_contact_counts)
    contig_contact_counts_summary = contig_contact_counts_df.groupby(['name', 'color']).size().reset_index(name='value')

    inter_contig_contacts_df = pd.DataFrame(inter_contig_contacts)
    inter_contig_contacts_summary = inter_contig_contacts_df.groupby(['name', 'color']).sum().reset_index()

    data_dict = {
        'Inter Contig Contacts': interannotation_contacts_df,
        'Contig Contacts Counts': contig_contact_counts_summary,
        'Contig Contacts Value': inter_contig_contacts_summary
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

# Function to visualize contig relationships
def contig_visualization(selected_annotation, selected_contig, contig_information, unique_annotations):

    # Find the index of the selected contig
    selected_contig_index = contig_information[contig_information['Contig name'] == selected_contig].index[0]
    selected_annotation = contig_information.loc[selected_contig_index, 'Contig annotation']

    # Get all indices that have contact with the selected contig
    contacts_indices = dense_matrix[selected_contig_index].nonzero()[0]
    
    # Remove self-contact
    contacts_indices = contacts_indices[contacts_indices != selected_contig_index]
    
    contacts_annotation = contig_information.loc[contacts_indices, 'Contig annotation']
    contacts_contigs = contig_information.loc[contacts_indices, 'Contig name']
    
    G = nx.Graph()

    # Use a categorical color scale
    color_scale = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

    # Rank annotation based on the number of contigs with contact to the selected contig
    annotation_contact_counts = contacts_annotation.value_counts()
    annotation_contact_ranks = annotation_contact_counts.rank(method='first').astype(int)
    max_rank = annotation_contact_ranks.max()
    
    # Fetch contig indexes for all unique annotations at once
    unique_annotations = contacts_annotation.unique().tolist()
    annotation_indexes_dict = get_contig_indexes(unique_annotations, contig_information)

    # Add annotation nodes and their positions
    for annotation in contacts_annotation.unique():
        annotation_rank = annotation_contact_ranks[annotation]
        gradient_color = color_scale[int((annotation_rank / max_rank) * (len(color_scale) - 1))]
        G.add_node(annotation, size=1, color='#FFFFFF', border_color=gradient_color, border_width=2)  # White color for nodes, gradient color for border


    # Set k value to avoid overlap and generate positions for the graph nodes
    k_value = sqrt(len(G.nodes))
    pos = nx.spring_layout(G, k=k_value, iterations=50, weight='weight')

    # Add contig nodes to the graph
    for annotation in contacts_annotation.unique():
        annotation_contigs = contacts_contigs[contacts_annotation == annotation]
        contig_positions = arrange_contigs(annotation_contigs, [], distance=2, center_position=pos[annotation],selected_contig=selected_contig if annotation == selected_annotation else None)
        for contig, (x, y) in contig_positions.items():
            G.add_node(contig, size=1 if contig != selected_contig else 5, color='black' if contig == selected_contig else G.nodes[annotation]['border_color'], parent=annotation)  # Same color as annotation, black for selected contig
            if contig != selected_contig:
                G.add_edge(selected_contig, contig, weight=dense_matrix[selected_contig_index, contig_information[contig_information['Contig name'] == contig].index[0]])
            pos[contig] = (x, y)  # Use positions directly from arrange_contigs

    # Ensure the selected contig node is positioned above all other contigs
    pos[selected_contig] = pos[selected_annotation]

    cyto_elements = nx_to_cyto_elements(G, pos)
    
    # Prepare data for bar chart
    contig_contact_values = dense_matrix[selected_contig_index, contacts_indices]
    contig_data = pd.DataFrame({'name': contacts_contigs, 'value': contig_contact_values, 'color': [G.nodes[contig]['color'] for contig in contacts_contigs]})

    annotation_contact_values = []
    contig_contact_counts_per_annotation = [] 
    for annotation in unique_annotations:
        annotation_indexes = annotation_indexes_dict[annotation]  # Use the pre-fetched indexes
        contact_value = dense_matrix[selected_contig_index, annotation_indexes].sum()
        annotation_contact_values.append(contact_value)
        contig_contact_counts_per_annotation.append(len(annotation_indexes))

    annotation_data = pd.DataFrame({'name': contacts_annotation.unique(), 'value': annotation_contact_values, 'color': [G.nodes[annotation]['color'] for annotation in contacts_annotation.unique()]})
    contig_contact_counts_data = pd.DataFrame({'name': contacts_annotation.unique(), 'value': contig_contact_counts_per_annotation, 'color': [G.nodes[annotation]['color'] for annotation in contacts_annotation.unique()]})

    data_dict = {
        'Contig Contacts': contig_data,
        'Annotation Contacts': annotation_data,
        'Contig Contact Counts': contig_contact_counts_data  # New trace
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

def prepare_data(contig_information_intact, dense_matrix, taxonomy_level = 'Family'):
    global contig_information
    global unique_annotations
    global contact_matrix
    global contact_matrix_display
    
    if taxonomy_level is None:
        taxonomy_level = 'Family'

    contig_information = contig_information_intact.copy()
    contig_information['Contig annotation'] = contig_information[taxonomy_level]
    taxonomy_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    contig_information = contig_information.drop(columns=taxonomy_columns)
    
    unique_annotations = contig_information['Contig annotation'].unique()

    contact_matrix = pd.DataFrame(0.0, index=unique_annotations, columns=unique_annotations)
    contig_indexes_dict = get_contig_indexes(unique_annotations, contig_information)

    # Use the pre-fetched indexes for calculating contacts
    for annotation_i in unique_annotations:
        for annotation_j in unique_annotations:   
            indexes_i = contig_indexes_dict[annotation_i]
            indexes_j = contig_indexes_dict[annotation_j]
            sub_matrix = dense_matrix[np.ix_(indexes_i, indexes_j)]
            
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
contig_info_path = '../1_Data_Processing/output/contig_info_final.csv'
raw_contact_matrix_path= '../1_Data_Processing/output/contact_matrix_final.npz'

# Load the data
logger.info('Loading data')
contig_information_intact = pd.read_csv(contig_info_path)
contact_matrix_data = np.load(raw_contact_matrix_path)
data = contact_matrix_data['data']
indices = contact_matrix_data['indices']
indptr = contact_matrix_data['indptr']
shape = contact_matrix_data['shape']
sparse_matrix = csc_matrix((data, indices, indptr), shape=shape)
dense_matrix = sparse_matrix.toarray()

if dense_matrix.shape[0] != contig_information_intact.shape[0]:
    logger.info(f'Dimention dismatch: {dense_matrix.shape[0]}, {contig_information_intact.shape[0]}')
    dense_matrix = dense_matrix[:contig_information_intact.shape[0], :contig_information_intact.shape[0]]

matrix_columns = {
    'Contig name': 'Contig',
    'Bin': 'Bin',
    'Domain': 'Domain',
    'Kingdom': 'Kingdom',
    'Phylum': 'Phylum',
    'Class': 'Class',
    'Order': 'Order',
    'Family': 'Family',
    'Genus': 'Genus',
    'Species': 'Species',
    'Number of restriction sites': 'Restriction sites',
    'Contig length': 'Contig length',
    'Contig coverage': 'Contig coverage',
    'Hi-C contacts mapped to the same contigs': 'Intra-contig contact'
}
contig_information_display = contig_information_intact[list(matrix_columns.keys())].rename(columns=matrix_columns)
# Add a "Visibility" column to the contig_information_display DataFrame
contig_information_display['Visibility'] = 1  # Default value to 1 (visible)

logger.info('Data loading completed')  

contig_information, unique_annotations, contact_matrix, contact_matrix_display = prepare_data(contig_information_intact, dense_matrix)
  
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
            {"headerName": "Contig", "field": "Contig", "pinned": 'left', "width": 120},
            {"headerName": "Bin", "field": "Bin", "width": 120}
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
            {"headerName": "Contig length", "field": "Contig length", "width": 140, "wrapHeaderText": True},
            {"headerName": "Contig coverage", "field": "Contig coverage", "width": 140, "wrapHeaderText": True},
            {"headerName": "Intra-contig contact", "field": "Intra-contig contact", "width": 140, "wrapHeaderText": True},
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
                {'label': 'Contig', 'value': 'contig'}
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
            style={}
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
                rowData=contig_information_display.to_dict('records'),
                defaultColDef={},
                style={'height': '40vh', 'width': '30vw', 'display': 'inline-block'},
                dashGridOptions={
                    'headerPinned': 'top',
                    'rowSelection': 'single'  # Enable single row selection
                }
            )
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            dcc.Graph(id='treemap-graph', figure=treemap_fig, style={'height': '80vh', 'width': '48vw', 'display': 'inline-block'}),
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
        html.Div(id='hover-info', style={'height': '20vh', 'width': '20vw', 'background-color': 'white', 'padding': '5px', 'border': '1px solid #ccc', 'margin-top': '3px'}),
            html.Div([
                dcc.Textarea(
                    id='chatgpt-input',
                    placeholder='Enter your query here...',
                    style={'width': '100%', 'height': '15vh', 'display': 'inline-block'}
                ),
                html.Button('Interpret Data', id='interpret-button', n_clicks=0, style={'width': '100%', 'display': 'inline-block'})
            ], style={'width': '20vw', 'display': 'inline-block'}),
            html.Div(id='gpt-answer', 
                     style={'height': '45vh', 
                            'width': '20vw', 
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
    
    contig_information, unique_annotations, contact_matrix, contact_matrix_display = prepare_data(contig_information_intact, dense_matrix, taxonomy_level)    
    
    global current_visualization_mode
    
    current_visualization_mode = {
        'visualization_type': 'taxonomy_hierarchy',
        'taxonomy_level': taxonomy_level,
        'selected_annotation': None,
        'secondary_annotation': None,
        'selected_contig': None
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
     Output('taxonomy-level-selector', 'style'),
     Output('annotation-selector', 'value'),
     Output('annotation-selector', 'style'),
     Output('secondary-annotation-selector', 'value'),
     Output('secondary-annotation-selector', 'style'),
     Output('contig-selector', 'value'),
     Output('contig-selector', 'style'),
     Output('contact-table', 'active_cell'),
     Output('contig-info-table', 'selectedRows')],
    [Input('reset-btn', 'n_clicks'),
     Input('visualization-selector', 'value'),
     Input('contact-table', 'active_cell'),
     Input('contig-info-table', 'selectedRows'),
     Input('cyto-graph', 'selectedNodeData'),
     Input('cyto-graph', 'selectedEdgeData')],
     State('taxonomy-level-selector', 'value'),
     prevent_initial_call=True
)
def sync_selectors(reset_clicks,visualization_type, contact_table_active_cell, contig_info_selected_rows, selected_node_data, selected_edge_data, taxonomy_level):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    selected_annotation, secondary_annotation, selected_contig = synchronize_selections(
        triggered_id, selected_node_data, selected_edge_data, contig_info_selected_rows, contact_table_active_cell, taxonomy_level )
    
    taxonomy_level_style = {'display': 'none'}
    annotation_selector_style = {'display': 'none'}
    secondary_annotation_style = {'display': 'none'}
    contig_selector_style = {'display': 'none'}
    
    # Reset all the selections
    if triggered_id == 'reset-btn':
        visualization_type = 'taxonomy_hierarchy'
        selected_annotation = None
        secondary_annotation = None
        selected_contig = None
        

    # If a contig is selected in the network or contig table
    if selected_contig:
        visualization_type = 'contig'
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
        contig_selector_style = {'width': '300px', 'display': 'inline-block'}

    # If a annotation is selected in the network or annotation table
    if selected_annotation and not selected_contig:
        if secondary_annotation:
            visualization_type = 'inter_annotation'
            annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
            secondary_annotation_style = {'width': '300px', 'display': 'inline-block'}
        else:
            visualization_type = 'intra_annotation'
            annotation_selector_style = {'width': '300px', 'display': 'inline-block'}

    # Default cases based on visualization_type
    if visualization_type == 'taxonomy_hierarchy':
        taxonomy_level_style = {'width': '300px', 'display': 'inline-block'}
    elif visualization_type == 'intra_annotation':
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
    elif visualization_type == 'inter_annotation':
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
        secondary_annotation_style = {'width': '300px', 'display': 'inline-block'}
    elif visualization_type == 'contig':
        annotation_selector_style = {'width': '300px', 'display': 'inline-block'}
        contig_selector_style = {'width': '300px', 'display': 'inline-block'}

    return visualization_type, taxonomy_level_style, selected_annotation, annotation_selector_style, secondary_annotation, secondary_annotation_style, selected_contig, contig_selector_style, None, []
 
def synchronize_selections(triggered_id, selected_node_data, selected_edge_data, contig_info_selected_rows, contact_table_active_cell, taxonomy_level ):
    # Initialize the return values
    selected_annotation = None
    secondary_annotation = None
    selected_contig = None
    

    # If a node in the network is selected
    if triggered_id == 'cyto-graph' and selected_node_data:
        selected_node_id = selected_node_data[0]['id']
        # Check if the selected node is a contig or a annotation
        if selected_node_id in contig_information['Contig name'].values:
            contig_info = contig_information[contig_information['Contig name'] == selected_node_id].iloc[0]
            selected_annotation = contig_info['Contig annotation']
            selected_contig = contig_info['Contig name']
        else:
            selected_annotation = selected_node_id

    # If an edge in the network is selected
    elif triggered_id == 'cyto-graph' and selected_edge_data:
        source_annotation = selected_edge_data[0]['source']
        target_annotation = selected_edge_data[0]['target']
        selected_annotation = source_annotation
        secondary_annotation = target_annotation

    # If a row in the contig-info-table is selected
    elif triggered_id == 'contig-info-table' and contig_info_selected_rows:
        print(contig_info_selected_rows)
        selected_row = contig_info_selected_rows[0]
        if taxonomy_level in selected_row and 'Contig' in selected_row:
            selected_annotation = selected_row[taxonomy_level]
            selected_contig = selected_row['Contig']

    # If a cell in the contact-table is selected
    elif triggered_id == 'contact-table' and contact_table_active_cell:
        row_id = contact_table_active_cell['row']
        column_id = contact_table_active_cell['column_id']
        row_annotation = contact_matrix_display.iloc[row_id]['Annotation']
        col_annotation = column_id if column_id != 'Annotation' else None
        selected_annotation = row_annotation
        secondary_annotation = col_annotation
    
    logger.info(f'Current selected: \n{selected_annotation}, \n{secondary_annotation}, \n{selected_contig}')

    return selected_annotation, secondary_annotation, selected_contig

# Callback to update the visualizationI want to 
@app.callback(
    [Output('cyto-graph', 'elements'),
     Output('cyto-graph', 'style'),
     Output('bar-chart', 'figure'),
     Output('treemap-graph', 'figure'),
     Output('treemap-graph', 'style'),
     Output('contig-info-table', 'defaultColDef')],
    [Input('reset-btn', 'n_clicks'),
     Input('confirm-btn', 'n_clicks')],
    [State('visualization-selector', 'value'),
     State('annotation-selector', 'value'),
     State('secondary-annotation-selector', 'value'),
     State('contig-selector', 'value'),
     State('contact-table', 'data')],
     prevent_initial_call=True
)
def update_visualization(reset_clicks, confirm_clicks, visualization_type, selected_annotation, secondary_annotation, selected_contig, table_data):
    logger.info('update_visualization triggered')
    logger.info(f'Visualization type: {visualization_type}')
    logger.info(f'Selected annotation: {selected_annotation}')
    logger.info(f'Secondary annotation: {secondary_annotation}')
    logger.info(f'Selected contig: {selected_contig}')
    
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
        current_visualization_mode['secondary_annotation'] = secondary_annotation
        current_visualization_mode['selected_contig'] = selected_contig

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
        elif visualization_type == 'contig':
            logger.info('show contig visulization')
            cyto_elements, bar_fig = contig_visualization(selected_annotation, selected_contig, contig_information, unique_annotations)
            treemap_fig = go.Figure()
            treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            cyto_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}

    # Update column definitions with style conditions
    contig_colors, annotation_colors = get_contig_and_annotation_colors(cyto_elements, contig_information)
    styleConditions = styling_contig_table(contig_colors, annotation_colors, unique_annotations)
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
     Input('contig-selector', 'value')],
     prevent_initial_call=True
)
def update_selected_styles(selected_annotation, secondary_annotation, selected_contig):
    selected_nodes = []
    selected_edges = []
    hover_info = "No selection"

    if selected_annotation and secondary_annotation:
        selected_edges.append((selected_annotation, secondary_annotation))
        selected_nodes.append(selected_annotation)
        selected_nodes.append(secondary_annotation)
        hover_info = f"Edge between {selected_annotation} and {secondary_annotation}"
    elif selected_contig:
        selected_nodes.append(selected_contig)
        contig_info = contig_information[contig_information['Contig name'] == selected_contig].iloc[0]
        hover_info = f"Contig: {selected_contig}<br>Annotation: {contig_info['Contig annotation']}"

        if current_visualization_mode['visualization_type'] == 'inter_annotation':
            if contig_info['Contig annotation'] == current_visualization_mode['secondary_annotation']:
                # Find contigs from the selected annotation that have contact with the selected contig
                selected_contig_index = contig_information[contig_information['Contig name'] == selected_contig].index[0]
                selected_annotation_indices = get_contig_indexes(current_visualization_mode['selected_annotation'], contig_information)
                connected_contigs = []

                for j in selected_annotation_indices:
                    if dense_matrix[j, selected_contig_index] != 0:  # Check contact from selected annotation to the selected contig
                        connected_contig = contig_information.at[j, 'Contig name']
                        connected_contigs.append(connected_contig)

                # Add the connected contigs and edges to the lists
                selected_nodes.extend(connected_contigs)
                for contig in connected_contigs:
                    selected_edges.append((contig, selected_contig))  # Edge goes from the connected contig to the selected contig

            else:
                # Find the contigs in the secondary annotation that have contact with the selected contig
                selected_contig_index = contig_information[contig_information['Contig name'] == selected_contig].index[0]
                secondary_annotation_indices = get_contig_indexes(current_visualization_mode['secondary_annotation'], contig_information)
                connected_contigs = []

                for j in secondary_annotation_indices:
                    if dense_matrix[selected_contig_index, j] != 0:  # Check contact from selected contig to secondary annotation
                        connected_contig = contig_information.at[j, 'Contig name']
                        connected_contigs.append(connected_contig)

                # Add the connected contigs and edges to the lists
                selected_nodes.extend(connected_contigs)
                for contig in connected_contigs:
                    selected_edges.append((selected_contig, contig))  # Edge goes from selected contig to the connected contig

    elif selected_annotation:
        selected_nodes.append(selected_annotation)
        hover_info = f"Annotation: {selected_annotation}"

    # Add selection styles for the selected nodes and edges
    stylesheet = add_selection_styles(selected_nodes, selected_edges)

    return stylesheet, hover_info

@app.callback(
    [Output('contig-info-table', 'rowData'), 
     Output('contig-info-table', 'filterModel'),
     Output('row-count', 'children')],
    [Input('annotation-selector', 'value'),
     Input('secondary-annotation-selector', 'value'),
     Input('visibility-filter', 'value')],
    [State('taxonomy-level-selector', 'value'),
     State('contig-info-table', 'rowData')]
)
def update_filter_model_and_row_count(selected_annotation, secondary_annotation, filter_value, taxonomy_level, contig_data):
    filter_model = {}
    filtered_data = contig_data
    
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
            row_indices = get_contig_indexes(current_visualization_mode['selected_annotation'], contig_information)
            col_indices = get_contig_indexes(current_visualization_mode['secondary_annotation'], contig_information)
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
    [Output('annotation-selector', 'options'),
     Output('secondary-annotation-selector', 'options'),
     Output('contig-selector', 'options')],
    [Input('annotation-selector', 'value')],
    [State('visualization-selector', 'value')],
    prevent_initial_call=True
)
def update_dropdowns(selected_annotation, visualization_type):
    # Initialize empty lists for options
    annotation_options = []
    secondary_annotation_options = []
    contig_options = []

    annotation_options = [{'label': annotation, 'value': annotation} for annotation in unique_annotations]
        
    # Only show secondary annotation options if visualization type is 'inter_annotation'
    if visualization_type == 'inter_annotation':
        secondary_annotation_options = annotation_options  # Same options for secondary annotation dropdown
    else:
        secondary_annotation_options = []  # Hide secondary annotation dropdown if not in inter_annotation mode

    # Only show contig options if visualization type is 'contig'
    if visualization_type == 'contig' and selected_annotation:
        contigs = contig_information.loc[get_contig_indexes(selected_annotation, contig_information), 'Contig name']
        contig_options = [{'label': contig, 'value': contig} for contig in contigs]

    return annotation_options, secondary_annotation_options, contig_options

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