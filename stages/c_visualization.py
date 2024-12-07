import numpy as np
import pandas as pd
import networkx as nx
import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import dash_ag_grid as dag
import plotly.graph_objects as go
from dash import callback_context
import plotly.express as px
import plotly.colors as colors
import math 
from math import sqrt, sin, cos
import os
import io
import logging
import json
import py7zr
from joblib import Parallel, delayed
from itertools import combinations
from stages.helper import (
    get_indexes,
    calculate_submatrix_sum,
    save_to_redis,
    load_from_redis
)

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
    elements = []
    nodes_to_remove = set()

    # Identify nodes to remove
    for node in G.nodes:
        if "Unclassified" in node or "Unclassified" in str(G.nodes[node].get('parent', '')):
            nodes_to_remove.add(node)

    # Filter out nodes and add visible ones to elements
    for node in G.nodes:
        if node not in nodes_to_remove:
            elements.append({
                'data': {
                    'id': node,
                    'label': node,
                    'label_size': 20 if G.nodes[node].get('parent') is None else 15,
                    'size': G.nodes[node].get('size', 1),
                    'color': G.nodes[node].get('color', '#000'),
                    'parent': G.nodes[node].get('parent', None),
                    'visible': 'none' if node in invisible_nodes else 'element'
                },
                'position': {
                    'x': pos[node][0] * 100,
                    'y': pos[node][1] * 100
                },
                'style': {
                    'text-margin-y': -5,
                    'font-style': 'italic' if G.nodes[node].get('parent') is None else 'normal'
                }
            })

    # Filter out edges involving nodes to remove
    for edge in G.edges(data=True):
        if edge[0] not in nodes_to_remove and edge[1] not in nodes_to_remove:
            elements.append({
                'data': {
                    'source': edge[0],
                    'target': edge[1],
                    'width': edge[2].get('width', 1),
                    'color': edge[2].get('color', '#bbb'),
                    'visible': 'none' if (edge[0], edge[1]) in invisible_edges or (edge[1], edge[0]) in invisible_edges else 'element',
                    'selectable': False
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
                    'line-color': '#bbb',
                    'display': 'element'
                }
            }
            cyto_stylesheet.append(edge_style)

    return cyto_stylesheet

def create_bar_chart(data_dict):
    traces = []
    buttons = []

    for idx, (trace_name, data_frame) in enumerate(data_dict.items()):
        if data_frame.empty or 'value' not in data_frame.columns:
            logger.warning(f"No data or 'value' column missing for {trace_name}, skipping trace.")
            continue

        bar_data = data_frame.sort_values(by='value', ascending=False)
        bar_colors = bar_data['color']
        hover_text = bar_data.get('hover', None)

        bar_trace = go.Bar(
            x=bar_data['name'],
            y=bar_data['value'],
            name=trace_name,
            marker_color=bar_colors,
            visible=(idx == 0),  # Default visibility
            hovertext=hover_text if hover_text is not None else None,
            hoverinfo='text' if hover_text is not None else None
        )
        traces.append(bar_trace)

        # Configure dropdown button
        buttons.append(
            dict(
                label=trace_name,
                method="update",
                args=[
                    {"visible": [i == idx for i in range(len(data_dict))]},
                    {"yaxis.autorange": True}
                ]
            )
        )

    if not traces:
        logger.warning("No valid traces created, returning empty figure.")
        return go.Figure()

    bar_layout = go.Layout(
        xaxis=dict(
            title="",
            tickangle=-45,
            tickfont=dict(size=12),
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        yaxis=dict(
            title="Value",
            tickfont=dict(size=15),
            autorange=True,
        ),
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=buttons,
                showactive=True,
                x=0.4,
                xanchor="center",
                y=1,
                yanchor="bottom",
                font=dict(size=16),
            )
        ]
    )

    return go.Figure(data=traces, layout=bar_layout)

# Function to add opacity to a hex color
def add_opacity_to_color(color, opacity):
    if color.startswith('#') and len(color) == 7:
        hex_color = color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'
    
    elif color.startswith('rgb(') and color.endswith(')'):
        rgb_color = color[4:-1].split(',')
        if len(rgb_color) == 3:
            r, g, b = (int(value.strip()) for value in rgb_color)
            return f'rgba({r}, {g}, {b}, {opacity})'
        
    else:
        return f'rgba(255, 255, 255, {opacity})'
    
# Function to get bin colors from Cytoscape elements or use annotation color if not found
def get_id_colors(cyto_elements):
    filtered_cyto_elements = [
        element for element in cyto_elements or []
        if 'data' in element and 'source' not in element['data']
    ]
    
    id_colors = {}
    for element in filtered_cyto_elements or []:
        try:
            element_id = element['data']['id']
            element_color = element['data']['color']
            id_colors[element_id] = element_color
        except KeyError:
            continue

    return id_colors

# Function to style annotation contact table using Blugrn color scheme
def styling_annotation_table(row_data, bin_information, unique_annotations):
    styles = []
    opacity = 0.6

    # Styling for numeric columns
    numeric_cols = [key for key in row_data[0].keys() if key != "index"]

    for col in numeric_cols:
        # Extract numeric data
        numeric_df = pd.DataFrame([row[col] for row in row_data if isinstance(row[col], (int, float))], columns=[col])
        
        # Skip columns with no valid numeric data
        if numeric_df.empty or numeric_df[col].isnull().all():
            continue

        # Add 1 to avoid log issues and calculate bounds
        numeric_df += 1
        col_min = np.log1p(numeric_df[col].min())
        col_max = np.log1p(numeric_df[col].max())

        # Skip if all values are the same (no range for styling)
        if col_min == col_max:
            continue

        col_range = col_max - col_min
        n_bins = 10
        bounds = [i * (col_range / n_bins) + col_min for i in range(n_bins + 1)]

        for i in range(1, len(bounds)):
            min_bound = bounds[i - 1]
            max_bound = bounds[i]
            if i == len(bounds) - 1:
                max_bound += 1

            bg_color = f"rgba({255 - int((min_bound - col_min) / col_range * 255)}, {255 - int((min_bound - col_min) / col_range * 255)}, 255, {opacity})"
            styles.append({
                "condition": f"params.colDef.field == '{col}' && Math.log1p(params.value) >= {min_bound} && Math.log1p(params.value) < {max_bound}",
                "style": {
                    'backgroundColor': bg_color,
                    'color': "white" if i > len(bounds) / 2 else "inherit"
                }
            })
    
    styles.append({
        "condition": "params.colDef.field == 'index' && params.data.index.endsWith('_p')",
        "style": {
            "fontStyle": "italic",
            "backgroundColor": add_opacity_to_color('#D5ED9F', opacity),  # Plasmid color
            "color": "black"
        }
    })
    
    styles.append({
        "condition": "params.colDef.field == 'index' && params.data.index.endsWith('_v')",
        "style": {
            "fontStyle": "italic",
            "backgroundColor": add_opacity_to_color('#AE445A', opacity),  # Phage color
            "color": "black"
        }
    })
    
    styles.append({
        "condition": "params.colDef.field == 'index' && params.data.index.endsWith('_c')",  # Default case for chromosome
        "style": {
            "fontStyle": "italic",
            "backgroundColor": add_opacity_to_color('#81BFDA', opacity),  # Chromosome color
            "color": "black"
        }
    })

    return styles

def styling_information_table(information_data, id_colors, unique_annotations, table_type='bin', taxonomy_columns=None, taxonomy_level=None):
    if taxonomy_level is None:
        taxonomy_level = taxonomy_columns[-1]
    columns = ['The number of restriction sites', 'Contig length', 'Contig coverage']
    styles = []
    # Precompute numeric data
    numeric_data = information_data.select_dtypes(include=[np.number]).copy()
    numeric_data += 1
    col_min = np.log1p(numeric_data.min(axis=0))
    col_max = np.log1p(numeric_data.max(axis=0))
    col_range = col_max - col_min
    n_bins = 10
    bounds = {col: [i * (col_range[col] / n_bins) + col_min[col] for i in range(n_bins + 1)] for col in columns}

    # Style numeric columns
    def style_numeric_column(col):
        col_styles = []
        for i in range(1, len(bounds[col])):
            min_bound = bounds[col][i - 1]
            max_bound = bounds[col][i]
            if i == len(bounds[col]) - 1:
                max_bound += 1
    
            # Handle NaN and zero range
            col_min_value = col_min.get(col, 0)  # Default to 0 if missing
            col_range_value = col_range.get(col, 1)  # Default to 1 if missing or 0
    
            if np.isnan(col_min_value) or np.isnan(col_range_value):
                continue  # Skip this column style if data is invalid
    
            if col_range_value == 0:
                col_range_value = 1  # Prevent division by zero
    
            opacity = 0.6
            adjusted_value = 255 - int((min_bound - col_min_value) / col_range_value * 255)
    
            # Add style condition
            col_styles.append({
                "condition": f"params.colDef.field == '{col}' && Math.log1p(params.value) >= {min_bound} && Math.log1p(params.value) < {max_bound}",
                "style": {
                    'backgroundColor': f"rgba({adjusted_value}, {adjusted_value}, 255, {opacity})",
                    'color': "white" if i > len(bounds[col]) / 2.0 else "inherit"
                }
            })
        return col_styles

    # Parallelize numeric column styling
    numeric_styles = Parallel(n_jobs=-1)(delayed(style_numeric_column)(col) for col in columns)
    for col_style in numeric_styles:
        styles.extend(col_style)
        
    # Style annotation column
    for type_key, color in type_colors.items():
        annotation_color_with_opacity = add_opacity_to_color(color, 0.6)
        annotation_style = {
            "condition": f"params.colDef.field == '{taxonomy_level}' && params.data.Category == '{type_key}'",
            "style": {
                'backgroundColor': annotation_color_with_opacity,
                "fontStyle": "italic",
                'color': 'black'
            }
        }
        styles.append(annotation_style)
    
    # Style ID column
    id_column = 'Bin index' if table_type == 'bin' else 'Contig index'
    
    # Function to compute color, create styles, and append them to the styles list
    def style_id_column(item, item_type, styles):
        # Combine logic to calculate colors
        type_color = type_colors.get(item_type, default_color)  # Fallback to white if annotation not found 
        item_color = id_colors.get(item, type_color)  # Fallback to annotation_color if item not found
        item_color_with_opacity = add_opacity_to_color(item_color, 0.6)        
        # Create and append style for the ID column
        id_style = {
            "condition": f"params.colDef.field == '{id_column}' && params.value == '{item}'",
            "style": {
                'backgroundColor': item_color_with_opacity,
                'color': 'black'
            }
        }
        styles.append(id_style)
    
    # Iterate over rows and compute styles
    for _, row in information_data.iterrows():
        if row['Annotation'] in unique_annotations:
            style_id_column(row[id_column],  row['Category'], styles)
            
    # Add styles for italic taxonomy columns
    for col in taxonomy_columns:
        italic_style = {
            "condition": f"params.colDef.field == '{col}'",
            "style": {
                "fontStyle": "italic"
            }
        }
        styles.append(italic_style)

    return styles

# Function to arrange bins
def arrange_nodes(bins, inter_bin_edges, distance, selected_element=None, center_position=(0, 0)):
    distance /= 100 
    phi = (1 + sqrt(5)) / 2
    pi = math.pi

    # Identify bins that connect to other annotation
    connecting_bins = [bin for bin in bins if bin in inter_bin_edges and bin != selected_element]
    other_bins = [bin for bin in bins if bin not in inter_bin_edges and bin != selected_element]

    # Arrange inner bins in a sunflower pattern
    inner_positions = {}
    angle_stride = 2 * pi / phi ** 2

    max_inner_radius = 0  # To keep track of the maximum radius used for inner nodes

    for k, bin in enumerate(other_bins, start=1):
        r = distance * sqrt(k)
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

def taxonomy_visualization(bin_information, unique_annotations, contact_matrix, taxonomy_columns):
    taxonomy_columns = taxonomy_columns.tolist() if isinstance(taxonomy_columns, np.ndarray) else taxonomy_columns
    logger.debug("Starting taxonomy_visualization function.")

    # Check input validity
    if taxonomy_columns is None or len(taxonomy_columns) == 0:
        logger.error("taxonomy_columns is empty or not provided.")
        raise ValueError("taxonomy_columns must be provided.")

    logger.debug(f"taxonomy_columns: {taxonomy_columns}")
    logger.debug(f"bin_information columns: {bin_information.columns}")
    logger.debug(f"Number of unique_annotations: {len(unique_annotations)}")

    # Dynamically map taxonomy levels
    level_mapping = {"Community": 1, "Category": 2}  # Assign Community and Category fixed levels
    level_mapping.update({level: idx + 3 for idx, level in enumerate(taxonomy_columns)})  # Taxonomy levels start at 3
    logger.debug(f"Level mapping: {level_mapping}")

    logger.debug(f"Level mapping: {level_mapping}")

    records = []
    existing_annotations = set()

    # Add "Community" as the root node
    records.append({
        "annotation": "Community",
        "parent": "",
        "level": level_mapping["Community"],
        "level_name": "Community",
        "type": "Community",
        "total coverage": 0,
        "border_color": "black",
        "bin": bin_information['Bin index'].unique()
    })
    existing_annotations.add("Community")

    categories = bin_information['Category'].unique()
    for category in categories:
        records.append({
            "annotation": category,
            "parent": "Community",
            "level": level_mapping["Category"],
            "level_name": "Category",
            "type": category,
            "total coverage": 0,
            "border_color": type_colors.get(category, "gray"),
            "bin": []
        })
        existing_annotations.add(category)

    # Populate hierarchy dynamically
    for _, row in bin_information.iterrows():
        parent = row['Category']
        for level in taxonomy_columns:
            annotation = row[level]

            # Skip unclassified annotations
            if "Unclassified" in annotation:
                continue

            if annotation not in existing_annotations:
                records.append({
                    "annotation": annotation,
                    "parent": parent,
                    "level": level_mapping[level],
                    "level_name": level.capitalize(),
                    "type": row['Category'],
                    "total coverage": row['Contig coverage'],
                    "border_color": type_colors.get(row['Category'], "gray"),
                    "bin": [row['Bin index']]
                })
                existing_annotations.add(annotation)
            else:
                # Update existing record
                for rec in records:
                    if rec['annotation'] == annotation:
                        rec['total coverage'] += row['Contig coverage']
                        rec['bin'].append(row['Bin index'])
                        break

            # Set current annotation as parent for the next level
            parent = annotation

    logger.debug(f"Number of records created: {len(records)}")
    
    hierarchy_df = pd.DataFrame(records)
    logger.debug(f"Hierarchy DataFrame head:\n{hierarchy_df.head()}")

    hierarchy_df['scaled_coverage'] = generate_gradient_values(hierarchy_df['total coverage'], 10, 30)
    hierarchy_df.loc[hierarchy_df['type'] == 'virus', 'scaled_coverage'] *= 20

    def format_bins(bin_list, max_bins=5):
        if len(bin_list) > max_bins:
            return ', '.join(bin_list[:max_bins]) + f"... (+{len(bin_list) - max_bins} more)"
        return ', '.join(bin_list)

    hierarchy_df['limited_bins'] = hierarchy_df['bin'].apply(lambda bins: format_bins(bins))
    logger.debug("Processed bin formatting.")

    # Create Treemap
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
        customdata=hierarchy_df[['level_name', 'limited_bins']],
        hovertemplate='<b>%{label}</b><br>Level: %{customdata[0]}<br>Coverage: %{value}<br>Bins: %{customdata[1]}'
    )

    fig.update_layout(
        coloraxis_showscale=False,
        font=dict(family="Arial", size=20, style="italic"),
        autosize=True,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    logger.debug("Treemap created.")

    # Generate classification bar chart
    classification_data = []
    for level in taxonomy_columns:
        total_count = len(bin_information[level])
        unclassified_count = bin_information[level].str.contains("Unclassified", na=False).sum()
        classified_count = total_count - unclassified_count
        classified_ratio = (classified_count / total_count) * 100 if total_count > 0 else 0

        classification_data.append({
            "name": level,
            "value": classified_ratio,
            "color": "gray"
        })

    classification_data_df = pd.DataFrame(classification_data)
    logger.debug(f"Classification data:\n{classification_data_df}")

    data_dict = {
        'Ratio of classified bins by taxonomic Level': classification_data_df
    }

    bar_fig = create_bar_chart(data_dict)
    logger.debug("Bar chart created.")

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
    total_bin_coverage = bin_information.groupby('Annotation')['Contig coverage'].sum().reindex(connected_nodes)

    node_colors = {}
    for annotation in total_bin_coverage.index:
        bin_type = bin_information.loc[
            bin_information['Annotation'] == annotation, 'Category'
        ].values[0]
        color = type_colors.get(bin_type, default_color)
        node_colors[annotation] = color
        G.add_node(annotation, size=20, color=color, parent=None)

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


    if edge_weights:
        normalized_weights = generate_gradient_values(np.array(edge_weights), 1, 2)
        for (i, (u, v)) in enumerate(G.edges()):
            G[u][v]['weight'] = normalized_weights[i]
    

    # Initial node positions using a force-directed layout with increased dispersion
    if selected_node:
        pos = nx.spring_layout(G, dim=2, k=2, iterations=200, weight='weight', scale=10.0, fixed=[selected_node], pos={selected_node: (0, 0)})
    else:
        pos = nx.spring_layout(G, dim=2, k=2, iterations=200, weight='weight', scale=10.0)


    cyto_elements = nx_to_cyto_elements(G, pos, invisible_edges=invisible_edges)
    cyto_style = {
        'name': 'preset',
        'fit': False
    }
    
    if not selected_node:
        inter_annotation_contact_sum = contact_matrix.sum(axis=1) - np.diag(contact_matrix.values)
        total_bin_coverage_sum = bin_information.groupby('Annotation').apply(
            lambda group: group['Contig coverage'].mul(group['Contig length']).sum() /
                          group['The number of restriction sites'].sum()
        ).reindex(unique_annotations).fillna(0).values
    
        data_dict = {
            'Across Taxonomy Hi-C Contacts': 
                pd.DataFrame({'name': unique_annotations, 
                              'value': inter_annotation_contact_sum, 
                              'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]
                              }).query('value != 0'),
            'The coverage of different taxa': 
                pd.DataFrame({'name': unique_annotations, 
                              'value': total_bin_coverage_sum, 
                              'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]
                              }).query('value != 0')
        }
    
        bar_fig = create_bar_chart(data_dict)
        return cyto_elements, bar_fig, cyto_style

    # If a node is selected, don't return the histogram
    return cyto_elements, None, cyto_style

# Function to visualize bin relationships
def bin_visualization(selected_annotation, selected_bin, bin_information, bin_dense_matrix, unique_annotations):
    
    # Find the index of the selected bin
    selected_bin_index = bin_information[bin_information['Bin index'] == selected_bin].index[0]
    selected_annotation = bin_information.loc[selected_bin_index, 'Annotation']

    # Get all indices that have contact with the selected bin
    contacts_indices = bin_dense_matrix[selected_bin_index].nonzero()[0]
    contacts_indices = contacts_indices[contacts_indices != selected_bin_index]

    # If no contacts found, raise a warning
    if len(contacts_indices) == 0:
        logger.warning(f'No contacts found for the selected bin: {selected_bin}')
        contacts_annotation = pd.Series([selected_annotation])
        contacts_bins = pd.Series([])
    else:
        contacts_annotation = bin_information.loc[contacts_indices, 'Annotation']
        contacts_bins = bin_information.loc[contacts_indices, 'Bin index']

    G = nx.Graph()

    # Use a categorical color scale
    color_scale_mapping = {
        'virus': colors.sequential.Redor,
        'plasmid': colors.sequential.YlGn,
        'chromosome': colors.sequential.BuPu
    }

    # Rank annotation based on the number of bins with contact to the selected bin
    annotation_contact_counts = contacts_annotation.value_counts()
    annotation_contact_ranks = annotation_contact_counts.rank(method='first').astype(int)
    max_rank = annotation_contact_ranks.max()

    # Add annotation nodes
    for annotation in contacts_annotation.unique():
        annotation_type = bin_information.loc[bin_information['Annotation'] == annotation, 'Category'].values[0]
        color_scale = color_scale_mapping.get(annotation_type, [default_color])  
        annotation_rank = annotation_contact_ranks.get(annotation, int(max_rank/2))
        gradient_color = color_scale[int((annotation_rank / max_rank) * (len(color_scale) - 1))]
        G.add_node(annotation, size=1, color='#FFFFFF', border_color=gradient_color, border_width=2)

    # Ensure selected_annotation is added
    if selected_annotation not in G.nodes:
        annotation_type = bin_information.loc[bin_information['Annotation'] == selected_annotation, 'Category'].values[0]
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
    
    scaled_weights = generate_gradient_values(contact_values, 1, 2) if contact_values else [1] * len(contacts_annotation.unique())
    for i, annotation in enumerate(contacts_annotation.unique()):
        annotation_bins = bin_information[bin_information['Annotation'] == annotation].index
        annotation_bins = annotation_bins[annotation_bins < bin_dense_matrix.shape[1]]  # Filter indices
        contact_value = bin_dense_matrix[selected_bin_index, annotation_bins].sum()
        
        if contact_value > 0:
            G.add_edge(selected_annotation, annotation, weight=scaled_weights[i])


    fixed_positions = {selected_annotation: (0, 0)}
    pos = nx.spring_layout(G, iterations=200, k=1, fixed=[selected_annotation], pos=fixed_positions, weight='weight')

    # Remove the edges after positioning
    G.remove_edges_from(list(G.edges()))

    # Handle selected_bin node
    selected_bin_type = bin_information.loc[selected_bin_index, 'Category']
    selected_bin_color = type_colors.get(selected_bin_type, default_color)
    G.add_node(selected_bin, size=15, color=selected_bin_color, parent=selected_annotation)
    pos[selected_bin] = (0, 0)

    # Add bin nodes to the graph
    for annotation in contacts_annotation.unique():
        annotation_bins = contacts_bins[contacts_annotation == annotation]
        bin_positions = arrange_nodes(annotation_bins, [], distance=25, center_position=pos[annotation], selected_element=None)
        for bin, (x, y) in bin_positions.items():
            G.add_node(bin, 
                       size=10, 
                       color=G.nodes[annotation]['border_color'],  # Use the color from the annotation
                       parent=annotation)
            G.add_edge(selected_bin, bin)
            pos[bin] = (x, y)
    
    cyto_elements = nx_to_cyto_elements(G, pos)
    
    # Prepare data for histogram
    bin_contact_values = bin_dense_matrix[selected_bin_index, contacts_indices]
    
    try:
        bin_data = pd.DataFrame({
            'name': contacts_bins, 
            'value': bin_contact_values, 
            'color': [G.nodes[bin]['color'] for bin in contacts_bins],
            'hover': [f"({bin_information.loc[bin_information['Bin index'] == bin, 'Annotation'].values[0]}, {value})" for bin, value in zip(contacts_bins, bin_contact_values)]
        })
    except:
        bin_data = pd.DataFrame()
    
    # Attempt to create annotation_data DataFrame
    try:
        annotation_data = pd.DataFrame({
            'name': contacts_annotation.unique(),
            'value': contact_values,
            'color': [G.nodes[annotation]['border_color'] for annotation in contacts_annotation.unique()]
        })
    except:
        annotation_data = pd.DataFrame()  # or assign None if that fits better with your app

    data_dict = {
        'Across Bin Hi-C Contacts': bin_data,
        'Across Taxa Hi-C Contact': annotation_data
    }

    bar_fig = create_bar_chart(data_dict)

    return cyto_elements, bar_fig

def contig_visualization(selected_annotation, selected_contig, contig_information, contig_dense_matrix, unique_annotations):
    # Find the index of the selected contig
    selected_contig_index = contig_information[contig_information['Contig index'] == selected_contig].index[0]
    selected_annotation = contig_information.loc[selected_contig_index, 'Annotation']

    # Get all indices that have contact with the selected contig
    contacts_indices = contig_dense_matrix[selected_contig_index].nonzero()[0]
    contacts_indices = contacts_indices[contacts_indices != selected_contig_index]

    # If no contacts found, raise a warning
    if len(contacts_indices) == 0:
        logger.warning(f'No contacts found for the selected contig: {selected_contig}')
        contacts_annotation = pd.Series([selected_annotation])
        contacts_contigs = pd.Series([])
    else:
        contacts_annotation = contig_information.loc[contacts_indices, 'Annotation']
        contacts_contigs = contig_information.loc[contacts_indices, 'Contig index']

    G = nx.Graph()

    # Use a categorical color scale
    color_scale_mapping = {
        'virus': colors.sequential.Redor,
        'plasmid': colors.sequential.YlGn,
        'chromosome': colors.sequential.BuPu
    }

    # Rank annotation based on the number of contigs with contact to the selected contig
    annotation_contact_counts = contacts_annotation.value_counts()
    annotation_contact_ranks = annotation_contact_counts.rank(method='first').astype(int)
    max_rank = annotation_contact_ranks.max()

    # Add annotation nodes
    for annotation in contacts_annotation.unique():
        annotation_type = contig_information.loc[contig_information['Annotation'] == annotation, 'Category'].values[0]
        color_scale = color_scale_mapping.get(annotation_type, [default_color])
        annotation_rank = annotation_contact_ranks.get(annotation, int(max_rank/2))
        gradient_color = color_scale[int((annotation_rank / max_rank) * (len(color_scale) - 1))]
        G.add_node(annotation, size=1, color='#FFFFFF', border_color=gradient_color, border_width=2)

    # Ensure selected_annotation is added
    if selected_annotation not in G.nodes:
        annotation_type = contig_information.loc[contig_information['Annotation'] == selected_annotation, 'Category'].values[0]
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

    scaled_weights = generate_gradient_values(contact_values, 1, 2) if contact_values else [1] * len(contacts_annotation.unique())
    for i, annotation in enumerate(contacts_annotation.unique()):
        annotation_contigs = contig_information[contig_information['Annotation'] == annotation].index
        annotation_contigs = annotation_contigs[annotation_contigs < contig_dense_matrix.shape[1]]  # Filter indices
        contact_value = contig_dense_matrix[selected_contig_index, annotation_contigs].sum()
        
        if contact_value > 0:
            G.add_edge(selected_annotation, annotation, weight=scaled_weights[i])

    fixed_positions = {selected_annotation: (0, 0)}
    pos = nx.spring_layout(G, iterations=200, k=1, fixed=[selected_annotation], pos=fixed_positions, weight='weight')

    # Remove the edges after positioning
    G.remove_edges_from(list(G.edges()))

    # Handle selected_contig node
    selected_contig_type = contig_information.loc[selected_contig_index, 'Category']
    selected_contig_color = type_colors.get(selected_contig_type, default_color)
    G.add_node(selected_contig, size=15, color=selected_contig_color, parent=selected_annotation)
    pos[selected_contig] = (0, 0)

    # Add contig nodes to the graph
    for annotation in contacts_annotation.unique():
        annotation_contigs = contacts_contigs[contacts_annotation == annotation]
        contig_positions = arrange_nodes(annotation_contigs, [], distance=25, center_position=pos[annotation], selected_element=None)
        for contig, (x, y) in contig_positions.items():
            G.add_node(contig, 
                       size=10, 
                       color=G.nodes[annotation]['border_color'],  # Use the color from the annotation
                       parent=annotation)
            G.add_edge(selected_contig, contig)
            pos[contig] = (x, y)

    cyto_elements = nx_to_cyto_elements(G, pos)
    
    # Prepare data for histogram
    contig_contact_values = contig_dense_matrix[selected_contig_index, contacts_indices]

    try:
        contig_data = pd.DataFrame({
            'name': contacts_contigs, 
            'value': contig_contact_values, 
            'color': [G.nodes[contig]['color'] for contig in contacts_contigs],
            'hover': [f"({contig_information.loc[contig_information['Contig index'] == contig, 'Annotation'].values[0]}, {value})" for contig, value in zip(contacts_contigs, contig_contact_values)]
        })
    except:
        contig_data = pd.DataFrame()
    
    # Attempt to create annotation_data DataFrame
    try:
        annotation_data = pd.DataFrame({
            'name': contacts_annotation.unique(),
            'value': contact_values,
            'color': [G.nodes[annotation]['border_color'] for annotation in contacts_annotation.unique()]
        })
    except:
        annotation_data = pd.DataFrame()


    data_dict = {
        'Across Contig Hi-C Contacts': contig_data,
        'Across Taxa Hi-C Contact': annotation_data
    }

    bar_fig = create_bar_chart(data_dict)

    logger.info("Finished contig_visualization function")

    return cyto_elements, bar_fig

def prepare_data(bin_information, contig_information, bin_dense_matrix, taxonomy_level):
    
    bin_information['Annotation'] = bin_information[taxonomy_level]
    contig_information['Annotation'] = contig_information[taxonomy_level]
    bin_information['Visibility'] = 1
    contig_information['Visibility'] = 1

    unique_annotations = bin_information['Annotation'].unique()
    
    contact_matrix = pd.DataFrame(0.0, index=unique_annotations, columns=unique_annotations)
    bin_indexes_dict = get_indexes(unique_annotations, bin_information, 'Annotation')

    non_self_pairs = list(combinations(unique_annotations, 2))
    self_pairs = [(x, x) for x in unique_annotations]
    all_pairs = non_self_pairs + self_pairs
    
    results = Parallel(n_jobs=-1)(
        delayed(calculate_submatrix_sum)(pair, bin_indexes_dict, bin_dense_matrix) for pair in all_pairs
    )
    
    for annotation_i, annotation_j, value in results:
        contact_matrix.at[annotation_i, annotation_j] = value
        contact_matrix.at[annotation_j, annotation_i] = value

    return bin_information, contig_information, unique_annotations, contact_matrix

# Create a logger and add the custom Dash logger handler
logger = logging.getLogger("app_logger")
 
type_colors = {
    'chromosome': '#81BFDA',
    'virus': '#AE445A',
    'plasmid': '#D5ED9F'
}
default_color = '#808080' 

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
            'display': 'data(visible)'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'width': 'data(width)',
            'line-color': 'data(color)',
            'opacity': 0.6,
            'display': 'data(visible)'
        }
    }
]

hover_info = {
    'download-btn': 
       ("The Download Button allows users to download the dataset they are currently viewing or working with.  \n\n"
        
         "This downloaded data can be re-uploaded to the app when visiting the website again, "
         "enabling returning users to continue from where they left off with their previously analyzed data.  \n\n"
         
         "Use this to save your progress or share your dataset for collaborative analysis."),
        
    'reset-btn': 
        ("The Reset Button resets all user selections and filters, bringing the visualization back to the 'Cross-Taxa Hi-C Interaction' visualization."),
        
    'confirm-btn': 
        ("The Confirm Button confirms the current selections made in the dropdowns and updates the visualization accordingly.  \n\n"
         
         "After making selections, use this button to refresh and reflect your choices in the visualizations."),
        
    "tooltip-toggle-container":
        ("Check the box next to 'Enable Help Tooltip' to activate tooltips.  \n\n"
         
         "Once enabled, hovering over buttons, charts, tables, or other interactive elements will display helpful instructions.  \n\n"
         
         "Use tooltips for guidance on how to interpret and interact with elements on the interface."),
        
    'dropdowns':
        ("The Taxonomy Level Selector allows users to choose the taxonomic level they wish to focus on in the dataset. \n"
         "The bar chart in the Taxonomic Framework, titled 'Ratio of classified bins across taxonomic levels,' can help users identify the taxonomic level with the highest saturability. \n"
         "Once a taxonomic level is selected, the visualization and data tables will update IMMEDIATELY without clicking confirm button. \n\n"
         
         "The rest of dropdown selectors allow users to filter and select specific annotations, bins, or contigs within the dataset. \n\n"
         
         "Click the Confirm button to apply the selection in selectors."),
        
    'bar-chart-container':
        ("The bar chart provides a dynamic and interactive way to explore various aspects of the dataset. \n"
         "Each aspect highlights a different characteristic of the data.  \n"
         "Users can select which aspect to display by choosing from the available options.  \n\n"
         
         "In the Taxonomic Framework, the bar chart titled 'Ratio of classified bins across taxonomy levels' highlights the percentage of classified bins at each taxonomy level. \n"
         "This visualization helps users identify taxonomy levels with the highest saturability, where a higher percentage indicates more complete classification information.  \n\n"
         
         "In other visualizations, the bar chart presents metrics relevant to the selected view.   \n"
         "For instance, in Bin and Contig visualizations, the bar chart displays interaction or coverage data that helps users analyze relationships and connections within the dataset.  \n\n"
         
         "A scroll bar is available at the bottom of the chart, allowing users to set the number of bars to display at once.  \n\n"
         
         "The color coding and labels align with the elements shown in the main visualization, providing a consistent reference across views.  \n\n"),

    'info-table-container': 
        ("The Visibility Filter checkbox filters the table to display only elements represented in the main visualization.   \n\n"
                  
         "Cells in the 'Index'column are colored based on the corresponding node in the Cytoscape Graph.  \n"
         "Cells in the column of the selected taxonomy level are colored using the unified color for the type (chromosome, plasmid, or virus).  \n"
         "Cells in numeric columns are highlighted with a bluish color. Higher values have a deeper color.  \n\n"
         
         "Users can click individual rows to select specific bins or contigs.  \n"
         "The first click selects the corresponding annotation.  \n"
         "The second click selects the specific bin/contig.   \n"
         "Remove the selection by clicking the reset button.   \n\n"
         
         "The table allows users to filter and sort columns to organize data according to taxonomic groups and metrics."),
    
    'treemap-graph-container': 
        ("The Treemap Graph provides a hierarchical view of taxonomic data.  \n\n"
         
         "Each box represents a taxonomic level or group. Visual encoding allows users to quickly identify dominant groups, their levels, and types within the dataset.  \n"
         "Darker colors indicate higher taxonomy levels (e.g., Domain), while lighter colors represent lower levels within the hierarchy (e.g., Species).  \n"
         "The size of each box reflects the coverage of that group, and the border color indicates the type (e.g., chromosome, plasmid, virus).  \n\n"),
        
    'cyto-graph-container': 
        ("The Cytoscape Graph is a network-style visualization that represents relationships between annotations, "
         "bins, or contigs, depending on the selected visualization type.  \n\n"
         
         "Selecting a node in 'Cross-Taxa Hi-C Interaction' may highlight its connected nodes, showing relationships within the network.  \n"
         "Selection made on Cytoscape Graph in 'Cross-Taxa Hi-C Interaction' can be remove by clicking on blank area.  \n\n"
         
         "Selecting a node in 'Cross-Bin Hi-C Interactions' or 'Cross-Contig Hi-C Interactions' may filter the rows being displayed in the information table.  \n\n"
         
         "Colors of nodes may vary based on element types:  \n"
         "Chromosomes are in bluish color, plasmids are in greenish color, and virus are in reddish color.  \n"
         "This color coding helps users quickly identify the biological role of each node.  \n\n"
         
         "Nodes that are densely connected may form clusters. There might be a higher contact between these nodes.  \n"
         "The geometry distribution of nodes allows users to identify groups of closely related elements. "),
    
    'contact-table-container': 
        ("The Contact Table provides a tabular view of interactions between annotations, allowing users to explore contact data in more detail.  \n\n"
         
         "Click on the row header to select an annotation.  \n"
         "Click on the column header to sort the values.  \n"
         "Click on the 'Index' header to unsort.  \n\n"
         
         "Color coding is based on the logarithmic value of contacts to make the cells with higher contacts more conspicuous.")
}

def create_visualization_layout():
    logger.info("Generating layout.")
    
    common_text_style = {
        'height': '38px',
        'width': '200px',
        'display': 'inline-block',
        'margin-right': '10px',
        'vertical-align': 'middle'
    }
    
    # Use the styling functions in the Dash layout
    return html.Div(
        children=[
            html.Div(
                id="main-controls",
                children=[
                    dcc.Store(id='data-loaded', data=False),
                    dcc.Download(id="download"),
    
                    html.Button("Download Files", id="download-btn", style={**common_text_style}),
                    html.Button("Reset Selection", id="reset-btn", style={**common_text_style}),
                    html.Div(
                        id="tooltip-toggle-container",
                        children=[
                            dcc.Checklist(
                                id='tooltip-toggle',
                                options=[{'label': '  Enable Help Tooltip', 'value': 'show-tooltip'}],
                                value=['show-tooltip'],
                                inline=True,
                            ),
                        ],
                        style={
                            'height': '38px',
                            'width': '200px',
                            'display': 'inline-block',
                            'margin-right': '10px',
                            'padding-top': '5px',
                            'vertical-align': 'middle',
                            'border': '2px solid black',
                            'borderRadius': '0px',
                            'backgroundColor': '#f8f9fa',
                            'textAlign': 'center'
                        }
                    ),
                    html.Div(
                        id="dropdowns",
                        children=[
                            dcc.Dropdown(
                                id='visualization-selector',
                                options=[
                                    {'label': 'Taxonomic Framework', 'value': 'taxonomy_hierarchy'},
                                    {'label': 'Cross-Taxa Hi-C Interaction', 'value': 'basic'},
                                    {'label': 'Cross-Bin Hi-C Interactions', 'value': 'bin'},
                                    {'label': 'Cross-Contig Hi-C Interactions', 'value': 'contig'}
                                ],
                                value='taxonomy_hierarchy',
                                style={'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
                            ),
                            dcc.Dropdown(
                                id='taxonomy-level-selector',
                                options=[],
                                value=None,
                                placeholder="Select Taxonomy Level",
                                style={}
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
                            )
                        ]
                    ),
                    html.Button("Confirm Selection", id="confirm-btn", style={**common_text_style}),
                ], 
                style={
                    'display': 'flex',
                    'justify-content': 'space-between',
                    'align-items': 'center',
                    'position': 'fixed',
                    'top': '0',
                    'left': '0',
                    'width': '100%',
                    'z-index': '1000',
                    'background-color': 'lightgrey',
                    'padding': '10px',
                    'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
                }
            ),
            html.Div(style={'height': '70px'}),  # Placeholder for header height
            
            html.Div(
                id="visualization-container",
                children=[
                    html.Div(
                        id="left-column",
                        children=[
                            html.Button(id='logger-button-visualization', n_clicks=0, style={'display': 'none'}),
                            dcc.Textarea(
                                id="log-box-visualization",
                                style={
                                    'height': '30vh',
                                    'width': '30vw',
                                    'font-size': '12px',
                                    'fontFamily': 'Arial, sans-serif',
                                    'overflowY': 'scroll', 
                                    'background-color': 'white',
                                    'padding': '5px',
                                    'border': '1px solid #ccc',
                                    'margin-top': '3px',
                                    'resize': 'none'},
                             readOnly=True
                            ),
                            dcc.Loading(
                                id="loading-spinner",
                                type="default",
                                delay_show=2000,
                                children=[
                                    html.Div(
                                        id='info-table-container',
                                        children=[
                                            dcc.Checklist(
                                                id='visibility-filter',
                                                options=[{'label': '  Only show elements present in the diagram', 'value': 'filter'}],
                                                value=['filter'],
                                                style={'display': 'inline-block', 'width': '25vw'}
                                            ),
                                            dcc.Tabs(id='table-tabs', value='bin', 
                                                children=[
                                                    dcc.Tab(label='Bin Info', value='bin', className="p-0"),
                                                    dcc.Tab(label='Contig Info', value='contig', className="p-0")
                                                ],
                                                style={"display": "none"}
                                            ),
                                            html.Div(
                                                children=[
                                                    dag.AgGrid(
                                                        id='bin-info-table',
                                                        columnDefs=[],
                                                        rowData=[],
                                                        defaultColDef={},
                                                        style={'display': 'none'},
                                                        dashGridOptions={
                                                            "pagination": True,
                                                            'paginationPageSize': 20,
                                                            'rowSelection': 'single',
                                                            'headerPinned': 'top',}
                                                    ),
                                                    dag.AgGrid(
                                                        id='contig-info-table',
                                                        columnDefs=[],
                                                        rowData=[],
                                                        defaultColDef={},
                                                        style={'display': 'none'},
                                                        dashGridOptions={
                                                            "pagination": True,
                                                            'paginationPageSize': 20,
                                                            'rowSelection': 'single',
                                                            'headerPinned': 'top',}
                                                    )
                                                ], 
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ], 
                        style={'display': 'inline-block', 'vertical-align': 'top', 'height': '85vh', 'width': '30vw'}
                    ),
                    html.Div(
                        id="middle-column",
                        children=[
                            html.Div(
                                id="treemap-graph-container",
                                children=[
                                    dcc.Graph(
                                        id='treemap-graph', 
                                        figure=go.Figure(), 
                                        config={'displayModeBar': False}, 
                                        style={'height': '85vh', 'width': '48vw', 'display': 'inline-block'}
                                    )
                                ]
                            ),
                            html.Div(
                                id="cyto-graph-container",
                                children=[
                                    cyto.Cytoscape(
                                        id='cyto-graph',
                                        elements=[],
                                        stylesheet=base_stylesheet,
                                        style={},
                                        layout={'name': 'preset'},
                                        zoom=1,
                                        userZoomingEnabled=True,
                                        wheelSensitivity=0.1
                                    ),
                                ]
                            ),
                        ], 
                        style={'display': 'inline-block', 'vertical-align': 'top', 'height': '85vh', 'width': '49vw'}
                    ),
                    html.Div(
                        id="right-column",
                        children=[
                            dcc.Textarea(
                                id='hover-info',
                                style={'height': '10vh',
                                       'width': '19vw',
                                       'font-size': '12px',
                                       'background-color': 'white',
                                       'padding': '5px',
                                       'border': '1px solid #ccc',
                                       'margin-top': '3px',
                                       'resize': 'none'},
                                readOnly=True
                            ),
                            html.Div(
                                id="bar-chart-container",
                                children=[
                                    dcc.Graph(id='bar-chart', 
                                              config={'displayModeBar': False}, 
                                              figure=go.Figure(), 
                                              style={'height': '75vh', 'width': '19vw', 'display': 'inline-block'}                                 
                                    ),
                                ]
                            ),
                        ],
                        style={'display': 'inline-block', 'vertical-align': 'top', 'height': '85vh', 'width': '19vw'}
                    ),
                ], 
                style={'display': 'inline-block', 'vertical-align': 'top', 'height': '85vh', 'width': '98vw'}
            ),
            dcc.Loading(
                id="loading-spinner",
                type="default",
                delay_show=2000,
                children=[
                    html.Div(
                        id="contact-table-container",
                        children=[
                            dag.AgGrid(
                                id='contact-table',
                                rowData=[],
                                columnDefs=[], 
                                dashGridOptions = {
                                    "rowSelection": "single"
                                }
                            )
                        ], 
                        style={'width': '98vw', 'display': 'inline-block', 'vertical-align': 'top', 'margin-top': '3px'}
                    )
                ]
            )
        ]
    )
           
def register_visualization_callbacks(app):
    @app.callback(
        [Output('taxonomy-level-selector', 'options'),
         Output('taxonomy-level-selector', 'value'),
         Output('bin-info-table', 'columnDefs'),
         Output('contig-info-table', 'columnDefs')],
        [Input('user-folder', 'data')]
    )
    def update_taxonomy_selector(user_folder):
        taxonomy_levels = load_from_redis(f'{user_folder}:taxonomy-levels')

        # Generate taxonomy options from taxonomy levels
        taxonomy_options = [{'label': level, 'value': level} for level in taxonomy_levels]
        default_taxonomy_level = taxonomy_levels[-1]  # Set the last level as the default value
    
        # Dynamically create taxonomy columns for the bin info table
        taxonomy_columns = [
            {"headerName": level, "field": level, "width": 150, "wrapHeaderText": True} for level in taxonomy_levels
        ]
    
        # Add hidden "Category" column
        taxonomy_columns.append({"headerName": "Category", "field": "Category", "hide": True})
    
        # Construct the full column definitions for the bin info table
        bin_column_defs = [
            {
                "headerName": "Index",
                "children": [
                    {"headerName": "Index", "field": "Bin index", "pinned": 'left', "width": 120}
                ]
            },
            {
                "headerName": "Taxonomy",
                "children": taxonomy_columns
            },
            {
                "headerName": "Contact Information",
                "children": [
                    {"headerName": "The number of restriction sites", "field": "The number of restriction sites", "width": 150, "wrapHeaderText": True},
                    {"headerName": "Bin Size", "field": "Contig length", "width": 150, "wrapHeaderText": True},
                    {"headerName": "Bin coverage", "field": "Contig coverage", "width": 150, "wrapHeaderText": True},
                    {"headerName": "Visibility", "field": "Visibility", "hide": True}
                ]
            }
        ]
    
        # Construct the full column definitions for the contig info table
        contig_column_defs = [
            {
                "headerName": "Index",
                "children": [
                    {"headerName": "Index", "field": "Contig index", "pinned": 'left', "width": 120}
                ]
            },
            {
                "headerName": "Taxonomy",
                "children": taxonomy_columns
            },
            {
                "headerName": "Contact Information",
                "children": [
                    {"headerName": "The number of restriction sites", "field": "The number of restriction sites", "width": 150, "wrapHeaderText": True},
                    {"headerName": "Contig length", "field": "Contig length", "width": 150, "wrapHeaderText": True},
                    {"headerName": "Contig coverage", "field": "Contig coverage", "width": 150, "wrapHeaderText": True},
                    {"headerName": "Visibility", "field": "Visibility", "hide": True}
                ]
            }
        ]
    
        return taxonomy_options, default_taxonomy_level, bin_column_defs, contig_column_defs
    
    @app.callback(
        [Output('data-loaded', 'data'),
         Output('contact-table', 'rowData'),
         Output('contact-table', 'columnDefs'),
         Output('contact-table', 'defaultColDef'),
         Output('contact-table', 'styleConditions'),
         Output('annotation-selector', 'options'),
         Output('bin-selector', 'options'),
         Output('contig-selector', 'options'),
         Output('reset-btn', 'n_clicks'),
         Output('logger-button-visualization', 'n_clicks', allow_duplicate=True)],
        [Input('taxonomy-level-selector', 'value')],
        [State('reset-btn', 'n_clicks'),
         State('user-folder', 'data')],
        prevent_initial_call=True
    )
    def update_data(taxonomy_level, reset_clicks, user_folder):
        logger.info("Updating taxonomy level data.")

        # Define Redis keys that incorporate the user_folder to avoid concurrency issues
        bin_matrix_key = f'{user_folder}:bin-dense-matrix'
        bin_info_key = f'{user_folder}:bin-information'
        contig_info_key = f'{user_folder}:contig-information'
        unique_annotations_key = f'{user_folder}:unique-annotations'
        contact_matrix_key = f'{user_folder}:contact-matrix'
        visualization_mode_key = f'{user_folder}:current-visualization-mode'
    
        # Load user-specific data from Redis
        try:
            bin_information = load_from_redis(bin_info_key)
            contig_information = load_from_redis(contig_info_key)
            bin_dense_matrix = load_from_redis(bin_matrix_key)
        except KeyError as e:
            logger.error(f"KeyError while loading data from Redis: {e}")
            raise PreventUpdate
           
        bin_information, contig_information, unique_annotations, contact_matrix = prepare_data(
            bin_information, contig_information, bin_dense_matrix, taxonomy_level
        )
        
        # Update visualization mode state
        current_visualization_mode = {
            'visualization_type': 'taxonomy_hierarchy',
            'taxonomy_level': taxonomy_level,
            'selected_annotation': None,
            'selected_bin': None,
            'selected_contig': None
        }
        
        # Create Annotation Table
        column_defs = [
            {"headerName": "Index", "field": "index", "pinned": "left", "width": 120, 
             "sortable": True, "suppressMovable": True,"unSortIcon": True, "sortingOrder": [None]}
        ] + [
            {"headerName": col, "field": col, "width": 120, "wrapHeaderText": True, "autoHeaderHeight": True, "suppressMovable": True} 
            for col in contact_matrix.columns
        ]

        row_data = contact_matrix.reset_index().to_dict('records')

        style_conditions = styling_annotation_table(row_data, bin_information, unique_annotations)
        
        default_col_def = {
            "cellStyle": {
                "textAlign": "center",
                "styleConditions": style_conditions
            }
        }
    
        # Save updated data to Redis
        save_to_redis(bin_info_key, bin_information)
        save_to_redis(contig_info_key, contig_information)
        save_to_redis(unique_annotations_key, unique_annotations)
        save_to_redis(contact_matrix_key, contact_matrix)
        save_to_redis(visualization_mode_key, current_visualization_mode)
        
        annotation_options = [{'label': annotation, 'value': annotation} for annotation in unique_annotations]
        
        # Initialize empty lists for dropdown options
        bin_options = []
        contig_options = []
        
        # Determine which dataset to use based on selected_tab
        bin_information = load_from_redis(f'{user_folder}:bin-information')
        bins = bin_information['Bin index']
        bin_options = [{'label': bin, 'value': bin} for bin in bins]
    
        contig_information = load_from_redis(f'{user_folder}:contig-information')
        contigs = contig_information['Contig index']
        contig_options = [{'label': contig, 'value': contig} for contig in contigs]
                
        reset_clicks = (reset_clicks or 0) + 1
                    
        return 1, row_data, column_defs, default_col_def, style_conditions, annotation_options, bin_options, contig_options,reset_clicks, 1

    @app.callback(
        [Output('bin-info-table', 'rowData'),
         Output('bin-info-table', 'style'),
         Output('bin-info-table', 'filterModel'),
         Output('contig-info-table', 'rowData'),
         Output('contig-info-table', 'style'),
         Output('contig-info-table', 'filterModel'),
         Output('logger-button-visualization', 'n_clicks', allow_duplicate=True)],
        [Input('reset-btn', 'n_clicks'),
         Input('confirm-btn', 'n_clicks'),
         Input('table-tabs', 'value'),
         Input('annotation-selector', 'value'),
         Input('visibility-filter', 'value')],
        [State('cyto-graph', 'elements'),
         State('taxonomy-level-selector', 'value'),
         State('user-folder', 'data'),
         State('data-loaded', 'data')]
    )
    def display_info_table(reset_clicks, confirm_clicks, selected_tab, selected_annotation, filter_value, cyto_elements, taxonomy_level, user_folder, data_loaded):
        if not data_loaded:
            raise PreventUpdate
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == 'reset-btn':  
            selected_annotation = None
            
        current_visualization_mode = load_from_redis(f'{user_folder}:current-visualization-mode')
        
        # Default styles to hide tables initially
        bin_style = {'display': 'none'}
        contig_style = {'display': 'none'}
        
        # Apply filtering logic for the selected table and annotation
        def apply_filter_logic(row_data, dense_matrix):
            # Filter out rows containing 'Unclassified' in the taxonomy level
            row_data = [
                row for row in row_data
                if 'Unclassified' not in row[taxonomy_level]
            ]

            # Apply visibility filter if checkbox is selected
            filter_model = {}
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
            
            if triggered_id == 'reset-btn': 
                return filter_model, row_data
            
            else:
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
                                "type": "equals",
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
                        selected_bin_index = bin_information[bin_information['Bin index'] == current_visualization_mode['selected_bin']].index[0]
                
                        connected_bins = set()
                        for j in range(dense_matrix.shape[0]):
                            if dense_matrix[selected_bin_index, j] != 0:
                                connected_bins.add(bin_information.at[j, 'Bin index'])
                
                        for row in row_data:
                            if row['Bin index'] not in connected_bins and row['Bin index'] != current_visualization_mode['selected_bin']:
                                row['Visibility'] = 0
                elif selected_tab == 'contig':
                    if current_visualization_mode['selected_contig']:
                        selected_contig_index = contig_information[contig_information['Contig index'] == current_visualization_mode['selected_contig']].index[0]
                
                        connected_contigs = set()
                        for j in range(dense_matrix.shape[0]):
                            if dense_matrix[selected_contig_index, j] != 0:
                                connected_contigs.add(contig_information.at[j, 'Contig index'])
                
                        for row in row_data:
                            if row['Contig index'] not in connected_contigs and row['Contig index'] != current_visualization_mode['selected_contig']:
                                row['Visibility'] = 0
                        
            return filter_model, row_data
    
        # Update based on the selected tab and apply filters
        if selected_tab == 'bin':
            bin_dense_matrix = load_from_redis(f'{user_folder}:bin-dense-matrix')
            bin_information = load_from_redis(f'{user_folder}:bin-information')

            bin_style = {'display': 'block', 'height': '47vh'}
            contig_style = {'display': 'none'}
        
            # Apply filter logic to all rows
            bin_filter_model, edited_bin_data = apply_filter_logic(bin_information.to_dict('records'), bin_dense_matrix)
            return (
                edited_bin_data, bin_style, bin_filter_model,
                [], contig_style, {}, 1
            )
            
        elif selected_tab == 'contig':
            contig_dense_matrix = load_from_redis(f'{user_folder}:contig-dense-matrix')
            contig_information = load_from_redis(f'{user_folder}:contig-information')

            bin_style = {'display': 'none'}
            contig_style = {'display': 'block', 'height': '47vh'}
        
            # Apply filter logic to all rows
            contig_filter_model, edited_contig_data = apply_filter_logic(contig_information.to_dict('records'), contig_dense_matrix)
            return (
                [], bin_style, {}, # Bin table remains empty
                edited_contig_data, contig_style, contig_filter_model, 1
            )
        
    @app.callback(
        [Output('bin-info-table', 'defaultColDef'),
         Output('contig-info-table', 'defaultColDef')],
        [Input('bin-info-table', 'virtualRowData'),
         Input('contig-info-table', 'virtualRowData')],
        [State('cyto-graph', 'elements'),
         State('taxonomy-level-selector', 'value'),
         State('user-folder', 'data'),
         State('data-loaded', 'data')],
        prevent_initial_call=True
    )
    def update_table_styles(bin_virtual_row_data, contig_virtual_row_data, cyto_elements, taxonomy_level, user_folder, data_loaded):
        if not data_loaded:
            raise PreventUpdate
        unique_annotations = load_from_redis(f'{user_folder}:unique-annotations')
        taxonomy_level_list = load_from_redis(f'{user_folder}:taxonomy-levels')
        
        # Bin Table Styling
        if bin_virtual_row_data:
            bin_row_data_df = pd.DataFrame(bin_virtual_row_data)
            bin_colors = get_id_colors(cyto_elements)
            bin_style_conditions = styling_information_table(
                bin_row_data_df, bin_colors, unique_annotations, 'bin', taxonomy_level_list, taxonomy_level
            )
        
            bin_col_def = {
                "sortable": True,
                "filter": True,
                "resizable": True,
                "cellStyle": {
                    "styleConditions": bin_style_conditions
                }
            }
        else:
            bin_col_def = {"sortable": True, "filter": True, "resizable": True}
        
        # Contig Table Styling
        if contig_virtual_row_data:
            contig_row_data_df = pd.DataFrame(contig_virtual_row_data)
            contig_colors = get_id_colors(cyto_elements)
            contig_style_conditions = styling_information_table(
                contig_row_data_df, contig_colors, unique_annotations, 'contig', taxonomy_level_list, taxonomy_level
            )
        
            contig_col_def = {
                "sortable": True,
                "filter": True,
                "resizable": True,
                "cellStyle": {
                    "styleConditions": contig_style_conditions
                }
            }
        else:
            contig_col_def = {"sortable": True, "filter": True, "resizable": True}
        return bin_col_def, contig_col_def

    @app.callback(
        [Output('visualization-selector', 'value'),
         Output('taxonomy-level-selector', 'style'),
         Output('annotation-selector', 'value'),
         Output('annotation-selector', 'style'),
         Output('bin-selector', 'value'),
         Output('bin-selector', 'style'),
         Output('contig-selector', 'value'),
         Output('contig-selector', 'style'),
         Output('table-tabs', 'value'),
         Output('logger-button-visualization', 'n_clicks', allow_duplicate=True)],
        [Input('reset-btn', 'n_clicks'),
         Input('visualization-selector', 'value'),
         Input('contact-table', 'selectedRows'),
         Input('bin-info-table', 'selectedRows'),
         Input('contig-info-table', 'selectedRows'),
         Input('cyto-graph', 'selectedNodeData')],
        [State('table-tabs', 'value'),
         State('taxonomy-level-selector', 'value'),
         State('user-folder', 'data'),
         State('data-loaded', 'data')]
    )
    def sync_selectors(reset_clicks, visualization_type, contact_table_selected_rows, bin_info_selected_rows, contig_info_selected_rows, selected_node_data, current_tab, taxonomy_level, user_folder, data_loaded):      
        if not data_loaded:
            raise PreventUpdate
            
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
        # Initialize the styles as hidden
        annotation_selector_style = {'display': 'none'}
        bin_selector_style = {'display': 'none'}
        contig_selector_style = {'display': 'none'}
        taxonomy_selector_style = {'display': 'none'}
        tab_value = current_tab
    
        # Reset selections if reset button is clicked
        if triggered_id == 'reset-btn':
                visualization_type = 'basic'
                selected_annotation = None
                selected_bin = None
                selected_contig = None
                tab_value = 'bin'
                taxonomy_selector_style = {'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
                annotation_selector_style = {'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
        else:
            bin_information = load_from_redis(f'{user_folder}:bin-information')
            contig_information = load_from_redis(f'{user_folder}:contig-information')
            contact_matrix = load_from_redis(f'{user_folder}:contact-matrix')

            selected_annotation, selected_bin, selected_contig = synchronize_selections(
                triggered_id, selected_node_data, bin_info_selected_rows, contig_info_selected_rows, contact_table_selected_rows, 
                taxonomy_level, current_tab, bin_information, contig_information, contact_matrix
            )

            # Update styles, tab, and visualization type based on selections
            if selected_bin or visualization_type == 'bin':
                visualization_type = 'bin'
                taxonomy_selector_style = {'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
                bin_selector_style = {'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
                tab_value = 'bin'  # Switch to bin tab
        
            elif selected_contig or visualization_type == 'contig':
                visualization_type = 'contig'
                taxonomy_selector_style = {'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
                contig_selector_style = {'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
                tab_value = 'contig'  # Switch to contig tab
                
            elif selected_annotation or visualization_type == 'basic':
                visualization_type = 'basic'
                taxonomy_selector_style = {'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
                annotation_selector_style = {'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}

            else:
                visualization_type = 'taxonomy_hierarchy'


        return (visualization_type, taxonomy_selector_style, selected_annotation, annotation_selector_style, 
                selected_bin, bin_selector_style, selected_contig, contig_selector_style, tab_value, 1)
               
    def synchronize_selections(
            triggered_id, selected_node_data, bin_info_selected_rows, contig_info_selected_rows, contact_table_selected_rows, 
            taxonomy_level, table_tab_value, bin_information, contig_information, contact_matrix):
        selected_annotation = None
        selected_bin = None
        selected_contig = None
    
        # Node selected in the network
        if triggered_id == 'cyto-graph' and selected_node_data:
            selected_node_id = selected_node_data[0]['id']
            
            # Check if the node exists in both bin and contig information
            is_in_bin = selected_node_id in bin_information['Bin index'].values
            is_in_contig = selected_node_id in contig_information['Contig index'].values
    
            # Decide based on the active tab
            if is_in_bin and is_in_contig:
                if table_tab_value == 'bin':
                    # Use bin info if the bin tab is active
                    bin_info = bin_information[bin_information['Bin index'] == selected_node_id].iloc[0]
                    selected_annotation = bin_info['Annotation']
                    selected_bin = bin_info['Bin index']
                elif table_tab_value == 'contig':
                    # Use contig info if the contig tab is active
                    contig_info = contig_information[contig_information['Contig index'] == selected_node_id].iloc[0]
                    selected_annotation = contig_info['Annotation']
                    selected_contig = contig_info['Contig index']
            elif is_in_bin:
                # If only in bin, select bin info
                bin_info = bin_information[bin_information['Bin index'] == selected_node_id].iloc[0]
                selected_annotation = bin_info['Annotation']
                selected_bin = bin_info['Bin index']
            elif is_in_contig:
                # If only in contig, select contig info
                contig_info = contig_information[contig_information['Contig index'] == selected_node_id].iloc[0]
                selected_annotation = contig_info['Annotation']
                selected_contig = contig_info['Contig index']
            else:
                # Default to treating it as an annotation
                selected_annotation = selected_node_id
    
        # Row selected in bin-info-table
        elif triggered_id == 'bin-info-table' and bin_info_selected_rows:
            selected_row = bin_info_selected_rows[0]
            if taxonomy_level in selected_row and 'Bin index' in selected_row:
                selected_annotation = selected_row[taxonomy_level]
                selected_bin = selected_row['Bin index']
    
        # Row selected in contig-info-table
        elif triggered_id == 'contig-info-table' and contig_info_selected_rows:
            selected_row = contig_info_selected_rows[0]
            if taxonomy_level in selected_row and 'Contig index' in selected_row:
                selected_annotation = selected_row[taxonomy_level]
                selected_contig = selected_row['Contig index']
    
        # Cell selected in contact-table
        elif triggered_id == 'contact-table' and contact_table_selected_rows:
            selected_row = contact_table_selected_rows[0]
            selected_annotation = selected_row['index']
    
        logger.info(f'Current selected: Annotation={selected_annotation}, Bin={selected_bin}, Contig={selected_contig}')
        return selected_annotation, selected_bin, selected_contig
    
    # Callback to update the visualizationI want to 
    @app.callback(
        [Output('cyto-graph', 'elements'),
         Output('cyto-graph', 'style'),
         Output('bar-chart', 'figure'),
         Output('treemap-graph', 'figure'),
         Output('treemap-graph', 'style'),
         Output('logger-button-visualization', 'n_clicks', allow_duplicate=True)],
        [Input('reset-btn', 'n_clicks'),
         Input('confirm-btn', 'n_clicks')],
        [State('visualization-selector', 'value'),
         State('annotation-selector', 'value'),
         State('bin-selector', 'value'),
         State('contig-selector', 'value'),
         State('table-tabs', 'value'),
         State('user-folder', 'data'),
         State('data-loaded', 'data')],
         prevent_initial_call=True
    )
    def update_visualization(reset_clicks, confirm_clicks, visualization_type, selected_annotation, selected_bin, selected_contig, selected_tab, user_folder, data_loaded):
        if not data_loaded:
            raise PreventUpdate

        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        logger.info(f"Updating visulization. Triggered by: {triggered_id}")
        logger.info(f"Visualization type: {visualization_type}")
        logger.info(f"Selected annotation: {selected_annotation}")
        logger.info(f"Selected bin: {selected_bin}")
        logger.info(f"Selected contig: {selected_contig}")
        
        # Load user-specific data from Redis
        unique_annotations = load_from_redis(f'{user_folder}:unique-annotations')
        current_visualization_mode = load_from_redis(f'{user_folder}:current-visualization-mode')
        taxonomy_level_list = load_from_redis(f'{user_folder}:taxonomy-levels')
    
        if triggered_id == 'reset-btn':
            contact_matrix = load_from_redis(f'{user_folder}:contact-matrix')
            bin_information = load_from_redis(f'{user_folder}:bin-information')

            if reset_clicks == 1:
                logger.info("Visualization initiated, displaying taxonomy framework visualization.")
                current_visualization_mode = {
                    'visualization_type': 'taxonomy_hierarchy',
                    'selected_annotation': None,
                    'selected_bin': None,
                    'selected_contig': None
                }
                treemap_fig, bar_fig = taxonomy_visualization(bin_information, unique_annotations, contact_matrix, taxonomy_level_list)
                treemap_style = {'height': '83vh', 'width': '48vw', 'display': 'inline-block'}
                cyto_elements = []
                cyto_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            else:    
                logger.info("Reset button clicked, switching to Taxonomy Interaction visualization.")
                current_visualization_mode = {
                    'visualization_type': 'basic',
                    'selected_annotation': None,
                    'selected_bin': None,
                    'selected_contig': None
                }
                cyto_elements, bar_fig, _ = annotation_visualization(bin_information, unique_annotations, contact_matrix)
                treemap_fig = go.Figure()
                treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
                cyto_style = {'height': '85vh', 'width': '48vw', 'display': 'inline-block'}
    
        else:
            current_visualization_mode.update({
                'visualization_type': visualization_type,
                'selected_annotation': selected_annotation,
                'selected_bin': selected_bin,
                'selected_contig': selected_contig
            })
            save_to_redis(f'{user_folder}:current-visualization-mode', current_visualization_mode)
    
            if visualization_type == 'taxonomy_hierarchy':
                contact_matrix = load_from_redis(f'{user_folder}:contact-matrix')
                bin_information = load_from_redis(f'{user_folder}:bin-information')

                logger.info("Displaying Taxonomy Framework visualization.")
                treemap_fig, bar_fig = taxonomy_visualization(bin_information, unique_annotations, contact_matrix, taxonomy_level_list)
                treemap_style = {'height': '85vh', 'width': '48vw', 'display': 'inline-block'}
                cyto_elements = []
                cyto_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
                
            elif visualization_type == 'basic' or (selected_annotation and not selected_bin and not selected_contig):
                contact_matrix = load_from_redis(f'{user_folder}:contact-matrix')
                bin_information = load_from_redis(f'{user_folder}:bin-information')

                logger.info("Displaying Taxonomy Interaction.")
                cyto_elements, bar_fig, _ = annotation_visualization(bin_information, unique_annotations, contact_matrix)
                treemap_fig = go.Figure()
                treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
                cyto_style = {'height': '85vh', 'width': '48vw', 'display': 'inline-block'}
    
            elif visualization_type == 'bin' and selected_bin:
                bin_dense_matrix = load_from_redis(f'{user_folder}:bin-dense-matrix')
                bin_information = load_from_redis(f'{user_folder}:bin-information')

                logger.info(f"Displaying bin Interaction for selected bin: {selected_bin}.")
                cyto_elements, bar_fig = bin_visualization(selected_annotation, selected_bin, bin_information, bin_dense_matrix, unique_annotations)
                treemap_fig = go.Figure()
                treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
                cyto_style = {'height': '85vh', 'width': '48vw', 'display': 'inline-block'}
    
            elif visualization_type == 'contig' and selected_contig:
                contig_dense_matrix = load_from_redis(f'{user_folder}:contig-dense-matrix')
                contig_information = load_from_redis(f'{user_folder}:contig-information')

                logger.info(f"Displaying contig Interaction for selected contig: {selected_contig}.")
                cyto_elements, bar_fig = contig_visualization(selected_annotation, selected_contig, contig_information, contig_dense_matrix, unique_annotations)
                treemap_fig = go.Figure()
                treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
                cyto_style = {'height': '85vh', 'width': '48vw', 'display': 'inline-block'}
                
        save_to_redis(f'{user_folder}:current-visualization-mode', current_visualization_mode)
        return cyto_elements, cyto_style, bar_fig, treemap_fig, treemap_style, 1
    
    @app.callback(
        [Output('cyto-graph', 'elements', allow_duplicate=True),
         Output('cyto-graph', 'stylesheet'),
         Output('hover-info', 'value'),
         Output('cyto-graph', 'layout'),
         Output('logger-button-visualization', 'n_clicks', allow_duplicate=True)],
        [Input('annotation-selector', 'value'),
         Input('bin-selector', 'value'),
         Input('contig-selector', 'value')],
        [State('user-folder', 'data'),
        State('data-loaded', 'data')],
        prevent_initial_call=True
    )
    def update_selected_styles(selected_annotation, selected_bin, selected_contig, user_folder, data_loaded):
        if not data_loaded:
            raise PreventUpdate

        current_visualization_mode = load_from_redis(f'{user_folder}:current-visualization-mode')    
            
        selected_nodes = []
        selected_edges = []
        hover_info = "No selection"
        cyto_elements = dash.no_update
        cyto_style = dash.no_update
    
        # Determine which selection to use based on the active tab
        if  selected_bin:
            bin_information = load_from_redis(f'{user_folder}:bin-information')
            selected_nodes.append(selected_bin)
            bin_info = bin_information[bin_information['Bin index'] == selected_bin].iloc[0]
            hover_info = f"Bin: {selected_bin}  \nTaxonomic annotation: {bin_info['Annotation']}"
            # No need to update cyto_elements or layout if a bin is selected
    
        elif selected_contig:
            contig_information = load_from_redis(f'{user_folder}:contig-information')
            selected_nodes.append(selected_contig)
            contig_info = contig_information[contig_information['Contig index'] == selected_contig].iloc[0]
            hover_info = f"Taxonomic annotation: {contig_info['Annotation']}  \nContig: {selected_contig}"
            # No need to update cyto_elements or layout if a contig is selected
    
        if current_visualization_mode['visualization_type'] == 'basic':
            bin_information = load_from_redis(f'{user_folder}:bin-information')
            unique_annotations = load_from_redis(f'{user_folder}:unique-annotations')
            contact_matrix = load_from_redis(f'{user_folder}:contact-matrix')
            selected_nodes.append(selected_annotation)
            hover_info = f"Taxonomic annotation: {selected_annotation}"   
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
        elif current_visualization_mode['visualization_type'] == 'taxonomy_hierarchy':
            hover_info = ""
    
        # Add selection styles for the selected nodes and edges
        stylesheet = add_selection_styles(selected_nodes, selected_edges)
        
        return cyto_elements, stylesheet, hover_info, cyto_style, 1
    
    @app.callback(
        Output('log-box-visualization', 'value'),
        Input('logger-button-visualization', 'n_clicks'),
        State('user-folder', 'data'),
        prevent_initial_call=True
    )
    def update_log_box(n_intervals, session_id):
        from app import r
        redis_key = f"{session_id}:log"
        logs = r.get(redis_key)
        
        if logs:
            logs = json.loads(logs.decode())
            return "\n".join(logs)
        
        return "No logs yet."
    
    app.clientside_callback(
        """
        function(n_intervals) {
            const logBox = document.getElementById('log-box-visualization');
            if (logBox) {
                logBox.scrollTop = logBox.scrollHeight;  // Scroll to the bottom
            }
            return null;  // No output needed
        }
        """,
        Output('log-box-visualization', 'value', allow_duplicate=True),  # Dummy output to trigger the callback
        Input('logger-button-visualization', 'n_clicks')
    )
    
    @app.callback(
        [Output("download", "data"),
         Output('logger-button-visualization', 'n_clicks', allow_duplicate=True)],
        [Input("download-btn", "n_clicks")],
        [State("user-folder", "data")]
    )
    def download_user_folder(n_clicks, user_folder):
        if not n_clicks:
            raise PreventUpdate
    
        # Path to the user folder
        folder_path = f"output/{user_folder}"
    
        # Create a 7z archive in memory
        memory_file = io.BytesIO()
        with py7zr.SevenZipFile(memory_file, 'w') as archive:
            # Add files to the archive
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    archive.write(file_path, arcname=os.path.relpath(file_path, folder_path))
        memory_file.seek(0)
    
        # Return the 7z file to download
        return dcc.send_bytes(
            memory_file.getvalue(),
            filename=f"{user_folder}.7z"
        ), 1
    
    @app.callback(
        [Output('download-btn', 'title'),
         Output('reset-btn', 'title'),
         Output('confirm-btn', 'title'),
         Output('tooltip-toggle-container', 'title'),
         Output('dropdowns', 'title'),
         Output('bar-chart-container', 'title'),
         Output('info-table-container', 'title'),
         Output('treemap-graph-container', 'title'),
         Output('cyto-graph-container', 'title'),
         Output('contact-table-container', 'title')],
        [Input('tooltip-toggle', 'value')]
    )
    def update_tooltips(show_tooltip):
        if 'show-tooltip' in show_tooltip:
            return (hover_info['download-btn'], 
                    hover_info['reset-btn'], 
                    hover_info['confirm-btn'], 
                    hover_info['tooltip-toggle-container'],
                    hover_info['dropdowns'],
                    hover_info['bar-chart-container'],
                    hover_info['info-table-container'], 
                    hover_info['treemap-graph-container'],
                    hover_info['cyto-graph-container'],
                    hover_info['contact-table-container'])
        else:
            return ("", "", "", "", "", "", "", "", "", "")