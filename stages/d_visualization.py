import numpy as np
import pandas as pd
import networkx as nx
from dash.exceptions import PreventUpdate
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_ag_grid as dag
import plotly.graph_objects as go
from dash import callback_context
import plotly.express as px
import math 
from math import sqrt, sin, cos
import time
import logging
import json
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
        if edge[0] in G.nodes and edge[1] in G.nodes:
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
        "condition": "params.colDef.field == 'index' && !params.data.index.endsWith('_v') && !params.data.index.endsWith('_p')",  # Default case for chromosome
        "style": {
            "fontStyle": "italic",
            "backgroundColor": add_opacity_to_color('#81BFDA', opacity),  # Chromosome color
            "color": "black"
        }
    })

    return styles

def styling_information_table(information_data, id_colors, unique_annotations, taxonomy_columns=None, taxonomy_level=None):
    if taxonomy_level is None:
        taxonomy_level = taxonomy_columns[-1]
    columns = ['The number of restriction sites', 'Contig length', 'Contig coverage', 'Connected bins']
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
    id_column = 'Bin index'
    
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
        if row[taxonomy_level] in unique_annotations:
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
def arrange_nodes(bins, distance, center_position):
    distance /= 100 
    phi = (1 + sqrt(5)) / 2
    pi = math.pi

    # Arrange inner bins in a sunflower pattern
    inner_positions = {}
    angle_stride = 2 * pi / phi ** 2

    max_inner_radius = 0  # To keep track of the maximum radius used for inner nodes

    for k, bin in enumerate(bins, start=1):
        r = distance * sqrt(k)
        theta = k * angle_stride
        x = center_position[0] + r * cos(theta)
        y = center_position[1] + r * sin(theta)
        inner_positions[bin] = (x, y)
        if r > max_inner_radius:
            max_inner_radius = r

    return {**inner_positions}

def create_legend_html(id_colors):
    legend_items = []
    for node_id, color in id_colors.items():
        legend_items.append(
            html.Div(
                style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'marginBottom': '5px'
                },
                children=[
                    html.Div(
                        style={
                            'width': '20px',
                            'height': '20px',
                            'backgroundColor': color,
                            'marginRight': '10px',
                            'border': '1px solid #000'
                        }
                    ),
                    html.Span(node_id)
                ]
            )
        )
    return html.Div(legend_items, 
                    style={'width': '19vw', 'height': '25vh', 'overflowY': 'scroll',
                           'margin-top': '5px', 'margin-bottom': '5px',
                           'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px'})

def create_bar_chart(data_dict, taxonomy_level=[]):
    # Default dropdown and figure for empty or invalid input
    dropdown = dcc.Dropdown(
        id='bar-chart-dropdown',
        style={'width': '100%', 'margin-top': '5px', 'margin-bottom': '5px',}
    )
    
    figure = dcc.Graph(
        id='bar-chart', 
        figure=go.Figure(),
        style={'width': '19vw', 'height': '50vh', 'padding': '10px',
               'margin-top': '5px', 'margin-bottom': '5px',
               'border': '1px solid #ccc', 'borderRadius': '5px'}
    )

    if not data_dict:
        logger.warning("data_dict is empty, returning empty dropdown and figure.")
        return [dropdown, figure]
    
    traces = []
    options = []
    
    # Iterate over the input data dictionary to create traces and dropdown options
    for idx, (trace_name, data_frame) in enumerate(data_dict.items()):
        if data_frame.empty or 'value' not in data_frame.columns:
            logger.warning(f"No data for {trace_name}, skipping trace.")
            continue

        # Sort data based on the trace name and provided taxonomy levels
        if trace_name == "Fraction of classified bins by ranks":
            taxonomy_order = {taxonomy: idx for idx, taxonomy in enumerate(taxonomy_level)}
            data_frame['taxonomy_rank'] = data_frame['name'].apply(lambda x: taxonomy_order.get(x, float('inf')))
            bar_data = data_frame.sort_values(by=['taxonomy_rank', 'value'], ascending=[True, False])
        else:
            bar_data = data_frame.sort_values(by='value', ascending=False)
    
            # Create a bar trace
            bar_trace = go.Bar(
                x=bar_data['name'],
                y=bar_data['value'],
                name=trace_name,
                marker_color=bar_data['color'],
                visible=(idx == 0),  # Show only the first trace by default
                hovertext=bar_data.get('hover', None),
                hoverinfo='text' if 'hover' in bar_data.columns else None
            )
            traces.append(bar_trace)
    
            # Add dropdown options for this trace
            options.append({'label': trace_name, 'value': trace_name})

    if not traces:
        logger.warning("No valid traces created, returning empty figure.")
        return [dropdown, figure]
    
    dropdown = dcc.Dropdown(
        id='bar-chart-dropdown',
        options=options,
        value=options[0]['value'],
        style={'width': '100%', 'margin-top': '5px', 'margin-bottom': '5px',}
    )

    # Create the bar chart layout
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
    )
    
    figure = dcc.Graph(
        id='bar-chart', 
        figure=go.Figure(data=traces, layout=bar_layout),
        style={'width': '19vw', 'height': '50vh', 'padding': '10px',
               'margin-top': '5px', 'margin-bottom': '5px',
               'border': '1px solid #ccc', 'borderRadius': '5px'}
    )
    
    return [dropdown, figure]

def taxonomy_visualization(bin_information, unique_annotations, contact_matrix, taxonomy_columns):
    data_dict = {}
    taxonomy_columns = taxonomy_columns.tolist() if isinstance(taxonomy_columns, np.ndarray) else taxonomy_columns

    # Check input validity
    if taxonomy_columns is None or len(taxonomy_columns) == 0:
        logger.error("taxonomy_columns is empty or not provided.")
        return go.Figure(), create_bar_chart(data_dict)

    # Dynamically map taxonomy levels
    level_mapping = {"Microbial Community": 1, "Category": 2}  # Assign Community and Category fixed levels
    level_mapping.update({level: idx + 3 for idx, level in enumerate(taxonomy_columns)})  # Taxonomy levels start at 3

    records = []
    existing_annotations = set()

    # Add "Microbial Community" as the root node
    records.append({
        "annotation": "Microbial Community",
        "parent": "",
        "level": level_mapping["Microbial Community"],
        "level_name": "Microbial Community",
        "type": "Microbial Community",
        "total coverage": 0,
        "border_color": "black",
        "bin": bin_information['Bin index'].unique()
    })
    existing_annotations.add("Microbial Community")

    categories = bin_information['Category'].unique()
    for category in categories:
        records.append({
            "annotation": category,
            "parent": "Microbial Community",
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
            if len(annotation) == 2:
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

    
    hierarchy_df = pd.DataFrame(records)
    hierarchy_df['scaled_coverage'] = generate_gradient_values(hierarchy_df['total coverage'], 10, 30)
    hierarchy_df.loc[hierarchy_df['type'] == 'virus', 'scaled_coverage'] *= 20

    def format_bins(bin_list, max_bins=5):
        if len(bin_list) > max_bins:
            return ', '.join(bin_list[:max_bins]) + f"... (+{len(bin_list) - max_bins} more)"
        return ', '.join(bin_list)

    hierarchy_df['limited_bins'] = hierarchy_df['bin'].apply(lambda bins: format_bins(bins))

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

    # Generate classification bar chart
    classification_data = []
    for level in taxonomy_columns:
        total_count = len(bin_information[level])
        unclassified_count = bin_information[level].apply(lambda x: len(str(x)) == 2).sum()
        classified_count = total_count - unclassified_count
        classified_ratio = (classified_count / total_count) * 100 if total_count > 0 else 0

        classification_data.append({
            "name": level,
            "value": classified_ratio,
            "color": "gray"
        })

    classification_data_df = pd.DataFrame(classification_data)
    

    data_dict = {
        'Fraction of Classified Bins by Taxonmic Ranks': classification_data_df
    }
    
    if classification_data:
        classification_data_df = pd.DataFrame(classification_data)
        data_dict['Fraction of Classified Bins by Taxonmic Ranks'] = classification_data_df

    bar_fig = create_bar_chart(data_dict, taxonomy_columns)

    return fig, bar_fig

#Function to visualize annotation relationship
def annotation_visualization(bin_information, unique_annotations, contact_matrix, taxonomy_level, selected_node=None):
    data_dict = {}
    G = nx.Graph()
    
    if selected_node and len(selected_node) == 2:
        logger.warning(f"Selected node '{selected_node}' is not classified.")
        return [], create_bar_chart(data_dict), {}
    
    # Filter out nodes with names of length 2
    filtered_unique_annotations = [annotation for annotation in unique_annotations if len(annotation) != 2]
    
    if selected_node:
        connected_nodes = [selected_node]
    
        # Only iterate over nodes connected to the selected node (i == selected_index)
        for annotation_j in filtered_unique_annotations:
            if annotation_j != selected_node:  # Skip the selected node itself
                weight = contact_matrix.at[selected_node, annotation_j]
                if weight > 0:
                    connected_nodes.append(annotation_j)

    else:
        connected_nodes = filtered_unique_annotations
        
    node_colors = {}
    for annotation in connected_nodes:
        bin_type = bin_information.loc[
            bin_information[taxonomy_level] == annotation, 'Category'
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
        total_bin_coverage_sum = bin_information.groupby(taxonomy_level).apply(
            lambda group: group['Contig coverage'].mul(group['Contig length']).sum() / 
                          group['The number of restriction sites'].sum()
        ).reindex(unique_annotations).fillna(0).values
        
        # For 'Across Taxonomy Hi-C Contacts' DataFrame
        df_contacts = pd.DataFrame({'name': unique_annotations, 
                                    'value': inter_annotation_contact_sum, 
                                    'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]
                                   }).query('value != 0')
        
        # Filter out rows where 'name' length is 2
        df_contacts = df_contacts[df_contacts['name'].str.len() != 2]
        
        if not df_contacts.empty:
            data_dict['Across Taxonomy Hi-C Contacts'] = df_contacts
        
        # For 'The coverage of different taxa' DataFrame
        df_coverage = pd.DataFrame({'name': unique_annotations, 
                                     'value': total_bin_coverage_sum, 
                                     'color': [node_colors.get(annotation, 'rgba(0,128,0,0.8)') for annotation in unique_annotations]
                                    }).query('value != 0')
        
        # Filter out rows where 'name' length is 2
        df_coverage = df_coverage[df_coverage['name'].str.len() != 2]
        
        if not df_coverage.empty:
            data_dict[f'The Coverage of Different {taxonomy_level}s'] = df_coverage
    
        bar_fig = create_bar_chart(data_dict)
        return cyto_elements, bar_fig, cyto_style
    
    else:
        selected_node_contacts = []
        for annotation in connected_nodes:
            if annotation != selected_node:
                weight = contact_matrix.at[selected_node, annotation]
                if weight > 0:
                    selected_node_contacts.append({'name': annotation, 'value': weight, 
                                                   'color': node_colors.get(annotation, 'rgba(0,128,0,0.8)')})
        
        if selected_node_contacts:
            df_selected_contacts = pd.DataFrame(selected_node_contacts)
            data_dict[f'Contacts with {selected_node}'] = df_selected_contacts
            
        return cyto_elements, create_bar_chart(data_dict) , cyto_style

# Function to visualize bin relationships
def bin_visualization(bin_information, unique_annotations, bin_dense_matrix, taxonomy_level, selected_bin):
    data_dict = {}
    
    selected_bin_index = bin_information[bin_information['Bin index'] == selected_bin].index[0]
    selected_annotation = bin_information.loc[selected_bin_index, taxonomy_level]

    # Get all indices that have contact with the selected bin
    contacts_indices = bin_dense_matrix[selected_bin_index].nonzero()[0]

    # If no contacts found, raise a warning
    if len(contacts_indices) == 0:
        logger.warning(f'No contacts found for the selected bin: {selected_bin}')
        return [], create_bar_chart(data_dict)
    else:
        original_contacts_annotation = bin_information.loc[contacts_indices, taxonomy_level]
        contacts_bins = bin_information.loc[contacts_indices, 'Bin index']

    G = nx.Graph()

    # Use a categorical color scale
    color_scale_mapping = {
        'virus': reds,
        'plasmid': greens,
        'chromosome': blues
    }

    color_index = 0
    if selected_annotation not in original_contacts_annotation.unique():
        contacts_annotation = pd.concat([original_contacts_annotation, pd.Series(selected_annotation)])
    else:
        contacts_annotation = original_contacts_annotation
        
    # Add annotation nodes
    for annotation in contacts_annotation.unique():
        annotation_type = bin_information.loc[bin_information[taxonomy_level] == annotation, 'Category'].values[0]
        color_scale = color_scale_mapping.get(annotation_type, [default_color])  
        color = color_scale[color_index % len(color_scale)]
        color_index += 1
        G.add_node(annotation, size=1, color='#FFFFFF', border_color=color)

    # Collect all contact values between selected_annotation and other annotations
    contact_values = []
    
    for annotation in contacts_annotation.unique():
        annotation_bins = bin_information[bin_information[taxonomy_level] == annotation].index
        annotation_bins = annotation_bins[annotation_bins < bin_dense_matrix.shape[1]]
        contact_value = bin_dense_matrix[selected_bin_index, annotation_bins].sum()
        
        if contact_value > 0:
            contact_values.append(contact_value)
            if G.has_node(annotation):
                G.add_edge(selected_annotation, annotation, weight=1)
        elif annotation == selected_annotation:
            contact_values.append(0)
                
    filtered_contact_values = list(filter(None, contact_values))
    scaled_weights = generate_gradient_values(filtered_contact_values, 1, 2) if contact_values else [1] * len(contacts_annotation.unique())
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['weight'] = scaled_weights[i]
        
    fixed_positions = {selected_annotation: (0, 0)}
    pos = nx.spring_layout(G, iterations=200, k=2, fixed=[selected_annotation], pos=fixed_positions, weight='weight')
    
    # Remove the edges after positioning
    G.remove_edges_from(list(G.edges()))

    # Handle selected_bin node
    selected_bin_type = bin_information.loc[selected_bin_index, 'Category']
    selected_bin_color = type_colors.get(selected_bin_type, default_color)
    G.add_node(selected_bin, size=15, color=selected_bin_color, parent=selected_annotation)
    pos[selected_bin] = (0, 0)

    # Add bin nodes to the graph
    for annotation in contacts_annotation.unique():
        annotation_bins = contacts_bins[original_contacts_annotation == annotation]
        bin_positions = arrange_nodes(annotation_bins, distance=40, center_position=pos[annotation])
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
    
    bin_data = pd.DataFrame({
        'name': contacts_bins, 
        'value': bin_contact_values, 
        'color': [G.nodes[bin]['color'] for bin in contacts_bins],
        'hover': [f"({bin_information.loc[bin_information['Bin index'] == bin, taxonomy_level].values[0]}, {value})" for bin, value in zip(contacts_bins, bin_contact_values)]
    }).query('value != 0')
    if not bin_data.empty:
        data_dict[f'Hi-C Contacts with {selected_bin}'] = bin_data

    unique_annotations = contacts_annotation.unique()
    annotation_data = pd.DataFrame({
        'name': unique_annotations,
        'value': contact_values,
        'color': [G.nodes[annotation]['border_color'] for annotation in unique_annotations]
    }).query('value != 0')
    if not annotation_data.empty:
        data_dict[f'Hi-C Contacts with {selected_annotation}'] = annotation_data

    bar_fig = create_bar_chart(data_dict)
    print(2)
    return cyto_elements, bar_fig

# Create a logger and add the custom Dash logger handler
logger = logging.getLogger("app_logger")
 
type_colors = {
    'chromosome': '#81BFDA',
    'virus': '#AE445A',
    'plasmid': '#D5ED9F'
}
default_color = '#808080' 
reds = ['#F49AC2', '#CF71AF', '#FF007F', '#872657', '#E32636', 
           '#C41E3A', '#960018', '#65000B', '#FF9966', '#FF5A36']
greens = ['#BFFF00', '#D1E231', '#A4C639', '#808000', '#4B5320',
          '#B2EC5D', '#00CC99', '#87A96B', '#138808', '#00FF6F']
blues = ['#00FFFF', '#89CFF0', '#008B8B', '#6495ED', '#007FFF',
         '#6082B6', '#1560BD', '#0047AB', '#CCCCFF', '#002FA7']

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
        'width': '300px',
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
                    dcc.Store(id='current-visualization-mode', data={}),

                    dbc.Button("Switch to Nomalization Results", id="switch-visualization-network", color="primary",
                                style={'height': '38px',
                                       'width': '300px',
                                       'display': 'inline-block',
                                       'margin-right': '10px',
                                       'margin-top': '0px',
                                       'vertical-align': 'middle'}),
                    
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
                                    {'label': 'Taxonomic Framework', 'value': 'taxonomy'},
                                    {'label': 'Cross-Taxa Hi-C Interaction', 'value': 'basic'},
                                    {'label': 'Cross-Bin Hi-C Interactions', 'value': 'bin'},
                                ],
                                value='basic',
                                style={'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
                            ),
                            dcc.Dropdown(
                                id='annotation-selector',
                                options=[],
                                value=None,
                                placeholder="Select an annotation",
                                style={'display': 'none'}
                            ),
                            dcc.Dropdown(
                                id='bin-selector',
                                options=[],
                                value=None,
                                placeholder="Select a bin",
                                style={'display': 'none'}
                            )
                        ]
                    ),
                    html.Button("Reset Selection", id="reset-btn", style={**common_text_style}),
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
                                                    'headerPinned': 'top'
                                                }
                                            )
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
                            dcc.Loading(
                                id="loading-spinner",
                                type="default",
                                delay_show=1000,
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
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ],
                        style={'display': 'inline-block', 'vertical-align': 'top', 'height': '85vh', 'width': '49vw'}
                    ),

                    html.Div(
                        id="right-column",
                        children=[
                            html.Div(
                                id="legand-container",
                                children=[
                                    dcc.Dropdown(
                                        id='taxonomy-level-selector',
                                        options=[],
                                        value=None,
                                        placeholder="Select Taxonomy Level",
                                        style={'width': '100%', 'margin-top': '5px', 'margin-bottom': '5px'}
                                    ),
                                    
                                    html.Div(id='legend-div')
                                ], 
                                style={'display': 'inline-block', 'vertical-align': 'top', 'height': '30vh', 'width': '19vw'}
                            ),

                            html.Div(
                                id="bar-chart-container",
                                children=[],
                                style={'display': 'inline-block', 'vertical-align': 'top', 'height': '55vh', 'width': '19vw'}
                            ),
                        ],
                        style={'display': 'inline-block', 'vertical-align': 'top', 'height': '85vh', 'width': '19vw'}
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
                ], 
                style={'display': 'inline-block', 'vertical-align': 'top', 'width': '98vw'}
            ),
        ]
    )
           
def register_visualization_callbacks(app):
    @app.callback(
        [Output('taxonomy-level-selector', 'options'),
         Output('taxonomy-level-selector', 'value'),
         Output('bin-info-table', 'columnDefs'),
         Output('bin-selector', 'options'),
         Output('current-visualization-mode', 'data')],  # Add dcc.Store output for visualization mode
        [Input('user-folder', 'data')]
    )
    def Initialize_selector(user_folder):
        taxonomy_levels = load_from_redis(f'{user_folder}:taxonomy-levels')
    
        # Generate taxonomy options from taxonomy levels
        taxonomy_options = [{'label': level, 'value': level} for level in taxonomy_levels]
        default_taxonomy_level = taxonomy_levels[0]  # Set the last level as the default value
    
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
                    {"headerName": "Bin Size/ Contig length", "field": "Contig length", "width": 150, "wrapHeaderText": True},
                    {"headerName": "Bin / Contig coverage", "field": "Contig coverage", "width": 150, "wrapHeaderText": True},
                    {"headerName": "Number of Connected bins", "field": "Connected bins", "width": 150, "wrapHeaderText": True},
                    {"headerName": "Visibility", "field": "Visibility", "hide": True}
                ]
            }
        ]
    
        # Initialize empty lists for dropdown options
        bin_options = []
    
        # Determine which dataset to use based on selected_tab
        bin_information = load_from_redis(f'{user_folder}:bin-information')
        bins = bin_information['Bin index']
        bin_options = [{'label': bin, 'value': bin} for bin in bins]
    
        # Update visualization mode state
        current_visualization_mode = {
            'visualization_type': 'taxonomy',
            'taxonomy_level': default_taxonomy_level,
            'selected_annotation': None,
            'selected_bin': None
        }
    
        return taxonomy_options, default_taxonomy_level, bin_column_defs, bin_options, current_visualization_mode

    @app.callback(
        [Output('data-loaded', 'data'),
         Output('contact-table', 'rowData'),
         Output('contact-table', 'columnDefs'),
         Output('contact-table', 'defaultColDef'),
         Output('contact-table', 'styleConditions'),
         Output('annotation-selector', 'options'),
         Output('logger-button-visualization', 'n_clicks', allow_duplicate=True),
         Output('current-visualization-mode', 'data', allow_duplicate=True)],
        [Input('taxonomy-level-selector', 'value')],
        [State('user-folder', 'data')],
        prevent_initial_call=True
    )
    def generate_contact_matrix(taxonomy_level, user_folder):
        logger.info("Updating taxonomy level data.")
        # Define Redis keys that incorporate the user_folder to avoid concurrency issues
        bin_matrix_key = f'{user_folder}:bin-dense-matrix'
        bin_info_key = f'{user_folder}:bin-information'
        unique_annotations_key = f'{user_folder}:unique-annotations'
        contact_matrix_key = f'{user_folder}:contact-matrix'
    
        bin_information = load_from_redis(bin_info_key)
        bin_dense_matrix = load_from_redis(bin_matrix_key)
    
        np.fill_diagonal(bin_dense_matrix, 0)
            
        unique_annotations = bin_information[taxonomy_level].unique()
        
        contact_matrix = pd.DataFrame(0.0, index=unique_annotations, columns=unique_annotations)
        bin_indexes_dict = get_indexes(unique_annotations, bin_information, taxonomy_level)

        non_self_pairs = list(combinations(unique_annotations, 2))
        self_pairs = [(x, x) for x in unique_annotations]
        all_pairs = non_self_pairs + self_pairs
        
        results = Parallel(n_jobs=-1)(
            delayed(calculate_submatrix_sum)(pair, bin_indexes_dict, bin_dense_matrix) for pair in all_pairs
        )
        
        for annotation_i, annotation_j, value in results:
            contact_matrix.at[annotation_i, annotation_j] = value
            contact_matrix.at[annotation_j, annotation_i] = value
    
        # Create Annotation Table
        column_defs = [
            {"headerName": "Index", "field": "index", "pinned": "left", "width": 120,
             "sortable": True, "suppressMovable": True, "unSortIcon": True, "sortingOrder": [None]}
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
        save_to_redis(unique_annotations_key, unique_annotations)
        save_to_redis(contact_matrix_key, contact_matrix)
    
        annotation_options = [{'label': annotation, 'value': annotation} for annotation in unique_annotations]
        
        current_visualization_mode = {
            'visualization_type': 'taxonomy',
            'taxonomy_level': taxonomy_level,
            'selected_annotation': None,
            'selected_bin': None
        }
    
        return 1, row_data, column_defs, default_col_def, style_conditions, annotation_options, 1, current_visualization_mode

    @app.callback(
        [Output('bin-info-table', 'rowData'),
         Output('bin-info-table', 'style'),
         Output('bin-info-table', 'filterModel'),
         Output('logger-button-visualization', 'n_clicks', allow_duplicate=True)],
        [Input('reset-btn', 'n_clicks'),
         Input('data-loaded', 'data'),
         Input('annotation-selector', 'value'),
         Input('visibility-filter', 'value')],
        [State('cyto-graph', 'elements'),
         State('user-folder', 'data'),
         State('current-visualization-mode', 'data')],  # Use current-visualization-mode instead of taxonomy_level
        prevent_initial_call=True
    )
    def generate_info_table(reset_clicks, data_loaded, selected_annotation, filter_value, cyto_elements, user_folder, current_visualization_mode):
        if not data_loaded:
            raise PreventUpdate
    
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == 'reset-btn':  
            selected_annotation = None
    
        # Extract taxonomy_level from current_visualization_mode
        taxonomy_level = current_visualization_mode.get('taxonomy_level')
        selected_bin = current_visualization_mode.get('selected_bin')
    
        # Default styles to hide tables initially
        bin_style = {'display': 'none'}
    
        # Apply filtering logic for the selected table and annotation
        def apply_filter_logic(row_data, dense_matrix):
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
        
                # Set visibility based on the current visualization mode
                elif selected_bin:
                    bin_information = load_from_redis(f'{user_folder}:bin-information')
                    selected_bin_index = bin_information[bin_information['Bin index'] == selected_bin].index[0]
            
                    connected_bins = set()
                    for j in range(dense_matrix.shape[0]):
                        if dense_matrix[selected_bin_index, j] != 0:
                            connected_bins.add(bin_information.at[j, 'Bin index'])
            
                    for row in row_data:
                        if row['Bin index'] not in connected_bins and row['Bin index'] != selected_bin:
                            row['Visibility'] = 0
                            
                else:
                    filter_model = {}
                        
            return filter_model, row_data
    
        # Update based on the selected tab and apply filters
        bin_dense_matrix = load_from_redis(f'{user_folder}:bin-dense-matrix')
        bin_information = load_from_redis(f'{user_folder}:bin-information')
    
        bin_style = {'display': 'block', 'height': '52vh'}
    
        # Apply filter logic to all rows
        bin_filter_model, edited_bin_data = apply_filter_logic(bin_information.to_dict('records'), bin_dense_matrix)
        return (edited_bin_data, bin_style, bin_filter_model, 1)
            
    @app.callback(
         Output('bin-info-table', 'defaultColDef'),
        [Input('bin-info-table', 'virtualRowData')],
        [State('cyto-graph', 'elements'),
         State('taxonomy-level-selector', 'value'),
         State('user-folder', 'data'),
         State('data-loaded', 'data')],
        prevent_initial_call=True
    )
    def update_info_table(bin_virtual_row_data, cyto_elements, taxonomy_level, user_folder, data_loaded):
        if not data_loaded:
            raise PreventUpdate
        unique_annotations = load_from_redis(f'{user_folder}:unique-annotations')
        taxonomy_level_list = load_from_redis(f'{user_folder}:taxonomy-levels')
        
        # Bin Table Styling
        if bin_virtual_row_data:
            bin_row_data_df = pd.DataFrame(bin_virtual_row_data)
            bin_colors = get_id_colors(cyto_elements)
            bin_style_conditions = styling_information_table(
                bin_row_data_df, bin_colors, unique_annotations, taxonomy_level_list, taxonomy_level
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
        
        return bin_col_def

    @app.callback(
        [Output('visualization-selector', 'value'),
         Output('annotation-selector', 'value'),
         Output('annotation-selector', 'style'),
         Output('bin-selector', 'value'),
         Output('bin-selector', 'style'),
         Output('logger-button-visualization', 'n_clicks', allow_duplicate=True),
         Output('current-visualization-mode', 'data', allow_duplicate=True)],
        [Input('reset-btn', 'n_clicks'),
         Input('visualization-selector', 'value'),
         Input('annotation-selector', 'value'),
         Input('bin-selector', 'value'),
         Input('contact-table', 'selectedRows'),
         Input('bin-info-table', 'selectedRows'),
         Input('cyto-graph', 'selectedNodeData'),
         Input('data-loaded', 'data')],
        [State('taxonomy-level-selector', 'value'),
         State('user-folder', 'data'),
         State('current-visualization-mode', 'data')]
    )
    def sync_selectors(reset_clicks, visualization_type, selected_annotation, selected_bin,
                       contact_table_selected_rows, bin_info_selected_rows, selected_node_data, data_loaded,
                       taxonomy_level, user_folder, current_visualization_mode):
        if not data_loaded:
            raise PreventUpdate
    
        ctx = callback_context
        triggered_props = [t['prop_id'].split('.')[0] for t in ctx.triggered]

        # Save initial values
        initial_visualization_type = current_visualization_mode.get('visualization_type')
        initial_selected_annotation = current_visualization_mode.get('selected_annotation')
        initial_selected_bin = current_visualization_mode.get('selected_bin')
        initial_taxonomy_level = current_visualization_mode.get('taxonomy_level')
    
        # Initialize the styles as hidden
        bin_information = load_from_redis(f'{user_folder}:bin-information')
        annotation_selector_style = {'display': 'none'}
        bin_selector_style = {'display': 'none'}
    
        # If reset button is clicked
        if 'reset-btn' in triggered_props:
            visualization_type = 'basic'
            selected_annotation = None
            selected_bin = None
    
        # If all triggers are from the selectors, skip redundant logic
        elif all(t in {'visualization-selector', 'annotation-selector', 'bin-selector'} for t in triggered_props):
            pass
    
        # If data-loaded is triggered
        elif 'data-loaded' in triggered_props:
            time.sleep(2)
            visualization_type = visualization_type
            selected_bin = selected_bin
    
        # If cyto-graph node is selected
        elif 'cyto-graph' in triggered_props:
            if selected_node_data:
                selected_node_id = selected_node_data[0]['id']
                if selected_node_id in bin_information['Bin index'].values:
                    visualization_type = 'bin'
                    selected_bin = selected_node_id
                    selected_annotation = None  # Ensure only one selection
                elif selected_node_id in bin_information[taxonomy_level].values:
                    visualization_type = 'basic'
                    selected_annotation = selected_node_id
                    selected_bin = None  # Ensure only one selection
            else:
                selected_annotation = None
                selected_bin = None
    
        # If a row is selected in bin-info-table
        elif 'bin-info-table' in triggered_props:
            if bin_info_selected_rows:
                selected_row = bin_info_selected_rows[0]
                if 'Bin index' in selected_row:
                    visualization_type = 'bin'
                    selected_bin = selected_row['Bin index']
                    selected_annotation = None  # Ensure only one selection
    
        # If a row is selected in contact-table
        elif 'contact-table' in triggered_props:
            if contact_table_selected_rows:
                selected_row = contact_table_selected_rows[0]
                selected_annotation = selected_row['index']
                visualization_type = 'basic'
                selected_bin = None  # Ensure only one selection
    
        else:
            raise PreventUpdate
    
        # Build the current visualization mode
        current_visualization_mode = {
            'visualization_type': visualization_type,
            'taxonomy_level': taxonomy_level,
            'selected_annotation': selected_annotation,
            'selected_bin': selected_bin
        }
    
        # Compare initial and new values
        if (visualization_type == initial_visualization_type and
            selected_annotation == initial_selected_annotation and
            selected_bin == initial_selected_bin and
            taxonomy_level == initial_taxonomy_level):
            # No change in critical outputs, prevent update
            raise PreventUpdate
    
        # Update styles based on selections
        if visualization_type == 'bin':
            bin_selector_style = {'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
            annotation_selector_style = {'display': 'none'}
        elif visualization_type == 'basic':
            annotation_selector_style = {'width': '250px', 'display': 'inline-block', 'margin-top': '4px'}
            bin_selector_style = {'display': 'none'}
        else:
            # Default styles
            annotation_selector_style = {'display': 'none'}
            bin_selector_style = {'display': 'none'}
    
        return (visualization_type, selected_annotation, annotation_selector_style,
                selected_bin, bin_selector_style, 1, current_visualization_mode)

    @app.callback(
        [Output('cyto-graph', 'elements'),
         Output('cyto-graph', 'style'),
         Output('bar-chart-container', 'children'),
         Output('treemap-graph', 'figure'),
         Output('treemap-graph', 'style'),
         Output('cyto-graph', 'stylesheet'),
         Output('cyto-graph', 'layout'),
         Output('legend-div', 'children'),
         Output('logger-button-visualization', 'n_clicks', allow_duplicate=True)],
        [Input('current-visualization-mode', 'data'),
         Input('data-loaded', 'data')],
        [State('visualization-selector', 'value'),
         State('user-folder', 'data')],
        prevent_initial_call=True
    )
    def update_visualization(current_visualization_mode, data_loaded, visualization_type, user_folder):
        if not data_loaded:
            raise PreventUpdate
    
        ctx = callback_context
        triggered_props = [t['prop_id'].split('.')[0] for t in ctx.triggered]
        if 'data-loaded' in triggered_props:
            print(1)
            cyto_elements = []
            return cyto_elements, no_update, no_update, no_update, no_update, no_update, no_update, no_update, 1
        
        logger.info(f"Updating visualization. Triggered by: {triggered_props}")
        logger.info(f"Visualization type: {visualization_type}")
            
        unique_annotations = load_from_redis(f'{user_folder}:unique-annotations')
        taxonomy_level_list = load_from_redis(f'{user_folder}:taxonomy-levels')
    
        selected_nodes = []
        selected_edges = []
        stylesheet = no_update
        layout = no_update
        legend = None
    
        # Extract data from current_visualization_mode
        visualization_type = current_visualization_mode.get('visualization_type', 'taxonomy')
        taxonomy_level = current_visualization_mode.get('taxonomy_level')
        selected_annotation = current_visualization_mode.get('selected_annotation')
        selected_bin = current_visualization_mode.get('selected_bin')
            
        if visualization_type == 'taxonomy':
            contact_matrix = load_from_redis(f'{user_folder}:contact-matrix')
            bin_information = load_from_redis(f'{user_folder}:bin-information')
    
            logger.info("Displaying Taxonomy Framework visualization.")
            treemap_fig, bar_fig = taxonomy_visualization(bin_information, unique_annotations, contact_matrix, taxonomy_level_list)
            treemap_style = {'height': '85vh', 'width': '48vw', 'display': 'inline-block'}
            cyto_elements = []
            cyto_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            
            type_colors = {
                'chromosome': '#81BFDA',
                'virus': '#AE445A',
                'plasmid': '#D5ED9F'
            }
            legend = create_legend_html(type_colors)
    
        elif visualization_type == 'basic':
            bin_information = load_from_redis(f'{user_folder}:bin-information')
            unique_annotations = load_from_redis(f'{user_folder}:unique-annotations')
            contact_matrix = load_from_redis(f'{user_folder}:contact-matrix')
            logger.info("Displaying Taxonomy Interaction.")
            selected_nodes.append(selected_annotation)
    
            if selected_annotation:
                annotation_index = unique_annotations.tolist().index(selected_annotation)
                for i, contact_value in enumerate(contact_matrix.iloc[annotation_index]):
                    if contact_value > 0:
                        connected_annotation = unique_annotations[i]
                        selected_edges.append((selected_annotation, connected_annotation))
    
                cyto_elements, bar_fig, layout = annotation_visualization(
                    bin_information, unique_annotations, contact_matrix, taxonomy_level, selected_node=selected_annotation
                )
            else:
                cyto_elements, bar_fig, layout = annotation_visualization(
                    bin_information, unique_annotations, contact_matrix, taxonomy_level
                )
            treemap_fig = go.Figure()
            treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            cyto_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}
            
            type_colors = {
                'chromosome': '#81BFDA',
                'virus': '#AE445A',
                'plasmid': '#D5ED9F'
            }
            legend = create_legend_html(type_colors)
    
        elif visualization_type == 'bin' and selected_bin:
            print(2)
            bin_dense_matrix = load_from_redis(f'{user_folder}:bin-dense-matrix')
            bin_information = load_from_redis(f'{user_folder}:bin-information')
            selected_nodes.append(selected_bin)
    
            logger.info(f"Displaying bin Interaction for selected bin: {selected_bin}.")
            cyto_elements, bar_fig = bin_visualization(bin_information, unique_annotations, bin_dense_matrix, taxonomy_level, selected_bin)
            treemap_fig = go.Figure()
            treemap_style = {'height': '0vh', 'width': '0vw', 'display': 'none'}
            cyto_style = {'height': '80vh', 'width': '48vw', 'display': 'inline-block'}
            
            id_color_map = get_id_colors(cyto_elements)
            
            if id_color_map:
                color_annotation_map = {}
                unique_colors = {color: node_id for node_id, color in id_color_map.items() if color != '#FFFFFF'}.items()
                for color, node_id in unique_colors:
                    row = bin_information.loc[bin_information['Bin index'] == node_id]
                    if not row.empty:
                        annotation = row[taxonomy_level].iloc[0]
                        color_annotation_map[annotation] = color
                        
                legend = create_legend_html(color_annotation_map)

        else:
            raise PreventUpdate
    
        # Update selected styles and hover info
        stylesheet = add_selection_styles(selected_nodes, selected_edges)
    
        return cyto_elements, cyto_style, bar_fig, treemap_fig, treemap_style, stylesheet, layout, legend, 1
    
    @app.callback(
        Output('bar-chart', 'figure', allow_duplicate=True),
        Input('bar-chart-dropdown', 'value'),
        State('bar-chart', 'figure')
    )
    def update_bar_chart(selected_trace, current_figure):
        if not selected_trace:
            raise PreventUpdate
    
        # Update visibility for the selected trace
        for trace in current_figure['data']:
            trace['visible'] = trace['name'] == selected_trace
    
        return current_figure

    @app.callback(
        Output('visualization-status', 'data', allow_duplicate=True),
        Input('switch-visualization-network', 'n_clicks'),
        prevent_initial_call=True
    )
    def switch_to_result(n_clicks):
        if n_clicks:
            return 'results'
        
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
        [Output('download-btn', 'title'),
         Output('reset-btn', 'title'),
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
                    hover_info['tooltip-toggle-container'],
                    hover_info['dropdowns'],
                    hover_info['bar-chart-container'],
                    hover_info['info-table-container'], 
                    hover_info['treemap-graph-container'],
                    hover_info['cyto-graph-container'],
                    hover_info['contact-table-container'])
        else:
            return ("", "", "", "", "", "", "", "", "")