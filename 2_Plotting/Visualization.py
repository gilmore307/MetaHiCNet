import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from scipy.sparse import csc_matrix
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash import callback_context

# File paths
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

# Form the inter_species_contacts DataFrame with the contacts
for i, annotation_i in enumerate(unique_annotations):
    for j, annotation_j in enumerate(unique_annotations):
        if i != j:
            contacts = dense_matrix[
                contig_information[contig_information['Contig annotation'] == annotation_i].index
            ][:, 
                contig_information[contig_information['Contig annotation'] == annotation_j].index
            ].sum()
            inter_species_contacts.at[annotation_i, annotation_j] = contacts

# Create a graph for the Arc Diagram
G = nx.Graph()

# Add nodes with size based on intra-species contacts
max_intra_species_contacts = intra_species_contacts.max()
is_viral_colors = {'True': 'red', 'False': 'blue'}
for annotation, contacts in intra_species_contacts.items():
    is_viral = contig_information.loc[
        contig_information['Contig annotation'] == annotation, 'Is Viral'
    ].values[0]
    color = is_viral_colors[str(is_viral)]
    size = (contacts / max_intra_species_contacts) * 100  # Scale node size for plotly
    G.add_node(annotation, size=size, color=color, contacts=contacts)

# Add edges with weight based on inter-species contacts
max_inter_species_contacts = inter_species_contacts.values.max()
for annotation_i in unique_annotations:
    for annotation_j in unique_annotations:
        if annotation_i != annotation_j:
            weight = inter_species_contacts.at[annotation_i, annotation_j]
            if weight > 0:
                scaled_weight = (weight / max_inter_species_contacts) * 100  # Increase edge width
                G.add_edge(annotation_i, annotation_j, weight=scaled_weight, contacts=weight)

# Calculate node positions using a force-directed layout with increased dispersion
pos = nx.spring_layout(G, dim=3, k=2, iterations=50)

# Create plotly figure
edge_trace = []
for edge in G.edges(data=True):
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    
    node1_size = G.nodes[edge[0]]['size']
    node2_size = G.nodes[edge[1]]['size']
    
    # Calculate the gradient width
    start_width = node1_size / 2
    end_width = node2_size / 2
    
    # Create a gradient width between node1 and node2
    num_points = 15
    x_values = np.linspace(x0, x1, num_points)
    y_values = np.linspace(y0, y1, num_points)
    z_values = np.linspace(z0, z1, num_points)
    widths = np.linspace(start_width, end_width, num_points)
    
    # Create individual segments for the gradient effect
    for i in range(num_points - 1):
        trace = go.Scatter3d(
            x=[x_values[i], x_values[i + 1], None],
            y=[y_values[i], y_values[i + 1], None],
            z=[z_values[i], z_values[i + 1], None],
            line=dict(width=widths[i], color='rgba(0,0,0,0.3)'),
            hoverinfo='text',
            text=f"{edge[0]} - {edge[1]}: {edge[2]['contacts']}",
            mode='lines',
            name=f"{edge[0]}-{edge[1]}"
        )
        edge_trace.append(trace)

# Node trace
node_trace = go.Scatter3d(
    x=[pos[node][0] for node in G.nodes],
    y=[pos[node][1] for node in G.nodes],
    z=[pos[node][2] for node in G.nodes],
    text=[node for node in G.nodes],
    hovertext=[f"{node}: {G.nodes[node]['contacts']}" for node in G.nodes],
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        showscale=False,  # Hide the color scale bar
        colorscale='Viridis',
        size=[G.nodes[node]['size'] for node in G.nodes],
        color=[G.nodes[node]['color'] for node in G.nodes],
        line_width=2
    ),
    name='Nodes'
)

# Layout for the Arc Diagram
layout = go.Layout(
    title={'text': 'MetaHi-C Visualization', 'x': 0.5},
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    )
)

fig = go.Figure(data=edge_trace + [node_trace], layout=layout)

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
        xaxis=dict(showticklabels=True),
        yaxis=dict(title="Contact Count"),
        height=720,  # Adjusted height
        margin=dict(t=30, b=0, l=0, r=0)  # Adjusted margin
    )
    bar_fig = go.Figure(data=[bar_trace], layout=bar_layout)
    return bar_fig

# Function to create annotations for the 3D plot
def create_annotations(G, pos, selected_node):
    annotations = []
    if '-' in selected_node:
        return annotations
    neighbors = set(G.neighbors(selected_node))
    neighbors.add(selected_node)
    for node in G.nodes:
        if node not in neighbors:
            x, y = pos[node][0], pos[node][1]
            annotations.append(
                dict(
                    x=x, y=y,
                    text=node,
                    showarrow=False,
                    font=dict(color="lightgrey")
                )
            )
    return annotations

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

def create_heatmap(species1, species2):
    # Get the combined matrix for the selected species
    matrix_df = get_combined_matrix(species1, species2)
    
    # Sort contigs based on annotation
    contig_subset = contig_information[contig_information['Contig annotation'].isin([species1, species2])]
    sorted_contig_subset = contig_subset.sort_values(by='Contig annotation')
    sorted_contig_names = sorted_contig_subset['Contig name']
    
    # Reorder the matrix based on sorted contig names
    sorted_matrix_df = matrix_df.loc[sorted_contig_names, sorted_contig_names]
    
    # Log transform the matrix values
    log_matrix_df = np.log1p(sorted_matrix_df)
    
    # Create heatmap
    fig = px.imshow(log_matrix_df, color_continuous_scale='Reds', labels=dict(color='Log(Contact Count)'))
    
    # Update x-axis and y-axis to indicate species ranges
    species1_range = len(sorted_contig_subset[sorted_contig_subset['Contig annotation'] == species1])
    species2_range = len(sorted_contig_subset) - species1_range
    
    # Add ticks to divided the axises into two parts
    fig.update_xaxes(tickvals=[0, species1_range - 1, species1_range + species2_range - 1],
                     ticktext=[species1, species2, ''])
    fig.update_yaxes(tickvals=[0, species1_range - 1, species1_range + species2_range - 1],
                     ticktext=[species1, species2, ''])
    
    # Working on add indication bar
    fig.update_layout(
        title=f'Contact Matrix for {species1} and {species2}',
        xaxis_title=None,
        yaxis_title=None,
        autosize=True,
        margin=dict(t=40, b=40, l=40, r=40),
        annotations=[
            dict(
                x=species1_range / 2, y=1.1,
                xref='x', yref='paper',
                text=species1,
                showarrow=False,
                font=dict(size=12, color='black'),
                align='center'
            ),
            dict(
                x=species1_range + species2_range / 2, y=1.1,
                xref='x', yref='paper',
                text=species2,
                showarrow=False,
                font=dict(size=12, color='black'),
                align='center'
            ),
            dict(
                x=-0.1, y=species1_range / 2,
                xref='paper', yref='y',
                text=species1,
                showarrow=False,
                font=dict(size=12, color='black'),
                align='center',
                textangle=-90
            ),
            dict(
                x=-0.1, y=species1_range + species2_range / 2,
                xref='paper', yref='y',
                text=species2,
                showarrow=False,
                font=dict(size=12, color='black'),
                align='center',
                textangle=-90
            )
        ]
    )
    
    return fig

# Convert inter_species_contacts and intra_species_contacts to a DataFrame
inter_intra_species_df = inter_species_contacts.copy()
for annotation in unique_annotations:
    inter_intra_species_df.at[annotation, annotation] = intra_species_contacts.get(annotation, 0)

inter_intra_species_df.insert(0, 'Species', inter_intra_species_df.index)
inter_intra_species_df = inter_intra_species_df.rename(columns={'Contig annotation': 'Species'})

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
        dcc.Download(id="download-dataframe-csv")
    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin': '20px'}),
    html.Div([
        html.Div([
            dcc.Graph(id='bar-chart', style={'height': '80vh', 'width': '30vw', 'display': 'inline-block'})
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            dcc.Graph(id='3d-graph', figure=fig, style={'height': '80vh', 'width': '60vw', 'display': 'inline-block'})
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
    html.Div([
        dcc.Graph(id='heatmap', style={'height': '80vh', 'width': '100vw'})
    ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
    help_modal
], style={'height': '100vh', 'overflowY': 'auto', 'width': '100%'})

# Update the layout object with the new title
layout = go.Layout(
    title={'text': 'MetaHi-C Visualization', 'x': 0.5},
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    )
)

# Updated reset button callback and layout update code
@app.callback(
    [Output('3d-graph', 'figure'),
     Output('bar-chart', 'figure')],
    [Input('contact-table', 'active_cell'),
     Input('reset-btn', 'n_clicks')],
    [State('contact-table', 'data')]
)
def update_figure(active_cell, reset_clicks, table_data):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'reset-btn' or not active_cell:
        # Reset all selections
        for trace in fig.data:
            if 'name' in trace:
                if trace['name'] == 'Nodes':
                    trace.marker.color = [G.nodes[node]['color'] for node in G.nodes]
                else:
                    trace['line']['color'] = 'rgba(0,0,0,0.3)'
        return fig, go.Figure()

    row_contig = table_data[active_cell['row']]['Species']
    col_contig = active_cell['column_id']

    if row_contig in pos and col_contig in pos:
        center_x, center_y, center_z = np.mean([pos[row_contig], pos[col_contig]], axis=0)
    else:
        center_x, center_y, center_z = 0, 0, 0

    annotations = create_annotations(G, pos, row_contig)
    visibility = []
    node_colors = [G.nodes[node]['color'] for node in G.nodes]
    for trace in fig.data:
        if 'name' in trace:
            if trace['name'] == 'Nodes': #for node
                for i, node in enumerate(trace.text):
                    if node == row_contig:
                        node_colors[i] = 'green'
                    elif node == col_contig:
                        node_colors[i] = 'yellow'
                trace.marker.color = node_colors
                visibility.append(True)
            else: #for edge
                node1, node2 = trace['name'].split("-")
                if node1 == row_contig or node2 == row_contig:
                    visibility.append(True)
                    trace['line']['color'] = 'rgba(0,128,0,0.8)' 
                elif node1 == col_contig or node2 == col_contig:
                    visibility.append(True)
                    trace['line']['color'] = 'rgba(255,165,0,0.8)'
                else:
                    visibility.append(False)

    # Highlight the selected edge with a different color
    for trace in fig.data:
        if trace.name == f"{row_contig}-{col_contig}" or trace.name == f"{col_contig}-{row_contig}":
            trace['line']['color'] = 'rgba(128,0,128,0.8)'  # Highlight color (purple)

    new_fig = go.Figure(data=fig.data, layout=layout)
    new_fig.update_layout(annotations=annotations)
    for i, vis in enumerate(visibility):
        new_fig.data[i].visible = vis

    new_fig.update_layout(scene_camera=dict(
        center=dict(x=center_x, y=center_y, z=center_z),
        eye=dict(x=center_x, y=center_y, z=center_z)
    ))

    # Create bar chart
    bar_fig = create_bar_chart(row_contig, col_contig)

    return new_fig, bar_fig

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-btn", "n_clicks")],
    [State('contact-table', 'active_cell'), State('contact-table', 'data')],
    prevent_initial_call=True
)
def download_csv(n_clicks, active_cell, table_data):
    if not active_cell:
        return None
    row_contig = table_data[active_cell['row']]['Species']
    col_contig = active_cell['column_id']

    matrix_df = get_combined_matrix(row_contig, col_contig)

    return dcc.send_data_frame(matrix_df.to_csv, f"{row_contig}_{col_contig}_matrix.csv")

@app.callback(
    Output("modal", "is_open"),
    [Input("open-help", "n_clicks"), Input("close-help", "n_clicks")],
    [State("modal", "is_open")]
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Callback to generate heatmap when a cell in the matrix is selected
@app.callback(
    Output('heatmap', 'figure'),
    [Input('contact-table', 'active_cell')],
    [State('contact-table', 'data')]
)
def update_heatmap(active_cell, table_data):
    if not active_cell:
        return go.Figure()

    row_contig = table_data[active_cell['row']]['Species']
    col_contig = active_cell['column_id']

    if row_contig and col_contig:
        return create_heatmap(row_contig, col_contig)
    else:
        return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
