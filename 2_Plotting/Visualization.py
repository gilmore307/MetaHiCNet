import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from scipy.sparse import csc_matrix
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
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
for annotation_i in unique_annotations:
    for annotation_j in unique_annotations:
        if annotation_i != annotation_j:
            if inter_species_contacts.at[annotation_i, annotation_j] > 0:
                G.add_edge(annotation_i, annotation_j, contacts=inter_species_contacts.at[annotation_i, annotation_j])

# Initial node positions using a force-directed layout with increased dispersion
pos = nx.spring_layout(G, dim=2, k=2, iterations=50)

# Create plotly figure
edge_trace = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    
    trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        line=dict(width=5, color='rgba(0,0,0,0.3)'),  # Fixed width for the edges
        hoverinfo='none',  # Remove hover information
        mode='lines',
        name=f"{edge[0]}-{edge[1]}"
    )
    edge_trace.append(trace)

# Node trace
node_trace = go.Scatter(
    x=[pos[node][0] for node in G.nodes],
    y=[pos[node][1] for node in G.nodes],
    text=['' for node in G.nodes],  # Remove node names
    hoverinfo='none',  # Remove hover information
    mode='markers',
    marker=dict(
        showscale=False,  # Hide the color scale bar
        colorscale='Viridis',
        size=[G.nodes[node]['size'] for node in G.nodes],
        color=[G.nodes[node]['color'] for node in G.nodes],
        sizemode='area',  # Ensure the size changes with zoom
        sizeref=2.*max([G.nodes[node]['size'] for node in G.nodes])/100**2,  # Set sizeref for scaling
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
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    dragmode='pan',  # Enable panning
    newshape=dict(line_color="cyan"),
    modebar_add=['zoom', 'pan', 'resetScale2d'],
    modebar_remove=['select2d', 'lasso2d']
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
        xaxis=dict(showticklabels=False),  # Remove x-axis tick labels
        yaxis=dict(title="Contact Count", showticklabels=False),  # Remove y-axis tick labels
        height=720,  # Adjusted height
        margin=dict(t=30, b=0, l=0, r=0)  # Adjusted margin
    )
    bar_fig = go.Figure(data=[bar_trace], layout=bar_layout)
    return bar_fig

# Function to create annotations for the 2D plot
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
                    text='',  # Remove text
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
        dcc.Dropdown(
            id='visualization-selector',
            options=[
                {'label': 'Species', 'value': 'species'},
                {'label': 'Species Contact', 'value': 'species_contact'},
                {'label': 'Contig', 'value': 'contig'}
            ],
            value='species',
            style={'width': '300px', 'display': 'inline-block'}
        )
    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin': '20px'}),
    html.Div([
        html.Div([
            dcc.Graph(id='bar-chart', style={'height': '80vh', 'width': '30vw', 'display': 'inline-block'})
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            dcc.Graph(id='2d-graph', figure=fig, style={'height': '80vh', 'width': '60vw', 'display': 'inline-block'}, config={'scrollZoom': True})
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

def rearrange_nodes(selected_node):
    num_nodes = len(G.nodes) - 1  # Exclude the selected node
    angles = np.linspace(-np.pi/4, np.pi/4, num_nodes//2).tolist() + np.linspace(3*np.pi/4, 5*np.pi/4, num_nodes//2).tolist()
    
    new_pos = {selected_node: (0, 0)}
    i = 0
    radius = 1  # Radius for placing the nodes around the center
    for node in G.nodes:
        if node != selected_node:
            angle = angles[i]
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            new_pos[node] = (x, y)
            i += 1
    return new_pos

def get_contig_nodes(species):
    contigs = contig_information[contig_information['Contig annotation'] == species]['Contig name']
    return contigs

# Visualization functions
def species_visualization(active_cell, selected_species, table_data):
    row_contig = table_data[active_cell['row']]['Species'] if active_cell else selected_species
    col_contig = active_cell['column_id'] if active_cell else None

    G_copy = G.copy()

    row_contacts = inter_species_contacts.loc[row_contig]
    max_row_contacts = row_contacts.max()

    for node in G_copy.nodes:
        if node == row_contig:
            G_copy.nodes[node]['size'] = 5000
            G_copy.nodes[node]['color'] = 'rgba(0,255,0,0.1)'  # Green with high transparency
        else:
            G_copy.nodes[node]['size'] = (row_contacts[node] / max_row_contacts) * 100

    new_pos = rearrange_nodes(row_contig)

    node_sizes = [G_copy.nodes[node]['size'] for node in G_copy.nodes]
    node_x = [new_pos[node][0] for node in G_copy.nodes]
    node_y = [new_pos[node][1] for node in G_copy.nodes]

    annotations = create_annotations(G_copy, new_pos, row_contig)
    visibility = []
    node_colors = [G_copy.nodes[node]['color'] for node in G_copy.nodes]
    for trace in [node_trace]:
        if 'name' in trace:
            if trace['name'] == 'Nodes':
                trace.marker.color = node_colors
                trace.marker.size = node_sizes
                trace.x = node_x
                trace.y = node_y
                trace.text = ['' for _ in trace.text]  # Remove node names
                trace.hoverinfo = 'none'  # Remove hover information
                visibility.append(True)
                trace.marker.line = dict(width=[5 if node == row_contig else 2 for node in trace.text], color=['green' if node == row_contig else 'black' for node in trace.text])

    new_fig = go.Figure(data=[node_trace], layout=layout)
    new_fig.update_layout(annotations=annotations)
    for i, vis in enumerate(visibility):
        new_fig.data[i].visible = vis

    contigs = get_contig_nodes(row_contig)
    indices = contig_information[contig_information['Contig annotation'] == row_contig].index
    inter_contig_edges = set()

    for i in indices:
        for j in range(dense_matrix.shape[0]):
            if dense_matrix[i, j] != 0 and contig_information.at[j, 'Contig annotation'] != row_contig:
                inter_contig_edges.add(contig_information.at[i, 'Contig name'])
                inter_contig_edges.add(contig_information.at[j, 'Contig name'])

    contig_positions = arrange_contigs(contigs, inter_contig_edges)

    for contig, (x, y) in contig_positions.items():
        color = 'green' if contig in inter_contig_edges else 'yellow'
        new_fig.add_trace(go.Scatter(
            x=[new_pos[row_contig][0] + x],
            y=[new_pos[row_contig][1] + y],
            hoverinfo='none',  # Remove hover information
            mode='markers',
            marker=dict(
                symbol='circle',
                showscale=False,
                colorscale='Viridis',
                size=10,
                color=color,
                line_width=2
            ),
            name='Contigs'
        ))

    bar_fig = create_bar_chart(row_contig, col_contig)

    return new_fig, bar_fig, row_contig

def species_contact_visualization(active_cell, selected_species, secondary_species, table_data):
    if not selected_species or not secondary_species:
        return fig, go.Figure(), selected_species

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
    contig_positions = arrange_contigs(all_contigs, list(), radius=0.5, jitter=0.05)

    for contig, (x, y) in contig_positions.items():
        color = 'green' if contig in inter_contigs_row else 'yellow'
        G_copy.add_node(contig, size=2, color=color)
        new_pos[contig] = (new_pos[row_contig][0] + x, new_pos[row_contig][1] + y) if contig in inter_contigs_row else (new_pos[col_contig][0] + x, new_pos[col_contig][1] + y)

    edge_trace = []
    node_trace = go.Scatter(
        x=[new_pos[node][0] for node in G_copy.nodes],
        y=[new_pos[node][1] for node in G_copy.nodes],
        hoverinfo='none',  # Remove hover information
        mode='markers',
        marker=dict(
            showscale=False,
            colorscale='Viridis',
            size=[G_copy.nodes[node]['size'] for node in G_copy.nodes],
            color=[G_copy.nodes[node]['color'] for node in G_copy.nodes],
            sizemode='area',
            sizeref=2.*max([G_copy.nodes[node]['size'] for node in G_copy.nodes])/450**2,
            line_width=2
        ),
        name='Nodes'
    )

    new_fig = go.Figure(data=edge_trace + [node_trace], layout=layout)
    annotations = create_annotations(G_copy, new_pos, row_contig)
    new_fig.update_layout(annotations=annotations)

    bar_fig = create_bar_chart(row_contig, col_contig)

    return new_fig, bar_fig, row_contig


def contig_visualization(active_cell, selected_species, selected_contig, table_data):
    # Empty function for now
    return go.Figure(), go.Figure(), selected_species

@app.callback(
    [Output('2d-graph', 'figure'),
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

    # Initialize default values for fig and bar_fig
    fig = go.Figure(data=edge_trace + [node_trace], layout=layout)
    bar_fig = go.Figure()
    secondary_species_style = {'display': 'none'}
    contig_selector_style = {'display': 'none'}

    if triggered_id == 'reset-btn' or not active_cell:
        # Reset all selections to show the original plot
        for trace in fig.data:
            if 'name' in trace:
                if trace['name'] == 'Nodes':
                    trace.marker.color = [G.nodes[node]['color'] for node in G.nodes]
                    trace.marker.size = [G.nodes[node]['size'] for node in G.nodes]  # Reset node sizes
                    trace.x = [pos[node][0] for node in G.nodes]  # Reset node positions
                    trace.y = [pos[node][1] for node in G.nodes]  # Reset node positions
                    trace.text = ['' for _ in G.nodes]  # Remove node names
                else:
                    trace['line']['color'] = 'rgba(0,0,0,0.3)'
        return fig, bar_fig, None, 'species', None, secondary_species_style, contig_selector_style

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
        fig, bar_fig, selected_species = species_visualization(active_cell, selected_species, table_data)
        return fig, bar_fig, selected_species, visualization_type, None, secondary_species_style, contig_selector_style
    elif visualization_type == 'species_contact':
        fig, bar_fig, selected_species = species_contact_visualization(active_cell, selected_species, secondary_species, table_data)
        return fig, bar_fig, selected_species, visualization_type, secondary_species, secondary_species_style, contig_selector_style
    elif visualization_type == 'contig':
        fig, bar_fig, selected_species = contig_visualization(active_cell, selected_species, selected_contig, table_data)
        contig_selector_style = {'width': '300px', 'display': 'inline-block'}
        return fig, bar_fig, selected_species, visualization_type, secondary_species, secondary_species_style, contig_selector_style

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
