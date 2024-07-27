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

# Calculate node positions using a force-directed layout with increased dispersion
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
        hoverinfo='text',
        text=f"{edge[0]} - {edge[1]}: {edge[2]['contacts']}",
        mode='lines',
        name=f"{edge[0]}-{edge[1]}"
    )
    edge_trace.append(trace)

# Node trace
node_trace = go.Scatter(
    x=[pos[node][0] for node in G.nodes],
    y=[pos[node][1] for node in G.nodes],
    text=[node for node in G.nodes],
    hovertext=[f"{node}: {G.nodes[node]['contacts']}" for node in G.nodes],
    mode='markers+text',
    hoverinfo='text',
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
        xaxis=dict(showticklabels=True),
        yaxis=dict(title="Contact Count"),
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

# Updated reset button callback and layout update code
@app.callback(
    [Output('2d-graph', 'figure'),
     Output('bar-chart', 'figure'),
     Output('species-selector', 'value')],
    [Input('contact-table', 'active_cell'),
     Input('reset-btn', 'n_clicks'),
     Input('species-selector', 'value')],
    [State('contact-table', 'data')]
)
def update_figure(active_cell, reset_clicks, selected_species, table_data):
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
        return fig, go.Figure(), None

    row_contig = table_data[active_cell['row']]['Species'] if triggered_id != 'species-selector' else selected_species
    col_contig = active_cell['column_id'] if triggered_id != 'species-selector' else None

    annotations = create_annotations(G, pos, row_contig)
    visibility = []
    node_colors = [G.nodes[node]['color'] for node in G.nodes]
    for trace in fig.data:
        if 'name' in trace:
            if trace['name'] == 'Nodes':  # for node
                for i, node in enumerate(trace.text):
                    if node == row_contig:
                        node_colors[i] = 'green'
                    elif node == col_contig:
                        node_colors[i] = 'yellow'
                trace.marker.color = node_colors
                visibility.append(True)
            else:  # for edge
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

    # Create bar chart
    bar_fig = create_bar_chart(row_contig, col_contig)

    return new_fig, bar_fig, row_contig

@app.callback(
    Output("modal", "is_open"),
    [Input("open-help", "n_clicks"), Input("close-help", "n_clicks")],
    [State("modal", "is_open")]
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)
