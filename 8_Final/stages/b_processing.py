import os
import io
import pandas as pd
import base64
from dash import html, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from helper import save_file_to_user_folder, query_plasmid_id
import logging

# Initialize logger
logger = logging.getLogger("app_logger")

# Helper function to adjust taxonomy
def adjust_taxonomy(row, taxonomy_columns, prefixes):
    last_non_blank = ""
    
    for tier in taxonomy_columns:
        row[tier] = str(row[tier]) if pd.notna(row[tier]) else ""

    if row['Type'] != 'unmapped':
        for tier in taxonomy_columns:
            if row[tier]:
                last_non_blank = row[tier]
            else:
                row[tier] = f"Unspecified {last_non_blank}"
    else:
        for tier in taxonomy_columns:
            row[tier] = "unmapped"

    if row['Type'] == 'phage':
        row['Domain'] = 'Virus'
        row['Phylum'] = 'Virus'
        row['Class'] = 'Virus'
        for tier in taxonomy_columns:
            row[tier] = row[tier] + '_v'
        row['Contig'] = row['Contig'] + "_v"
        row['Bin'] = row['Bin'] + "_v"

    if row['Type'] == 'plasmid':
        # Add suffix '_p' to all taxonomy levels for plasmids
        for tier in taxonomy_columns:
            row[tier] = row[tier] + '_p'
        row['Contig'] = row['Contig'] + "_p"
        row['Bin'] = row['Bin'] + "_p"

    for tier, prefix in prefixes.items():
        row[tier] = f"{prefix}{row[tier]}" if row[tier] else "N/A"

    return row

# Main function to process data
def process_data(contig_info_path, binning_info_path, taxonomy_path, user_folder):
    try:
        logger.info("Starting data processing...")
        
        # **1. Load contig_information.csv**
        contig_data = pd.read_csv(contig_info_path)

        # **2. Load binning_information.csv**
        binning_data = pd.read_csv(binning_info_path)

        # **3. Load taxonomy.csv**
        taxonomy_data = pd.read_csv(taxonomy_path)

        # Query plasmid classification for any available plasmid IDs
        logger.info("Querying plasmid IDs for classification...")
        plasmid_ids = taxonomy_data['Plasmid ID'].dropna().unique().tolist()
        plasmid_classification_df = query_plasmid_id(plasmid_ids)[[
            'NUCCORE_ACC', 
            'TAXONOMY_superkingdom', 
            'TAXONOMY_phylum', 
            'TAXONOMY_class', 
            'TAXONOMY_order', 
            'TAXONOMY_family', 
            'TAXONOMY_genus', 
            'TAXONOMY_species'
        ]]

        # Rename columns to match internal structure
        plasmid_classification_df.rename(columns={
            'NUCCORE_ACC': 'Plasmid ID',
            'TAXONOMY_superkingdom': 'Kingdom',
            'TAXONOMY_phylum': 'Phylum',
            'TAXONOMY_class': 'Class',
            'TAXONOMY_order': 'Order',
            'TAXONOMY_family': 'Family',
            'TAXONOMY_genus': 'Genus',
            'TAXONOMY_species': 'Species'
        }, inplace=True)

        # Define prefixes for taxonomy tiers
        prefixes = {
            'Domain': 'd_',
            'Kingdom': 'k_',
            'Phylum': 'p_',
            'Class': 'c_',
            'Order': 'o_',
            'Family': 'f_',
            'Genus': 'g_',
            'Species': 's_'
        }

        # Replace certain text in the classification dataframe
        plasmid_classification_df = plasmid_classification_df.replace(r"\s*\(.*\)", "", regex=True)
        plasmid_classification_df['Domain'] = plasmid_classification_df['Kingdom']

        # Merge plasmid classification with taxonomy data
        taxonomy_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        taxonomy_data = taxonomy_data.merge(
            plasmid_classification_df[['Plasmid ID'] + taxonomy_columns],
            on='Plasmid ID',
            how='left',
            suffixes=('', '_new')
        )

        # Fill taxonomy columns with new classification where available
        for column in taxonomy_columns:
            taxonomy_data[column] = taxonomy_data[column + '_new'].combine_first(taxonomy_data[column])

        # Drop unnecessary columns after merge
        taxonomy_data = taxonomy_data.drop(columns=['Plasmid ID'] + [col + '_new' for col in taxonomy_columns])

        # **4. Merge contig, binning, and taxonomy data**
        combined_data = pd.merge(contig_data, binning_data, on="Contig", how="left")
        combined_data = pd.merge(combined_data, taxonomy_data, on="Bin", how="left")

        # **5. Apply taxonomy adjustments**
        combined_data = combined_data.apply(lambda row: adjust_taxonomy(row, taxonomy_columns, prefixes), axis=1)

        # Fill missing bins with 'Unbinned MAG'
        combined_data['Bin'] = combined_data['Bin'].fillna('Unbinned MAG')

        # **6. Save the processed contig info as CSV**
        csv_buffer = io.StringIO()
        combined_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Convert csv_content to base64-encoded string with required prefix
        encoded_csv_content = base64.b64encode(csv_content.encode()).decode()
        encoded_csv_content = f"data:text/csv;base64,{encoded_csv_content}"
        
        # Use save_file_to_user_folder to save the processed CSV with encoded content
        save_file_to_user_folder(encoded_csv_content, 'contig_info_complete.csv', user_folder)
        
        logger.info("Data processed successfully, generating preview.")
        return combined_data  # Return the processed DataFrame for preview

    except Exception as e:
        logger.error(f"Error during data processing: {e}; no preview will be generated.")
        return None

# Function to create the processed data preview layout
def create_processed_data_preview(combined_data):
    if not combined_data.empty:
        preview_table = dbc.Table.from_dataframe(combined_data.head(), striped=True, bordered=True, hover=True)
        return html.Div([
            html.H5('Processed Data Preview'),
            preview_table,
            dbc.Button("Next", id="next-button-processing", color="success", className="mt-3")  # Add the Next button
        ])
    return None

# Callback to update the preview of the processed data in the 'Data Processing' stage
def register_processing_callbacks(app):
    @app.callback(
        Output('processing-preview-method1', 'children'),
        [Input('current-stage-method1', 'data')],
        [State('user-folder', 'data')],
        prevent_initial_call=True
    )
    def update_processed_data_preview(stage_method1, user_folder):
        if stage_method1 != 'Data Processing':
            return no_update  # Keeps existing content if stage isn't 'Data Processing'

        # Define paths for the user-uploaded files
        contig_info_path = os.path.join('assets/output', user_folder, 'contig_information.csv')
        binning_info_path = os.path.join('assets/output', user_folder, 'binning_information.csv')
        taxonomy_path = os.path.join('assets/output', user_folder, 'bin_taxonomy.csv')

        # Process data and get the processed DataFrame
        combined_data = process_data(contig_info_path, binning_info_path, taxonomy_path, user_folder)

        # Generate a preview of the processed data
        return create_processed_data_preview(combined_data)

    # Callback to handle the "Next" button click
    @app.callback(
        Output('current-stage-method1', 'data'),
        [Input('next-button-processing', 'n_clicks')],
        [State('current-stage-method1', 'data')],
        prevent_initial_call=True
    )
    def proceed_to_normalization(next_clicks, current_stage):
        if next_clicks is None or current_stage != 'Data Processing':
            raise PreventUpdate
        return 'Normalization'
