import pandas as pd
import numpy as np
from plsdbapi import query
from scipy.sparse import csc_matrix
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# **Helper Functions**
def get_contig_indexes(annotations, contig_information):
    num_threads = 4 * os.cpu_count()
    
    if isinstance(annotations, str):
        annotations = [annotations]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for annotation in annotations:
            futures[executor.submit(
                lambda ann: (ann, contig_information[contig_information['Binning information'] == ann].index.tolist()),
                annotation)] = annotation
        
        contig_indexes = {}

        for future in futures:
            annotation = futures[future]
            try:
                annotation, indexes = future.result()
                contig_indexes[annotation] = indexes
            except Exception as e:
                print(f'Error fetching contig indexes for annotation: {annotation}, error: {e}')
        
    if len(contig_indexes) == 1:
        return list(contig_indexes.values())[0]

    return contig_indexes

def filter_contigs(group):
    
    if len(group) > 1:
        # Rule 1: Highest Percent_of_votes
        group = group[group["Percent_of_votes"] == group["Percent_of_votes"].max()]
        group = group[group["Percent_of_votes.1"] == group["Percent_of_votes.1"].max()]
    
    if len(group) > 1:
        # Rule 2: Completeness of the Sequence
        group = group[group["Partial"] == 'partial=00']
    
    if len(group) > 1:
        # Rule 3: Length of the Sequence
        group = group.sort_values(by="Length", ascending=False)

    
    if len(group) > 1:
        # Rule 4: Consistency of Taxonomic Assignment
        group = group[(~group["Order"].str.contains("no_order")) & (~group["Family"].str.contains("no_family"))]
    
    return group.head(1)

def classify(row):
    classification = []
    if row['phage_score'] > 0.5:
        classification = "phage"
    elif row['chromosome_score'] > 0.5:
        classification = "chromosome"
    elif row['plasmid_score'] > 0.5:
        classification = "plasmid"
    else:
        classification = "unmapped"
    return classification

def adjust_classification(row):
    if row['type'] == 'plasmid' and pd.isna(row['classification']):
        row['E-value'] = 100
        row['Bit Score'] = -100
    return row

def split_classification(classification):
    result = {tier: "" for tier in prefix_to_tier.values()}
    
    if isinstance(classification, str):
        components = classification.split(";")
        
        for component in components:
            for prefix, tier in prefix_to_tier.items():
                if component.startswith(prefix):
                    result[tier] = component.split("__")[1]
    
    return pd.Series(result)

def adjust_taxonomy(row):
    last_non_blank = ""

    if row['type'] != 'unmapped':
        for tier in tiers:
            if row[tier] and row[tier] != "N/A":
                last_non_blank = row[tier]
            else:
                row[tier] = f"Unspecified {last_non_blank}"
    else:
        for tier in tiers:
            row[tier] = "unmapped"
            
    if row['type'] == 'phage':
        row['Domain'] = 'Virus'
        row['Phylum'] = 'Virus'
        row['Class'] = 'Virus'
        for tier in tiers:
            row[tier] = row[tier] + '_v'
        row['Contig name'] = row['Contig name'] + "_v"
        
    if row['type'] == 'plasmid':
        # Add suffix '_p' to all taxonomy levels for plasmids
        for tier in tiers:
            row[tier] = row[tier] + '_p'
        row['Contig name'] = row['Contig name'] + "_p"
    
    return row

# **1. contig_info.csv**
file_path = 'input/contig_info.csv'
contig_data = pd.read_csv(file_path)
new_column_names = [
    "Contig name",
    "Restriction sites",
    "Contig length",
    "Contig coverage",
    "Intra-contig contact"
]
contig_data.columns = new_column_names

# **2. binned_contig.txt**
binned_file_path = 'input/binned_contig.txt'
binned_contig_data = pd.read_csv(binned_file_path, sep='\t', header=None)
binned_contig_data.columns = ["Contig name", "Bin"]

# **3. DemoVir_assignments.txt**
demo_vir_file_path = 'input/DemoVir_assignments.txt'
demo_vir_data = pd.read_csv(demo_vir_file_path, sep='\t')
split_columns = demo_vir_data['Sequence_ID'].str.split(r'[ #;]+', expand=True)
split_columns.columns = [
    'Contig', 'Start', 'End', 'Strand', 'ID', 
    'Partial', 'Start_Type', 'RBS_Motif', 'RBS_Spacer', 'GC_Content'
]

demo_vir_data = pd.concat([demo_vir_data, split_columns], axis=1)
demo_vir_data['Contig'] = demo_vir_data['Contig'].apply(lambda x: '_'.join(x.split('_')[:2]))
demo_vir_data['Start'] = demo_vir_data['Start'].astype(int)
demo_vir_data['End'] = demo_vir_data['End'].astype(int)
demo_vir_data['Length'] = demo_vir_data['End'] - demo_vir_data['Start']

filtered_vir_data = demo_vir_data.groupby("Contig", as_index=False).apply(filter_contigs).reset_index(drop=True)

filtered_vir_data["Order"] = "o__" + filtered_vir_data["Order"].astype(str)
filtered_vir_data["Order"] = filtered_vir_data["Order"].apply(lambda x: "" if "no_order_" in x else x)
filtered_vir_data["Family"] = "f__" + filtered_vir_data["Family"].astype(str)
filtered_vir_data["classification"] = filtered_vir_data["Order"] + ";" + filtered_vir_data["Family"]
filtered_vir_data.rename(columns={"Contig": "Contig name"}, inplace=True)

demo_vir_data = filtered_vir_data[["Contig name", "classification"]]

# **4. ppr_meta_result.csv**
ppr_meta_file_path = 'input/ppr_meta_result.csv'
ppr_meta_data = pd.read_csv(ppr_meta_file_path)

ppr_meta_data["type"] = ppr_meta_data.apply(classify, axis=1)
ppr_meta_data.rename(columns={"Header": "Contig name"}, inplace=True)
ppr_meta_data = ppr_meta_data[["Contig name", "type"]]

# **5. query_plasmid.txt**
query_plasmid_file_path = 'input/query_plasmid.txt'
query_plasmid_data = pd.read_csv(query_plasmid_file_path, sep='\t', header=None)
column_names = ["Contig name", "plasmid_ID", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8", "Column9", "Column10", "E-value", "Bit Score"]
query_plasmid_data.columns = column_names
query_plasmid_data = query_plasmid_data[["Contig name", "plasmid_ID","E-value", "Bit Score"]]

# **6. metacc.gtdbtk.bac120.summary.tsv**
metacc_file_path = 'input/metacc.gtdbtk.bac120.summary.tsv'
metacc_data = pd.read_csv(metacc_file_path, sep='\t')
metacc_data.rename(columns={"user_genome": "Bin"}, inplace=True)
metacc_data = metacc_data.iloc[:, :2]

# **7. Getting classifications**
contig_data = pd.merge(contig_data, binned_contig_data, on="Contig name", how="left")
contig_data = pd.merge(contig_data, ppr_meta_data, on="Contig name", how="left")

chromosome_data = contig_data[contig_data['type'] == "chromosome"]
chromosome_data = pd.merge(chromosome_data, metacc_data, on="Bin", how="left")
chromosome_data['classification'] = chromosome_data['classification'].apply(
    lambda x: np.nan if isinstance(x, str) and "Unclassified" in x else x
)

phage_data = contig_data[contig_data['type'] == "phage"]
phage_data = pd.merge(phage_data, demo_vir_data, on="Contig name", how="left")

unmapped_data = contig_data[contig_data['type'] == "unmapped"]
unmapped_data['classification'] = np.nan

# **8. Query classification for plasmids**
plasmid_data = contig_data[contig_data['type'] == "plasmid"]
plasmid_data = pd.merge(plasmid_data, query_plasmid_data, on="Contig name", how="left")
plasmid_ids = plasmid_data['plasmid_ID'].dropna().unique().tolist()
plasmid_classification_df = query.query_plasmid_id(plasmid_ids)[['NUCCORE_ACC', 'TAXONOMY_superkingdom', 'TAXONOMY_phylum', 'TAXONOMY_class', 'TAXONOMY_order', 'TAXONOMY_family', 'TAXONOMY_genus', 'TAXONOMY_species']]

# Processing the taxonomy information
taxonomy_columns = ['TAXONOMY_superkingdom', 'TAXONOMY_phylum', 'TAXONOMY_class', 'TAXONOMY_order', 'TAXONOMY_family', 'TAXONOMY_genus', 'TAXONOMY_species']
taxonomy_prefixes = {
    'TAXONOMY_superkingdom': 'd__',
    'TAXONOMY_phylum': 'p__',
    'TAXONOMY_class': 'c__',
    'TAXONOMY_order': 'o__',
    'TAXONOMY_family': 'f__',
    'TAXONOMY_genus': 'g__',
    'TAXONOMY_species': 's__'
}

for column, prefix in taxonomy_prefixes.items():
    plasmid_classification_df[column] = plasmid_classification_df[column].apply(lambda x: prefix + x.split(' (')[0].split('_')[-1] if pd.notna(x) and x != '' else '')

# Combine taxonomy columns into a classification string
plasmid_classification_df['classification'] = plasmid_classification_df[taxonomy_columns].apply(lambda row: ';'.join(filter(None, row)), axis=1)
plasmid_classification_df = plasmid_classification_df[['NUCCORE_ACC','classification']]
plasmid_classification_df.columns = ['plasmid_ID','classification']

plasmid_data = pd.merge(plasmid_data, plasmid_classification_df, on='plasmid_ID', how='left')

plasmid_data = plasmid_data.apply(adjust_classification, axis=1)
plasmid_data_sorted = plasmid_data.sort_values(by=['E-value', 'Bit Score'], ascending=[True, False])
plasmid_data_filtered = plasmid_data_sorted.drop_duplicates(subset=['Contig name'], keep='first')
plasmid_data = plasmid_data_filtered.drop(columns=['plasmid_ID','E-value', 'Bit Score'])

# **9. Combine sub-dataframe**
combined_data = pd.concat([chromosome_data, phage_data, plasmid_data, unmapped_data], ignore_index=True)
combined_data = combined_data.sort_values(by='Contig name').reset_index(drop=True)

prefix_to_tier = {
    'd__': 'Domain',
    'p__': 'Phylum',
    'c__': 'Class',
    'o__': 'Order',
    'f__': 'Family',
    'g__': 'Genus',
    's__': 'Species'
}

tiers = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
combined_data[tiers] = combined_data['classification'].apply(split_classification)
combined_data.loc[combined_data['classification'].isna(), 'type'] = 'unmapped'
combined_data = combined_data.apply(adjust_taxonomy, axis=1)
combined_data = combined_data.drop(columns=['classification'])
combined_data['Kingdom'] = combined_data['Domain'] 
cols = combined_data.columns.tolist()
cols.insert(cols.index('Domain') + 1, cols.pop(cols.index('Kingdom')))
combined_data = combined_data[cols]

# **10. Handle unmapped contigs**
remove_unmapped_choice = input("Do you want to remove unmapped contigs? (y/n): ").strip().lower()
remove_unmapped = remove_unmapped_choice == 'y'

bin_contact_matrix_path= 'input/Normalized_contact_matrix.npz'
bin_contact_matrix_data = np.load(bin_contact_matrix_path)
data = bin_contact_matrix_data['data']
indices = bin_contact_matrix_data['indices']
indptr = bin_contact_matrix_data['indptr']
shape = bin_contact_matrix_data['shape']
sparse_matrix = csc_matrix((data, indices, indptr), shape=shape)
dense_matrix = sparse_matrix.toarray()

if remove_unmapped:
    # Find the positions of unmapped contigs in combined_data
    unmapped_contigs = combined_data[combined_data['type'] == "unmapped"].index.tolist()
    filtered_data = combined_data.drop(unmapped_contigs).reset_index(drop=True)
    
    # Create a mask of rows and columns to keep
    keep_mask = np.ones(dense_matrix.shape[0], dtype=bool)
    keep_mask[unmapped_contigs] = False

    contig_contact_matrix = dense_matrix[keep_mask, :][:, keep_mask]
else:
    filtered_data = combined_data.copy()
    contig_contact_matrix = dense_matrix.copy()

# Generate 'Binning information'
filtered_data['Binning information'] = filtered_data.apply(
    lambda row: row['Bin'] if row['type'] in ['chromosome', 'unmapped'] else row['Contig name'], axis=1
)
filtered_data = filtered_data.drop(columns=['Bin'])

# Replace 'BIN' with 'MAG_' in 'Binning information'
filtered_data['Binning information'] = filtered_data['Binning information'].str.replace('BIN', 'MAG_')

# If 'Binning information' is NaN, replace it with 'Unbinned MAG'
filtered_data['Binning information'] = filtered_data['Binning information'].fillna('Unbinned MAG')

unmapped_contigs = combined_data[combined_data['type'] == "unmapped"].index
final_data = combined_data.drop(unmapped_contigs).reset_index(drop=True)

# **11. Handle host-host interactions**
remove_host_host_choice = input("Do you want to remove host-host interactions? (y/n): ").strip().lower()
remove_host_host = remove_host_host_choice == 'y'

grouped_data = filtered_data.groupby('Binning information').agg({
    'Contig name': lambda x: ', '.join(x),
    'Restriction sites': 'sum',
    'Contig length': 'sum',
    'Contig coverage': 'sum',
    'Intra-contig contact': 'sum',
    'type': 'first',
    'Domain': 'first',
    'Kingdom': 'first',
    'Phylum': 'first',
    'Class': 'first',
    'Order': 'first',
    'Family': 'first',
    'Genus': 'first',
    'Species': 'first'
}).reset_index()

# Define the prefixes for each column
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

# Apply the prefix to each relevant column
for column, prefix in prefixes.items():
    grouped_data[column] = grouped_data[column].apply(lambda x: f"{prefix}{x}" if pd.notna(x) else x)

unique_annotations = grouped_data['Binning information']
contig_indexes_dict = get_contig_indexes(unique_annotations, filtered_data)

host_annotations = grouped_data[grouped_data['type'] == 'chromosome']['Binning information'].tolist()
non_host_annotations = grouped_data[~grouped_data['type'].isin(['chromosome'])]['Binning information'].tolist()

bin_contact_matrix = pd.DataFrame(0.0, index=unique_annotations, columns=unique_annotations)

if remove_host_host:
    # Step 1: Process interactions between non-host annotations
    for annotation_i in tqdm(non_host_annotations, desc="Processing non-host interactions"):
        for annotation_j in non_host_annotations:
            indexes_i = contig_indexes_dict[annotation_i]
            indexes_j = contig_indexes_dict[annotation_j]
            
            # Extract the submatrix and sum the values
            sub_matrix = dense_matrix[np.ix_(indexes_i, indexes_j)]
            bin_contact_matrix.at[annotation_i, annotation_j] = sub_matrix.sum()

    # Step 2: Add interactions between host and non-host annotations
    for annotation_i in tqdm(host_annotations, desc="Processing host-non-host interactions"):
        for annotation_j in non_host_annotations:
            indexes_i = contig_indexes_dict[annotation_i]
            indexes_j = contig_indexes_dict[annotation_j]
            
            # Extract the submatrix and sum the values
            sub_matrix = dense_matrix[np.ix_(indexes_i, indexes_j)]
            bin_contact_matrix.at[annotation_i, annotation_j] = sub_matrix.sum()

            # Symmetric assignment for host-non-host
            bin_contact_matrix.at[annotation_j, annotation_i] = sub_matrix.sum()

else:
    # Original method: process all interactions including host-host
    for annotation_i in tqdm(unique_annotations, desc="Processing contact matrix"):
        for annotation_j in unique_annotations:
            indexes_i = contig_indexes_dict[annotation_i]
            indexes_j = contig_indexes_dict[annotation_j]
            
            # Extract the submatrix and sum the values
            sub_matrix = dense_matrix[np.ix_(indexes_i, indexes_j)]
            bin_contact_matrix.at[annotation_i, annotation_j] = sub_matrix.sum()


# **12. Save outputs**           
# Convert the DataFrame to a sparse matrix for saving
bin_contact_matrix = csc_matrix(bin_contact_matrix)
contig_contact_matrix = csc_matrix(contig_contact_matrix)

matrix_columns = {
    'Contig name': 'Contig',
    'Binning information': 'Bin',
    'type': 'type',
    'Domain': 'Domain',
    'Kingdom': 'Kingdom',
    'Phylum': 'Phylum',
    'Class': 'Class',
    'Order': 'Order',
    'Family': 'Family',
    'Genus': 'Genus',
    'Species': 'Species',
    'Restriction sites': 'Restriction sites',
    'Contig length': 'Length',
    'Contig coverage': 'Coverage',
    'Intra-contig contact': 'Signal'
}
grouped_data = grouped_data.rename(columns=matrix_columns)
grouped_data = grouped_data[list(matrix_columns.values())]
filtered_data = filtered_data.rename(columns=matrix_columns)
filtered_data = filtered_data[list(matrix_columns.values())]

# Extract the updated components
bin_contact_data = bin_contact_matrix.data
bin_contact_indices = bin_contact_matrix.indices
bin_contact_indptr = bin_contact_matrix.indptr
bin_contact_shape = bin_contact_matrix.shape

contig_contact_data = contig_contact_matrix.data
contig_contact_indices = contig_contact_matrix.indices
contig_contact_indptr = contig_contact_matrix.indptr
contig_contact_shape = contig_contact_matrix.shape

grouped_data_path = 'output/bin_info_final.csv'
filtered_data_path = 'output/contig_info_final.csv'
bin_contact_matrix_path = 'output/bin_contact_matrix.npz'
contig_contact_matrix_path = 'output/contig_contact_matrix.npz'

np.savez_compressed(bin_contact_matrix_path, data=bin_contact_data, indices=bin_contact_indices, indptr=bin_contact_indptr, shape=bin_contact_shape)
np.savez_compressed(contig_contact_matrix_path, data=contig_contact_data, indices=contig_contact_indices, indptr=contig_contact_indptr, shape=contig_contact_shape)
grouped_data.to_csv(grouped_data_path, index=False)
filtered_data.to_csv(filtered_data_path, index=False)