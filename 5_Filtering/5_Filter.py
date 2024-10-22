import pandas as pd
import numpy as np
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
                lambda ann: (ann, contig_information[contig_information['Bin'] == ann].index.tolist()),
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



# **1. contig_information.csv**
contig_data_path = 'input/contig_info_complete.csv'
contig_data = pd.read_csv(contig_data_path)

# **2. normalized_contact_matrix.npz**
bin_contact_matrix_path= 'input/normalized_contact_matrix.npz'
bin_contact_matrix_data = np.load(bin_contact_matrix_path)
data = bin_contact_matrix_data['data']
indices = bin_contact_matrix_data['indices']
indptr = bin_contact_matrix_data['indptr']
shape = bin_contact_matrix_data['shape']
sparse_matrix = csc_matrix((data, indices, indptr), shape=shape)
dense_matrix = sparse_matrix.toarray()

# **10. Handle unmapped contigs**
remove_unmapped_choice = input("Do you want to remove unmapped contigs? (y/n): ").strip().lower()
remove_unmapped = remove_unmapped_choice == 'y'

if remove_unmapped:
    # Find the positions of unmapped contigs in contig_data
    unmapped_contigs = contig_data[contig_data['type'] == "unmapped"].index.tolist()
    contig_data = contig_data.drop(unmapped_contigs).reset_index(drop=True)
    
    # Create a mask of rows and columns to keep
    keep_mask = np.ones(dense_matrix.shape[0], dtype=bool)
    keep_mask[unmapped_contigs] = False
    contig_contact_matrix = dense_matrix[keep_mask, :][:, keep_mask]
else:
    contig_data = contig_data.copy()
    contig_contact_matrix = dense_matrix.copy()

# **11. Handle host-host interactions**
remove_host_host_choice = input("Do you want to remove host-host interactions? (y/n): ").strip().lower()
remove_host_host = remove_host_host_choice == 'y'

bin_data = contig_data.groupby('Bin').agg({
    'Contig': lambda x: ', '.join(x),
    'Restriction sites': 'sum',
    'Length': 'sum',
    'Coverage': 'sum',
    'Self-contact': 'sum',
    'type': lambda x: x.mode()[0],
    'Domain': lambda x: x.mode()[0],
    'Kingdom': lambda x: x.mode()[0],
    'Phylum': lambda x: x.mode()[0],
    'Class': lambda x: x.mode()[0],
    'Order': lambda x: x.mode()[0],
    'Family': lambda x: x.mode()[0],
    'Genus': lambda x: x.mode()[0],
    'Species': lambda x: x.mode()[0]
}).reset_index()

unique_annotations = bin_data['Bin']
contig_indexes_dict = get_contig_indexes(unique_annotations, contig_data)

host_annotations = bin_data[bin_data['type'] == 'chromosome']['Bin'].tolist()
non_host_annotations = bin_data[~bin_data['type'].isin(['chromosome'])]['Bin'].tolist()

bin_contact_matrix = pd.DataFrame(0.0, index=unique_annotations, columns=unique_annotations)

if remove_host_host:
    # Step 1: Process interactions between non-host annotations
    for annotation_i in tqdm(non_host_annotations, desc="Processing non-host to non-host interactions"):
        for annotation_j in non_host_annotations:
            indexes_i = contig_indexes_dict[annotation_i]
            indexes_j = contig_indexes_dict[annotation_j]
            
            # Extract the submatrix and sum the values
            sub_matrix = dense_matrix[np.ix_(indexes_i, indexes_j)]
            bin_contact_matrix.at[annotation_i, annotation_j] = sub_matrix.sum()

    # Step 2: Add interactions between host and non-host annotations
    for annotation_i in tqdm(host_annotations, desc="Processing host to non-host interactions"):
        for annotation_j in non_host_annotations:
            indexes_i = contig_indexes_dict[annotation_i]
            indexes_j = contig_indexes_dict[annotation_j]
            
            # Extract the submatrix and sum the values
            sub_matrix = dense_matrix[np.ix_(indexes_i, indexes_j)]
            bin_contact_matrix.at[annotation_i, annotation_j] = sub_matrix.sum()

            # Symmetric assignment for host-non-host
            bin_contact_matrix.at[annotation_j, annotation_i] = sub_matrix.sum()
    
    # Step 3: Remove host-host interactions in contig_contact_matrix using contig_indexes_dict
    host_indexes = []
    for host_annotation in host_annotations:
        host_indexes.extend(contig_indexes_dict[host_annotation])
    
    # Create a mask for host-host contacts to remove them
    host_mask = np.ones(dense_matrix.shape[0], dtype=bool)
    host_mask[host_indexes] = False
    
    # Remove host-host interactions from contig_contact_matrix
    contig_contact_matrix = dense_matrix[host_mask, :][:, host_mask]

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

# Extract the updated components
bin_contact_data = bin_contact_matrix.data
bin_contact_indices = bin_contact_matrix.indices
bin_contact_indptr = bin_contact_matrix.indptr
bin_contact_shape = bin_contact_matrix.shape

contig_contact_data = contig_contact_matrix.data
contig_contact_indices = contig_contact_matrix.indices
contig_contact_indptr = contig_contact_matrix.indptr
contig_contact_shape = contig_contact_matrix.shape

bin_data_path = 'output/bin_info_final.csv'
contig_data_path = 'output/contig_info_final.csv'
bin_contact_matrix_path = 'output/bin_contact_matrix.npz'
contig_contact_matrix_path = 'output/contig_contact_matrix.npz'

np.savez_compressed(bin_contact_matrix_path, data=bin_contact_data, indices=bin_contact_indices, indptr=bin_contact_indptr, shape=bin_contact_shape)
np.savez_compressed(contig_contact_matrix_path, data=contig_contact_data, indices=contig_contact_indices, indptr=contig_contact_indptr, shape=contig_contact_shape)
bin_data.to_csv(bin_data_path, index=False)
contig_data.to_csv(contig_data_path, index=False)