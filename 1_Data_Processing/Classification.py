import pandas as pd
import numpy as np
from plsdbapi import query
from scipy.sparse import csc_matrix

# **1. contig_info.csv**
file_path = 'input/contig_info.csv'
contig_data = pd.read_csv(file_path)
new_column_names = [
    "Contig name",
    "Number of restriction sites",
    "Contig length",
    "Contig coverage",
    "Hi-C contacts mapped to the same contigs"
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

filtered_data = demo_vir_data.groupby("Contig", as_index=False).apply(filter_contigs).reset_index(drop=True)

filtered_data["Order"] = "o__" + filtered_data["Order"].astype(str)
filtered_data["Order"] = filtered_data["Order"].apply(lambda x: "" if "no_order_" in x else x)
filtered_data["Family"] = "f__" + filtered_data["Family"].astype(str)
filtered_data["classification"] = filtered_data["Order"] + ";" + filtered_data["Family"]
filtered_data.rename(columns={"Contig": "Contig name"}, inplace=True)

demo_vir_data = filtered_data[["Contig name", "classification"]]

# **4. ppr_meta_result.csv**
ppr_meta_file_path = 'input/ppr_meta_result.csv'
ppr_meta_data = pd.read_csv(ppr_meta_file_path)

def classify(row):
    classification = []
    if row['phage_score'] > 0.5:
        classification = "phage"
    elif row['chromosome_score'] > 0.5:
        classification = "chromosome"
    elif row['plasmid_score'] > 0.5:
        classification = "plasmid"
    else:
        classification = "Unmapped"
    return classification

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

unmapped_data = contig_data[contig_data['type'] == "Unmapped"]
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

def adjust_classification(row):
    if row['type'] == 'plasmid' and pd.isna(row['classification']):
        row['E-value'] = 100
        row['Bit Score'] = -100
    return row

plasmid_data = plasmid_data.apply(adjust_classification, axis=1)
plasmid_data_sorted = plasmid_data.sort_values(by=['E-value', 'Bit Score'], ascending=[True, False])
plasmid_data_filtered = plasmid_data_sorted.drop_duplicates(subset=['Contig name'], keep='first')
plasmid_data = plasmid_data_filtered.drop(columns=['plasmid_ID','E-value', 'Bit Score'])

# **9. Combine sub-dataframe**
combined_data = pd.concat([chromosome_data, phage_data, plasmid_data, unmapped_data], ignore_index=True)
combined_data = combined_data.sort_values(by='Contig name').reset_index(drop=True)


prefix_to_tier = {
    'd__': 'domain',
    'p__': 'phylum',
    'c__': 'class',
    'o__': 'order',
    'f__': 'family',
    'g__': 'genus',
    's__': 'species'
}

def split_classification(classification):
    result = {tier: "" for tier in prefix_to_tier.values()}
    
    if isinstance(classification, str):
        components = classification.split(";")
        
        for component in components:
            for prefix, tier in prefix_to_tier.items():
                if component.startswith(prefix):
                    result[tier] = component.split("__")[1]
    
    return pd.Series(result)

tiers = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
combined_data[tiers] = combined_data['classification'].apply(split_classification)
combined_data.loc[combined_data['classification'].isna(), 'type'] = 'Unmapped'

tiers = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

def fill_unspecified(row):
    if row['type'] == 'phage':
        row['domain'] = 'Virus'
        row['phylum'] = 'Virus'
        row['class'] = 'Virus'
        
    last_non_blank = ""

    if row['type'] != 'Unmapped':
        for tier in tiers:
            if row[tier] and row[tier] != "N/A":
                last_non_blank = row[tier]
            else:
                row[tier] = f"Unspecified {last_non_blank}"

    return row

combined_data = combined_data.apply(fill_unspecified, axis=1)
combined_data = combined_data.drop(columns=['Bin', 'classification'])
combined_data['kingdom'] = combined_data['domain'] 
cols = combined_data.columns.tolist()
cols.insert(cols.index('domain') + 1, cols.pop(cols.index('kingdom')))
combined_data = combined_data[cols]

# **10. Remove unmapped contig**
contact_matrix_path= 'input/Normalized_contact_matrix.npz'
contact_matrix_data = np.load(contact_matrix_path)
data = contact_matrix_data['data']
indices = contact_matrix_data['indices']
indptr = contact_matrix_data['indptr']
shape = contact_matrix_data['shape']
sparse_matrix = csc_matrix((data, indices, indptr), shape=shape)
dense_matrix = sparse_matrix.toarray()

unmapped_contigs = combined_data[combined_data['type'] == "Unmapped"].index
final_data = combined_data.drop(unmapped_contigs).reset_index(drop=True)

dense_matrix = np.delete(dense_matrix, unmapped_contigs, axis=0)
dense_matrix = np.delete(dense_matrix, unmapped_contigs, axis=1)

dense_matrix = csc_matrix(dense_matrix)

# Extract the updated components
updated_data = dense_matrix.data
updated_indices = dense_matrix.indices
updated_indptr = dense_matrix.indptr
updated_shape = dense_matrix.shape

final_data_path = 'output/contig_info_final.csv'
dense_matrix_path = 'output/contact_matrix_final.npz'

np.savez_compressed(dense_matrix_path, data=updated_data, indices=updated_indices, indptr=updated_indptr, shape=updated_shape)
final_data.to_csv(final_data_path, index=False)