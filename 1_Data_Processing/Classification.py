import pandas as pd
from plsdbapi import query

# **1. contig_info.csv**
file_path = 'contig_info.csv'
contig_data = pd.read_csv(file_path)
new_column_names = [
    "Contig name",
    "Column A",
    "Column B",
    "Column C",
    "Column D"
]
contig_data.columns = new_column_names

# **2. binned_contig.txt**
binned_file_path = 'binned_contig.txt'
binned_contig_data = pd.read_csv(binned_file_path, sep='\t', header=None)
binned_contig_data.columns = ["Contig name", "Bin"]

# **3. DemoVir_assignments.txt**
demo_vir_file_path = 'DemoVir_assignments.txt'
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

filtered_data["Order"] = "o_" + filtered_data["Order"].astype(str)
filtered_data["Order"] = filtered_data["Order"].apply(lambda x: "" if "no_order_" in x else x)
filtered_data["Family"] = "f_" + filtered_data["Family"].astype(str)
filtered_data["classification"] = filtered_data["Order"] + ";" + filtered_data["Family"]
filtered_data.rename(columns={"Contig": "Contig name"}, inplace=True)

demo_vir_data = filtered_data[["Contig name", "classification"]]

# **4. ppr_meta_result.csv**
ppr_meta_file_path = 'ppr_meta_result.csv'
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
query_plasmid_file_path = 'query_plasmid.txt'
query_plasmid_data = pd.read_csv(query_plasmid_file_path, sep='\t', header=None)
column_names = ["Contig name", "plasmid_ID", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8", "Column9", "Column10", "E-value", "Bit Score"]
query_plasmid_data.columns = column_names
query_plasmid_data = query_plasmid_data[["Contig name", "plasmid_ID","E-value", "Bit Score"]]

# **6. metacc.gtdbtk.bac120.summary.tsv**
metacc_file_path = 'metacc.gtdbtk.bac120.summary.tsv'
metacc_data = pd.read_csv(metacc_file_path, sep='\t')
metacc_data.rename(columns={"user_genome": "Bin"}, inplace=True)
metacc_data = metacc_data.iloc[:, :2]

# **7. Getting classifications**
contig_data = pd.merge(contig_data, binned_contig_data, on="Contig name", how="left")
contig_data = pd.merge(contig_data, ppr_meta_data, on="Contig name", how="left")

chromosome_data = contig_data[contig_data['type'] == "chromosome"]
chromosome_data = pd.merge(chromosome_data, metacc_data, on="Bin", how="left")


phage_data = contig_data[contig_data['type'] == "phage"]
phage_data = pd.merge(phage_data, demo_vir_data, on="Contig name", how="left")

unmapped_data = contig_data[contig_data['type'] == "Unmapped"]
unmapped_data['classification'] = "Unmapped"

# **8. Query classification for plasmids**
plasmid_data = contig_data[contig_data['type'] == "plasmid"]
plasmid_data = pd.merge(plasmid_data, query_plasmid_data, on="Contig name", how="left")
plasmid_ids = plasmid_data['plasmid_ID'].dropna().unique().tolist()
plasmid_classification_df = query.query_plasmid_id(plasmid_ids)[['NUCCORE_ACC', 'TAXONOMY_superkingdom', 'TAXONOMY_phylum', 'TAXONOMY_class', 'TAXONOMY_order', 'TAXONOMY_family', 'TAXONOMY_genus', 'TAXONOMY_species']]

# Processing the taxonomy information
taxonomy_columns = ['TAXONOMY_superkingdom', 'TAXONOMY_phylum', 'TAXONOMY_class', 'TAXONOMY_order', 'TAXONOMY_family', 'TAXONOMY_genus', 'TAXONOMY_species']
taxonomy_prefixes = {
    'TAXONOMY_superkingdom': 'd_',
    'TAXONOMY_phylum': 'p_',
    'TAXONOMY_class': 'c_',
    'TAXONOMY_order': 'o_',
    'TAXONOMY_family': 'f_',
    'TAXONOMY_genus': 'g_',
    'TAXONOMY_species': 's_'
}

for column, prefix in taxonomy_prefixes.items():
    plasmid_classification_df[column] = plasmid_classification_df[column].apply(lambda x: prefix + x.split(' (')[0].split('_')[-1] if pd.notna(x) and x != '' else '')

# Combine taxonomy columns into a classification string
plasmid_classification_df['combined_taxonomy'] = plasmid_classification_df[taxonomy_columns].apply(lambda row: ';'.join(filter(None, row)), axis=1)
plasmid_classification_df = plasmid_classification_df[['NUCCORE_ACC','combined_taxonomy']]
plasmid_classification_df.columns = ['plasmid_ID','combined_taxonomy']

# Merging plasmid classification data with the original plasmid data
plasmid_data = pd.merge(plasmid_data, plasmid_classification_df, on='plasmid_ID', how='left')
plasmid_data = plasmid_data.drop(columns=['plasmid_ID'])


'''
# Adjust classification for plasmid rows
plasmid_data['classification'] = plasmid_data.apply(
    lambda row: 'Unmapped' if pd.isna(row['combined_taxonomy']) else row['classification'], axis=1
)

# Step 3: Combining all sub-dataframes back together
#combined_data = pd.concat([phage_data, plasmid_data, unmapped_data, contig_data[~contig_data['type'].isin(['phage', 'plasmid', 'Unmapped'])]])

# Step 4: Final classification adjustments
#combined_data['classification'] = combined_data['classification'].apply(
    #lambda x: "Unmapped" if pd.isna(x) or "Unclassified" in x else x
#)

# combined_data now contains the processed and classified contig data based on the logic described.

# Dropping unnecessary columns
#final_data = combined_data.drop(columns=['Bin', 'type', 'classification_plasmid', 'classification_vir', 'combined_taxonomy'])

# Saving the final data to CSV
#final_data_file_path = 'contig_info_final.csv'
#final_data.to_csv(final_data_file_path, index=False)
'''